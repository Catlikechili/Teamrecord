r"""
Bộ kiểm thử Trạm Ghi Nguyên Văn Cuộc Họp Khoa Học (`d:\Gravity\Dich\test_suite.py`)
==================================================================================
1. DSP: resample 48k stereo -> 16k mono; lọc 85–7600 Hz giảm mạnh tiếng ù 50 Hz, giữ nguyên dải tiếng nói.
2. Chống ảo giác.
3. Phần cứng WASAPI Loopback.
4. Cắt câu: câu khoa học dài giữ nguyên 1 đoạn; nói liền > 18 s được cắt và không mất âm thanh.
5. Chuẩn hóa thuật ngữ, khử chữ lặp ranh giới, gom mảnh từ Whisper, hợp nhất 2 động cơ.
6. End-to-end câu khoa học (Whisper + Google) + ghi .txt/.md/.jsonl.
7. WAV đối chiếu + xuất .docx/văn bản liền mạch cho phiên giả lập 3 giờ.
8. Giao diện: khởi tạo, ẩn khi chia sẻ màn hình, hiển thị 1 câu, đóng an toàn.
"""

from __future__ import annotations

import asyncio
import ctypes
import datetime
import io
import json
import queue
import sys
import tempfile
import time
import tkinter as tk
import unittest
import wave
from pathlib import Path
from typing import List

import av
import edge_tts
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_engine import (
    AudioChunkPacket,
    AudioDSP,
    AudioLoopbackCapture,
    EngineConfig,
    ScientificVerbatimASR,
    TranscriptSegment,
    TranscriptStore,
    UtteranceSegmenter,
    WavRecorder,
    is_hallucinated_text,
)
from scientific_normalizer import (
    EnsembleArbitrator,
    ScientificTermNormalizer,
    WordInfo,
    build_initial_prompt,
    collapse_repeated_sentences,
    deduplicate_overlap_text,
    group_whisper_words,
)


def synthesize_english_speech_16k(text: str) -> np.ndarray:
    """Tạo giọng đọc tiếng Anh 16 kHz mono float32 bằng Microsoft Edge Neural TTS (cần mạng)."""
    mp3_io = io.BytesIO()

    async def _run():
        comm = edge_tts.Communicate(text, voice="en-US-GuyNeural")
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                mp3_io.write(chunk["data"])

    asyncio.run(_run())
    mp3_io.seek(0)
    container = av.open(mp3_io, format="mp3")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    raw = b"".join(bytes(f.planes[0]) for fr in container.decode(audio=0) for f in resampler.resample(fr))
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


SCI_SENTENCE = (
    "In this study, we performed a Monte Carlo simulation to estimate the standard deviation, "
    "and the regression model achieved an R-squared of 0.87 with a p-value below 0.05."
)


class TestVerbatimScientificRecorder(unittest.TestCase):

    def test_01_dsp_resample_and_scientific_bandpass(self) -> None:
        print("\n[TEST 1] DSP: resample + lọc Butterworth 85–7600 Hz")
        sr_orig = 48000
        t = np.linspace(0, 1.0, sr_orig, endpoint=False)
        stereo = np.column_stack(
            ((np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16), (np.sin(2 * np.pi * 880 * t) * 16000).astype(np.int16))
        ).flatten()
        mono = AudioDSP.pcm16_to_float32(stereo.tobytes(), channels=2)
        self.assertEqual(AudioDSP.resample_to_16khz(mono, sr_orig).size, 16000)

        t16 = np.linspace(0, 2.0, 32000, endpoint=False)
        hum = 0.3 * np.sin(2 * np.pi * 50 * t16)
        voice = 0.3 * np.sin(2 * np.pi * 1000 * t16)
        out_hum = AudioDSP.scientific_enhance(hum.astype(np.float32), preemphasis=0.0)
        out_mix = AudioDSP.scientific_enhance((hum + voice).astype(np.float32), preemphasis=0.0)
        # Đo riêng thành phần 50 Hz và 1 kHz trong tín hiệu trộn bằng FFT
        spec = np.abs(np.fft.rfft(out_mix))
        freqs = np.fft.rfftfreq(out_mix.size, 1 / 16000)
        a50 = spec[np.argmin(np.abs(freqs - 50))]
        a1k = spec[np.argmin(np.abs(freqs - 1000))]
        print(f"  -> Tỉ lệ 1kHz/50Hz sau lọc: {a1k / max(a50, 1e-9):.1f} lần (trước lọc: 1.0)")
        self.assertGreater(a1k / max(a50, 1e-9), 10.0)
        self.assertEqual(out_hum.dtype, np.float32)
        self.assertEqual(AudioDSP.scientific_enhance(np.zeros(10, dtype=np.float32)).size, 10)

    def test_02_anti_hallucination(self) -> None:
        print("\n[TEST 2] Chống ảo giác")
        for h in ["", "Thank you.", "Subtitles by Amara.org", "um", "going to going to going to going to going to going to"]:
            self.assertTrue(is_hallucinated_text(h), h)
        for v in ["The p-value was below the significance threshold.", "We used a randomized controlled trial."]:
            self.assertFalse(is_hallucinated_text(v), v)

    def test_03_wasapi_loopback_probe(self) -> None:
        print("\n[TEST 3] WASAPI Loopback")
        devices = AudioLoopbackCapture.list_loopback_devices()
        print(f"  -> {[d['name'] for d in devices]}")
        self.assertGreaterEqual(len(devices), 1)

    def test_04_segmenter_keeps_long_sentence_and_splits_safely(self) -> None:
        print("\n[TEST 4] Cắt câu theo nhịp nói khoa học")
        cfg = EngineConfig()
        speech = synthesize_english_speech_16k(SCI_SENTENCE)
        out: List[tuple] = []
        seg = UtteranceSegmenter(cfg, on_utterance=lambda a, d, p, n, ov: out.append((a, d, ov)))
        stream = np.concatenate([speech, np.zeros(int(16000 * 1.2), dtype=np.float32)])
        for i in range(0, stream.size, 640):
            seg.process_frame(stream[i:i + 640], "t")
        seg.flush("t")
        print(f"  -> Câu {speech.size / 16000:.1f}s -> {len(out)} đoạn: {[round(x[1], 2) for x in out]}")
        self.assertEqual(len(out), 1, "Câu khoa học dài có ngắt phẩy phải được giữ nguyên 1 đoạn")

        # Nói liền không nghỉ ~38 s (cắt bỏ khoảng lặng đầu/cuối của từng câu rồi nối sát nhau):
        # buộc phải dùng nhánh cắt tại điểm năng lượng thấp nhất; mỗi đoạn <= 18 s và không mất âm thanh.
        active = np.flatnonzero(np.abs(speech) > 0.01)
        trimmed = speech[active[0]: active[-1] + 1]
        long_speech = np.concatenate([trimmed, trimmed, trimmed])
        out2: List[tuple] = []
        seg2 = UtteranceSegmenter(cfg, on_utterance=lambda a, d, p, n, ov: out2.append((a, d, ov)))
        stream2 = np.concatenate([long_speech, np.zeros(int(16000 * 1.2), dtype=np.float32)])
        for i in range(0, stream2.size, 640):
            seg2.process_frame(stream2[i:i + 640], "t")
        seg2.flush("t")
        durs = [round(x[1], 2) for x in out2]
        print(f"  -> Chuỗi liền {long_speech.size / 16000:.1f}s -> các đoạn {durs} (gối đầu: {[x[2] for x in out2]})")
        self.assertGreaterEqual(len(out2), 3, "Chuỗi nói liền > 36 s phải được cắt thành >= 3 đoạn")
        self.assertTrue(all(d <= cfg.max_phrase_sec + 0.1 for d in durs))
        total = sum(x[1] for x in out2)
        self.assertGreaterEqual(total, long_speech.size / 16000, "Không được làm mất âm thanh khi cắt")

    def test_05_normalizer_dedup_grouping_and_ensemble(self) -> None:
        print("\n[TEST 5] Chuẩn hóa thuật ngữ / khử lặp / gom từ / hợp nhất")
        n = ScientificTermNormalizer.normalize("the p value and r squared from the anova, as smith et al reported, co2 in vitro")
        print(f"  -> {n}")
        for term in ("p-value", "R-squared", "ANOVA", "et al.", "CO2", "in vitro"):
            self.assertIn(term, n)
        self.assertNotIn("et al..", ScientificTermNormalizer.normalize("as shown by Smith et al."))

        self.assertEqual(
            deduplicate_overlap_text("we migrated the production database", "the production database to PostgreSQL."),
            "to PostgreSQL.",
        )
        self.assertEqual(deduplicate_overlap_text("results are shown", "the results show"), "the results show")

        grouped = group_whisper_words([(" an", 0.9), (" R", 0.8), ("-squared", 0.64), (" of", 0.95)])
        self.assertEqual([g.text for g in grouped], ["an", "R-squared", "of"])
        self.assertAlmostEqual(grouped[1].probability, 0.64)

        arb = EnsembleArbitrator()
        words = [WordInfo("The", 0.95), WordInfo("casual", 0.30), WordInfo("effect", 0.93), WordInfo("was", 0.97), WordInfo("large.", 0.96)]
        res = arb.merge(words, "The casual effect was large.", [("the causal effect was large", 0.92)])
        print(f"  -> Hợp nhất: '{res.text}' thay thế={res.substitutions}")
        self.assertEqual(res.text, "The causal effect was large.")
        # Whisper tự tin thì KHÔNG bị Google ghi đè
        words2 = [WordInfo("Bayesian", 0.91), WordInfo("inference.", 0.93)]
        res2 = arb.merge(words2, "Bayesian inference.", [("Asian inference", 0.85)])
        self.assertEqual(res2.text, "Bayesian inference.")
        # Google nghe thêm chữ: không tự chèn
        res3 = arb.merge([WordInfo("Results", 0.9), WordInfo("improved.", 0.9)], "Results improved.", [("results really improved", 0.9)])
        self.assertEqual(res3.text, "Results improved.")

        prompt = build_initial_prompt("ai_cs_math", "LoRA, diffusion model", "previous sentence context.")
        self.assertTrue(prompt.endswith("previous sentence context."))
        self.assertIn("LoRA", prompt)

        # Vòng lặp ảo giác của Whisper bị cắt, câu thật giữ nguyên
        loop = [WordInfo(w, 0.9) for w in "The model converged. We'll see you soon. We'll see you soon. We'll see you soon.".split()]
        self.assertEqual(" ".join(w.text for w in collapse_repeated_sentences(loop)), "The model converged. We'll see you soon.")

    def test_05b_skip_when_lagging(self) -> None:
        print("\n[TEST 5B] Cơ chế bỏ qua khi xử lý không kịp")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EngineConfig(transcript_dir=tmp, max_lag_sec=30.0)
            store = TranscriptStore(tmp)
            aq: queue.Queue = queue.Queue()
            uq: queue.Queue = queue.Queue()
            asr = ScientificVerbatimASR(cfg, aq, uq, store)
            now = time.time()
            mk = lambda seq, age, off: AudioChunkPacket(np.zeros(16000 * 5, dtype=np.float32), 5.0, 0.1, now - age, sequence_id=seq, audio_offset_sec=off)
            first = mk(1, 120, 100.0)
            for seq, age, off in ((2, 110, 110.0), (3, 100, 120.0), (4, 2, 220.0)):
                aq.put(mk(seq, age, off))
            kept = asr._skip_if_lagging(first)
            self.assertEqual(kept.sequence_id, 4, "Phải giữ lại câu mới nhất")
            self.assertTrue(aq.empty())
            self.assertEqual(asr.skipped_segments, 3)
            marker = asr.history[-1]
            print(f"  -> {marker.text}")
            self.assertTrue(marker.is_skip)
            self.assertIn("00:01:40", marker.text)          # trỏ đúng vị trí WAV của đoạn đầu tiên bị bỏ
            self.assertIn("[BỎ QUA 3 đoạn", store.txt_path.read_text(encoding="utf-8"))
            # Không trễ -> không bỏ qua
            fresh = mk(5, 1, 230.0)
            self.assertIs(asr._skip_if_lagging(fresh), fresh)
            # Tắt cơ chế (0) -> không bao giờ bỏ qua
            cfg.max_lag_sec = 0.0
            old = mk(6, 500, 240.0)
            self.assertIs(asr._skip_if_lagging(old), old)

    def test_06_end_to_end_scientific_sentence(self) -> None:
        print("\n[TEST 6] End-to-end câu khoa học (mặc định: Whisper base offline, beam 5)")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EngineConfig(transcript_dir=tmp)
            store = TranscriptStore(tmp)
            aq: queue.Queue = queue.Queue()
            uq: queue.Queue = queue.Queue()
            asr = ScientificVerbatimASR(cfg, aq, uq, store)
            capture = AudioLoopbackCapture(cfg, aq)
            speech = synthesize_english_speech_16k(SCI_SENTENCE)
            capture.inject_test_audio(np.concatenate([speech, np.zeros(int(16000 * 1.2), dtype=np.float32)]), "e2e")
            self.assertEqual(aq.qsize(), 1)
            pkt: AudioChunkPacket = aq.get_nowait()
            seg = asr.process_packet_sync(pkt)
            self.assertIsNotNone(seg)
            assert seg is not None
            print(f"  -> Whisper : {seg.whisper_text}")
            print(f"  -> Google  : {seg.google_alternatives[:2]}")
            print(f"  -> KẾT QUẢ : {seg.text}")
            print(f"  -> {seg.engine} | tin cậy {seg.confidence * 100:.0f}% | từ cần rà soát {seg.low_confidence_words} | {seg.latency_ms:.0f} ms cho {pkt.duration_sec:.1f}s âm thanh")
            low = seg.text.lower()
            for term in ("monte carlo", "standard deviation", "r-squared", "p-value", "0.87", "0.05"):
                self.assertIn(term, low, f"Thiếu thuật ngữ '{term}'")
            self.assertIn(seg.text, store.txt_path.read_text(encoding="utf-8"))
            rec = json.loads(store.jsonl_path.read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertEqual(rec["text"], seg.text)
            self.assertIn("whisper_text", rec)

    def test_07_wav_and_three_hour_exports(self) -> None:
        print("\n[TEST 7] WAV đối chiếu + xuất Word/văn bản liền mạch cho phiên 3 giờ")
        with tempfile.TemporaryDirectory() as tmp:
            store = TranscriptStore(tmp)
            rec = WavRecorder(store.wav_path)
            rec.open()
            for _ in range(50):
                rec.write(np.zeros(640, dtype=np.float32) + 0.01)
            # File phải đọc được ngay cả khi CHƯA đóng (mô phỏng tắt đột ngột)
            with wave.open(str(store.wav_path), "rb") as wf:
                self.assertEqual(wf.getnframes(), 50 * 640)
            rec.close()

            base = datetime.datetime(2026, 11, 7, 9, 0, 0)
            segs = []
            for i in range(1080):  # 1 câu / 10 s trong 3 giờ
                st = base + datetime.timedelta(seconds=10 * i)
                segs.append(TranscriptSegment(
                    sequence_id=i + 1, timestamp_start=st, timestamp_end=st + datetime.timedelta(seconds=8),
                    text=f"Sentence {i + 1} reports a p-value of 0.0{i % 10} for the hypothesis.",
                    confidence=0.93, engine="Whisper+Google", latency_ms=2500.0,
                    low_confidence_words=["hypothesis."] if i % 7 == 0 else [], audio_offset_sec=10.0 * i,
                ))
            t0 = time.perf_counter()
            for s in segs:
                store.log_segment(s)
            log_s = time.perf_counter() - t0
            docx = store.export_docx(segs, {"Thời lượng": "03:00:00"})
            clean = store.export_clean_text(segs)
            print(f"  -> Ghi 1080 câu (flush+fsync) trong {log_s:.2f}s | docx {docx.stat().st_size // 1024} KB | txt liền mạch {clean.stat().st_size // 1024} KB")
            self.assertEqual(len(store.jsonl_path.read_text(encoding="utf-8").strip().splitlines()), 1080)
            self.assertIn("hypothesis.[?]", store.md_path.read_text(encoding="utf-8"))
            self.assertGreater(docx.stat().st_size, 20000)

    def test_08_ui_smoke(self) -> None:
        print("\n[TEST 8] Giao diện")
        from main import VerbatimRecorderApp, Win32WindowHelper, WDA_EXCLUDEFROMCAPTURE

        with tempfile.TemporaryDirectory() as tmp:
            root = tk.Tk()
            app = VerbatimRecorderApp(root, auto_start=False)
            root.update()
            app.stealth_share.set(True)
            app._on_stealth()
            self.assertEqual(Win32WindowHelper.get_display_affinity(root), WDA_EXCLUDEFROMCAPTURE)
            now = datetime.datetime.now()
            seg = TranscriptSegment(1, now, now, "The p-value was 0.03.", 0.9, "Whisper-base", 100.0, ["0.03."])
            app.controller.asr.history.append(seg)
            app.controller.ui_queue.put(seg)
            root.update()
            app._poll_ui()
            self.assertIn("The p-value was", app.txt.get("1.0", tk.END))
            self.assertTrue(app.txt.tag_ranges("low"))
            app._closing = True
            app.controller.stop()
            root.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
