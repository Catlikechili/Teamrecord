r"""
Lõi Ghi Nhận Lời Nói Nguyên Văn — Cuộc Họp Khoa Học Tiếng Anh (`d:\Gravity\Dich\core_engine.py`)
=============================================================================================
Mục tiêu: ghi lại lời nói tiếng Anh ổn định, chính xác trong cuộc họp tới 3 tiếng — không dịch, không tóm tắt.

Luồng xử lý:
1. `AudioLoopbackCapture` : Thu âm WASAPI Loopback (khóa 1 thiết bị), ghi song song WAV 16 kHz làm bản đối chiếu.
2. `UtteranceSegmenter`   : Cắt theo câu trọn vẹn (nghỉ >= 0.85 s, tối đa 18 s, cắt tại điểm năng lượng thấp nhất).
3. `ScientificVerbatimASR`: Xử lý tuần tự từng câu bằng Whisper offline (mặc định `base`, beam 5,
                            câu mồi học thuật + hotwords + ngữ cảnh câu trước) kèm 4 lớp chống ảo giác.
                            Tùy chọn kết hợp Google Speech. Nếu máy xử lý không kịp (câu chờ quá
                            `max_lag_sec`), bỏ qua phần tồn đọng và ghi mốc [BỎ QUA] trỏ tới vị trí trong WAV.
4. `TranscriptStore`      : Ghi tức thì (flush + fsync) ra .txt / .md / .jsonl; xuất .docx / văn bản liền mạch.
"""

from __future__ import annotations

import collections
import datetime
import json
import logging
import math
import os
import queue
import re
import socket
import threading
import time
import warnings
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import numpy as np
import requests
import urllib3.util.connection as urllib3_cn
from requests.adapters import HTTPAdapter
from scipy.signal import butter, resample_poly, sosfiltfilt

# Ưu tiên IPv4 (chỉ ảnh hưởng chế độ có Google): IPv6 fallback trên máy này làm mỗi request chậm 0.7–1.4 s.
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET

try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIOWPATCH = True
except ImportError:
    pyaudio = None
    HAS_PYAUDIOWPATCH = False

from scientific_normalizer import (
    DEFAULT_DOMAIN,
    DOMAIN_PROFILES,
    EnsembleArbitrator,
    ScientificTermNormalizer,
    build_hotwords,
    build_initial_prompt,
    collapse_repeated_sentences,
    deduplicate_overlap_text,
    group_whisper_words,
    normalize_sentence,
    prompt_echo_ratio,
)

logger = logging.getLogger("VerbatimRecorder")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# 1. CẤU HÌNH
# ============================================================================

TARGET_SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
LANGUAGE = "en"
GOOGLE_LANGUAGE = "en-US"

HALLUCINATION_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"^\s*(thank\s+you(\s+very\s+much)?|thanks(\s+for\s+watching)?|thank\s+you\s+for\s+watching)[\.\!\?\s]*$",
        re.I,
    ),
    re.compile(r"subtitles?\s+by|amara\.org|transcribed\s+by|translated\s+by", re.I),
    re.compile(r"^\s*(you|bye|bye-bye|okay|ok|uh|um|hmm|ah|oh)[\.\!\?\s]*$", re.I),
    re.compile(r"^\s*[\.\,\!\?\-\_\~\s]+\s*$"),
    re.compile(r"copyright\s+\d{4}|all\s+rights\s+reserved", re.I),
]

RECORDING_PROFILES: Dict[str, Dict[str, float | str]] = {
    "scientific_verbatim": {
        "label": "🔬 Khoa học – trọn câu dài (nghỉ 0.85s, tối đa 18s)",
        "vad_pause_sec": 0.85, "soft_cut_sec": 10.0, "soft_pause_sec": 0.35,
        "max_phrase_sec": 18.0, "min_phrase_sec": 0.45, "pre_roll_sec": 0.35, "overlap_sec": 0.40,
    },
    "standard_verbatim": {
        "label": "📝 Hội thoại – câu vừa (nghỉ 0.6s, tối đa 12s)",
        "vad_pause_sec": 0.60, "soft_cut_sec": 7.0, "soft_pause_sec": 0.30,
        "max_phrase_sec": 12.0, "min_phrase_sec": 0.40, "pre_roll_sec": 0.30, "overlap_sec": 0.35,
    },
}
DEFAULT_RECORDING_PROFILE = "scientific_verbatim"

# key -> (nhãn, kiểu, mô hình Whisper). Số liệu RTF đo trên máy này (i7-3720QM, máy rảnh, câu khoa học 13 s).
ENGINE_MODES: Dict[str, Tuple[str, str, str]] = {
    "whisper_base": ("🛡️ Ổn định – Whisper base (offline, ~0.25× thời gian thực)", "whisper", "base"),
    "whisper_base_en": ("🎯 Chính xác hơn – Whisper base.en (offline, ~0.25–0.5×)", "whisper", "base.en"),
    "whisper_small_en": ("🧠 Chính xác nhất – Whisper small.en (offline, ~0.8×, dễ phải bỏ qua khi máy bận)", "whisper", "small.en"),
    "ensemble_base": ("🔬 Whisper base + Google (cần mạng, khóa Google có thể bị giới hạn)", "ensemble", "base"),
}
DEFAULT_ENGINE_MODE = "whisper_base"

# Ngưỡng bỏ qua khi xử lý không kịp (giây trễ so với lúc câu kết thúc). 0 = không bao giờ bỏ qua.
LAG_OPTIONS: Dict[str, float] = {
    "Bỏ qua khi trễ > 30s": 30.0,
    "Bỏ qua khi trễ > 60s": 60.0,
    "Bỏ qua khi trễ > 120s": 120.0,
    "Không bỏ qua (chờ hết)": 0.0,
}
DEFAULT_MAX_LAG_SEC = 60.0


@dataclass
class EngineConfig:
    engine_mode: str = DEFAULT_ENGINE_MODE
    whisper_model_size: str = "base"
    beam_size: int = 5
    domain_key: str = DEFAULT_DOMAIN
    user_hotwords: str = ""
    recording_profile: str = DEFAULT_RECORDING_PROFILE
    vad_energy_threshold: float = 0.007
    vad_pause_sec: float = 0.85
    soft_cut_sec: float = 10.0
    soft_pause_sec: float = 0.35
    max_phrase_sec: float = 18.0
    min_phrase_sec: float = 0.45
    pre_roll_sec: float = 0.35
    overlap_sec: float = 0.40
    max_lag_sec: float = DEFAULT_MAX_LAG_SEC
    dsp_enhance: bool = True
    preemphasis: float = 0.15
    save_wav: bool = True
    device_index: Optional[int] = None
    cpu_threads: int = 4
    transcript_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parent / "transcripts"))

    @property
    def engine_kind(self) -> str:
        return ENGINE_MODES.get(self.engine_mode, ENGINE_MODES[DEFAULT_ENGINE_MODE])[1]

    def apply_engine_mode(self, mode_key: str) -> None:
        self.engine_mode = mode_key if mode_key in ENGINE_MODES else DEFAULT_ENGINE_MODE
        self.whisper_model_size = ENGINE_MODES[self.engine_mode][2]

    def apply_recording_profile(self, profile_key: str) -> None:
        self.recording_profile = profile_key if profile_key in RECORDING_PROFILES else DEFAULT_RECORDING_PROFILE
        prof = RECORDING_PROFILES[self.recording_profile]
        for key in ("vad_pause_sec", "soft_cut_sec", "soft_pause_sec", "max_phrase_sec",
                    "min_phrase_sec", "pre_roll_sec", "overlap_sec"):
            setattr(self, key, float(prof[key]))


@dataclass
class AudioChunkPacket:
    """Một câu âm thanh 16 kHz mono float32 đã cắt xong, chờ nhận diện."""
    audio_f32: np.ndarray
    duration_sec: float
    rms_energy: float
    captured_at: float                         # time.time() lúc câu kết thúc
    device_name: str = "System Loopback"
    sequence_id: int = 0
    audio_offset_sec: Optional[float] = None   # Vị trí bắt đầu câu trong file WAV
    has_overlap: bool = False                  # Bắt đầu bằng đoạn gối đầu của câu trước


@dataclass
class TranscriptSegment:
    sequence_id: int
    timestamp_start: datetime.datetime
    timestamp_end: datetime.datetime
    text: str
    confidence: float
    engine: str
    latency_ms: float
    low_confidence_words: List[str] = field(default_factory=list)
    audio_offset_sec: Optional[float] = None
    whisper_text: str = ""
    google_alternatives: List[Tuple[str, float]] = field(default_factory=list)
    substitutions: List[Tuple[str, str]] = field(default_factory=list)
    is_skip: bool = False


def is_hallucinated_text(text: str) -> bool:
    """Trống/quá ngắn, khớp danh sách đen, hoặc lặp từ bệnh lý."""
    if not text:
        return True
    cleaned = text.strip()
    if len(cleaned) < 2:
        return True
    if any(p.search(cleaned) for p in HALLUCINATION_PATTERNS):
        return True
    words = re.findall(r"\w+", cleaned.lower())
    return len(words) >= 6 and len(set(words)) / float(len(words)) < 0.28


def _fmt_hms(sec: Optional[float]) -> str:
    if sec is None:
        return "--:--:--"
    s = int(sec)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# ============================================================================
# 2. XỬ LÝ TÍN HIỆU
# ============================================================================

class AudioDSP:
    _SOS_BANDPASS = butter(4, [85.0, 7600.0], btype="bandpass", fs=TARGET_SAMPLE_RATE, output="sos")

    @staticmethod
    def pcm16_to_float32(pcm_bytes: bytes, channels: int = 1) -> np.ndarray:
        if not pcm_bytes:
            return np.zeros(0, dtype=np.float32)
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1 and samples.size >= channels:
            usable = (samples.size // channels) * channels
            samples = samples[:usable].reshape(-1, channels).mean(axis=1)
        return samples

    @staticmethod
    def resample_to_16khz(mono: np.ndarray, orig_sr: int) -> np.ndarray:
        if mono.size == 0:
            return mono
        orig_sr = int(round(orig_sr))
        if orig_sr == TARGET_SAMPLE_RATE:
            return mono
        g = math.gcd(orig_sr, TARGET_SAMPLE_RATE)
        up, down = TARGET_SAMPLE_RATE // g, orig_sr // g
        if up <= 160 and down <= 480:
            return resample_poly(mono, up, down).astype(np.float32, copy=False)
        target_len = max(1, int(round(mono.size / float(orig_sr) * TARGET_SAMPLE_RATE)))
        x_old = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        return np.interp(x_new, x_old, mono).astype(np.float32)

    @staticmethod
    def trim_trailing_silence(audio: np.ndarray, keep_sec: float = 0.3) -> np.ndarray:
        """
        Bỏ khoảng lặng cuối câu, chỉ giữ `keep_sec`. Khoảng lặng dài ở cuối là nơi Whisper hay "bịa"
        câu (đã quan sát: "We'll see you in a little bit." lặp 10 lần với base.en).
        """
        if audio.size == 0:
            return audio
        frame = int(TARGET_SAMPLE_RATE * 0.02)
        n = audio.size // frame
        if n < 2:
            return audio
        rms = np.sqrt(np.mean(np.square(audio[: n * frame].reshape(n, frame)), axis=1))
        thr = max(0.004, 0.06 * float(rms.max()))
        voiced = np.flatnonzero(rms > thr)
        if voiced.size == 0:
            return audio
        end = min(audio.size, (voiced[-1] + 1) * frame + int(TARGET_SAMPLE_RATE * keep_sec))
        return audio[:end]

    @staticmethod
    def scientific_enhance(mono: np.ndarray, preemphasis: float = 0.15) -> np.ndarray:
        """Butterworth 85–7600 Hz zero-phase + pre-emphasis nhẹ + chuẩn hóa đỉnh về ~0.7 khi giọng nhỏ."""
        if mono.size < 64:
            return mono.astype(np.float32, copy=False)
        y = sosfiltfilt(AudioDSP._SOS_BANDPASS, mono.astype(np.float64))
        if preemphasis > 0.0:
            y = np.append(y[0], y[1:] - preemphasis * y[:-1])
        peak = float(np.max(np.abs(y)))
        if 0.005 < peak < 0.7:
            y = y * min(6.0, 0.7 / peak)
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def float32_to_pcm16_bytes(mono: np.ndarray) -> bytes:
        return (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes() if mono.size else b""

    @staticmethod
    def compute_rms_and_zcr(mono: np.ndarray) -> Tuple[float, float]:
        if mono.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(np.mean(np.square(mono))))
        signs = np.signbit(mono)
        zcr = float(np.mean(signs[1:] != signs[:-1])) if mono.size > 1 else 0.0
        return rms, zcr


class WavRecorder:
    """Ghi liên tục luồng 16 kHz mono ra WAV (~115 MB/giờ). Header được cập nhật sau mỗi lần ghi."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._wf: Optional[wave.Wave_write] = None
        self.samples_written = 0

    def open(self) -> None:
        with self._lock:
            if self._wf is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                wf = wave.open(str(self.path), "wb")
                wf.setnchannels(1)
                wf.setsampwidth(SAMPLE_WIDTH_BYTES)
                wf.setframerate(TARGET_SAMPLE_RATE)
                self._wf = wf

    def write(self, mono: np.ndarray) -> None:
        if mono.size == 0:
            return
        with self._lock:
            if self._wf is not None:
                self._wf.writeframes(AudioDSP.float32_to_pcm16_bytes(mono))
                self.samples_written += mono.size

    @property
    def seconds_written(self) -> float:
        return self.samples_written / float(TARGET_SAMPLE_RATE)

    @property
    def is_open(self) -> bool:
        return self._wf is not None

    def close(self) -> None:
        with self._lock:
            if self._wf is not None:
                try:
                    self._wf.close()
                except Exception:
                    pass
                self._wf = None


# ============================================================================
# 3. CẮT CÂU
# ============================================================================

class UtteranceSegmenter:
    """
    1. Chốt câu khi im lặng >= `vad_pause_sec`.
    2. Câu dài >= `soft_cut_sec`: chốt ở nhịp nghỉ >= `soft_pause_sec`.
    3. Chạm `max_phrase_sec`: cắt tại khung năng lượng thấp nhất trong 2.5 s cuối; phần còn lại sang câu sau.
       Không có điểm trũng rõ -> gối đầu `overlap_sec` và đánh dấu để khử chữ lặp.
    """

    def __init__(self, config: EngineConfig, on_utterance: Callable[[np.ndarray, float, float, str, bool], None]) -> None:
        self.config = config
        self.on_utterance = on_utterance
        self._pre_roll: collections.deque = collections.deque()
        self._pre_roll_dur = 0.0
        self._noise_floor = 0.0025
        self._reset()

    def _reset(self) -> None:
        self._frames: List[np.ndarray] = []
        self._frame_rms: List[float] = []
        self._active_duration = 0.0
        self._silence_duration = 0.0
        self._in_speech = False
        self._peak_rms = 0.0
        self._has_overlap = False

    def _append_active(self, frame: np.ndarray, rms: float) -> None:
        self._frames.append(frame)
        self._frame_rms.append(rms)
        self._active_duration += frame.size / float(TARGET_SAMPLE_RATE)

    def process_frame(self, frame: np.ndarray, device_name: str) -> None:
        if frame.size == 0:
            return
        frame_dur = frame.size / float(TARGET_SAMPLE_RATE)
        rms, zcr = AudioDSP.compute_rms_and_zcr(frame)
        threshold = max(self.config.vad_energy_threshold, self._noise_floor * 2.0)
        zcr_ok = 0.004 <= zcr <= 0.50

        if not self._in_speech:
            if rms >= threshold and zcr_ok:
                self._in_speech = True
                self._peak_rms = rms
                for pre in self._pre_roll:
                    self._append_active(pre, AudioDSP.compute_rms_and_zcr(pre)[0])
                self._pre_roll.clear()
                self._pre_roll_dur = 0.0
                self._append_active(frame, rms)
            else:
                self._noise_floor = 0.94 * self._noise_floor + 0.06 * min(rms, 0.015)
                self._pre_roll.append(frame)
                self._pre_roll_dur += frame_dur
                while self._pre_roll and self._pre_roll_dur > self.config.pre_roll_sec:
                    self._pre_roll_dur -= self._pre_roll.popleft().size / float(TARGET_SAMPLE_RATE)
            return

        self._append_active(frame, rms)
        self._peak_rms = max(self._peak_rms * 0.995, rms)
        if rms >= max(threshold, self._peak_rms * 0.16) and zcr_ok:
            self._silence_duration = 0.0
        else:
            self._silence_duration += frame_dur

        cfg = self.config
        if self._silence_duration >= cfg.vad_pause_sec or (
            self._active_duration >= cfg.soft_cut_sec and self._silence_duration >= cfg.soft_pause_sec
        ):
            self._emit_all(device_name)
        elif self._active_duration >= cfg.max_phrase_sec:
            self._emit_split(device_name)

    def flush(self, device_name: str) -> None:
        if self._in_speech and self._frames:
            self._emit_all(device_name)

    def _deliver(self, frames: List[np.ndarray], peak: float, device_name: str, has_overlap: bool) -> None:
        if not frames:
            return
        audio = np.concatenate(frames).astype(np.float32, copy=False)
        dur = audio.size / float(TARGET_SAMPLE_RATE)
        if dur >= self.config.min_phrase_sec and peak >= self.config.vad_energy_threshold:
            self.on_utterance(audio, dur, peak, device_name, has_overlap)

    def _emit_all(self, device_name: str) -> None:
        frames, peak, ov = self._frames, self._peak_rms, self._has_overlap
        self._reset()
        self._deliver(frames, peak, device_name, ov)

    def _emit_split(self, device_name: str) -> None:
        frames, rms_list, peak, head_ov = self._frames, self._frame_rms, self._peak_rms, self._has_overlap
        durs = [f.size / float(TARGET_SAMPLE_RATE) for f in frames]
        n = len(frames)
        start, acc = n - 1, 0.0
        while start > 0 and acc < 2.5:
            acc += durs[start]
            start -= 1
        guard, acc = n, 0.0
        while guard > start + 1 and acc < 0.2:
            guard -= 1
            acc += durs[guard]
        split = min(range(start, max(start + 1, guard)), key=lambda k: rms_list[k])
        clear_gap = rms_list[split] < 0.35 * max(peak, 1e-6)

        head, carry, carry_rms = frames[: split + 1], frames[split + 1:], rms_list[split + 1:]
        if not clear_gap and self.config.overlap_sec > 0:
            head_audio = np.concatenate(head)
            tail = head_audio[-int(TARGET_SAMPLE_RATE * self.config.overlap_sec):]
            carry = [tail] + carry
            carry_rms = [AudioDSP.compute_rms_and_zcr(tail)[0]] + carry_rms

        self._reset()
        self._deliver(head, peak, device_name, head_ov)
        if carry:
            self._in_speech = True
            for f, r in zip(carry, carry_rms):
                self._append_active(f, r)
            self._peak_rms = max(carry_rms)
            self._has_overlap = not clear_gap


# ============================================================================
# 4. THU ÂM WASAPI LOOPBACK
# ============================================================================

class AudioLoopbackCapture:
    """
    Mở song song mọi thiết bị loopback nhưng khóa vào 1 thiết bị đang có tiếng (chỉ đổi khi thiết bị đang khóa
    im lặng >= 2 s), ghi song song WAV, tự kết nối lại khi cắm/rút tai nghe.
    """

    def __init__(
        self,
        config: EngineConfig,
        audio_queue: queue.Queue,
        level_callback: Optional[Callable[[float, str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        wav_recorder: Optional[WavRecorder] = None,
    ) -> None:
        self.config = config
        self.audio_queue = audio_queue
        self.level_callback = level_callback
        self.status_callback = status_callback
        self.wav_recorder = wav_recorder
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq_lock = threading.Lock()
        self._seq_counter = 0
        self.active_device_name = "Chưa kết nối"

    @staticmethod
    def list_loopback_devices() -> List[Dict]:
        if not HAS_PYAUDIOWPATCH:
            return []
        p = None
        try:
            p = pyaudio.PyAudio()
            return [dict(d) for d in p.get_loopback_device_info_generator()]
        except Exception as exc:
            logger.warning("Không thể liệt kê WASAPI Loopback: %s", exc)
            return []
        finally:
            if p is not None:
                try:
                    p.terminate()
                except Exception:
                    pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="WASAPILoopbackThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def set_paused(self, paused: bool) -> None:
        (self._pause_event.set if paused else self._pause_event.clear)()

    def inject_test_audio(self, mono_16k: np.ndarray, source_label: str = "Simulated Speaker") -> None:
        """Bơm âm thanh 16 kHz mono trực tiếp vào bộ cắt câu (kiểm thử / nút Test)."""
        frame_len = int(TARGET_SAMPLE_RATE * 0.04)
        segmenter = UtteranceSegmenter(self.config, on_utterance=self._enqueue_utterance)
        for idx in range(0, mono_16k.size, frame_len):
            frame = mono_16k[idx: idx + frame_len]
            if frame.size < frame_len:
                frame = np.pad(frame, (0, frame_len - frame.size))
            if self.level_callback:
                self.level_callback(AudioDSP.compute_rms_and_zcr(frame)[0], source_label)
            segmenter.process_frame(frame, source_label)
        segmenter.flush(source_label)

    def _enqueue_utterance(self, audio: np.ndarray, duration: float, rms: float, device: str, has_overlap: bool = False) -> None:
        with self._seq_lock:
            self._seq_counter += 1
            seq = self._seq_counter
        offset = None
        if self.wav_recorder is not None and self.wav_recorder.is_open:
            offset = max(0.0, self.wav_recorder.seconds_written - duration)
        self.audio_queue.put(AudioChunkPacket(
            audio_f32=audio, duration_sec=duration, rms_energy=rms, captured_at=time.time(),
            device_name=device, sequence_id=seq, audio_offset_sec=offset, has_overlap=has_overlap,
        ))

    def _open_streams(self, p) -> Tuple[List[Tuple[dict, object]], Dict[int, collections.deque]]:
        devices: List[dict] = []
        if self.config.device_index is not None:
            devices.append(dict(p.get_device_info_by_index(int(self.config.device_index))))
        else:
            try:
                devices.append(dict(p.get_default_wasapi_loopback()))
            except Exception:
                pass
            for dev in p.get_loopback_device_info_generator():
                d = dict(dev)
                if all(d["index"] != e["index"] for e in devices):
                    devices.append(d)

        streams: List[Tuple[dict, object]] = []
        buffers: Dict[int, collections.deque] = {}
        for dev in devices:
            idx = int(dev["index"])
            rate = int(dev.get("defaultSampleRate", 48000))
            buf: collections.deque = collections.deque(maxlen=200)
            buffers[idx] = buf

            def _cb(in_data, frame_count, time_info, status, q_ref=buf):
                if in_data:
                    q_ref.append(in_data)
                return (None, pyaudio.paContinue)

            st = p.open(
                format=pyaudio.paInt16, channels=max(1, min(2, int(dev.get("maxInputChannels", 2)))),
                rate=rate, frames_per_buffer=max(256, int(rate * 0.035)), input=True,
                input_device_index=idx, stream_callback=_cb,
            )
            st.start_stream()
            streams.append((dev, st))
        return streams, buffers

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if not HAS_PYAUDIOWPATCH:
                if self.status_callback:
                    self.status_callback("Thiếu thư viện PyAudioWPatch")
                time.sleep(2.0)
                continue

            p, streams, segmenter = None, [], None
            try:
                p = pyaudio.PyAudio()
                streams, buffers = self._open_streams(p)
                if not streams:
                    if self.status_callback:
                        self.status_callback("Không tìm thấy loa/tai nghe Loopback — đang chờ...")
                    time.sleep(1.5)
                    continue

                locked_idx = int(streams[0][0]["index"])
                last_locked_active = time.perf_counter()
                self.active_device_name = streams[0][0].get("name", "WASAPI Loopback")
                if self.status_callback:
                    self.status_callback(f"Đang ghi từ: {self.active_device_name}")
                segmenter = UtteranceSegmenter(self.config, on_utterance=self._enqueue_utterance)
                chosen_setting = self.config.device_index

                while not self._stop_event.is_set() and self.config.device_index == chosen_setting:
                    if self._pause_event.is_set():
                        for b in buffers.values():
                            b.clear()
                        time.sleep(0.05)
                        continue

                    now = time.perf_counter()
                    frames: Dict[int, Tuple[np.ndarray, float, str]] = {}
                    for dev, _st in streams:
                        idx = int(dev["index"])
                        buf = buffers[idx]
                        if not buf:
                            continue
                        chunks = [buf.popleft() for _ in range(len(buf))]
                        mono = AudioDSP.pcm16_to_float32(b"".join(chunks), max(1, min(2, int(dev.get("maxInputChannels", 2)))))
                        f16 = AudioDSP.resample_to_16khz(mono, int(dev.get("defaultSampleRate", 48000)))
                        frames[idx] = (f16, AudioDSP.compute_rms_and_zcr(f16)[0], dev.get("name", "Loopback"))

                    selected = None
                    if locked_idx in frames:
                        if frames[locked_idx][1] >= 0.0022:
                            last_locked_active = now
                        if now - last_locked_active < 2.0:
                            selected = frames[locked_idx]
                    if selected is None and frames:
                        cand_idx, cand = max(frames.items(), key=lambda it: it[1][1])
                        if cand[1] >= 0.005 or locked_idx not in frames:
                            locked_idx, last_locked_active, selected = cand_idx, now, cand
                        else:
                            selected = frames[locked_idx]

                    if selected is not None and selected[0].size:
                        f16, rms, name = selected
                        if rms > 0.003:
                            self.active_device_name = name
                    else:
                        # Loopback ngừng phát khung khi hệ thống im lặng -> bơm khung im lặng để chốt câu và giữ trục thời gian WAV
                        f16, rms = np.zeros(int(TARGET_SAMPLE_RATE * 0.04), dtype=np.float32), 0.0

                    if self.config.save_wav and self.wav_recorder is not None:
                        self.wav_recorder.write(f16)
                    if self.level_callback:
                        self.level_callback(rms, self.active_device_name)
                    segmenter.process_frame(f16, self.active_device_name)
                    time.sleep(0.04)

                segmenter.flush(self.active_device_name)
            except Exception as exc:
                logger.warning("WASAPI Loopback gián đoạn (%s) — đang kết nối lại...", exc)
                if segmenter is not None:
                    try:
                        segmenter.flush(self.active_device_name)
                    except Exception:
                        pass
                if self.status_callback:
                    self.status_callback("Đang tự động kết nối lại thiết bị âm thanh...")
                time.sleep(1.0)
            finally:
                for _dev, st in streams:
                    try:
                        st.stop_stream()
                        st.close()
                    except Exception:
                        pass
                if p is not None:
                    try:
                        p.terminate()
                    except Exception:
                        pass


# ============================================================================
# 5. GOOGLE SPEECH (TÙY CHỌN)
# ============================================================================

_ASR_SESSION: Optional[requests.Session] = None
_GOOGLE_ENDPOINT = "https://www.google.com/speech-api/v2/recognize"


def google_asr_nbest(pcm16_bytes: bytes, timeout: float = 8.0) -> List[Tuple[str, float]]:
    """
    Google Speech API v2 với khóa mặc định của thư viện SpeechRecognition (chỉ dành cho dùng cá nhân,
    có thể bị giới hạn bất cứ lúc nào). Trả về [(transcript, confidence)], tốt nhất đứng đầu.
    """
    global _ASR_SESSION
    import speech_recognition as sr
    import speech_recognition.recognizers.google as sr_google

    if _ASR_SESSION is None:
        _ASR_SESSION = requests.Session()
        _ASR_SESSION.mount("https://", HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=1))
    url = sr_google.create_request_builder(
        endpoint=_GOOGLE_ENDPOINT, key=None, language=GOOGLE_LANGUAGE, filter_level=0
    ).build_url()
    flac = sr.AudioData(pcm16_bytes, TARGET_SAMPLE_RATE, SAMPLE_WIDTH_BYTES).get_flac_data(
        convert_rate=TARGET_SAMPLE_RATE, convert_width=SAMPLE_WIDTH_BYTES
    )
    resp = _ASR_SESSION.post(url, data=flac, headers={"Content-Type": f"audio/x-flac; rate={TARGET_SAMPLE_RATE}"}, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Google HTTP {resp.status_code}")

    blocks: List[List[Tuple[str, float]]] = []
    for line in resp.text.splitlines():
        try:
            obj = json.loads(line) if line.strip() else {}
        except ValueError:
            continue
        for res in obj.get("result", []):
            alts = [(a.get("transcript", "").strip(), float(a.get("confidence", 0.0))) for a in res.get("alternative", [])]
            alts = [a for a in alts if a[0]]
            if alts:
                blocks.append(alts)
    if not blocks:
        return []
    if len(blocks) == 1:
        return blocks[0]
    confs = [b[0][1] for b in blocks if b[0][1] > 0]
    return [(" ".join(b[0][0] for b in blocks), sum(confs) / len(confs) if confs else 0.0)]


# ============================================================================
# 6. LƯU TRỮ
# ============================================================================

class TranscriptStore:
    """Ghi tức thì từng câu ra .txt / .md / .jsonl (flush + fsync); xuất .docx / văn bản liền mạch khi cần."""

    def __init__(self, transcript_dir: str) -> None:
        self.base_dir = Path(transcript_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.datetime.now()
        stem = f"Verbatim_{self.started_at.strftime('%Y%m%d_%H%M%S')}"
        self.txt_path = self.base_dir / f"{stem}.txt"
        self.md_path = self.base_dir / f"{stem}.md"
        self.jsonl_path = self.base_dir / f"{stem}.jsonl"
        self.wav_path = self.base_dir / f"{stem}.wav"
        self.docx_path = self.base_dir / f"{stem}.docx"
        self.clean_txt_path = self.base_dir / f"{stem}_lien_mach.txt"
        self._lock = threading.Lock()
        now_s = self.started_at.strftime("%Y-%m-%d %H:%M:%S")
        self.txt_path.write_text(f"=== BẢN GHI NGUYÊN VĂN CUỘC HỌP — bắt đầu {now_s} ===\n\n", encoding="utf-8")
        self.md_path.write_text(
            f"# Bản ghi nguyên văn cuộc họp\n- Bắt đầu: `{now_s}`\n"
            f"- File âm thanh đối chiếu: `{self.wav_path.name}` (cột *Vị trí WAV*)\n"
            f"- Từ có hậu tố **[?]** là từ có độ tin cậy thấp, nên nghe lại.\n\n"
            f"| Giờ | Vị trí WAV | Tin cậy | Nội dung nguyên văn |\n| :---: | :---: | :---: | :--- |\n",
            encoding="utf-8",
        )
        self.jsonl_path.write_text("", encoding="utf-8")

    @staticmethod
    def _append(path: Path, text: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def mark_low_confidence(text: str, low_words: List[str]) -> str:
        if not low_words:
            return text
        remaining = collections.Counter(low_words)
        out = []
        for tok in text.split():
            if remaining[tok] > 0:
                remaining[tok] -= 1
                out.append(f"{tok}[?]")
            else:
                out.append(tok)
        return " ".join(out)

    def log_segment(self, seg: TranscriptSegment) -> None:
        with self._lock:
            t_s = seg.timestamp_start.strftime("%H:%M:%S")
            self._append(self.txt_path, f"[{t_s}] {seg.text}\n")
            marked = self.mark_low_confidence(seg.text, seg.low_confidence_words).replace("|", "\\|")
            self._append(self.md_path, f"| `{t_s}` | `{_fmt_hms(seg.audio_offset_sec)}` | `{seg.confidence * 100:.0f}%` | {marked} |\n")
            self._append(self.jsonl_path, json.dumps({
                "seq": seg.sequence_id,
                "start": seg.timestamp_start.isoformat(timespec="milliseconds"),
                "end": seg.timestamp_end.isoformat(timespec="milliseconds"),
                "wav_offset_sec": None if seg.audio_offset_sec is None else round(seg.audio_offset_sec, 2),
                "text": seg.text,
                "is_skip": seg.is_skip,
                "confidence": round(seg.confidence, 4),
                "low_confidence_words": seg.low_confidence_words,
                "engine": seg.engine,
                "whisper_text": seg.whisper_text,
                "google_alternatives": seg.google_alternatives,
                "substitutions": seg.substitutions,
                "latency_ms": round(seg.latency_ms, 1),
            }, ensure_ascii=False) + "\n")

    def export_clean_text(self, segments: List[TranscriptSegment]) -> Path:
        """Văn bản liền mạch không mốc giờ; xuống đoạn khi ngắt nghỉ > 20 s."""
        paragraphs: List[List[str]] = []
        last_end: Optional[datetime.datetime] = None
        for s in segments:
            if last_end is None or (s.timestamp_start - last_end).total_seconds() > 20:
                paragraphs.append([])
            paragraphs[-1].append(s.text)
            last_end = s.timestamp_end
        self.clean_txt_path.write_text("\n\n".join(" ".join(p) for p in paragraphs) + "\n", encoding="utf-8")
        return self.clean_txt_path

    def export_docx(self, segments: List[TranscriptSegment], meta: Dict[str, str]) -> Path:
        from docx import Document
        from docx.enum.text import WD_COLOR_INDEX
        from docx.shared import Inches, Pt, RGBColor

        doc = Document()
        for section in doc.sections:
            section.top_margin = section.bottom_margin = Inches(0.8)
            section.left_margin = section.right_margin = Inches(0.9)
        doc.styles["Normal"].font.name = "Times New Roman"
        doc.styles["Normal"].font.size = Pt(12)

        doc.add_heading("BẢN GHI NGUYÊN VĂN CUỘC HỌP", level=0)
        table = doc.add_table(rows=0, cols=2)
        table.style = "Table Grid"
        for k, v in meta.items():
            cells = table.add_row().cells
            cells[0].text, cells[1].text = k, v
        doc.add_paragraph(
            "Ghi chú: từ tô vàng có độ tin cậy nhận diện thấp; mục [BỎ QUA] là đoạn chưa được ghi chữ — "
            "nghe lại file WAV tại vị trí tương ứng."
        ).runs[0].italic = True

        for s in segments:
            p = doc.add_paragraph()
            r = p.add_run(f"[{s.timestamp_start.strftime('%H:%M:%S')} | WAV {_fmt_hms(s.audio_offset_sec)}]  ")
            r.bold = True
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(100, 116, 139)
            if s.is_skip:
                run = p.add_run(s.text)
                run.italic = True
                run.font.color.rgb = RGBColor(185, 28, 28)
                continue
            remaining = collections.Counter(s.low_confidence_words)
            tokens = s.text.split()
            for i, tok in enumerate(tokens):
                run = p.add_run(tok + (" " if i < len(tokens) - 1 else ""))
                if remaining[tok] > 0:
                    remaining[tok] -= 1
                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        doc.save(str(self.docx_path))
        return self.docx_path


# ============================================================================
# 7. NHẬN DIỆN NGUYÊN VĂN
# ============================================================================

class ScientificVerbatimASR:
    """
    Xử lý tuần tự từng câu (giữ thứ tự + ngữ cảnh):
    1. Bỏ khoảng lặng cuối câu, lọc tín hiệu.
    2. Whisper: beam 5, temperature 0 (không lấy mẫu ngẫu nhiên -> kết quả ổn định), câu mồi học thuật +
       hotwords + ngữ cảnh câu trước, giới hạn số token theo độ dài câu, bỏ qua khoảng lặng khi nghi ảo giác.
    3. (Tùy chọn) Google chạy song song rồi hợp nhất thận trọng.
    4. Lọc ảo giác, bỏ câu lặp, khử chữ lặp ranh giới, chuẩn hóa thuật ngữ.
    Nếu câu đang chờ đã trễ hơn `max_lag_sec`, bỏ qua toàn bộ phần tồn đọng (trừ câu mới nhất) và ghi mốc [BỎ QUA].
    """

    def __init__(
        self,
        config: EngineConfig,
        audio_queue: queue.Queue,
        ui_queue: queue.Queue,
        store: TranscriptStore,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.audio_queue = audio_queue
        self.ui_queue = ui_queue
        self.store = store
        self.status_callback = status_callback
        self.arbitrator = EnsembleArbitrator()
        self._model = None
        self._model_size: Optional[str] = None
        self._model_lock = threading.Lock()
        self._google_pool: Optional[ThreadPoolExecutor] = None
        self._stop_event = threading.Event()
        self._busy = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_text = ""
        self.current_lag_sec = 0.0
        self.skipped_segments = 0
        self.history: List[TranscriptSegment] = []

    # -- Vòng đời -----------------------------------------------------------
    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._process_loop, name="VerbatimASRWorker", daemon=True)
        self._worker.start()
        threading.Thread(target=self.preload, name="WhisperPreload", daemon=True).start()

    def preload(self) -> None:
        try:
            self._get_model()
        except Exception as exc:
            logger.error("Không nạp được Whisper: %s", exc)
            if self.status_callback:
                self.status_callback(f"Lỗi nạp Whisper: {exc}")

    def stop(self, drain_timeout: float = 0.0) -> None:
        """Dừng worker; nếu `drain_timeout` > 0 thì chờ xử lý hết hàng đợi trước."""
        deadline = time.time() + max(0.0, drain_timeout)
        while drain_timeout > 0 and time.time() < deadline and (not self.audio_queue.empty() or self._busy.is_set()):
            time.sleep(0.2)
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5.0)

    @property
    def backlog(self) -> int:
        return self.audio_queue.qsize() + (1 if self._busy.is_set() else 0)

    def _get_model(self):
        with self._model_lock:
            size = self.config.whisper_model_size
            if self._model is None or self._model_size != size:
                if self.status_callback:
                    self.status_callback(f"Đang nạp Whisper '{size}' (lần đầu sẽ tải mô hình về máy)...")
                from faster_whisper import WhisperModel
                self._model = WhisperModel(size, device="cpu", compute_type="int8",
                                           cpu_threads=self.config.cpu_threads, num_workers=1)
                self._model_size = size
                if self.status_callback:
                    self.status_callback(f"Whisper '{size}' sẵn sàng")
            return self._model

    # -- Whisper ------------------------------------------------------------
    def _run_whisper(self, audio: np.ndarray):
        model = self._get_model()
        prompt = build_initial_prompt(self.config.domain_key, self.config.user_hotwords, self._last_text)
        beam = max(1, int(self.config.beam_size))
        dur = audio.size / float(TARGET_SAMPLE_RATE)
        segments, _info = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=beam,
            best_of=beam,
            patience=1.2,
            temperature=0.0,                          # tắt lấy mẫu ngẫu nhiên: ổn định, lặp lại được
            condition_on_previous_text=False,
            initial_prompt=prompt,
            hotwords=build_hotwords(self.config.domain_key, self.config.user_hotwords),
            word_timestamps=True,
            hallucination_silence_threshold=0.5,      # bỏ qua khoảng lặng > 0.5 s khi nghi ảo giác
            max_new_tokens=int(dur * 7) + 12,         # chặn vòng lặp vô hạn (~3 từ/giây nói)
            no_speech_threshold=0.6,
            vad_filter=False,
        )
        seg_list = list(segments)
        words = group_whisper_words([(w.word, float(w.probability)) for s in seg_list for w in (s.words or [])])
        text = " ".join(s.text.strip() for s in seg_list if s.text.strip())
        avg_lp = (sum(s.avg_logprob for s in seg_list) / len(seg_list)) if seg_list else -5.0
        no_speech = max((s.no_speech_prob for s in seg_list), default=1.0)
        return text, words, avg_lp, no_speech, prompt

    # -- Bỏ qua khi quá tải -------------------------------------------------
    def _skip_if_lagging(self, packet: AudioChunkPacket) -> AudioChunkPacket:
        """Nếu câu này đã trễ quá `max_lag_sec`, gom toàn bộ phần tồn đọng, chỉ giữ câu mới nhất để xử lý."""
        self.current_lag_sec = max(0.0, time.time() - packet.captured_at)
        limit = self.config.max_lag_sec
        if limit <= 0 or self.current_lag_sec <= limit:
            return packet
        skipped = [packet]
        while True:
            try:
                skipped.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        newest = skipped.pop()
        if skipped:
            self._log_skip(skipped)
        return newest

    def _log_skip(self, skipped: List[AudioChunkPacket]) -> None:
        first, last = skipped[0], skipped[-1]
        total = sum(p.duration_sec for p in skipped)
        end_dt = datetime.datetime.fromtimestamp(last.captured_at)
        start_dt = datetime.datetime.fromtimestamp(first.captured_at) - datetime.timedelta(seconds=first.duration_sec)
        where = f"nghe lại WAV từ {_fmt_hms(first.audio_offset_sec)}" if first.audio_offset_sec is not None else "không có WAV"
        seg = TranscriptSegment(
            sequence_id=first.sequence_id, timestamp_start=start_dt, timestamp_end=end_dt,
            text=f"[BỎ QUA {len(skipped)} đoạn ({total:.0f}s lời nói) do máy xử lý không kịp — {where}]",
            confidence=0.0, engine="skip", latency_ms=0.0, audio_offset_sec=first.audio_offset_sec, is_skip=True,
        )
        self.skipped_segments += len(skipped)
        self._last_text = ""   # ngữ cảnh bị đứt: không mồi bằng câu trước nữa
        self._commit(seg)
        logger.warning("Bỏ qua %d đoạn (%.0fs) do trễ %.0fs", len(skipped), total, self.current_lag_sec)

    # -- Xử lý một câu ------------------------------------------------------
    def process_packet_sync(self, packet: AudioChunkPacket) -> Optional[TranscriptSegment]:
        t0 = time.perf_counter()
        audio = AudioDSP.trim_trailing_silence(packet.audio_f32)
        if self.config.dsp_enhance:
            audio = AudioDSP.scientific_enhance(audio, self.config.preemphasis)

        google_future = None
        if self.config.engine_kind == "ensemble":
            if self._google_pool is None:
                self._google_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GoogleASR")
            google_future = self._google_pool.submit(google_asr_nbest, AudioDSP.float32_to_pcm16_bytes(audio))

        try:
            w_text, w_words, avg_lp, no_speech, prompt = self._run_whisper(audio)
        except Exception as exc:
            logger.error("Lỗi Whisper: %s", exc)
            w_text, w_words, avg_lp, no_speech, prompt = "", [], -5.0, 1.0, None
        if w_words:
            w_words = collapse_repeated_sentences(w_words)
            w_text = " ".join(w.text for w in w_words)

        google_alts: List[Tuple[str, float]] = []
        if google_future is not None:
            try:
                google_alts = google_future.result(timeout=10.0)
            except Exception as exc:
                logger.warning("Google lỗi (%s) — câu này chỉ dùng Whisper", exc)

        # Lọc ảo giác (bỏ qua nếu Google nghe ra y hệt Whisper)
        g_plain = re.sub(r"[^\w\s]", "", google_alts[0][0].lower()).strip() if google_alts else ""
        agreed = bool(g_plain) and g_plain == re.sub(r"[^\w\s]", "", w_text.lower()).strip()
        if w_text and not agreed and (
            is_hallucinated_text(w_text)
            or (no_speech > 0.6 and avg_lp < -1.0)
            or (prompt_echo_ratio(w_text, prompt) > 0.6 and avg_lp < -0.6)
        ):
            w_text, w_words = "", []

        result = self.arbitrator.merge(w_words, w_text, google_alts)
        if not result.text or (result.source == "google" and is_hallucinated_text(result.text)):
            return None
        text = deduplicate_overlap_text(self._last_text, result.text, max_overlap_words=4) if packet.has_overlap else result.text
        if not text:
            return None
        text = normalize_sentence(ScientificTermNormalizer.normalize(text))

        end_dt = datetime.datetime.fromtimestamp(packet.captured_at)
        seg = TranscriptSegment(
            sequence_id=packet.sequence_id,
            timestamp_start=end_dt - datetime.timedelta(seconds=packet.duration_sec),
            timestamp_end=end_dt,
            text=text,
            confidence=float(result.confidence),
            engine={"ensemble": "Whisper+Google", "google": "Google"}.get(result.source, f"Whisper-{self.config.whisper_model_size}"),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            low_confidence_words=[ScientificTermNormalizer.normalize(w) for w in result.low_confidence_words],
            audio_offset_sec=packet.audio_offset_sec,
            whisper_text=w_text,
            google_alternatives=google_alts,
            substitutions=result.substitutions,
        )
        self._last_text = text
        self._commit(seg)
        return seg

    def _commit(self, seg: TranscriptSegment) -> None:
        self.history.append(seg)
        self.store.log_segment(seg)
        self.ui_queue.put(seg)

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet: AudioChunkPacket = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                self.current_lag_sec = 0.0
                continue
            self._busy.set()
            try:
                self.process_packet_sync(self._skip_if_lagging(packet))
            except Exception as exc:
                logger.error("Lỗi xử lý câu #%s: %s", packet.sequence_id, exc)
            finally:
                self._busy.clear()


# ============================================================================
# 8. BỘ ĐIỀU PHỐI
# ============================================================================

class VerbatimRecorderController:
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        level_callback: Optional[Callable[[float, str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.audio_queue: queue.Queue = queue.Queue()
        self.ui_queue: queue.Queue = queue.Queue()
        self.store = TranscriptStore(self.config.transcript_dir)
        self.wav_recorder = WavRecorder(self.store.wav_path)
        self.asr = ScientificVerbatimASR(self.config, self.audio_queue, self.ui_queue, self.store, status_callback)
        self.audio_capture = AudioLoopbackCapture(
            self.config, self.audio_queue, level_callback, status_callback, self.wav_recorder
        )
        self.is_running = False
        self.is_paused = False

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running, self.is_paused = True, False
        if self.config.save_wav:
            self.wav_recorder.open()
        self.asr.start()
        self.audio_capture.start()

    def stop(self, drain_timeout: float = 0.0) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self.audio_capture.stop()
        self.asr.stop(drain_timeout=drain_timeout)
        self.wav_recorder.close()

    def toggle_pause(self) -> bool:
        self.is_paused = not self.is_paused
        self.audio_capture.set_paused(self.is_paused)
        return self.is_paused

    @property
    def history(self) -> List[TranscriptSegment]:
        return self.asr.history

    def session_meta(self) -> Dict[str, str]:
        hist = [s for s in self.asr.history if not s.is_skip]
        dur = _fmt_hms((hist[-1].timestamp_end - hist[0].timestamp_start).total_seconds()) if hist else "00:00:00"
        return {
            "Bắt đầu phiên ghi": self.store.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Khoảng thời gian có lời nói": dur,
            "Số câu / số từ": f"{len(hist)} câu / {sum(len(s.text.split()) for s in hist)} từ",
            "Số đoạn bị bỏ qua (quá tải)": str(self.asr.skipped_segments),
            "Lĩnh vực": DOMAIN_PROFILES.get(self.config.domain_key, {}).get("label", self.config.domain_key),
            "Từ khóa riêng": self.config.user_hotwords or "(không)",
            "Chế độ nhận diện": ENGINE_MODES[self.config.engine_mode][0],
            "File âm thanh đối chiếu": self.store.wav_path.name if self.config.save_wav else "(không ghi)",
        }

    def export_docx(self) -> Path:
        return self.store.export_docx(list(self.asr.history), self.session_meta())

    def export_clean_text(self) -> Path:
        return self.store.export_clean_text(list(self.asr.history))
