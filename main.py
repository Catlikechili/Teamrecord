r"""
Trạm Ghi Nguyên Văn Cuộc Họp Khoa Học Tiếng Anh (`d:\Gravity\Dich\main.py`)
==========================================================================
Giao diện 1 cửa sổ, chỉ phục vụ việc GHI LẠI lời nói (không dịch, không tóm tắt):
- Hàng 1: Ghi/Tạm dừng, Test giọng mẫu, Lĩnh vực khoa học, Chế độ nhận diện.
- Hàng 2: Kiểu cắt câu, Cơ chế quá tải (bỏ qua khi trễ quá X giây), Nguồn âm, đồng hồ tín hiệu.
- Hàng 3: Ô từ khóa chuyên ngành (hotwords).
- Giữa : Khung văn bản nguyên văn; từ độ tin cậy thấp tô vàng, mốc [BỎ QUA] tô đỏ.
- Dưới : Copy toàn bộ, mở .txt, xuất văn bản liền mạch, xuất Word, mở thư mục + các tùy chọn hiển thị;
         thanh trạng thái hiển thị thời lượng, số câu/số từ, số câu đang chờ, độ trễ và số đoạn đã bỏ qua.
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import math
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Dict, List, Optional

import numpy as np

from core_engine import (
    DEFAULT_MAX_LAG_SEC,
    ENGINE_MODES,
    LAG_OPTIONS,
    RECORDING_PROFILES,
    AudioLoopbackCapture,
    EngineConfig,
    TranscriptSegment,
    VerbatimRecorderController,
)
from scientific_normalizer import DOMAIN_PROFILES

# ============================================================================
# WIN32: ẨN CỬA SỔ KHI CHIA SẺ MÀN HÌNH
# ============================================================================

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Windows 10 2004+ / Windows 11


class Win32WindowHelper:
    @staticmethod
    def get_hwnd(tk_widget: tk.Misc) -> int:
        tk_widget.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(tk_widget.winfo_id())
        return hwnd if hwnd != 0 else tk_widget.winfo_id()

    @staticmethod
    def set_stealth_screen_share(tk_window: tk.Misc, hide_on_capture: bool) -> bool:
        if sys.platform != "win32":
            return False
        try:
            hwnd = Win32WindowHelper.get_hwnd(tk_window)
            affinity = WDA_EXCLUDEFROMCAPTURE if hide_on_capture else WDA_NONE
            return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, affinity))
        except Exception:
            return False

    @staticmethod
    def get_display_affinity(tk_window: tk.Misc) -> int:
        value = ctypes.c_uint(0)
        hwnd = Win32WindowHelper.get_hwnd(tk_window)
        ctypes.windll.user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(value))
        return int(value.value)


# ============================================================================
# ỨNG DỤNG
# ============================================================================

class VerbatimRecorderApp:
    BG = "#0F172A"
    PANEL = "#0B1120"
    CARD = "#1E293B"
    BORDER = "#334155"
    ACCENT = "#4F46E5"
    OK = "#10B981"
    WARN = "#F59E0B"
    TEXT = "#F8FAFC"
    MUTED = "#94A3B8"

    def __init__(self, root: tk.Tk, auto_start: bool = True) -> None:
        self.root = root
        self.root.title("Trạm Ghi Nguyên Văn Cuộc Họp Khoa Học — Verbatim Scientific Recorder")
        self.root.configure(bg=self.BG)

        self.config = EngineConfig()
        self.controller = VerbatimRecorderController(
            config=self.config,
            level_callback=self._on_level_threadsafe,
            status_callback=self._on_status_threadsafe,
        )

        self.show_timestamps = tk.BooleanVar(value=True)
        self.save_wav = tk.BooleanVar(value=True)
        self.stealth_share = tk.BooleanVar(value=False)

        self._latest_rms = 0.0
        self._latest_status = "Sẵn sàng"
        self._session_start = time.time()
        self._closing = False
        self._word_count = 0
        self._sentence_count = 0

        self.loopback_devices = AudioLoopbackCapture.list_loopback_devices()
        self.device_map: Dict[str, Optional[int]] = {"🔊 Tự động (khóa thiết bị đang phát tiếng)": None}
        for d in self.loopback_devices:
            self.device_map[f"[{d['index']}] {d['name']}"] = int(d["index"])

        self._setup_style()
        self._build_ui()

        self.root.after(40, self._poll_ui)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        if auto_start:
            self.root.after(400, self.start_recording)

    # ------------------------------------------------------------------ UI
    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

    def _label(self, parent: tk.Misc, text: str, fg: Optional[str] = None) -> tk.Label:
        lbl = tk.Label(parent, text=text, font=("Segoe UI", 9), bg=self.BG, fg=fg or self.MUTED)
        lbl.pack(side=tk.LEFT, padx=(4, 3))
        return lbl

    def _combo(self, parent: tk.Misc, values: List[str], width: int, current: int, cb) -> ttk.Combobox:
        cmb = ttk.Combobox(parent, values=values, state="readonly", width=width, font=("Segoe UI", 9))
        cmb.current(current)
        cmb.pack(side=tk.LEFT, padx=(0, 6))
        cmb.bind("<<ComboboxSelected>>", cb)
        return cmb

    def _button(self, parent: tk.Misc, text: str, cmd, bg: str, side=tk.LEFT, bold: bool = True) -> tk.Button:
        btn = tk.Button(
            parent, text=text, command=cmd, bg=bg, fg="#FFFFFF", activebackground=bg, activeforeground="#FFFFFF",
            relief=tk.FLAT, padx=9, pady=4, cursor="hand2", font=("Segoe UI", 9, "bold" if bold else "normal"),
        )
        btn.pack(side=side, padx=(0, 6) if side == tk.LEFT else (6, 0))
        return btn

    def _build_ui(self) -> None:
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(1180, sw - 40), min(700, sh - 80)
        self.root.geometry(f"{w}x{h}+{max(10, (sw - w) // 2)}+{max(10, (sh - h) // 2 - 20)}")
        self.root.minsize(980, 520)

        # --- Hàng 1 ---
        row1 = tk.Frame(self.root, bg=self.BG, padx=10, pady=6)
        row1.pack(fill=tk.X)
        self.btn_run = self._button(row1, "⏸ Tạm dừng ghi", self.toggle_run_pause, self.OK)
        self.btn_demo = self._button(row1, "🔊 Test giọng khoa học", self.run_demo, self.ACCENT)

        self._label(row1, "Lĩnh vực:")
        self.domain_keys = list(DOMAIN_PROFILES.keys())
        self.cmb_domain = self._combo(
            row1, [DOMAIN_PROFILES[k]["label"] for k in self.domain_keys], 26,
            self.domain_keys.index(self.config.domain_key), self._on_domain,
        )

        self._label(row1, "Nhận diện:")
        self.engine_keys = list(ENGINE_MODES.keys())
        self.cmb_engine = self._combo(
            row1, [ENGINE_MODES[k][0] for k in self.engine_keys], 52,
            self.engine_keys.index(self.config.engine_mode), self._on_engine,
        )

        # --- Hàng 2 ---
        row2 = tk.Frame(self.root, bg=self.BG, padx=10, pady=3)
        row2.pack(fill=tk.X)
        self._label(row2, "Cắt câu:")
        self.profile_keys = list(RECORDING_PROFILES.keys())
        self.cmb_profile = self._combo(
            row2, [str(RECORDING_PROFILES[k]["label"]) for k in self.profile_keys], 40,
            self.profile_keys.index(self.config.recording_profile), self._on_profile,
        )

        self._label(row2, "Khi quá tải:")
        self.lag_labels = list(LAG_OPTIONS.keys())
        self.cmb_lag = self._combo(
            row2, self.lag_labels, 22,
            list(LAG_OPTIONS.values()).index(DEFAULT_MAX_LAG_SEC), self._on_lag,
        )

        self._label(row2, "Nguồn âm:")
        self.cmb_device = self._combo(row2, list(self.device_map.keys()), 34, 0, self._on_device)

        self._label(row2, "Tín hiệu:")
        self.vu = tk.Canvas(row2, width=80, height=13, bg=self.CARD, highlightthickness=1, highlightbackground=self.BORDER)
        self.vu.pack(side=tk.LEFT, padx=(0, 8))
        self.vu_bar = self.vu.create_rectangle(0, 0, 0, 13, fill=self.OK, width=0)

        # --- Hàng 3: Từ khóa chuyên ngành ---
        row3 = tk.Frame(self.root, bg=self.BG, padx=10, pady=3)
        row3.pack(fill=tk.X)
        self._label(row3, "🔑 Từ khóa chuyên ngành của buổi họp (cách nhau bởi dấu phẩy, Enter để áp dụng):", fg="#FDE047")
        self.ent_hotwords = tk.Entry(
            row3, font=("Segoe UI", 9), bg=self.CARD, fg=self.TEXT, insertbackground=self.TEXT, relief=tk.FLAT,
        )
        self.ent_hotwords.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)
        self.ent_hotwords.bind("<Return>", lambda _e: self._apply_hotwords())
        self._button(row3, "Áp dụng", self._apply_hotwords, "#D97706")

        # --- Khung văn bản ---
        body = tk.Frame(self.root, bg=self.BORDER, bd=1)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 4))
        hdr = tk.Frame(body, bg=self.CARD, padx=10, pady=5)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr, text="📜 BẢN GHI NGUYÊN VĂN  —  từ tô vàng = độ tin cậy thấp, nên nghe lại file WAV tại vị trí tương ứng",
            font=("Segoe UI", 9, "bold"), bg=self.CARD, fg="#38BDF8", anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.txt = scrolledtext.ScrolledText(
            body, wrap=tk.WORD, font=("Segoe UI", 11), bg=self.PANEL, fg=self.TEXT,
            insertbackground=self.TEXT, relief=tk.FLAT, padx=12, pady=10, undo=True,
        )
        self.txt.pack(fill=tk.BOTH, expand=True)
        self.txt.tag_configure("ts", foreground=self.MUTED, font=("Consolas", 9))
        self.txt.tag_configure("low", background="#713F12", foreground="#FDE68A")
        self.txt.tag_configure("skip", foreground="#FCA5A5", font=("Segoe UI", 10, "italic"))
        self.txt.insert(
            tk.END,
            "Đang chờ lời nói... Bấm [🔊 Test giọng khoa học] để kiểm tra nhanh.\n"
            "Mẹo: nhập trước các thuật ngữ / tên riêng / tên phương pháp của buổi họp vào ô Từ khóa để tăng độ chính xác.\n\n",
            "ts",
        )
        self._placeholder = True
        self.txt.edit_modified(False)

        # --- Hàng tác vụ ---
        actions = tk.Frame(self.root, bg=self.BG, padx=10, pady=4)
        actions.pack(fill=tk.X)
        self._button(actions, "📋 Copy toàn bộ", self.copy_all, "#0EA5E9")
        self._button(actions, "📄 Mở file .txt", self.open_txt, self.CARD, bold=False)
        self._button(actions, "🧾 Xuất văn bản liền mạch", self.export_clean, self.CARD, bold=False)
        self._button(actions, "📘 Xuất Word (.docx)", self.export_docx, "#0284C7")
        self._button(actions, "📂 Mở thư mục", self.open_folder, self.CARD, bold=False)
        for text, var, cmd in (
            ("🕵️ Ẩn khi share màn hình", self.stealth_share, self._on_stealth),
            ("🕒 Hiện mốc giờ", self.show_timestamps, self._rerender_transcript),
            ("🎙️ Lưu WAV đối chiếu", self.save_wav, self._on_save_wav),
        ):
            tk.Checkbutton(
                actions, text=text, variable=var, command=cmd, font=("Segoe UI", 9), bg=self.BG, fg=self.TEXT,
                selectcolor=self.CARD, activebackground=self.BG, activeforeground=self.TEXT,
            ).pack(side=tk.RIGHT, padx=(6, 0))

        # --- Thanh trạng thái ---
        status = tk.Frame(self.root, bg="#090D16", padx=10, pady=4)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        self.lbl_status = tk.Label(status, text="🟢 Đang khởi tạo...", font=("Segoe UI", 9), bg="#090D16", fg="#A7F3D0", anchor="w")
        self.lbl_status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_metrics = tk.Label(status, text="", font=("Consolas", 9), bg="#090D16", fg=self.MUTED, anchor="e")
        self.lbl_metrics.pack(side=tk.RIGHT)

    # ------------------------------------------------------------- sự kiện
    def _on_domain(self, _e=None) -> None:
        self.config.domain_key = self.domain_keys[self.cmb_domain.current()]
        self._set_status(f"Lĩnh vực: {DOMAIN_PROFILES[self.config.domain_key]['label']}")

    def _on_engine(self, _e=None) -> None:
        key = self.engine_keys[self.cmb_engine.current()]
        self.config.apply_engine_mode(key)
        threading.Thread(target=self.controller.asr.preload, daemon=True).start()
        self._set_status(f"Chế độ nhận diện: {ENGINE_MODES[key][0]}")

    def _on_lag(self, _e=None) -> None:
        label = self.lag_labels[self.cmb_lag.current()]
        self.config.max_lag_sec = LAG_OPTIONS[label]
        self._set_status(f"Cơ chế quá tải: {label}")

    def _on_profile(self, _e=None) -> None:
        key = self.profile_keys[self.cmb_profile.current()]
        self.config.apply_recording_profile(key)
        self._set_status(f"Kiểu cắt câu: {RECORDING_PROFILES[key]['label']}")

    def _on_device(self, _e=None) -> None:
        sel = self.cmb_device.get()
        self.config.device_index = self.device_map.get(sel)
        self._set_status(f"Nguồn âm: {sel}")

    def _on_save_wav(self) -> None:
        self.config.save_wav = bool(self.save_wav.get())
        if self.config.save_wav and self.controller.is_running:
            self.controller.wav_recorder.open()
        self._set_status("Đang lưu WAV đối chiếu" if self.config.save_wav else "Đã tắt lưu WAV (file hiện có vẫn giữ nguyên)")

    def _on_stealth(self) -> None:
        ok = Win32WindowHelper.set_stealth_screen_share(self.root, bool(self.stealth_share.get()))
        self._set_status("Cửa sổ sẽ ẩn khi chia sẻ màn hình" if self.stealth_share.get() and ok else "Cửa sổ hiển thị bình thường khi chia sẻ")

    def _apply_hotwords(self) -> None:
        self.config.user_hotwords = self.ent_hotwords.get().strip()
        self._set_status(f"Đã áp dụng từ khóa: {self.config.user_hotwords or '(trống)'}")

    # ---------------------------------------------------------- điều khiển
    def start_recording(self) -> None:
        self.config.save_wav = bool(self.save_wav.get())
        self.controller.start()
        self._session_start = time.time()
        self.btn_run.configure(text="⏸ Tạm dừng ghi", bg=self.OK)

    def toggle_run_pause(self) -> None:
        if not self.controller.is_running:
            self.start_recording()
            return
        paused = self.controller.toggle_pause()
        if paused:
            self.btn_run.configure(text="▶ Tiếp tục ghi", bg=self.WARN)
            self._set_status("⏸ Đã tạm dừng thu âm (các câu đã thu vẫn tiếp tục được nhận diện)")
        else:
            self.btn_run.configure(text="⏸ Tạm dừng ghi", bg=self.OK)
            self._set_status("🟢 Đang ghi")

    def run_demo(self) -> None:
        self.btn_demo.configure(state=tk.DISABLED, text="⏳ Đang phát mẫu...")

        def worker():
            try:
                import av
                import edge_tts

                sentences = [
                    "In this study, we performed a Monte Carlo simulation to estimate the standard deviation, "
                    "and the regression model achieved an R-squared of 0.87 with a p-value below 0.05.",
                    "Consistent with Smith et al., the in vitro assay showed a statistically significant "
                    "increase in enzyme activity after the treatment.",
                ]
                for sentence in sentences:
                    buf = io.BytesIO()

                    async def synth(txt: str):
                        comm = edge_tts.Communicate(txt, voice="en-US-GuyNeural")
                        async for chunk in comm.stream():
                            if chunk["type"] == "audio":
                                buf.write(chunk["data"])

                    asyncio.run(synth(sentence))
                    buf.seek(0)
                    container = av.open(buf, format="mp3")
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                    pcm = b"".join(bytes(f.planes[0]) for fr in container.decode(audio=0) for f in resampler.resample(fr))
                    mono = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    signal = np.concatenate([mono, np.zeros(int(16000 * 1.1), dtype=np.float32)])
                    self.controller.audio_capture.inject_test_audio(signal, source_label="Demo (EN)")
            except Exception as exc:
                self._on_status_threadsafe(f"Lỗi chạy mẫu: {exc}")
            finally:
                self.root.after(0, lambda: self.btn_demo.configure(state=tk.NORMAL, text="🔊 Test giọng khoa học"))

        threading.Thread(target=worker, daemon=True).start()

    def copy_all(self) -> None:
        content = self.txt.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._set_status(f"📋 Đã copy {len(content.split())} từ vào clipboard (gồm cả chỉnh sửa tay trong khung)")

    def open_txt(self) -> None:
        self._open_path(self.controller.store.txt_path)

    def export_clean(self) -> None:
        path = self.controller.export_clean_text()
        self._set_status(f"🧾 Đã xuất văn bản liền mạch: {path.name}")
        self._open_path(path)

    def export_docx(self) -> None:
        try:
            path = self.controller.export_docx()
            self._set_status(f"📘 Đã xuất Word: {path.name}")
            self._open_path(path)
        except Exception as exc:
            messagebox.showerror("Lỗi xuất Word", str(exc))

    def open_folder(self) -> None:
        self._open_path(self.controller.store.base_dir)

    @staticmethod
    def _open_path(path) -> None:
        try:
            os.startfile(str(path))
        except Exception:
            subprocess.Popen(["explorer", str(path)])

    # --------------------------------------------------------- hiển thị
    def _insert_segment(self, seg: TranscriptSegment) -> None:
        if self._placeholder:
            self.txt.delete("1.0", tk.END)
            self._placeholder = False
        sep = "\n\n" if self.show_timestamps.get() else " "
        if self.show_timestamps.get():
            info = "" if seg.is_skip else f"  •  {seg.confidence * 100:.0f}%  •  {seg.engine}"
            self.txt.insert(tk.END, f"[{seg.timestamp_start.strftime('%H:%M:%S')}{info}]\n", "ts")
        if seg.is_skip:
            self.txt.insert(tk.END, seg.text + sep, "skip")
            return
        self._word_count += len(seg.text.split())
        self._sentence_count += 1
        remaining: Dict[str, int] = {}
        for w in seg.low_confidence_words:
            remaining[w] = remaining.get(w, 0) + 1
        tokens = seg.text.split()
        for i, tok in enumerate(tokens):
            tag = None
            if remaining.get(tok, 0) > 0:
                remaining[tok] -= 1
                tag = "low"
            self.txt.insert(tk.END, tok, tag)
            if i < len(tokens) - 1:
                self.txt.insert(tk.END, " ")
        self.txt.insert(tk.END, sep)

    def _rerender_transcript(self) -> None:
        if self.txt.edit_modified():
            if not messagebox.askyesno(
                "Hiển thị lại",
                "Bạn đã chỉnh sửa tay trong khung văn bản. Hiển thị lại sẽ bỏ các chỉnh sửa đó "
                "(file .txt/.jsonl gốc không bị ảnh hưởng). Tiếp tục?",
            ):
                self.show_timestamps.set(not self.show_timestamps.get())
                return
        self.txt.delete("1.0", tk.END)
        self._placeholder = False
        self._word_count = 0
        self._sentence_count = 0
        for seg in list(self.controller.history):
            self._insert_segment(seg)
        self.txt.edit_modified(False)
        self.txt.see(tk.END)

    def _on_level_threadsafe(self, rms: float, _dev: str) -> None:
        self._latest_rms = rms

    def _on_status_threadsafe(self, msg: str) -> None:
        self._latest_status = msg

    def _set_status(self, msg: str) -> None:
        self._latest_status = msg
        self.lbl_status.configure(text=f"🟢 {msg}")

    def _poll_ui(self) -> None:
        if self._closing:
            return
        lvl = min(1.0, math.sqrt(max(0.0, self._latest_rms) * 18.0))
        self.vu.coords(self.vu_bar, 0, 0, int(lvl * 80), 13)
        self.vu.itemconfig(self.vu_bar, fill=self.OK if lvl < 0.75 else self.WARN)
        self.lbl_status.configure(text=f"🟢 {self._latest_status}")

        added = False
        modified_before = self.txt.edit_modified()
        while True:
            try:
                seg: TranscriptSegment = self.controller.ui_queue.get_nowait()
            except Exception:
                break
            # Chèn vào cuối; giữ nguyên vị trí cuộn nếu người dùng đang đọc ở trên
            at_bottom = self.txt.yview()[1] > 0.98
            self._insert_segment(seg)
            if at_bottom:
                self.txt.see(tk.END)
            added = True
        if added and not modified_before:
            self.txt.edit_modified(False)

        asr = self.controller.asr
        elapsed = int(time.time() - self._session_start)
        extra = ""
        if asr.backlog:
            extra += f" | ⏳ Chờ: {asr.backlog} câu, trễ {asr.current_lag_sec:.0f}s"
        if asr.skipped_segments:
            extra += f" | ⚠ Đã bỏ qua: {asr.skipped_segments}"
        n_sent = self._sentence_count
        self.lbl_metrics.configure(
            text=f"Phiên: {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d} | {n_sent} câu, {self._word_count} từ{extra}"
        )
        self.root.after(80, self._poll_ui)

    def on_close(self) -> None:
        backlog = self.controller.asr.backlog
        if backlog and not messagebox.askyesno(
            "Đóng ứng dụng",
            f"Còn {backlog} câu đang chờ nhận diện. Ứng dụng sẽ chờ tối đa 3 phút để ghi xong rồi mới đóng.\n"
            f"(Âm thanh đã nằm trong file WAV nên không mất.) Tiếp tục đóng?",
        ):
            return
        self._closing = True
        self.lbl_status.configure(text="⏳ Đang hoàn tất các câu còn lại và đóng file...")
        self.root.update_idletasks()

        def finish():
            try:
                self.controller.stop(drain_timeout=180.0)
                if self.controller.history:
                    self.controller.export_docx()
            except Exception:
                pass

        worker = threading.Thread(target=finish, daemon=True)
        worker.start()

        def wait_then_destroy():
            if worker.is_alive():
                self.root.after(200, wait_then_destroy)
            else:
                self.root.destroy()

        wait_then_destroy()


def main() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    root = tk.Tk()
    VerbatimRecorderApp(root, auto_start=True)
    root.mainloop()


if __name__ == "__main__":
    main()
