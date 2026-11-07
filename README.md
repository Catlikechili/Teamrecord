# 🎙️ Verbatim Scientific Recorder

Ứng dụng Windows **ghi lại nguyên văn lời nói tiếng Anh** trong cuộc họp (MS Teams, Zoom, Google Meet… hoặc bất kỳ âm thanh nào phát ra loa/tai nghe), tối ưu cho **văn phong khoa học** và cuộc họp dài tới **3 tiếng**.

- Không dịch, không tóm tắt — chỉ ghi chép chính xác để bạn tự tổng hợp sau.
- Chạy **offline** bằng Whisper (không cần mạng sau khi cài, không gửi âm thanh ra ngoài ở chế độ mặc định).
- Lưu tức thì sau mỗi câu ra `.txt`, `.md`, `.jsonl` + file âm thanh `.wav` để đối chiếu; xuất Word `.docx` bằng 1 nút.

## Cài đặt (dành cho người tải ZIP)

1. Bấm **Code → Download ZIP** trên trang GitHub, giải nén ra một thư mục bất kỳ.
2. Bấm đúp **`install.bat`**. Trình cài đặt sẽ tự:
   - tìm Python 3.10+ (nếu chưa có sẽ thử cài Python 3.12 bằng `winget`),
   - tạo môi trường ảo `.venv` riêng trong thư mục (không ảnh hưởng Python của máy),
   - cài thư viện, tải trước mô hình Whisper `base` (~145 MB),
   - kiểm tra thiết bị âm thanh và tạo shortcut **Verbatim Recorder** trên Desktop.
3. Mở ứng dụng bằng shortcut hoặc **`run.bat`**.

> Lần cài đầu cần mạng và mất khoảng 3–10 phút tùy tốc độ mạng. Nếu `winget` không có sẵn, trình cài đặt sẽ mở trang tải Python — cài Python 3.12 (tick **Add python.exe to PATH**) rồi chạy lại `install.bat`.

## Sử dụng

| Mục | Ý nghĩa |
| :--- | :--- |
| **Lĩnh vực** | Câu mồi + từ khóa theo ngành (Khoa học tổng hợp, AI/CS/Toán, Y sinh/Hóa, Vật lý/Kỹ thuật, Kinh tế lượng/Thống kê) |
| **🔑 Từ khóa chuyên ngành** | Nhập trước tên phương pháp, tên riêng, từ viết tắt của buổi họp → tăng độ chính xác rõ rệt |
| **Nhận diện** | `Whisper base` (mặc định, ổn định nhất) · `base.en` (chính xác hơn) · `small.en` (chính xác nhất, chậm ~3 lần) · `Whisper + Google` (cần mạng) |
| **Cắt câu** | Khoa học – trọn câu dài (nghỉ 0.85 s, tối đa 18 s) · Hội thoại – câu vừa |
| **Khi quá tải** | Nếu máy xử lý không kịp quá X giây, bỏ qua phần tồn đọng và ghi mốc `[BỎ QUA … nghe lại WAV từ hh:mm:ss]` |
| Từ **tô vàng** | Độ tin cậy thấp — nên nghe lại file WAV ở vị trí tương ứng |

Kết quả lưu trong thư mục `transcripts/`:

| File | Nội dung |
| :--- | :--- |
| `Verbatim_<thời gian>.txt` | Nguyên văn kèm giờ `[HH:MM:SS]` |
| `Verbatim_<thời gian>.md` | Bảng: giờ, vị trí trong WAV, độ tin cậy, nội dung (từ nghi ngờ có `[?]`) |
| `Verbatim_<thời gian>.jsonl` | Dữ liệu đầy đủ từng câu (dùng để khôi phục/phân tích) |
| `Verbatim_<thời gian>.wav` | Âm thanh 16 kHz liên tục (~115 MB/giờ) để đối chiếu |
| `.docx`, `_lien_mach.txt` | Tạo khi bấm *Xuất Word* / *Xuất văn bản liền mạch* (tự xuất Word khi đóng ứng dụng) |

## Yêu cầu hệ thống
- Windows 10 (2004+) hoặc Windows 11, CPU 4 nhân, RAM ≥ 8 GB.
- Mô hình `base` trên Intel i7-3720QM (2012): xử lý ~0,25 giây cho mỗi giây lời nói khi máy rảnh.

## Giới hạn đã biết
- Khi lời nói khó nghe, mô hình vẫn có thể sai (vd. "an" → "in"). Từ nghi ngờ được tô vàng; file WAV dùng để kiểm tra.
- Chế độ `Whisper + Google` dùng khóa mặc định của thư viện SpeechRecognition — chỉ dành cho dùng cá nhân, có thể bị giới hạn bất cứ lúc nào.
- Câu mồi lĩnh vực được thiết kế cho tiếng Anh.

## ⚠️ Lưu ý pháp lý
Hãy **thông báo và xin sự đồng ý** của những người tham gia trước khi ghi âm/ghi chép cuộc họp, theo quy định của tổ chức và pháp luật nơi bạn sinh sống.

## Dành cho nhà phát triển
```powershell
.venv\Scripts\python.exe test_suite.py   # 9 bài kiểm thử (một số bài cần mạng để tạo giọng mẫu)
```
| File | Vai trò |
| :--- | :--- |
| `core_engine.py` | Thu âm WASAPI loopback, cắt câu, Whisper, cơ chế bỏ qua, lưu trữ |
| `scientific_normalizer.py` | Câu mồi/từ khóa theo lĩnh vực, chuẩn hóa thuật ngữ, khử lặp, hợp nhất Whisper + Google |
| `main.py` | Giao diện Tkinter |
| `install.ps1` / `install.bat` / `run.bat` | Cài đặt và khởi chạy |
