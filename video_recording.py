import subprocess
import time
import traceback
import os
from queue import Empty
from config import RECORD_CHUNK_SEC, record_q, record_event
from upload_ggdrive import upload_to_drive

# Hàm xử lý ghi video
def record_worker(ev):
    ffmpeg_proc = None  # Biến lưu tiến trình FFmpeg
    start_ts = 0.0  # Thời điểm bắt đầu ghi video
    while not ev.is_set():  # Chạy vòng lặp cho đến khi sự kiện dừng được thiết lập
        try:
            frame = record_q.get(timeout=0.2)  # Lấy frame từ hàng đợi record_q
        except Empty:
            continue  # Nếu hàng đợi rỗng, tiếp tục vòng lặp
        if ffmpeg_proc is None:  # Nếu không có tiến trình FFmpeg
            start_ts = time.time()  # Ghi lại thời gian bắt đầu
            out_path = f"vid_{int(start_ts)}.mp4"  # Tạo tên file video
            h, w, _ = frame.shape  # Lấy kích thước khung hình
            cmd = [  # Tạo lệnh FFmpeg để ghi video
                'ffmpeg','-y',
                '-loglevel', 'quiet',
                '-f','rawvideo','-vcodec','rawvideo',
                '-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(25),
                '-i','-','-c:v','libx264','-preset','veryfast','-crf','23',out_path
            ]
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # Khởi tạo tiến trình FFmpeg
            print(f"[REC] Start {out_path}")  # In thông báo bắt đầu ghi
        ffmpeg_proc.stdin.write(frame.tobytes())  # Gửi khung hình vào tiến trình FFmpeg
        if time.time() - start_ts >= RECORD_CHUNK_SEC:  # Nếu đủ thời gian đoạn video (30 giây)
            ffmpeg_proc.stdin.close()  # Đóng đầu vào FFmpeg
            ffmpeg_proc.wait()  # Đợi tiến trình hoàn tất
            print(f"[REC] Saved vid_{int(start_ts)}.mp4")  # In thông báo lưu file
            upload_to_drive(out_path, out_path)  # Tải file lên Google Drive
            print("[REC] Uploaded to Drive →", out_path)  # In thông báo tải lên thành công
            try:
                os.remove(out_path)  # Xóa file cục bộ
            except Exception:
                print(traceback.format_exc())  # In lỗi nếu không xóa được
            ffmpeg_proc = None  # Đặt lại tiến trình FFmpeg
    if ffmpeg_proc:  # Nếu tiến trình FFmpeg vẫn đang chạy
        ffmpeg_proc.stdin.close()  # Đóng đầu vào
        ffmpeg_proc.wait()  # Đợi tiến trình hoàn tất
    print("[REC] Thread stop")  # In thông báo dừng luồng
