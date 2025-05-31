import subprocess
import time
from queue import Empty
from config import stream_q, stream_event, TARGET_FPS, UDP_IP, UDP_PORT, STREAM_BITRATE

# Hàm xử lý phát trực tuyến video
def stream_worker(ev, frame_period: float, udp_ip: str, udp_port: int, bitrate: str):
    ffmpeg_proc = None  # Biến lưu tiến trình FFmpeg
    while not ev.is_set():  # Chạy vòng lặp cho đến khi sự kiện dừng được thiết lập
        try:
            frame = stream_q.get(timeout=0.2)  # Lấy frame từ hàng đợi stream_q
        except Empty:
            continue  # Nếu hàng đợi rỗng, tiếp tục vòng lặp
        h, w, _ = frame.shape  # Lấy kích thước khung hình
        if ffmpeg_proc is None:  # Nếu không có tiến trình FFmpeg
            cmd = [  # Tạo lệnh FFmpeg để phát trực tuyến qua UDP
                "ffmpeg", "-y",
                "-loglevel", "quiet",
                "-fflags", "+genpts", "-flags", "low_delay", "-fflags", "nobuffer",
                "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", f"{w}x{h}", "-framerate", str(int(TARGET_FPS)), "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                "-b:v", bitrate,
                "-f", "mpegts",
                f"udp://{udp_ip}:{udp_port}?pkt_size=1316&overrun_nonfatal=1"
            ]
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # Khởi tạo tiến trình FFmpeg
            print(f"[STREAM] Start streaming → udp://{udp_ip}:{udp_port} @ {bitrate}")  # In thông báo bắt đầu phát
        ffmpeg_proc.stdin.write(frame.tobytes())  # Gửi khung hình vào tiến trình FFmpeg
        time.sleep(frame_period)  # Chờ đúng khoảng thời gian giữa các khung hình
    if ffmpeg_proc:  # Nếu tiến trình FFmpeg vẫn đang chạy
        ffmpeg_proc.stdin.close()  # Đóng đầu vào
        ffmpeg_proc.wait()  # Đợi tiến trình hoàn tất
    print("[STREAM] Thread stop")  # In thông báo dừng luồng
