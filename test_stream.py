#!/usr/bin/env python3
import cv2
import subprocess
import time
import sys

# === CẤU HÌNH ===
# Nguồn video: 0 = webcam, hoặc đường dẫn tới file
source = "tomtep.mp4"  # Ví dụ: "0" cho webcam, hoặc "video.mp4" cho file video

# Thông tin UDP stream
udp_ip   = "127.0.0.1"
udp_port = 1234

# Bitrate cho H.264 (ví dụ "800k", "1M", "2M")
bitrate = "800k"
# =================

# Mở nguồn video
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Cannot open source: {source}", file=sys.stderr)
    sys.exit(1)

# Lấy độ phân giải và FPS từ nguồn
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    print("Warning: cannot determine input FPS, defaulting to 30", file=sys.stderr)
    fps = 30.0
frame_time = 1.0 / fps

print(f"Input: {width}×{height} @ {fps:.2f} FPS")
print(f"Streaming to udp://{udp_ip}:{udp_port} @ {bitrate}")

# Xây lệnh FFmpeg
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-fflags", "+genpts", "-flags", "low_delay", "-fflags", "nobuffer",
    "-f", "rawvideo", "-pixel_format", "bgr24",
    "-video_size", f"{width}x{height}", "-framerate", str(int(fps)), "-i", "-",
    "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
    "-b:v", bitrate,
    "-f", "mpegts",
    f"udp://{udp_ip}:{udp_port}?pkt_size=1316&overrun_nonfatal=1"
]

# Khởi FFmpeg với pipe
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

try:
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of stream or cannot read frame", file=sys.stderr)
            break

        # Gửi frame thô vào FFmpeg
        proc.stdin.write(frame.tobytes())

        # Hiển thị preview
        cv2.imshow("Preview (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Throttle để khớp FPS gốc
        elapsed = time.time() - start
        to_wait = frame_time - elapsed
        if to_wait > 0:
            time.sleep(to_wait)

except KeyboardInterrupt:
    print("Interrupted by user", file=sys.stderr)
finally:
    # Cleanup
    cap.release()
    proc.stdin.close()
    proc.wait()
    cv2.destroyAllWindows()
