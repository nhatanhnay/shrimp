import cv2
import subprocess
import time

VIDEO_PATH = 'tomtep.mp4'
DEST_IP = '127.0.0.1'
DEST_PORT = 5000
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS  # ~0.0333 giây

# Lấy kích thước video
cap_probe = cv2.VideoCapture(VIDEO_PATH)
if not cap_probe.isOpened():
    raise RuntimeError("Không mở được video")
width  = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_probe.release()

# Lệnh FFmpeg
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', str(TARGET_FPS),
    '-i', '-',
    '-an',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-f', 'rtp',
    f'rtp://{DEST_IP}:{DEST_PORT}'
]

# Mở FFmpeg pipe
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

print(f"🔁 Đang stream lặp lại video ở {TARGET_FPS} FPS đến rtp://{DEST_IP}:{DEST_PORT}")
print("➡️ Mở VLC và dán: rtp://@:5000")

try:
    while True:
        cap = cv2.VideoCapture(VIDEO_PATH)
        while cap.isOpened():
            start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            proc.stdin.write(frame.tobytes())

            # Giữ đúng FPS bằng cách delay nếu cần
            elapsed = time.time() - start
            delay = FRAME_INTERVAL - elapsed
            if delay > 0:
                time.sleep(delay)

        cap.release()
        time.sleep(0.1)  # Chờ nhẹ khi loop
except KeyboardInterrupt:
    print("\n🛑 Dừng stream.")
finally:
    proc.stdin.close()
    proc.wait()
    cv2.destroyAllWindows()
