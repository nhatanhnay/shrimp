from queue import Queue
from collections import deque
from typing import List, Optional, Set
import threading

# Video configuration
VIDEO_SOURCE = "tomtep.mp4"  # 0 for webcam
TARGET_FPS = 25  # UI refresh rate
OCR_INTERVAL = 0.5  # seconds between OCR runs
OCR_CONFIDENCE_THRESHOLD = 0.8  # Ngưỡng độ tin cậy của OCR (0.8)
DETECTION_DISPLAY_SECONDS = 2.0  # Thời gian hiển thị số thẻ được phát hiện (2 giây)

# Detection zone as width-ratios
ZONE_L = 0.10  # Tỷ lệ chiều rộng của khung hình (10%) để xác định vùng ROI bên trái
ZONE_R = 0.45  # Tỷ lệ chiều rộng của khung hình (45%) để xác định vùng ROI bên phải
LINE_THICKNESS = 2  # Độ dày đường kẻ vẽ vùng ROI trên khung hình

# RS-485 serial
SERIAL_PORT_CANDIDATES = ("/dev/ttyTHS1", "/dev/ttyTHS0", "/dev/ttyUSB0")  # Danh sách các cổng serial để thử kết nối RS-485
BAUD_RATE = 9600  # Tốc độ baud cho giao tiếp serial (9600)

# Worker threads
MAX_WORKER_THREADS = 1  # OCR workers at once: Số luồng OCR tối đa chạy đồng thời (1)
RECORD_CHUNK_SEC = 30  # length of each video piece: Độ dài mỗi đoạn video được ghi (30 giây)

# Streaming config
UDP_IP = "100.122.59.77"  # Địa chỉ IP cho phát trực tuyến UDP
UDP_PORT = 5000  # Cổng cho phát trực tuyến UDP
STREAM_BITRATE = "600K"  # Bitrate cho video phát trực tuyến (600K)

# Global variables
serial_port = None  # Biến toàn cục cho kết nối RS-485
modes_set: Set[str] = {"detect"}  # Tập hợp các chế độ đang hoạt động (`detect` là mặc định)
fps_deque = deque(maxlen=10)  # Hàng đợi lưu FPS (tối đa 10 giá trị)
stop_event = threading.Event()  # Sự kiện luồng để điều khiển dừng chương trình
record_event = threading.Event()  # Sự kiện luồng để điều khiển ghi video
stream_event = threading.Event()  # Sự kiện luồng để điều khiển phát trực tuyến
worker_threads: List[threading.Thread] = []  # Danh sách các luồng OCR đang chạy
processing_times = deque(maxlen=10)  # Hàng đợi lưu thời gian xử lý OCR (tối đa 10 giá trị)
last_detected_number: Optional[str] = None  # Số thẻ được phát hiện gần nhất
last_detection_expire = 0.0  # Thời điểm hết hạn hiển thị số thẻ
record_q = Queue(maxsize=30)  # Hàng đợi lưu frame cho ghi video (tối đa 30 frame)
stream_q = Queue(maxsize=30)  # Hàng đợi lưu frame cho phát trực tuyến (tối đa 30 frame)