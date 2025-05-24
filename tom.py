import cv2  # Thư viện OpenCV để xử lý ảnh và video
import numpy as np  # Thư viện NumPy để xử lý mảng và tính toán
from paddleocr import PaddleOCR  # Thư viện PaddleOCR để nhận diện văn bản
import time  # Thư viện để đo thời gian
import threading  # Thư viện để quản lý luồng
from collections import deque  # Cấu trúc deque để lưu lịch sử thời gian
import queue  # Hàng đợi để truyền khung hình
import subprocess  # Chạy lệnh GStreamer cho RTSP
import os  # Thư viện để quản lý file và thư mục
import argparse  # Thư viện để đọc tham số dòng lệnh
from upload_ggdrive import upload_to_drive  # Thư viện để upload video lên Google Drive

# Khởi tạo PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
# - use_angle_cls=True: Bật phân loại góc để phát hiện hướng ảnh
# - lang='en': Ngôn ngữ tiếng Anh cho OCR
# - use_gpu=True: Dùng GPU để tăng tốc trên Jetson Nano
# - show_log=False: Tắt log chi tiết của PaddleOCR

# Cấu hình cơ bản
TARGET_FPS = 30  # Mục tiêu FPS (khung hình/giây)
FRAME_TIME = 1.0 / TARGET_FPS  # Thời gian giữa các khung hình (giây)
OCR_CONFIDENCE_THRESHOLD = 0.8  # Ngưỡng độ tin cậy tối thiểu cho OCR
LINE_THICKNESS = 2  # Độ dày đường viền (không dùng vì không có GUI)

# Vùng detect
DETECTION_ZONE_LEFT_X_RATIO = 0.1  # Tỷ lệ X trái của vùng phát hiện
DETECTION_ZONE_RIGHT_X_RATIO = 0.45  # Tỷ lệ X phải của vùng phát hiện

# Biến trạng thái cho OCR
last_detected_number = None  # Số được phát hiện lần cuối
last_detection_time = 0  # Thời điểm phát hiện lần cuối
DETECTION_COOLDOWN = 5  # Thời gian chờ trước khi phát hiện số mới
last_detection_info = None  # Thông tin phát hiện gần nhất
detection_expiry_time = 1  # Thời điểm hết hạn kết quả
DETECTION_DISPLAY_SECONDS = 2.0  # Thời gian lưu kết quả

# Threading
MAX_WORKER_THREADS = 1  # Số luồng tối đa cho OCR
thread_pool = []  # Danh sách các luồng
thread_lock = threading.Lock()  # Khóa để đồng bộ biến toàn cục
processing_times = deque(maxlen=10)  # Lưu thời gian xử lý OCR
frame_queue = queue.Queue(maxsize=100)  # Hàng đợi để truyền khung hình cho luồng ghi

# File log và video
LOG_FILE = "/home/jetson/ocr_results.txt"  # Đường dẫn lưu kết quả OCR
VIDEO_DIR = "/home/jetson/videos/"  # Thư mục lưu video
MODE_FILE = "/home/jetson/mode.txt"  # File lưu trạng thái chế độ
RECORD_COMMAND = "/home/jetson/record_command.txt"  # File nhận lệnh ghi video

# Cấu hình mạng
REMOTE_HOST = "YOUR_B_IP"  # IP máy tính tại điểm B (thay bằng IP thực)
REMOTE_USER = "your_user"  # Username trên máy tính điểm B
REMOTE_PATH = "/path/to/storage/"  # Đường dẫn lưu video trên điểm B
SSH_KEY_PATH = "/home/jetson/.ssh/id_rsa"  # Đường dẫn SSH key trên Jetson

def parse_arguments():
    """Đọc tham số dòng lệnh để chọn chế độ"""
    parser = argparse.ArgumentParser(description="Chạy chế độ Detect hoặc Record")
    parser.add_argument("--mode", choices=["detect", "record"], default="detect",
                        help="Chế độ: detect (OCR) hoặc record (ghi video)")
    return parser.parse_args()

def correct_image_orientation(image):
    """Xoay ảnh về đúng hướng dựa trên pattern văn bản"""
    try:
        ocr_result = ocr.ocr(image, cls=True)  # Chạy OCR để lấy văn bản và góc
        if not ocr_result or not ocr_result[0]:
            return image, 0

        pattern = "Số:"  # Pattern cố định
        detected_angle = 0
        for line in ocr_result[0]:
            text, confidence = line[1]
            if pattern in text and confidence > OCR_CONFIDENCE_THRESHOLD:
                box = np.array(line[0])
                if box[0][1] > box[1][1]:
                    detected_angle = 90
                elif box[2][1] > box[1][1]:
                    detected_angle = 180
                break

        if detected_angle == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif detected_angle == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif detected_angle == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image, detected_angle
    except Exception as e:
        print(f"Orientation Error: {e}")
        return image, 0

def perform_simple_ocr(image):
    """Thực hiện OCR, ưu tiên số gần pattern"""
    try:
        start_time = time.time()
        ocr_result = ocr.ocr(image, cls=True)
        ocr_time = time.time() - start_time
        processing_times.append(ocr_time)

        if ocr_result and ocr_result[0]:
            pattern = "Số:"
            pattern_box = None
            number_text = None
            number_box = None
            confidence = 0

            for line in ocr_result[0]:
                box, (text, conf) = line
                if pattern in text and conf > OCR_CONFIDENCE_THRESHOLD:
                    pattern_box = np.array(box)
                    break

            if pattern_box is not None:
                for line in ocr_result[0]:
                    box, (text, conf) = line
                    cleaned_text = ''.join(filter(str.isdigit, text))
                    if cleaned_text and conf > OCR_CONFIDENCE_THRESHOLD:
                        pattern_center = np.mean(pattern_box, axis=0)
                        number_center = np.mean(np.array(box), axis=0)
                        distance = np.linalg.norm(pattern_center - number_center)
                        if distance < 100:
                            number_text = cleaned_text
                            number_box = box
                            confidence = conf
                            break

            if number_text:
                return number_text, confidence, number_box
    except Exception as e:
        print(f"OCR Error: {e}")
    return None, 0, None

def ocr_worker(frame, thread_id):
    """Worker thread để xử lý OCR"""
    global last_detected_number, last_detection_time, last_detection_info, detection_expiry_time

    try:
        detect_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        detect_frame, detected_angle = correct_image_orientation(detect_frame)
        number_text, confidence, box = perform_simple_ocr(detect_frame)
        current_time = time.time()

        with thread_lock:
            if number_text:
                box = np.array(box)
                box = box * 2
                if ((box[0][0] > frame.shape[1] * 0.2 and
                     box[1][0] < frame.shape[1] * 0.8 and
                     box[0][1] > frame.shape[0] * 0.1 and
                     box[3][1] < frame.shape[0] * 0.9) and
                    (number_text != last_detected_number or
                     current_time - last_detection_time > DETECTION_COOLDOWN)):
                    
                    if number_text != last_detected_number:
                        log_message = f"{time.ctime()}: {number_text} (Conf: {confidence:.2f}, Angle: {detected_angle}°)"
                        print(log_message)
                        with open(LOG_FILE, "a") as f:
                            f.write(log_message + "\n")

                    last_detected_number = number_text
                    last_detection_time = current_time
                    last_detection_info = (int(box[0][0]), int(box[0][1]),
                                         int(box[2][0] - box[0][0]),
                                         int(box[2][1] - box[0][1]),
                                         number_text, confidence)
                    detection_expiry_time = current_time + DETECTION_DISPLAY_SECONDS
                    
    except Exception as e:
        print(f"Worker error: {e}")

def record_worker():
    """Worker thread để ghi video khi nhận lệnh"""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec để ghi video
    video_writer = None  # Đối tượng VideoWriter
    video_start_time = None  # Thời gian bắt đầu video
    while True:
        try:
            frame = frame_queue.get(timeout=5)  # Lấy khung hình
            current_time = time.time()
            
            if video_writer is None:  # Khởi tạo video writer
                video_path = f"video_{current_time}.avi"
                video_writer = cv2.VideoWriter(video_path, fourcc, TARGET_FPS,
                                                (frame.shape[1], frame.shape[0]))
                video_start_time = current_time
                print(f"Started Record: {video_path}")

            video_writer.write(frame)  # Ghi khung hình

            # Dừng sau 30 giây để tránh file quá lớn
            if current_time - video_start_time > 30:
                video_writer.release()
                video_writer = None
                print(f"Saved video: {video_path}")
                # Upload video lên Google Drive
                upload_to_drive(video_path, video_path)
                print(f"Uploaded vid to gg_drive")
                # Xóa video cũ
                os.remove(video_path)
                print(f"Deleted video: {video_path}")
            else:
                time.sleep(0.1)  # Nghỉ để giảm tải CPU khi không ghi
                
        except Exception as e:
            print(f"Record error: {e}")
            if video_writer is not None:
                video_writer.release()
                video_writer = None

def rtsp_worker():
    """Worker thread để stream RTSP"""
    try:
        # Pipeline GStreamer để stream camera
        pipeline = (
            "v4l2src ! videoconvert ! x264enc tune=zerolatency bitrate=1000 ! "
            "rtspclientsink location=rtsp://localhost:8554/stream"
        )
        # Nếu dùng CSI camera, thay v4l2src bằng:
        # nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGR ! videoconvert
        subprocess.run(["gst-launch-1.0", "-e", *pipeline.split()], check=True)
    except Exception as e:
        print(f"RTSP error: {e}")

def cleanup_threads():
    """Dọn dẹp các luồng đã hoàn thành"""
    global thread_pool
    thread_pool = [t for t in thread_pool if t.is_alive()]

def start_ocr_process(frame):
    """Khởi động OCR process"""
    if len([t for t in thread_pool if t.is_alive()]) >= MAX_WORKER_THREADS:
        return False
    
    cleanup_threads()
    thread_id = len(thread_pool) + 1
    thread = threading.Thread(
        target=ocr_worker,
        args=(frame.copy(), thread_id),
        daemon=True,
    )
    thread.start()
    thread_pool.append(thread)
    return True

def main():
    # Đọc chế độ từ tham số dòng lệnh
    args = parse_arguments()
    mode = args.mode
    print(f"Chạy ở chế độ: {mode}")

    # Khởi tạo camera
    # USB webcam
    cap = cv2.VideoCapture("tomtep.mp4")  # Thay bằng đường dẫn video của bạn
    # Nếu dùng CSI camera, uncomment dòng sau và comment dòng trên:
    # cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGR ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    last_frame_time = time.time()
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    # Khởi động luồng tương ứng với chế độ
    if mode == "record":
        # Khởi động luồng ghi video
        record_thread = threading.Thread(target=record_worker, daemon=True)
        record_thread.start()
        # Khởi động luồng RTSP
        # rtsp_thread = threading.Thread(target=rtsp_worker, daemon=True)
        # rtsp_thread.start()
        # print("Chế độ Record: Đang chờ lệnh ghi video. Stream RTSP tại rtsp://JETSON_IP:8554/stream")
    else:
        print("Chế độ Detect: Bắt đầu nhận diện số. Kết quả lưu vào", LOG_FILE)

    while True:
        current_time = time.time()
        if current_time - last_frame_time < FRAME_TIME:
            continue
            
        last_frame_time = current_time
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Giảm kích thước để tối ưu

        fps_frame_count += 1
        if current_time - fps_start_time > 1:
            fps = fps_frame_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time

        if mode == "detect":
            # Chạy OCR
            if fps_frame_count % 5 == 0:  # Giảm tần suất xoay để tối ưu
                frame, detected_angle = correct_image_orientation(frame)
            else:
                detected_angle = 0
            start_ocr_process(frame)
        else:
            # Chế độ Record: Thêm khung hình vào hàng đợi
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    print("Đang tắt...")
    for thread in thread_pool:
        if thread.is_alive():
            thread.join(timeout=1)
    cap.release()

if __name__ == "__main__":
    main()