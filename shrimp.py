import cv2, gc, traceback
import numpy as np
from paddleocr import PaddleOCR
import time
import threading
from collections import deque
import serial  # Thêm thư viện pyserial để giao tiếp RS485
from upload_ggdrive import upload_to_drive  
import os
from flask import Flask, request, jsonify
from queue import Queue, Full, Empty
frame_q = Queue(maxsize=5)

# Khởi tạo PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
mode = 'detect'  # Mặc định là chế độ phát hiện
stop_event = threading.Event()
worker_thread = None
# Cấu hình cơ bản
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS
OCR_CONFIDENCE_THRESHOLD = 0.8
LINE_THICKNESS = 2
OCR_INTERVAL = 0.5  # Chạy OCR mỗi 0.5 giây để giảm tải

# Vùng detect
DETECTION_ZONE_LEFT_X_RATIO = 0.1
DETECTION_ZONE_RIGHT_X_RATIO = 0.45

# Biến trạng thái
last_detected_number = None
last_detection_time = 0
DETECTION_COOLDOWN = 7
last_detection_info = None
detection_expiry_time = 1
DETECTION_DISPLAY_SECONDS = 2.0

# Threading
MAX_WORKER_THREADS = 1
thread_pool = []
thread_lock = threading.Lock()
processing_times = deque(maxlen=10)

# Cấu hình RS485
SERIAL_PORT = '/dev/ttyTHS1'  # Thay bằng cổng thực tế
BAUD_RATE = 9600  # Thay bằng baud rate phù hợp
serial_port = None

def init_serial():
    """Khởi tạo kết nối RS485."""
    global serial_port
    possible_ports = ['/dev/ttyTHS1', '/dev/ttyTHS0', '/dev/ttyUSB0']
    for port in possible_ports:
        try:
            serial_port = serial.Serial(
                port=port,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"Đã kết nối với RS485 trên {port} với baud rate {BAUD_RATE}")
            return True
        except serial.SerialException as e:
            print(f"Lỗi kết nối RS485 trên {port}: {e}")
            serial_port = None
    print("Không thể kết nối với bất kỳ cổng RS485 nào. Kiểm tra quyền hoặc kết nối phần cứng.")
    return False

def send_rs485_data(data):
    """Gửi dữ liệu qua RS485."""
    global serial_port
    if serial_port and serial_port.is_open:
        try:
            start_time = time.time()
            serial_port.write(data.encode('ascii'))
            print(f"Đã gửi dữ liệu qua RS485: {data} (Thời gian: {time.time() - start_time:.3f}s)")
        except serial.SerialException as e:
            print(f"Lỗi khi gửi dữ liệu qua RS485: {e}")
    else:
        print("Cổng RS485 không khả dụng, không thể gửi dữ liệu")

def correct_card_orientation_by_text_position(image):
    """Nếu text nằm ở nửa dưới ảnh → xoay lại 180°"""
    try:
        h = image.shape[0]
        result = ocr.ocr(image, cls=True)
        if result and result[0]:
            for line in result[0]:
                box = line[0]
                y_coords = [point[1] for point in box]
                center_y = sum(y_coords) / len(y_coords)

                if center_y > h / 2:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                break
    except Exception as e:
        print(f"Orientation check error: {e}")
    return image

def perform_simple_ocr(image):
    try:
        image = correct_card_orientation_by_text_position(image)
        start_time = time.time()
        ocr_result = ocr.ocr(image, cls=True)
        ocr_time = time.time() - start_time
        processing_times.append(ocr_time)

        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                text, confidence = line[1]
                cleaned_text = ''.join(filter(str.isdigit, text))
                if cleaned_text and len(cleaned_text) >= 2 and confidence > OCR_CONFIDENCE_THRESHOLD:
                    return cleaned_text, confidence
    except Exception as e:
        print(f"OCR Error: {e}")
    return None, 0

def ocr_worker(frame, coords):
    """Worker thread đơn giản"""
    global last_detected_number, last_detection_time, last_detection_info, detection_expiry_time
    
    try:
        x_start, y_start, x_end, y_end = coords
        processing_zone = frame[y_start:y_end, x_start:x_end]
        
        if processing_zone.size == 0:
            return

        number_text, confidence = perform_simple_ocr(processing_zone)
        
        if number_text:
            box_info = (x_start, y_start, x_end-x_start, y_end-y_start, 
                       number_text, confidence)
            current_time = time.time()

            with thread_lock:
                if current_time < detection_expiry_time:
                    return

                if number_text != last_detected_number:
                    print(f"THẺ MỚI: {number_text} (Độ tin cậy: {confidence:.2f})")
                    last_detected_number = number_text
                    last_detection_time = current_time
                    last_detection_info = box_info
                    detection_expiry_time = current_time + DETECTION_DISPLAY_SECONDS
                    
                    rs485_data = f"*{number_text}#"
                    send_rs485_data(rs485_data)
                    
    except Exception as e:
        print(f"Worker error: {e}")

def cleanup_threads():
    """Dọn dẹp thread"""
    global thread_pool
    thread_pool = [t for t in thread_pool if t.is_alive()]

def is_significantly_new_number(new, old):
    """Trả về True nếu số mới thực sự khác biệt"""
    if not old:
        return True
    if new == old:
        return False
    if new in old or old in new:
        return False
    return True

def shrimpt_detect(frame, stop_ev):
    last_frame_time = time.time()
    last_ocr_time = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    print("Start detect. Press 'q' to exit.")

    while not stop_ev.is_set():
        try:
            frame = frame_q.get(timeout=0.2)     # đợi frame mới
        except Empty:
            continue
        current_time = time.time()
        if current_time - last_frame_time < FRAME_TIME:
            continue
            
        last_frame_time = current_time

        fps_frame_count += 1
        if current_time - fps_start_time > 1:
            fps = fps_frame_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time

        display_frame = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        
        left_x = int(frame_w * DETECTION_ZONE_LEFT_X_RATIO)
        right_x = int(frame_w * DETECTION_ZONE_RIGHT_X_RATIO)
        cv2.line(display_frame, (left_x, 0), (left_x, frame_h), 
                (0, 0, 255), LINE_THICKNESS)
        cv2.line(display_frame, (right_x, 0), (right_x, frame_h), 
                (0, 0, 255), LINE_THICKNESS)

        if current_time - last_ocr_time >= OCR_INTERVAL:
            ocr_worker(frame.copy(), coords=(left_x, 0, right_x, frame_h))
            last_ocr_time = current_time
            
        with thread_lock:
            if last_detection_info:
                x, y, w, h, num, conf = last_detection_info
                cv2.rectangle(display_frame, (x,y), (x+w, y+h), 
                            (255,0,0), LINE_THICKNESS)
                cv2.putText(display_frame, f"{num} ({conf:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255,0,0), LINE_THICKNESS)

        active_threads = len([t for t in thread_pool if t.is_alive()])
        avg_time = np.mean(processing_times) if processing_times else 0
        
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display_frame, f"Threads: {active_threads}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display_frame, f"OCR Time: {avg_time:.3f}s", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow('Simple OCR Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting detection mode.")
            break


def shrimp_record(stop_ev):
    video_writer = None
    video_start_time = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    while not stop_ev.is_set():
        try:
            frame = frame_q.get(timeout=0.2)  # đợi frame mới
        except Empty:
            continue
        try:
            if video_writer is None:
                video_start_time = time.time()
                video_path = f"vid_{int(video_start_time)}.avi"
                video_writer = cv2.VideoWriter(video_path, fourcc, 30, (frame.shape[1], frame.shape[0]))
                print(f"Started Record: {video_path}")
            video_writer.write(frame)
            if time.time() - video_start_time > 30:  # Ghi mỗi 10 giây
                video_writer.release()
                video_writer = None
                print(f"Saved video: {video_path}")
                upload_to_drive(video_path, video_path)
                print("uploaded to Google Drive")
                try:
                    # chắc chắn không còn handle
                    gc.collect()
                    time.sleep(0.2)
                    os.remove(video_path)
                    print("Deleted:", video_path)
                except Exception:
                    print("Delete error:\n", traceback.format_exc())
        except Exception as e:
            print(f"Error in record thread: {e}")
            if video_writer:
                video_writer.release()
                video_writer = None
    print("Video đã được ghi.")

app = Flask(__name__)
@app.route("/mode", methods=["GET", "POST"])
def mode_endpoint():
    global mode
    """
    GET  /mode           → trả về {"mode": "detect"}
    POST /mode {"mode":"record"} → đổi sang record
    """
    if request.method == "GET":
        return jsonify(mode)

    data = request.get_json(silent=True) or {}
    new_mode = data.get("mode", "").lower()
    if new_mode not in ("detect", "record"):
        return jsonify(error="mode must be 'detect' or 'record'"), 400
    mode = new_mode
    return jsonify(status="ok", mode=new_mode)

def start_api_server():
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

def main():
    global worker_thread, serial_port, mode
    # Khởi tạo kết nối RS485
    threading.Thread(target=start_api_server, daemon=True).start()
    if not init_serial():
        print("Không thể khởi tạo RS485. Tiếp tục chạy mà không gửi dữ liệu RS485.")

    cap = cv2.VideoCapture("tomtep.mp4") # Thay bằng đường dẫn video hoặc camera
    if not cap.isOpened():
        print("Cant open camera/video.")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Giảm độ phân giải để tối ưu
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    current_mode = None
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_period = 1 / video_fps
    last_push = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            return
        now = time.time()
        wait = frame_period - (now - last_push)
        if wait > 0:
            time.sleep(wait)
        last_push = time.time()
        try:
            frame_q.put_nowait(frame)
        except Full:
            pass
        if mode != current_mode:
            if worker_thread and worker_thread.is_alive():
                stop_event.set()
                worker_thread.join()
                stop_event.clear()
            if mode == 'detect':
                try:
                    worker_thread = threading.Thread(target=shrimpt_detect,args=(frame, stop_event), name="DetectionThread")
                except KeyboardInterrupt:
                    print("Stopped detection.")
            elif mode == 'record':
                try:
                    worker_thread = threading.Thread(target=shrimp_record,args=(stop_event,), name="RecordThread")
                except KeyboardInterrupt:
                    print("Stopped recording.")
            current_mode = mode
            if worker_thread:
                worker_thread.start()
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down…")
        stop_event.set()