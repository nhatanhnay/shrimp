from paddleocr import PaddleOCR
import cv2
from config import OCR_CONFIDENCE_THRESHOLD, DETECTION_DISPLAY_SECONDS, processing_times, last_detected_number, last_detection_expire
from serial_comm import send_rs485
import time

# Khởi tạo engine PaddleOCR với phân loại góc, ngôn ngữ tiếng Anh, dùng GPU, và tắt log
ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)

# Điều chỉnh hướng ảnh dựa trên kết quả OCR
def correct_orientation(img):
    try:
        h = img.shape[0]  # Lấy chiều cao ảnh
        res = ocr_engine.ocr(img, cls=True)  # Chạy OCR với phân loại góc để xác định hướng
        if res and res[0]:
            box = res[0][0][0]  # Lấy tọa độ hộp văn bản đầu tiên
            if sum(p[1] for p in box)/len(box) > h/2:  # Nếu tọa độ Y trung bình vượt quá nửa chiều cao, ảnh bị ngược
                return cv2.rotate(img, cv2.ROTATE_180)  # Xoay ảnh 180 độ
    except Exception as exc:
        print(f"[OCR] orientation error: {exc}")  # In lỗi nếu có vấn đề
    return img  # Trả về ảnh gốc nếu không cần xoay hoặc có lỗi

# Thực hiện OCR đơn giản để trích xuất số
def simple_ocr(img):
    try:
        img = correct_orientation(img)  # Điều chỉnh hướng ảnh
        t0 = time.time()  # Ghi lại thời gian bắt đầu OCR
        res = ocr_engine.ocr(img, cls=True)  # Chạy OCR với phân loại góc
        processing_times.append(time.time() - t0)  # Lưu thời gian xử lý vào processing_times
        if res and res[0]:
            for _, (txt, conf) in res[0]:
                digits = ''.join(ch for ch in txt if ch.isdigit())  # Lọc các ký tự số từ văn bản
                if digits and len(digits) >= 2 and conf >= OCR_CONFIDENCE_THRESHOLD:  # Nếu chuỗi số có ít nhất 2 ký tự và độ tin cậy ≥ ngưỡng
                    return digits, conf  # Trả về chuỗi số và độ tin cậy
    except Exception as exc:
        print(f"[OCR] error: {exc}")  # In lỗi nếu có vấn đề
    return None, 0.0  # Trả về None và 0.0 nếu không tìm thấy số hoặc có lỗi

# Hàm xử lý OCR trong luồng
def ocr_worker(frame, coords):
    global last_detected_number, last_detection_expire
    x1, y1, x2, y2 = coords  # Lấy tọa độ vùng ROI
    roi = frame[y1:y2, x1:x2]  # Cắt vùng ROI từ khung hình
    if roi.size == 0:  # Nếu ROI rỗng, thoát
        return
    digits, conf = simple_ocr(roi)  # Chạy OCR trên ROI
    if not digits:  # Nếu không tìm thấy số, thoát
        return
    now = time.time()  # Lấy thời gian hiện tại
    if now < last_detection_expire or digits == last_detected_number:  # Nếu số trùng hoặc vẫn trong thời gian hiển thị, thoát
        return
    last_detected_number = digits  # Cập nhật số được phát hiện
    last_detection_expire = now + DETECTION_DISPLAY_SECONDS  # Cập nhật thời gian hết hạn hiển thị
    print(f"[DETECT] {digits}  conf={conf:.2f}")  # In thông tin số và độ tin cậy
    send_rs485(f"*{digits}#")  # Gửi số qua RS-485 với định dạng *<digits>#
