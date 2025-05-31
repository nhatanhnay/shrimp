import cv2
import threading
import time
from queue import Full
from config import VIDEO_SOURCE, TARGET_FPS, OCR_INTERVAL, ZONE_L, ZONE_R, LINE_THICKNESS, modes_set, stop_event, record_event, stream_event, worker_threads, record_q, stream_q, last_detected_number, last_detection_expire, UDP_IP, UDP_PORT, fps_deque, serial_port
from serial_comm import init_serial
from ocr_processing import ocr_worker
from video_recording import record_worker
from video_streaming import stream_worker

# Hàm chính điều phối ứng dụng
def main():
    init_serial()  # Khởi tạo kết nối RS-485
    threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':5000,'threaded':True,'debug':False}, daemon=True).start()  # Chạy Flask API trong một luồng riêng trên cổng 5000

    cap = cv2.VideoCapture(VIDEO_SOURCE)  # Mở nguồn video (webcam hoặc thiết bị)
    if not cap.isOpened():  # Nếu không mở được
        raise RuntimeError(f"Cannot open video source {VIDEO_SOURCE}")  # Ném lỗi
    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS  # Lấy FPS từ nguồn video, mặc định là TARGET_FPS
    frame_period = 1.0 / fps  # Tính thời gian giữa các khung hình

    last_ocr_time = 0.0  # Thời điểm chạy OCR lần cuối
    rec_thread: threading.Thread = None  # Biến lưu luồng ghi video
    stream_thread: threading.Thread = None  # Biến lưu luồng phát trực tuyến
    prev_time = time.time()
    while not stop_event.is_set():  # Vòng lặp chính, chạy đến khi stop_event được thiết lập
        now = time.time()
        interval = now - prev_time
        prev_time = now
        fps_deque.append(interval)
        # trung bình thời gian xử lý mỗi khung
        avg_interval = sum(fps_deque) / len(fps_deque)
        real_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
        ret, frame = cap.read()  # Đọc khung hình từ nguồn video
        if not ret:  # Nếu không đọc được
            print('[MAIN] End of stream')  # In thông báo
            break  # Thoát vòng lặp
        now = time.time()  # Lấy thời gian hiện tại

        if 'detect' in modes_set and not modes_set.intersection({'record','stream'}):  # Nếu ở chế độ `detect` và không có `record` hoặc `stream`
            if now - last_ocr_time >= OCR_INTERVAL:  # Kiểm tra nếu đã đến thời điểm chạy OCR
                lx = int(frame.shape[1]*ZONE_L)  # Tính tọa độ ROI trái
                rx = int(frame.shape[1]*ZONE_R)  # Tính tọa độ ROI phải
                coords = (lx, 0, rx, frame.shape[0])  # Tạo tọa độ ROI
                worker_threads[:] = [t for t in worker_threads if t.is_alive()]  # Lọc các luồng OCR còn hoạt động
                if len(worker_threads) < 1:  # Nếu số luồng < MAX_WORKER_THREADS
                    t = threading.Thread(target=ocr_worker, args=(frame.copy(), coords), daemon=True)  # Tạo luồng OCR mới
                    worker_threads.append(t)  # Thêm vào danh sách
                    t.start()  # Khởi động luồng
                last_ocr_time = now  # Cập nhật thời gian OCR
                record_q.queue.clear()  # Xóa hàng đợi record_q
                stream_q.queue.clear()  # Xóa hàng đợi stream_q

        if 'record' in modes_set:  # Nếu ở chế độ `record`
            if rec_thread is None or not rec_thread.is_alive():  # Nếu không có luồng ghi hoặc luồng đã dừng
                record_event.clear()  # Xóa sự kiện dừng ghi
                rec_thread = threading.Thread(target=record_worker, args=(record_event,), daemon=True)  # Khởi động luồng ghi mới
                rec_thread.start()  # Bắt đầu luồng
            try:
                record_q.put_nowait(frame.copy())  # Thêm khung hình vào record_q
            except Full:
                pass  # Bỏ qua nếu hàng đợi đầy
        else:
            if rec_thread and rec_thread.is_alive():  # Nếu không ở chế độ `record` và luồng ghi đang chạy
                record_event.set()  # Dừng luồng ghi

        if 'stream' in modes_set:  # Nếu ở chế độ `stream`
            if stream_thread is None or not stream_thread.is_alive():  # Nếu không có luồng phát hoặc luồng đã dừng
                stream_event.clear()  # Xóa sự kiện dừng phát
                stream_thread = threading.Thread(
                    target=stream_worker,
                    args=(stream_event, frame_period, UDP_IP, 5000, UDP_PORT),
                    daemon=True
                )  # Khởi động luồng phát mới
                stream_thread.start()  # Bắt đầu luồng
            try:
                stream_q.put_nowait(frame.copy())  # Thêm khung hình vào stream_q
            except Full:
                pass  # Bỏ qua nếu hàng đợi đầy
        else:
            if stream_thread and stream_thread.is_alive():  # Nếu không ở chế độ `stream` và luồng phát đang chạy
                stream_event.set()  # Dừng luồng phát

        lx = int(frame.shape[1]*ZONE_L)  # Tính tọa độ ROI trái
        rx = int(frame.shape[1]*ZONE_R)  # Tính tọa độ ROI phải
        cv2.line(frame, (lx,0), (lx,frame.shape[0]), (0,0,255), LINE_THICKNESS)  # Vẽ đường dọc đỏ cho ROI trái
        cv2.line(frame, (rx,0), (rx,frame.shape[0]), (0,0,255), LINE_THICKNESS)  # Vẽ đường dọc đỏ cho ROI phải
        if now < last_detection_expire and last_detected_number:  # Nếu số thẻ còn trong thời gian hiển thị
            cv2.putText(frame, f"CARD: {last_detected_number}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)  # Hiển thị số trên khung hình

        cv2.putText(frame, f"FPS: {real_fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)  # Hiển thị FPS trên khung hình
        cv2.imshow('OCR RS485', frame)  # Hiển thị khung hình trong cửa sổ OpenCV
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nếu nhấn phím 'q'
            break  # Thoát vòng lặp
        time.sleep(max(0, frame_period - (time.time() - now)))  # Chờ để đảm bảo tốc độ khung hình

    stop_event.set()  # Thiết lập sự kiện dừng chương trình
    record_event.set()  # Thiết lập sự kiện dừng ghi video
    stream_event.set()  # Thiết lập sự kiện dừng phát trực tuyến
    if rec_thread and rec_thread.is_alive(): rec_thread.join()  # Đợi luồng ghi hoàn tất
    if stream_thread and stream_thread.is_alive(): stream_thread.join()  # Đợi luồng phát hoàn tất
    cap.release()  # Giải phóng nguồn video
    cv2.destroyAllWindows()  # Đóng cửa sổ OpenCV
    if serial_port and serial_port.is_open: serial_port.close()  # Đóng cổng serial
    print('[MAIN] Shutdown complete')  # In thông báo hoàn tất tắt chương trình

if __name__ == '__main__':
    from api import app
    try:
        main()  # Chạy hàm chính
    except KeyboardInterrupt:
        print('[MAIN] Ctrl-C received')  # Xử lý ngắt bàn phím (Ctrl+C)
        stop_event.set()  # Thiết lập sự kiện dừng
