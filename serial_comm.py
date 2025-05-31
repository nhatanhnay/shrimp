import serial
from config import SERIAL_PORT_CANDIDATES, BAUD_RATE, serial_port

# Khởi tạo kết nối RS-485
def init_serial() -> None:  # Khởi tạo biến toàn cục serial_port
    for port in SERIAL_PORT_CANDIDATES:  # Thử kết nối với các cổng trong SERIAL_PORT_CANDIDATES
        try:
            serial_port = serial.Serial(port, BAUD_RATE, timeout=1)  # Tạo đối tượng serial với cổng và tốc độ baud
            print(f"[RS485] Connected → {port} @ {BAUD_RATE}")  # In thông báo khi kết nối thành công
            return
        except serial.SerialException as err:
            print(f"[RS485] {port}: {err}")  # In lỗi nếu kết nối thất bại
    print("[RS485] No port available – running without serial output")  # Nếu không tìm được cổng, tiếp tục chạy mà không có đầu ra serial

# Gửi dữ liệu qua RS-485
def send_rs485(payload: str) -> None:
    if serial_port and serial_port.is_open:  # Kiểm tra nếu serial_port tồn tại và đang mở
        try:
            serial_port.write(payload.encode("ascii"))  # Mã hóa payload thành ASCII và gửi
            serial_port.flush()  # Xóa bộ đệm sau khi gửi
            print(f"[RS485] Sent {payload}")  # In thông báo dữ liệu đã gửi
        except serial.SerialException as err:
            print(f"[RS485] Write error: {err}")  # In lỗi nếu có vấn đề khi gửi
