from flask import Flask, request, jsonify
from config import modes_set, stop_event, record_event, stream_event, UDP_IP, UDP_PORT

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# API để quản lý chế độ (GET/POST)
@app.route('/mode', methods=['GET','POST'])
def api_mode():
    if request.method == 'GET':  # Nếu là GET, trả về danh sách các chế độ hiện tại
        return jsonify({'modes': list(modes_set)})
    
    data = request.get_json(silent=True) or {}  # Lấy dữ liệu JSON từ yêu cầu POST
    new_modes = data.get('mode')  # Lấy trường `mode` từ JSON

    if isinstance(new_modes, str):  # Chuẩn hóa `new_modes` thành danh sách các chuỗi chữ thường
        new_modes = [m.strip().lower() for m in new_modes.split(',') if m.strip()]
    if not isinstance(new_modes, list):  # Nếu `mode` không phải danh sách hoặc chuỗi
        return jsonify(error="mode must be a list or comma-separated string"), 400  # Trả về lỗi 400
    
    valid = {'detect', 'record', 'stream'}  # Các chế độ hợp lệ
    mset = set(m.lower() for m in new_modes)  # Chuyển đổi thành tập hợp chữ thường
    if not mset.issubset(valid):  # Kiểm tra nếu có chế độ không hợp lệ
        return jsonify(error="invalid mode(s)", allowed=list(valid)), 400  # Trả về lỗi 400
    if 'stream' in mset:
        ip = data.get('udp_ip')
        port = data.get('udp_port')
        if not ip or not port:
            return jsonify(
                error="Missing 'udp_ip' or 'udp_port' for stream mode"
            ), 400
        # Bạn có thể thêm validation IP/port ở đây nếu cần
        UDP_IP = ip
        UDP_PORT = port
    
    if 'detect' in mset:  # Nếu `detect` được yêu cầu
        modes_set.clear()  # Xóa tất cả chế độ
        modes_set.add('detect')  # Chỉ bật `detect`
    else:
        modes_set.discard('detect')  # Xóa `detect`
        modes_set |= mset  # Thêm các chế độ được yêu cầu
        if not modes_set:  # Nếu không có chế độ nào, mặc định bật `detect`
            modes_set.add('detect')
    
    if 'stream' in modes_set:
        print(f"[API] Active modes → {modes_set} (udp_ip={UDP_IP}, udp_port={UDP_IP})")
        return jsonify(status='ok', modes=list(modes_set), udp_ip=UDP_IP, udp_port=UDP_PORT)
    print(f"[API] Active modes → {modes_set}")
    return jsonify(status='ok', modes=list(modes_set))

# API để tắt chương trình
@app.route('/shutdown', methods=['POST'])
def api_shutdown():
    stop_event.set()  # Thiết lập sự kiện dừng chương trình
    record_event.set()  # Thiết lập sự kiện dừng ghi video
    stream_event.set()  # Thiết lập sự kiện dừng phát trực tuyến
    return jsonify(status='shutting down')  # Trả về phản hồi JSON thông báo đang tắt
