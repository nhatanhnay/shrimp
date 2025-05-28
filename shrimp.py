import cv2
import gc
import traceback
import time
import threading
import os
import serial
from collections import deque
from queue import Queue, Empty, Full
from typing import List, Optional
from paddleocr import PaddleOCR
from flask import Flask, request, jsonify
from upload_ggdrive import upload_to_drive

"""
OCR + RS‑485 Server
===================
*   Compatible with Python ≥3.7.
*   Two run‑modes exposed via REST:
    * **detect** – run PaddleOCR on a ROI every `OCR_INTERVAL` s and push new
      card numbers via RS‑485 (`*digits#`).
    * **record** – record 30‑second AVI chunks and upload to Google Drive.
*   Clean shutdown: all threads, cv2 windows and the serial port are closed.
"""

# ─────────────────────────────────── Configuration ────────────────────────────
VIDEO_SOURCE = "tomtep.mp4"      # 0 for webcam
TARGET_FPS = 30                  # UI refresh rate
OCR_INTERVAL = 0.5               # seconds between OCR runs
OCR_CONFIDENCE_THRESHOLD = 0.8
DETECTION_DISPLAY_SECONDS = 2.0

# Detection zone as width‑ratios
ZONE_L = 0.10
ZONE_R = 0.45
LINE_THICKNESS = 2

# RS‑485 serial
SERIAL_PORT_CANDIDATES = ("/dev/ttyTHS1", "/dev/ttyTHS0", "/dev/ttyUSB0")
BAUD_RATE = 9600

MAX_WORKER_THREADS = 1           # OCR workers at once
RECORD_CHUNK_SEC = 30            # length of each video piece

# ───────────────────────────────────── Globals ────────────────────────────────
ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)
mode: str = "detect"             # mutable via REST
stop_event = threading.Event()
worker_threads: List[threading.Thread] = []
processing_times = deque(maxlen=10)

last_detected_number: Optional[str] = None
last_detection_expire = 0.0

serial_port: Optional[serial.Serial] = None
record_q: Queue = Queue(maxsize=90)  # ~3 s buffer @30 fps

# ────────────────────────────── RS‑485 helpers ────────────────────────────────

def init_serial() -> None:
    global serial_port
    for port in SERIAL_PORT_CANDIDATES:
        try:
            serial_port = serial.Serial(port, BAUD_RATE, timeout=1)
            print(f"[RS485] Connected → {port} @ {BAUD_RATE}")
            return
        except serial.SerialException as err:
            print(f"[RS485] {port}: {err}")
    serial_port = None
    print("[RS485] No port available – running without serial output")


def send_rs485(payload: str) -> None:
    if serial_port and serial_port.is_open:
        try:
            serial_port.write(payload.encode("ascii"))
            serial_port.flush()
            print(f"[RS485] Sent {payload}")
        except serial.SerialException as err:
            print(f"[RS485] Write error: {err}")

# ──────────────────────────────── OCR helpers ────────────────────────────────

def correct_orientation(img):
    try:
        h = img.shape[0]
        res = ocr_engine.ocr(img, cls=True)
        if res and res[0]:
            box = res[0][0][0]
            if sum(p[1] for p in box)/len(box) > h/2:
                return cv2.rotate(img, cv2.ROTATE_180)
    except Exception as exc:
        print(f"[OCR] orientation error: {exc}")
    return img


def simple_ocr(img):
    try:
        img = correct_orientation(img)
        t0 = time.time()
        res = ocr_engine.ocr(img, cls=True)
        processing_times.append(time.time() - t0)
        if res and res[0]:
            for _, (txt, conf) in res[0]:
                digits = ''.join(ch for ch in txt if ch.isdigit())
                if digits and len(digits) >= 2 and conf >= OCR_CONFIDENCE_THRESHOLD:
                    return digits, conf
    except Exception as exc:
        print(f"[OCR] error: {exc}")
    return None, 0.0

# ────────────────────────────── Worker threads ───────────────────────────────

def ocr_worker(frame, coords):
    global last_detected_number, last_detection_expire
    x1, y1, x2, y2 = coords
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    digits, conf = simple_ocr(roi)
    if not digits:
        return
    now = time.time()
    if now < last_detection_expire or digits == last_detected_number:
        return
    last_detected_number = digits
    last_detection_expire = now + DETECTION_DISPLAY_SECONDS
    print(f"[DETECT] {digits}  conf={conf:.2f}")
    send_rs485(f"*{digits}#")

# ───────────────────────────── Recording thread ──────────────────────────────

def record_worker(ev: threading.Event):
    writer = None
    start_ts = 0.0
    out_path = ''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    while not ev.is_set():
        try:
            frame = record_q.get(timeout=0.2)
        except Empty:
            continue
        if writer is None:
            start_ts = time.time()
            out_path = f"vid_{int(start_ts)}.avi"
            writer = cv2.VideoWriter(out_path, fourcc, 30, (frame.shape[1], frame.shape[0]))
            print(f"[REC] Start {out_path}")
        writer.write(frame)
        if time.time() - start_ts >= RECORD_CHUNK_SEC:
            writer.release(); writer = None
            print(f"[REC] Saved {out_path}")
            upload_to_drive(out_path, out_path)
            print("[REC] Uploaded to Drive →", out_path)
            try:
                os.remove(out_path)
            except Exception:
                print(traceback.format_exc())
    if writer:
        writer.release()
    print("[REC] Thread stop")

# ───────────────────────────────── Flask API ─────────────────────────────────
app = Flask(__name__)

@app.route('/mode', methods=['GET', 'POST'])
def api_mode():
    global mode
    if request.method == 'GET':
        return jsonify({'mode': mode})
    data = request.get_json(silent=True) or {}
    new_mode = str(data.get('mode', '')).lower()
    if new_mode not in {'detect', 'record'}:
        return jsonify(error="mode must be 'detect' or 'record'"), 400
    if new_mode != mode:
        print(f"[API] Mode → {new_mode}")
        mode = new_mode
    return jsonify(status='ok', mode=mode)

@app.route('/shutdown', methods=['POST'])
def api_shutdown():
    stop_event.set()
    return jsonify(status='shutting down')

# ───────────────────────────────── Utilities ─────────────────────────────────

def cleanup_workers():
    global worker_threads
    worker_threads = [t for t in worker_threads if t.is_alive()]

# ────────────────────────────────── Main loop ────────────────────────────────

def main():
    init_serial()
    threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':5000,'threaded':True,'debug':False}, daemon=True).start()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {VIDEO_SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    frame_period = 1.0 / fps

    last_ocr_time = 0.0
    fps_counter = 0
    fps_ts = time.time()
    cur_fps = 0.0
    rec_thread: Optional[threading.Thread] = None

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print('[MAIN] End of stream')
            break
        now = time.time()

        # mode‑specific work
        if mode == 'detect':
            if now - last_ocr_time >= OCR_INTERVAL:
                lx = int(frame.shape[1]*ZONE_L)
                rx = int(frame.shape[1]*ZONE_R)
                coords = (lx, 0, rx, frame.shape[0])
                cleanup_workers()
                if len(worker_threads) < MAX_WORKER_THREADS:
                    t = threading.Thread(target=ocr_worker, args=(frame.copy(), coords), daemon=True)
                    worker_threads.append(t); t.start()
                last_ocr_time = now
        elif mode == 'record':
            if rec_thread is None or not rec_thread.is_alive():
                rec_thread = threading.Thread(target=record_worker, args=(stop_event,), daemon=True)
                rec_thread.start()
            try:
                record_q.put_nowait(frame.copy())
            except Full:
                pass

        # overlay
        lx = int(frame.shape[1]*ZONE_L); rx = int(frame.shape[1]*ZONE_R)
        cv2.line(frame, (lx,0), (lx,frame.shape[0]), (0,0,255), LINE_THICKNESS)
        cv2.line(frame, (rx,0), (rx,frame.shape[0]), (0,0,255), LINE_THICKNESS)

        if now < last_detection_expire and last_detected_number:
            cv2.putText(frame, f"CARD: {last_detected_number}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        fps_counter += 1
        if now - fps_ts >= 1:
            cur_fps = fps_counter/(now-fps_ts)
            fps_counter = 0
            fps_ts = now
        cv2.putText(frame, f"FPS: {cur_fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        cv2.imshow('OCR RS485', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # maintain GUI rate
        remaining = frame_period - (time.time()-now)
        if remaining > 0:
            time.sleep(remaining)

    # cleanup
    stop_event.set()
    if rec_thread and rec_thread.is_alive():
        rec_thread.join()
    cap.release(); cv2.destroyAllWindows()
    if serial_port and serial_port.is_open:
        serial_port.close()
    print('[MAIN] shutdown complete')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('[MAIN] Ctrl‑C received')
        stop_event.set()
