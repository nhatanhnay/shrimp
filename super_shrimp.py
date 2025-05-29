import cv2
import subprocess
import traceback
import time
import threading
import os
import serial
from collections import deque
from queue import Queue, Empty, Full
from typing import List, Optional, Set
from paddleocr import PaddleOCR
from flask import Flask, request, jsonify
from upload_ggdrive import upload_to_drive

"""
OCR + RS‑485 Server
===================
*   Compatible with Python ≥3.7.
*   Run-modes via REST:
    * **detect**  – run PaddleOCR on a ROI every `OCR_INTERVAL` s and push new
      card numbers via RS‑485 (`*digits#`).
    * **record**  – record 30-second video chunks and upload to Google Drive.
    * **stream**  – stream raw frames via UDP.
*   Clean shutdown: all threads, cv2 windows and the serial port are closed.
"""

# ─────────────────────────────────── Configuration ────────────────────────────
VIDEO_SOURCE = "tomtep.mp4"      # 0 for webcam
TARGET_FPS = 25                  # UI refresh rate
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

# Streaming config
UDP_IP = "127.0.0.1"
UDP_PORT = 5000
STREAM_BITRATE = "600K"

# ───────────────────────────────────── Globals ────────────────────────────────
ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)
# active modes: any combination of 'record' and/or 'stream'; 'detect' exclusive
modes_set: Set[str] = {"detect"}
stop_event = threading.Event()
record_event = threading.Event()
stream_event = threading.Event()
worker_threads: List[threading.Thread] = []
processing_times = deque(maxlen=10)
last_detected_number: Optional[str] = None
last_detection_expire = 0.0
serial_port: Optional[serial.Serial] = None
# shared frame queue for record & stream
record_q: Queue = Queue(maxsize=30)
stream_q: Queue = Queue(maxsize=30)

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

# ────────────────────────────── Worker threads ────────────────────────────────

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
    ffmpeg_proc = None
    start_ts = 0.0
    while not ev.is_set():
        try:
            frame = record_q.get(timeout=0.2)
        except Empty:
            continue
        if ffmpeg_proc is None:
            start_ts = time.time()
            out_path = f"vid_{int(start_ts)}.mp4"
            h, w, _ = frame.shape
            cmd = [
                'ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
                '-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(TARGET_FPS),
                '-i','-','-c:v','libx264','-preset','veryfast','-crf','23',out_path
            ]
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            print(f"[REC] Start {out_path}")
        ffmpeg_proc.stdin.write(frame.tobytes())
        if time.time() - start_ts >= RECORD_CHUNK_SEC:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            print(f"[REC] Saved vid_{int(start_ts)}.mp4")
            upload_to_drive(out_path, out_path)
            print("[REC] Uploaded to Drive →", out_path)
            try:
                os.remove(out_path)
            except Exception:
                print(traceback.format_exc())
            ffmpeg_proc = None
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    print("[REC] Thread stop")

# ────────────────────────────── Stream thread ─────────────────────────────────
def stream_worker(ev: threading.Event, frame_period: float, udp_ip: str, udp_port: int, bitrate: str):
    ffmpeg_proc = None
    while not ev.is_set():
        try:
            frame = stream_q.get(timeout=0.2)
        except Empty:
            continue
        h, w, _ = frame.shape
        if ffmpeg_proc is None:
            cmd = [
                "ffmpeg", "-y",
                "-fflags", "+genpts", "-flags", "low_delay", "-fflags", "nobuffer",
                "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", f"{w}x{h}", "-framerate", str(int(TARGET_FPS)), "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                "-b:v", bitrate,
                "-f", "mpegts",
                f"udp://{udp_ip}:{udp_port}?pkt_size=1316&overrun_nonfatal=1"
            ]
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            print(f"[STREAM] Start streaming → udp://{udp_ip}:{udp_port} @ {bitrate}")
        ffmpeg_proc.stdin.write(frame.tobytes())
        time.sleep(frame_period)
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    print("[STREAM] Thread stop")

# ───────────────────────────────── Flask API ─────────────────────────────────
app = Flask(__name__)

@app.route('/mode', methods=['GET','POST'])
def api_mode():
    global modes_set
    if request.method == 'GET':
        return jsonify({'modes': list(modes_set)})

    data = request.get_json(silent=True) or {}
    new_modes = data.get('mode')

    # normalize input into a list of lowercase strings
    if isinstance(new_modes, str):
        new_modes = [m.strip().lower() for m in new_modes.split(',') if m.strip()]
    if not isinstance(new_modes, list):
        return jsonify(error="mode must be a list or comma-separated string"), 400

    valid = {'detect', 'record', 'stream'}
    mset = set(m.lower() for m in new_modes)
    if not mset.issubset(valid):
        return jsonify(error="invalid mode(s)", allowed=list(valid)), 400

    # If user requested 'detect', it becomes the *only* active mode
    if 'detect' in mset:
        modes_set.clear()
        modes_set.add('detect')
    else:
        # otherwise, remove detect and *add* any requested record/stream modes,
        # but never remove existing stream when asking for record (and vice versa)
        modes_set.discard('detect')
        modes_set |= mset
        # if they somehow end up with no modes, fall back to detect
        if not modes_set:
            modes_set.add('detect')

    print(f"[API] Active modes → {modes_set}")
    return jsonify(status='ok', modes=list(modes_set))

@app.route('/shutdown', methods=['POST'])
def api_shutdown():
    stop_event.set()
    record_event.set()
    stream_event.set()
    return jsonify(status='shutting down')

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
    rec_thread: Optional[threading.Thread] = None
    stream_thread: Optional[threading.Thread] = None

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print('[MAIN] End of stream')
            break
        now = time.time()

        # detect mode (exclusive)
        if 'detect' in modes_set and not modes_set.intersection({'record','stream'}):
            if now - last_ocr_time >= OCR_INTERVAL:
                lx = int(frame.shape[1]*ZONE_L)
                rx = int(frame.shape[1]*ZONE_R)
                coords = (lx, 0, rx, frame.shape[0])
                # spawn OCR worker
                worker_threads[:] = [t for t in worker_threads if t.is_alive()]
                if len(worker_threads) < MAX_WORKER_THREADS:
                    t = threading.Thread(target=ocr_worker, args=(frame.copy(), coords), daemon=True)
                    worker_threads.append(t)
                    t.start()
                last_ocr_time = now
                record_q.queue.clear()  # clear record queue to avoid stale frames
                stream_q.queue.clear()  # clear stream queue to avoid stale frames

        # record mode
        if 'record' in modes_set:
            if rec_thread is None or not rec_thread.is_alive():
                record_event.clear()
                rec_thread = threading.Thread(target=record_worker, args=(record_event,), daemon=True)
                rec_thread.start()
            try:
                record_q.put_nowait(frame.copy())
            except Full:
                pass
        else:
            if rec_thread and rec_thread.is_alive():
                record_event.set()

        # stream mode
        if 'stream' in modes_set:
            if stream_thread is None or not stream_thread.is_alive():
                stream_event.clear()
                stream_thread = threading.Thread(
                    target=stream_worker,
                    args=(stream_event, frame_period, UDP_IP, UDP_PORT, STREAM_BITRATE),
                    daemon=True
                )
                stream_thread.start()
            try:
                stream_q.put_nowait(frame.copy())
            except Full:
                pass
        else:
            if stream_thread and stream_thread.is_alive():
                stream_event.set()

        # overlay lines and text
        lx = int(frame.shape[1]*ZONE_L)
        rx = int(frame.shape[1]*ZONE_R)
        cv2.line(frame, (lx,0), (lx,frame.shape[0]), (0,0,255), LINE_THICKNESS)
        cv2.line(frame, (rx,0), (rx,frame.shape[0]), (0,0,255), LINE_THICKNESS)
        if now < last_detection_expire and last_detected_number:
            cv2.putText(frame, f"CARD: {last_detected_number}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.imshow('OCR RS485', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(max(0, frame_period - (time.time() - now)))

    # shutdown
    stop_event.set()
    record_event.set()
    stream_event.set()
    if rec_thread and rec_thread.is_alive(): rec_thread.join()
    if stream_thread and stream_thread.is_alive(): stream_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    if serial_port and serial_port.is_open: serial_port.close()
    print('[MAIN] Shutdown complete')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('[MAIN] Ctrl-C received')
        stop_event.set()
