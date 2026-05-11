import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from collections import deque, Counter
import uvicorn
import threading
import time

app = FastAPI()

# Model Yükleme
session = None
input_name = None
try:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 2
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    session = ort.InferenceSession("emotion-ferplus-8.onnx", opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print(f"SİSTEM: Model OK | Input adı: '{input_name}'")

    dummy = np.zeros((1, 1, 64, 64), dtype=np.float32)
    session.run(None, {input_name: dummy})
    print("SİSTEM: Model ısındırıldı (warmup)")
except Exception as e:
    print(f"KRİTİK HATA: {e}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

emotion_buffer = deque(maxlen=8)
EMOTIONS = ['TARAFSIZ', 'MUTLU', 'ŞAŞKIN', 'ÜZGÜN', 'ÖFKELİ', 'İĞRENMİŞ', 'KORKMUŞ', 'KÜÇÜMSEYEN']

# Neutral bias düzeltmesi: FER+ modeli eğitim verisinden dolayı TARAFSIZ'a çok yüksek skor verir.
# Bu çarpanlarla dengeliyoruz.
BIAS = np.array([0.35, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

current_data = {
    "duygu": "ANALİZ BEKLENİYOR...",
    "guven": "0.0",
    "gecikme": "0"
}

latest_frame = None
frame_lock = threading.Lock()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


blob_buffer = np.zeros((1, 1, 64, 64), dtype=np.float32)


def preprocess_face(gray_img, x, y, w, h):
    pad = int(w * 0.12)
    y1 = max(0, y - pad)
    y2 = min(gray_img.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(gray_img.shape[1], x + w + pad)

    face = cv2.resize(gray_img[y1:y2, x1:x2], (64, 64), interpolation=cv2.INTER_LINEAR)
    face = cv2.equalizeHist(face)
    np.copyto(blob_buffer[0, 0], face.astype(np.float32))
    return blob_buffer


last_face = None


def analysis_worker():
    """Ayrı thread'de çalışır — video akışını BLOKLAMAZ"""
    global latest_frame, last_face
    print("SİSTEM: Analiz thread'i başladı")

    log_counter = 0

    while True:
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame

        if frame is None:
            time.sleep(0.02)
            continue

        t0 = time.time()

        small = cv2.resize(frame, (240, 180), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(35, 35))

        if len(faces) > 0:
            (sx, sy, sw, sh) = max(faces, key=lambda f: f[2] * f[3])
            last_face = (sx, sy, sw, sh)

            blob = preprocess_face(gray, sx, sy, sw, sh)

            try:
                raw = session.run(None, {input_name: blob})[0][0]
                adjusted = raw * BIAS
                probs = softmax(adjusted)

                idx = np.argmax(probs)
                conf = probs[idx]
                detected = EMOTIONS[idx]

                log_counter += 1
                if log_counter % 8 == 0:
                    pairs = [f"{EMOTIONS[i]}:{probs[i]:.2f}" for i in range(8)]
                    print(f"[SKOR] {' | '.join(pairs)}")

                if conf > 0.2:
                    emotion_buffer.append(detected)

                if len(emotion_buffer) > 0:
                    stable = Counter(emotion_buffer).most_common(1)[0][0]
                    current_data["duygu"] = stable
                    current_data["guven"] = str(round(float(conf * 100), 1))

                ms = (time.time() - t0) * 1000
                current_data["gecikme"] = str(round(ms))

            except Exception as e:
                print(f"Analiz Hatası: {e}")
        else:
            current_data["duygu"] = "YÜZ ARANIYOR..."

        time.sleep(0.02)


@app.get("/api/v6/durum")
async def get_status():
    return current_data


def generate_frames():
    global latest_frame

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("HATA: Kamera açılamadı!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"SİSTEM: Kamera hazır ({int(cap.get(3))}x{int(cap.get(4))})")

    fc = 0
    t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fc += 1
        if fc % 4 == 0:
            with frame_lock:
                latest_frame = frame

        if fc % 120 == 0:
            fps = fc / (time.time() - t)
            print(f"[STREAM] {fps:.0f} FPS")

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("startup")
async def startup():
    threading.Thread(target=analysis_worker, daemon=True).start()
    print("SİSTEM: Tüm motorlar aktif!")


if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
