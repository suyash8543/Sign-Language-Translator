# main.py
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from threading import Lock

# IMPORTANT: keep heavy imports inside functions if those modules load big native libs
# If `realtimedetection` loads a Keras/TensorFlow model at import time and that caused problems,
# consider moving load into startup_event or refactor realtimedetection to expose a load() function.
from realtimedetection import model, label, extract_features

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = None
camera_lock = Lock()
model_ready = False


def open_camera_device(index: int = 0, backends=(cv2.CAP_DSHOW, cv2.CAP_MSMF)):
    """
    Try to open camera using preferred backends. Return the opened VideoCapture or None.
    """
    for backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap is not None and cap.isOpened():
                # Optional: set a common resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            else:
                # release if created but not opened
                if cap is not None:
                    cap.release()
        except Exception:
            # swallow backend-specific exceptions and try next
            pass
    return None


def load_model_sync():
    """If your realtimedetection module provides a load function, call it here.
       Otherwise this assumes model import already loaded it at top-level."""
    global model_ready
    # if additional synchronous model loading required, do it here
    # e.g. from realtimedetection import load_my_model; load_my_model()
    model_ready = True
    print("Model load flag set: ready")


@app.on_event("startup")
async def startup_event():
    global camera
    # load model in background executor so event loop is not blocked
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, load_model_sync)

    # open camera in executor (ensures this runs in the server process that handles requests)
    def _open():
        return open_camera_device(0, backends=(cv2.CAP_DSHOW, cv2.CAP_MSMF))
    camera = await loop.run_in_executor(None, _open)
    if camera is None or not camera.isOpened():
        print("Warning: camera could not be opened at startup. /start_model or /video_feed will try to reconnect.")


@app.on_event("shutdown")
async def shutdown_event():
    global camera
    try:
        if camera is not None:
            with camera_lock:
                camera.release()
            camera = None
    except Exception as e:
        print("Error releasing camera:", e)


@app.get("/start_model")
async def start_model():
    """
    tries to ensure camera is opened — useful to call from frontend before requesting /video_feed
    """
    global camera
    if camera is None or not camera.isOpened():
        loop = asyncio.get_running_loop()
        cap = await loop.run_in_executor(None, lambda: open_camera_device(0, backends=(cv2.CAP_DSHOW, cv2.CAP_MSMF)))
        if cap is None:
            return JSONResponse(content={"error": "Could not open camera"}, status_code=500)
        camera = cap
    return {"status": "Model started, camera active"}


def generate_frames():
    """
    Generator that yields JPEG multipart frames for StreamingResponse.
    This runs in a thread spawned by FastAPI/Starlette — protect camera with a lock.
    """
    global camera
    reconnect_attempts = 0
    while True:
        if camera is None or not camera.isOpened():
            # try reconnect a few times (synchronous, because this is already in a worker thread)
            reconnect_attempts += 1
            if reconnect_attempts > 5:
                print("generate_frames: failed to reconnect camera after attempts")
                break
            camera = open_camera_device(0, backends=(cv2.CAP_DSHOW, cv2.CAP_MSMF))
            if camera is None:
                import time
                time.sleep(0.5)
                continue
            reconnect_attempts = 0

        with camera_lock:
            success, frame = camera.read()

        if not success or frame is None:
            # brief wait and retry (camera might need warm-up)
            import time
            time.sleep(0.02)
            continue

        # Draw detection area
        cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 2)

        try:
            # Crop & preprocess
            cropframe = frame[40:300, 0:300]
            cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
            cropframe = cv2.resize(cropframe, (48, 48))
            cropframe = extract_features(cropframe)

            # Prediction
            pred = model.predict(cropframe, verbose=0)
            prediction_label = label[pred.argmax()]
            confidence = float(np.max(pred)) * 100.0

            # Draw prediction overlay
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
            cv2.putText(frame, f"{prediction_label} {confidence:.2f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        except Exception as e:
            # If inference fails, continue streaming raw frame (or draw error)
            print("Inference error:", e)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video_feed")
async def video_feed():
    # Return MJPEG streaming response
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
