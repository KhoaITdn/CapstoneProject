"""
app.py - Flask Web Application for Real-Time Facial Emotion Recognition in Video Calls
Features:
  - WebRTC-based video calling (create/join rooms)
  - Real-time facial emotion recognition using CBAM CNN model
  - Multi-participant emotion analytics dashboard
  - SocketIO for real-time communication
"""
import os
import sys
import time
import uuid
import base64
import threading
from collections import deque, defaultdict

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room

# Import model module
sys.path.insert(0, os.path.dirname(__file__))
from model import (
    load_emotion_model, preprocess_face, EMOTIONS, EMOTIONS_VI,
    EMOTION_EMOJIS, ENGAGEMENT_WEIGHTS, IMG_SIZE
)

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'emotion_rafdb_v2_final.keras')
EMA_ALPHA = 0.3          # Smoothing factor (0=smooth, 1=raw)

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion-video-call-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                    max_http_buffer_size=10 * 1024 * 1024)  # 10MB for frames

# ============================================================
# GLOBAL STATE
# ============================================================
model = None
face_cascade = None
lock = threading.Lock()

# Room management: { room_id: { user_id: { ... } } }
rooms = {}
# Per-user smoothed probabilities
user_smoothed_probs = defaultdict(lambda: np.zeros(7))
# Per-user emotion history
user_emotion_history = defaultdict(lambda: deque(maxlen=300))
# User info mapping
user_info = {}  # { sid: { user_id, room_id, name } }


def init_system():
    """Initialize model and face detector."""
    global model, face_cascade

    # Load EfficientNetB0+CBAM model
    model_path = os.path.abspath(MODEL_PATH)
    print(f"[INIT] Model path: {model_path}")
    model = load_emotion_model(model_path)
    if model is None:
        print("[INIT] FATAL: Cannot load model!")
        sys.exit(1)

    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    print("[INIT] System ready!")
    print(f"[INIT] Model: EfficientNetB0+CBAM | Input: {IMG_SIZE}x{IMG_SIZE} RGB")


# ============================================================
# EMOTION PROCESSING
# ============================================================
def process_frame_data(frame_data, user_id):
    """Process a base64-encoded frame from client.
    Returns emotion data dict or None if no face detected.
    """
    global user_smoothed_probs

    try:
        # Decode base64 image
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return {'face_detected': False}

        # Use the largest face
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        (x, y, w, h) = faces[idx]

        # Preprocess face ROI: BGR -> 224x224 RGB + efficientnet_preprocess
        roi_bgr = frame[y:y+h, x:x+w]
        roi_batch = preprocess_face(roi_bgr)  # (1, 224, 224, 3)

        # Predict
        with lock:
            raw_probs = model.predict(roi_batch, verbose=0)[0]

        # EMA smoothing per user
        smoothed = user_smoothed_probs[user_id]
        smoothed = EMA_ALPHA * raw_probs + (1 - EMA_ALPHA) * smoothed
        user_smoothed_probs[user_id] = smoothed

        # Calculate results
        dominant_idx = int(np.argmax(smoothed))
        confidence = float(smoothed[dominant_idx]) * 100
        engagement = calculate_engagement(smoothed)

        # Face bounding box (normalized)
        h_img, w_img = frame.shape[:2]
        face_box = {
            'x': round(x / w_img, 4),
            'y': round(y / h_img, 4),
            'w': round(w / w_img, 4),
            'h': round(h / h_img, 4)
        }

        result = {
            'face_detected': True,
            'emotions': {EMOTIONS[i]: round(float(smoothed[i]) * 100, 1) for i in range(7)},
            'dominant': EMOTIONS[dominant_idx],
            'dominant_vi': EMOTIONS_VI[dominant_idx],
            'emoji': EMOTION_EMOJIS[dominant_idx],
            'confidence': round(confidence, 1),
            'engagement': round(engagement, 1),
            'face_box': face_box,
            'user_id': user_id
        }

        # Save to history
        user_emotion_history[user_id].append({
            'time': time.time(),
            'dominant': result['dominant'],
            'engagement': result['engagement'],
            'probs': result['emotions']
        })

        return result

    except Exception as e:
        print(f"[PROCESS] Error for user {user_id}: {e}")
        return None


def calculate_engagement(probs):
    """Calculate engagement score (0-100) from probabilities."""
    score = 0
    for i, emotion in enumerate(EMOTIONS):
        score += probs[i] * ENGAGEMENT_WEIGHTS[emotion]
    return min(100, score * 100)


# ============================================================
# SOCKET.IO EVENTS - Video Call Signaling
# ============================================================
@socketio.on('connect')
def handle_connect():
    print(f"[WS] Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in user_info:
        info = user_info[sid]
        room_id = info.get('room_id')
        user_id = info.get('user_id')
        user_name = info.get('name', 'Unknown')

        if room_id and room_id in rooms:
            if user_id in rooms[room_id]:
                del rooms[room_id][user_id]

            # Notify others
            emit('user_left', {
                'user_id': user_id,
                'name': user_name,
                'participants': get_room_participants(room_id)
            }, room=room_id)

            leave_room(room_id)

            # Cleanup empty rooms
            if not rooms[room_id]:
                del rooms[room_id]

        # Cleanup user data
        if user_id in user_smoothed_probs:
            del user_smoothed_probs[user_id]
        if user_id in user_emotion_history:
            del user_emotion_history[user_id]
        del user_info[sid]

    print(f"[WS] Client disconnected: {sid}")


@socketio.on('create_room')
def handle_create_room(data):
    """Create a new video call room."""
    room_id = str(uuid.uuid4())[:8]
    user_id = data.get('user_id', str(uuid.uuid4())[:8])
    user_name = data.get('name', 'User')

    rooms[room_id] = {}
    rooms[room_id][user_id] = {
        'name': user_name,
        'sid': request.sid,
        'joined_at': time.time()
    }

    join_room(room_id)
    user_info[request.sid] = {
        'user_id': user_id,
        'room_id': room_id,
        'name': user_name
    }

    emit('room_created', {
        'room_id': room_id,
        'user_id': user_id,
        'participants': get_room_participants(room_id)
    })
    print(f"[ROOM] Created room {room_id} by {user_name}")


@socketio.on('join_room')
def handle_join_room(data):
    """Join an existing video call room."""
    room_id = data.get('room_id', '').strip()
    user_id = data.get('user_id', str(uuid.uuid4())[:8])
    user_name = data.get('name', 'User')

    if room_id not in rooms:
        emit('room_error', {'message': 'Phòng không tồn tại!'})
        return

    if len(rooms[room_id]) >= 6:
        emit('room_error', {'message': 'Phòng đã đầy (tối đa 6 người)!'})
        return

    rooms[room_id][user_id] = {
        'name': user_name,
        'sid': request.sid,
        'joined_at': time.time()
    }

    join_room(room_id)
    user_info[request.sid] = {
        'user_id': user_id,
        'room_id': room_id,
        'name': user_name
    }

    # Notify existing participants
    emit('user_joined', {
        'user_id': user_id,
        'name': user_name,
        'participants': get_room_participants(room_id)
    }, room=room_id)

    # Send room info to the new user
    emit('room_joined', {
        'room_id': room_id,
        'user_id': user_id,
        'participants': get_room_participants(room_id)
    })

    print(f"[ROOM] {user_name} joined room {room_id}")


@socketio.on('leave_room_request')
def handle_leave_room(data):
    """Leave the current room."""
    sid = request.sid
    if sid in user_info:
        info = user_info[sid]
        room_id = info.get('room_id')
        user_id = info.get('user_id')

        if room_id and room_id in rooms:
            if user_id in rooms[room_id]:
                del rooms[room_id][user_id]
            emit('user_left', {
                'user_id': user_id,
                'name': info.get('name'),
                'participants': get_room_participants(room_id)
            }, room=room_id)
            leave_room(room_id)

            if not rooms[room_id]:
                del rooms[room_id]

        del user_info[sid]
        emit('left_room', {})


# ---- WebRTC Signaling ----
@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    """Forward WebRTC offer to target peer."""
    target_sid = get_user_sid(data.get('target'), data.get('room_id'))
    if target_sid:
        emit('webrtc_offer', {
            'sdp': data['sdp'],
            'from_user': data.get('from_user')
        }, room=target_sid)


@socketio.on('webrtc_answer')
def handle_webrtc_answer(data):
    """Forward WebRTC answer to target peer."""
    target_sid = get_user_sid(data.get('target'), data.get('room_id'))
    if target_sid:
        emit('webrtc_answer', {
            'sdp': data['sdp'],
            'from_user': data.get('from_user')
        }, room=target_sid)


@socketio.on('webrtc_ice_candidate')
def handle_ice_candidate(data):
    """Forward ICE candidate to target peer."""
    target_sid = get_user_sid(data.get('target'), data.get('room_id'))
    if target_sid:
        emit('webrtc_ice_candidate', {
            'candidate': data['candidate'],
            'from_user': data.get('from_user')
        }, room=target_sid)


# ---- Emotion Frame Processing ----
@socketio.on('video_frame')
def handle_video_frame(data):
    """Receive a video frame from client, process emotions, send results back."""
    sid = request.sid
    if sid not in user_info:
        return

    info = user_info[sid]
    user_id = info['user_id']
    room_id = info.get('room_id')

    frame_data = data.get('frame')
    if not frame_data:
        return

    # Process the frame
    result = process_frame_data(frame_data, user_id)
    if result is None:
        return

    result['user_id'] = user_id
    result['user_name'] = info.get('name', 'Unknown')
    result['timestamp'] = time.time()

    # Broadcast emotion data to room
    if room_id:
        emit('emotion_result', result, room=room_id)
    else:
        emit('emotion_result', result)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_room_participants(room_id):
    """Get list of participants in a room."""
    if room_id not in rooms:
        return []
    return [
        {'user_id': uid, 'name': info['name']}
        for uid, info in rooms[room_id].items()
    ]


def get_user_sid(user_id, room_id):
    """Get Socket.IO SID for a user in a room."""
    if room_id and room_id in rooms:
        user = rooms[room_id].get(user_id)
        if user:
            return user.get('sid')
    return None


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/rooms')
def api_rooms():
    """List active rooms (for debugging)."""
    result = {}
    for room_id, participants in rooms.items():
        result[room_id] = {
            'count': len(participants),
            'participants': [
                {'user_id': uid, 'name': info['name']}
                for uid, info in participants.items()
            ]
        }
    return jsonify(result)


@app.route('/api/history/<user_id>')
def api_user_history(user_id):
    """Return emotion history for a specific user."""
    history = list(user_emotion_history.get(user_id, []))
    return jsonify(history)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssl', action='store_true',
                        help='Enable HTTPS (required for camera via IP address)')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    init_system()

    ssl_ctx = None
    if args.ssl:
        import ssl
        # Generate self-signed cert for HTTPS
        cert_file = os.path.join(os.path.dirname(__file__), 'cert.pem')
        key_file  = os.path.join(os.path.dirname(__file__), 'key.pem')

        if not os.path.exists(cert_file):
            print("[SSL] Generating self-signed certificate...")
            from subprocess import run as sp_run
            sp_run([
                'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
                '-keyout', key_file, '-out', cert_file,
                '-days', '365', '-nodes',
                '-subj', '/CN=localhost'
            ], check=True)
            print("[SSL] Certificate generated!")

        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert_file, key_file)
        protocol = "https"
    else:
        protocol = "http"

    print("\n" + "=" * 55)
    print("  EMOTION RECOGNITION VIDEO CALL SYSTEM")
    print(f"  Local:   {protocol}://localhost:{args.port}")
    if args.ssl:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        print(f"  Network: {protocol}://{ip}:{args.port}")
        print("  (Accept self-signed cert warning in browser)")
    else:
        print("  Camera only works on localhost (not IP)!")
        print("  For IP access: python app.py --ssl")
    print("=" * 55 + "\n")

    socketio.run(app, host='0.0.0.0', port=args.port, debug=False,
                 allow_unsafe_werkzeug=True, ssl_context=ssl_ctx)
