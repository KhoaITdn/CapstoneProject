/* ============================================
   EMOTION VIDEO CALL - MAIN JS
   WebRTC video calling + Real-time emotion recognition
   ============================================ */

// ============================================================
// STATE
// ============================================================
const state = {
    socket: null,
    localStream: null,
    peers: {},          // { user_id: RTCPeerConnection }
    remoteStreams: {},   // { user_id: MediaStream }
    roomId: null,
    userId: null,
    userName: null,
    isMicOn: true,
    isCameraOn: true,
    isAnalyticsOpen: true,
    callStartTime: null,
    frameInterval: null,
    emotionData: {},    // { user_id: { ...emotion data } }
};

// ---- CONSTANTS ----
const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
const EMOTIONS_VI = ['Giận dữ', 'Kinh tởm', 'Sợ hãi', 'Hạnh phúc', 'Bình thường', 'Buồn', 'Ngạc nhiên'];
const EMOTION_COLORS = [
    '#ef4444',  // Angry - red
    '#a855f7',  // Disgust - purple
    '#f59e0b',  // Fear - amber
    '#22c55e',  // Happy - green
    '#6366f1',  // Neutral - indigo
    '#3b82f6',  // Sad - blue
    '#f97316',  // Surprise - orange
];

const CIRCUMFERENCE = 2 * Math.PI * 52;
const MAX_TIMELINE_POINTS = 60;
const FRAME_SEND_INTERVAL = 500; // ms - send frame every 500ms

const ICE_SERVERS = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
    ]
};

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    initLobby();
    initCallControls();
    initCharts();
});


// ============================================================
// SOCKET.IO
// ============================================================
function initSocket() {
    state.socket = io();

    state.socket.on('connect', () => {
        console.log('[Socket] Connected:', state.socket.id);
        updateConnectionStatus(true);
    });

    state.socket.on('disconnect', () => {
        console.log('[Socket] Disconnected');
        updateConnectionStatus(false);
    });

    // Room events
    state.socket.on('room_created', handleRoomCreated);
    state.socket.on('room_joined', handleRoomJoined);
    state.socket.on('room_error', handleRoomError);
    state.socket.on('user_joined', handleUserJoined);
    state.socket.on('user_left', handleUserLeft);
    state.socket.on('left_room', handleLeftRoom);

    // WebRTC signaling
    state.socket.on('webrtc_offer', handleWebRTCOffer);
    state.socket.on('webrtc_answer', handleWebRTCAnswer);
    state.socket.on('webrtc_ice_candidate', handleICECandidate);

    // Emotion results
    state.socket.on('emotion_result', handleEmotionResult);
}


// ============================================================
// LOBBY
// ============================================================
function initLobby() {
    const btnCreate = document.getElementById('btn-create-room');
    const btnJoin = document.getElementById('btn-join-room');
    const nameInput = document.getElementById('user-name-input');
    const roomInput = document.getElementById('room-id-input');

    btnCreate.addEventListener('click', () => {
        const name = nameInput.value.trim() || 'User';
        state.userName = name;
        state.userId = generateId();
        state.socket.emit('create_room', { user_id: state.userId, name: name });
    });

    btnJoin.addEventListener('click', () => {
        const name = nameInput.value.trim() || 'User';
        const roomId = roomInput.value.trim();
        if (!roomId) {
            showLobbyError('Vui lòng nhập mã phòng!');
            return;
        }
        state.userName = name;
        state.userId = generateId();
        state.socket.emit('join_room', { room_id: roomId, user_id: state.userId, name: name });
    });

    // Enter key support
    roomInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') btnJoin.click();
    });
    nameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') btnCreate.click();
    });

    // Start camera preview
    startCameraPreview();
}


async function startCameraPreview() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: true
        });
        state.localStream = stream;

        const previewVideo = document.getElementById('preview-video');
        previewVideo.srcObject = stream;
        document.getElementById('preview-overlay').style.display = 'none';

        const previewStatus = document.getElementById('preview-status');
        previewStatus.textContent = '✅ Camera sẵn sàng';
        previewStatus.classList.add('visible');
        setTimeout(() => previewStatus.classList.remove('visible'), 2000);

    } catch (err) {
        console.error('[Camera] Error:', err);
        const previewStatus = document.getElementById('preview-status');
        previewStatus.textContent = '❌ Không thể truy cập camera: ' + err.message;
        previewStatus.classList.add('visible');
    }
}


function showLobbyError(msg) {
    const el = document.getElementById('lobby-error');
    el.textContent = msg;
    el.style.display = 'block';
    setTimeout(() => { el.style.display = 'none'; }, 4000);
}


// ============================================================
// ROOM HANDLERS
// ============================================================
function handleRoomCreated(data) {
    state.roomId = data.room_id;
    state.userId = data.user_id;
    enterCallScreen(data.participants);
    showToast(`Phòng đã tạo: ${data.room_id}`, 'success');
}

function handleRoomJoined(data) {
    state.roomId = data.room_id;
    state.userId = data.user_id;
    enterCallScreen(data.participants);
    showToast(`Đã tham gia phòng: ${data.room_id}`, 'success');

    // Create peer connections to existing participants
    data.participants.forEach(p => {
        if (p.user_id !== state.userId) {
            createPeerConnection(p.user_id, true);
        }
    });
}

function handleRoomError(data) {
    showLobbyError(data.message);
}

function handleUserJoined(data) {
    showToast(`${data.name} đã tham gia cuộc gọi`, 'info');
    updateParticipantCount(data.participants.length);

    // Create peer connection for the new user (they will send offer)
    if (data.user_id !== state.userId) {
        addRemoteVideoTile(data.user_id, data.name);
    }
}

function handleUserLeft(data) {
    showToast(`${data.name} đã rời cuộc gọi`, 'info');
    updateParticipantCount(data.participants.length);
    removePeerConnection(data.user_id);
    removeRemoteVideoTile(data.user_id);
    removeParticipantEmotionCard(data.user_id);
}

function handleLeftRoom() {
    leaveCallScreen();
}


// ============================================================
// CALL SCREEN
// ============================================================
function enterCallScreen(participants) {
    // Switch screens
    document.getElementById('lobby-screen').classList.remove('active');
    document.getElementById('call-screen').classList.add('active');

    // Setup local video
    const localVideo = document.getElementById('local-video');
    if (state.localStream) {
        localVideo.srcObject = state.localStream;
    }
    document.getElementById('local-name').textContent = state.userName + ' (Bạn)';

    // Room info
    document.getElementById('room-id-display').textContent = state.roomId;
    updateParticipantCount(participants.length);

    // Start call timer
    state.callStartTime = Date.now();
    startCallTimer();

    // Start sending frames for emotion detection
    startFrameSending();

    // Add participant cards
    addParticipantEmotionCard(state.userId, state.userName, true);
    participants.forEach(p => {
        if (p.user_id !== state.userId) {
            addRemoteVideoTile(p.user_id, p.name);
            addParticipantEmotionCard(p.user_id, p.name, false);
        }
    });

    updateVideoGrid();
}


function leaveCallScreen() {
    // Stop frame sending
    if (state.frameInterval) {
        clearInterval(state.frameInterval);
        state.frameInterval = null;
    }

    // Close peer connections
    Object.keys(state.peers).forEach(uid => removePeerConnection(uid));

    // Reset state
    state.roomId = null;
    state.emotionData = {};

    // Switch screens
    document.getElementById('call-screen').classList.remove('active');
    document.getElementById('lobby-screen').classList.add('active');

    // Remove remote tiles
    document.querySelectorAll('.video-tile:not(#local-tile)').forEach(el => el.remove());
    document.getElementById('participant-emotions').innerHTML = '';
}


function startCallTimer() {
    const timerEl = document.getElementById('call-timer');
    setInterval(() => {
        if (!state.callStartTime) return;
        const elapsed = Math.floor((Date.now() - state.callStartTime) / 1000);
        const min = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const sec = (elapsed % 60).toString().padStart(2, '0');
        timerEl.textContent = `${min}:${sec}`;
    }, 1000);
}


function updateParticipantCount(count) {
    document.getElementById('count-number').textContent = count;
}


function updateVideoGrid() {
    const grid = document.getElementById('video-grid');
    const tiles = grid.querySelectorAll('.video-tile');
    const count = tiles.length;

    // Remove old grid classes
    grid.className = 'video-grid';
    if (count <= 6) {
        grid.classList.add(`grid-${count}`);
    } else {
        grid.classList.add('grid-6');
    }
}


// ============================================================
// FRAME SENDING (for emotion detection)
// ============================================================
function startFrameSending() {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext('2d');

    state.frameInterval = setInterval(() => {
        if (!state.localStream || !state.isCameraOn) return;

        const video = document.getElementById('local-video');
        if (video.readyState < 2) return;

        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert to base64
        const frameData = canvas.toDataURL('image/jpeg', 0.6);

        // Send to server
        state.socket.emit('video_frame', { frame: frameData });

    }, FRAME_SEND_INTERVAL);
}


// ============================================================
// WebRTC PEER CONNECTIONS
// ============================================================
function createPeerConnection(remoteUserId, isInitiator) {
    if (state.peers[remoteUserId]) {
        console.log(`[RTC] Peer already exists: ${remoteUserId}`);
        return state.peers[remoteUserId];
    }

    console.log(`[RTC] Creating peer for ${remoteUserId}, initiator: ${isInitiator}`);
    const pc = new RTCPeerConnection(ICE_SERVERS);
    state.peers[remoteUserId] = pc;

    // Add local tracks
    if (state.localStream) {
        state.localStream.getTracks().forEach(track => {
            pc.addTrack(track, state.localStream);
        });
    }

    // ICE candidates
    pc.onicecandidate = (event) => {
        if (event.candidate) {
            state.socket.emit('webrtc_ice_candidate', {
                candidate: event.candidate,
                target: remoteUserId,
                from_user: state.userId,
                room_id: state.roomId
            });
        }
    };

    // Remote stream
    pc.ontrack = (event) => {
        console.log(`[RTC] Got remote track from ${remoteUserId}`);
        const remoteVideo = document.getElementById(`remote-video-${remoteUserId}`);
        if (remoteVideo && event.streams[0]) {
            remoteVideo.srcObject = event.streams[0];
            state.remoteStreams[remoteUserId] = event.streams[0];
        }
    };

    // Connection state
    pc.onconnectionstatechange = () => {
        console.log(`[RTC] ${remoteUserId} state: ${pc.connectionState}`);
        if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
            // Maybe retry
        }
    };

    // If initiator, create and send offer
    if (isInitiator) {
        pc.createOffer()
            .then(offer => pc.setLocalDescription(offer))
            .then(() => {
                state.socket.emit('webrtc_offer', {
                    sdp: pc.localDescription,
                    target: remoteUserId,
                    from_user: state.userId,
                    room_id: state.roomId
                });
            })
            .catch(err => console.error('[RTC] Offer error:', err));
    }

    return pc;
}


function removePeerConnection(userId) {
    if (state.peers[userId]) {
        state.peers[userId].close();
        delete state.peers[userId];
    }
    if (state.remoteStreams[userId]) {
        delete state.remoteStreams[userId];
    }
    if (state.emotionData[userId]) {
        delete state.emotionData[userId];
    }
}


// ---- WebRTC Signaling Handlers ----
async function handleWebRTCOffer(data) {
    const fromUser = data.from_user;
    console.log(`[RTC] Received offer from ${fromUser}`);

    const pc = createPeerConnection(fromUser, false);

    try {
        await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);

        state.socket.emit('webrtc_answer', {
            sdp: pc.localDescription,
            target: fromUser,
            from_user: state.userId,
            room_id: state.roomId
        });
    } catch (err) {
        console.error('[RTC] Answer error:', err);
    }
}


async function handleWebRTCAnswer(data) {
    const fromUser = data.from_user;
    const pc = state.peers[fromUser];
    if (pc) {
        try {
            await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
        } catch (err) {
            console.error('[RTC] Set answer error:', err);
        }
    }
}


async function handleICECandidate(data) {
    const fromUser = data.from_user;
    const pc = state.peers[fromUser];
    if (pc && data.candidate) {
        try {
            await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
        } catch (err) {
            console.error('[RTC] ICE error:', err);
        }
    }
}


// ============================================================
// VIDEO TILES
// ============================================================
function addRemoteVideoTile(userId, name) {
    if (document.getElementById(`remote-tile-${userId}`)) return;

    const tile = document.createElement('div');
    tile.className = 'video-tile';
    tile.id = `remote-tile-${userId}`;
    tile.dataset.userId = userId;
    tile.innerHTML = `
        <video id="remote-video-${userId}" autoplay playsinline></video>
        <div class="tile-overlay">
            <div class="emotion-badge" id="emotion-badge-${userId}">
                <span class="emotion-badge-emoji">😐</span>
                <span class="emotion-badge-text">Đang chờ...</span>
            </div>
            <div class="tile-name">${escapeHtml(name)}</div>
        </div>
        <div class="engagement-indicator" id="engagement-indicator-${userId}">
            <div class="engagement-fill"></div>
        </div>
    `;

    document.getElementById('video-grid').appendChild(tile);
    addParticipantEmotionCard(userId, name, false);
    updateVideoGrid();
}


function removeRemoteVideoTile(userId) {
    const tile = document.getElementById(`remote-tile-${userId}`);
    if (tile) {
        tile.remove();
        updateVideoGrid();
    }
}


// ============================================================
// PARTICIPANT EMOTION CARDS
// ============================================================
function addParticipantEmotionCard(userId, name, isLocal) {
    if (document.getElementById(`pcard-${userId}`)) return;

    const container = document.getElementById('participant-emotions');
    const initial = name.charAt(0).toUpperCase();

    const card = document.createElement('div');
    card.className = 'participant-emotion-card';
    card.id = `pcard-${userId}`;
    card.innerHTML = `
        <div class="participant-avatar">${initial}</div>
        <div class="participant-details">
            <div class="participant-name-row">
                <span class="participant-name">${escapeHtml(name)}${isLocal ? ' (Bạn)' : ''}</span>
                <span class="participant-emotion-emoji" id="pemoji-${userId}">😐</span>
            </div>
            <div class="participant-emotion-text" id="ptext-${userId}">Đang chờ...</div>
            <div class="participant-engagement-bar">
                <div class="participant-engagement-fill" id="pfill-${userId}" style="width:0%"></div>
            </div>
        </div>
    `;

    container.appendChild(card);
}


function removeParticipantEmotionCard(userId) {
    const card = document.getElementById(`pcard-${userId}`);
    if (card) card.remove();
}


// ============================================================
// EMOTION RESULT HANDLER
// ============================================================
function handleEmotionResult(data) {
    const userId = data.user_id;

    // Store emotion data
    state.emotionData[userId] = data;

    if (!data.face_detected) {
        // Update UI for no face
        updateEmotionBadge(userId, '😶', 'Không thấy mặt');
        return;
    }

    // Update emotion badge on video tile
    const label = `${data.dominant_vi} (${data.confidence.toFixed(0)}%)`;
    updateEmotionBadge(userId, data.emoji, label);

    // Update engagement indicator
    updateEngagementIndicator(userId, data.engagement);

    // Update participant emotion card
    updateParticipantCard(userId, data);

    // Update group analytics
    updateGroupAnalytics();

    // Update charts
    updateCharts(data);
}


function updateEmotionBadge(userId, emoji, text) {
    // Local user
    if (userId === state.userId) {
        const badge = document.getElementById('local-emotion-badge');
        if (badge) {
            badge.querySelector('.emotion-badge-emoji').textContent = emoji;
            badge.querySelector('.emotion-badge-text').textContent = text;
        }
    }

    // Any user (including remote)
    const badge = document.getElementById(`emotion-badge-${userId}`);
    if (badge) {
        badge.querySelector('.emotion-badge-emoji').textContent = emoji;
        badge.querySelector('.emotion-badge-text').textContent = text;
    }
}


function updateEngagementIndicator(userId, engagement) {
    let indicator;
    if (userId === state.userId) {
        indicator = document.getElementById('local-engagement-indicator');
    } else {
        indicator = document.getElementById(`engagement-indicator-${userId}`);
    }

    if (indicator) {
        const fill = indicator.querySelector('.engagement-fill');
        fill.style.width = `${engagement}%`;
    }
}


function updateParticipantCard(userId, data) {
    const emoji = document.getElementById(`pemoji-${userId}`);
    const text = document.getElementById(`ptext-${userId}`);
    const fill = document.getElementById(`pfill-${userId}`);

    if (emoji) emoji.textContent = data.emoji;
    if (text) text.textContent = `${data.dominant_vi} - ${data.confidence.toFixed(0)}% | Tương tác: ${data.engagement.toFixed(0)}%`;

    if (fill) {
        fill.style.width = `${data.engagement}%`;
        if (data.engagement >= 60) {
            fill.style.background = '#22c55e';
        } else if (data.engagement >= 30) {
            fill.style.background = '#6366f1';
        } else {
            fill.style.background = '#ef4444';
        }
    }
}


function updateGroupAnalytics() {
    const users = Object.values(state.emotionData).filter(d => d.face_detected);
    if (users.length === 0) return;

    // Average engagement
    const avgEngagement = users.reduce((sum, d) => sum + d.engagement, 0) / users.length;

    // Update circle
    const progressEl = document.getElementById('group-engagement-progress');
    const valueEl = document.getElementById('group-engagement-value');
    const labelEl = document.getElementById('group-engagement-label');

    if (valueEl) valueEl.textContent = Math.round(avgEngagement);

    if (progressEl) {
        const offset = CIRCUMFERENCE - (avgEngagement / 100) * CIRCUMFERENCE;
        progressEl.style.strokeDashoffset = offset;

        let color;
        if (avgEngagement >= 70) {
            color = '#22c55e';
            if (labelEl) labelEl.textContent = '🔥 Rất tích cực';
        } else if (avgEngagement >= 50) {
            color = '#6366f1';
            if (labelEl) labelEl.textContent = '👍 Tích cực';
        } else if (avgEngagement >= 30) {
            color = '#f59e0b';
            if (labelEl) labelEl.textContent = '😐 Trung bình';
        } else {
            color = '#ef4444';
            if (labelEl) labelEl.textContent = '⚠️ Thấp';
        }

        progressEl.style.stroke = color;
        if (valueEl) valueEl.style.color = color;
    }

    // Engagement breakdown
    const breakdownEl = document.getElementById('engagement-breakdown');
    if (breakdownEl) {
        breakdownEl.innerHTML = users.map(d =>
            `<span>${d.user_name || 'User'}: ${d.engagement.toFixed(0)}%</span>`
        ).join('');
    }
}


// ============================================================
// CHARTS
// ============================================================
let barChart, timelineChart;

function initCharts() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";

    // Bar Chart
    const barCtx = document.getElementById('emotionBarChart');
    if (!barCtx) return;

    barChart = new Chart(barCtx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: EMOTIONS_VI,
            datasets: [{
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: EMOTION_COLORS.map(c => c + '80'),
                borderColor: EMOTION_COLORS,
                borderWidth: 1.5,
                borderRadius: 6,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1a1f36',
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.x.toFixed(1)}%`
                    }
                }
            },
            scales: {
                x: {
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => v + '%', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 11, weight: 500 } }
                }
            },
            animation: { duration: 300 }
        }
    });

    // Timeline Chart (group engagement over time)
    const lineCtx = document.getElementById('engagementTimelineChart');
    if (!lineCtx) return;

    timelineChart = new Chart(lineCtx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Tương tác nhóm',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        boxWidth: 10,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        padding: 12,
                        font: { size: 10 }
                    }
                },
                tooltip: {
                    backgroundColor: '#1a1f36',
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { maxTicksLimit: 8, font: { size: 10 } }
                },
                y: {
                    min: 0, max: 100,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => v + '%', font: { size: 10 } }
                }
            },
            animation: { duration: 200 }
        }
    });
}


function updateCharts(data) {
    if (!barChart || !timelineChart) return;

    // Update bar chart with averaged emotions from all users
    const users = Object.values(state.emotionData).filter(d => d.face_detected && d.emotions);
    if (users.length === 0) return;

    const avgEmotions = EMOTIONS.map(e => {
        const sum = users.reduce((s, d) => s + (d.emotions[e] || 0), 0);
        return sum / users.length;
    });

    barChart.data.datasets[0].data = avgEmotions;
    barChart.update('none');

    // Update timeline
    const elapsed = state.callStartTime ? Math.floor((Date.now() - state.callStartTime) / 1000) : 0;
    const timeLabel = formatTime(elapsed);
    const avgEngagement = users.reduce((s, d) => s + d.engagement, 0) / users.length;

    timelineChart.data.labels.push(timeLabel);
    timelineChart.data.datasets[0].data.push(avgEngagement.toFixed(1));

    if (timelineChart.data.labels.length > MAX_TIMELINE_POINTS) {
        timelineChart.data.labels.shift();
        timelineChart.data.datasets[0].data.shift();
    }
    timelineChart.update('none');
}


// ============================================================
// CALL CONTROLS
// ============================================================
function initCallControls() {
    // Mic toggle
    document.getElementById('btn-toggle-mic').addEventListener('click', () => {
        state.isMicOn = !state.isMicOn;
        const btn = document.getElementById('btn-toggle-mic');

        if (state.localStream) {
            state.localStream.getAudioTracks().forEach(t => t.enabled = state.isMicOn);
        }

        btn.classList.toggle('muted', !state.isMicOn);
        btn.querySelector('.control-icon').textContent = state.isMicOn ? '🎤' : '🎤';
    });

    // Camera toggle
    document.getElementById('btn-toggle-camera').addEventListener('click', () => {
        state.isCameraOn = !state.isCameraOn;
        const btn = document.getElementById('btn-toggle-camera');

        if (state.localStream) {
            state.localStream.getVideoTracks().forEach(t => t.enabled = state.isCameraOn);
        }

        btn.classList.toggle('muted', !state.isCameraOn);
        document.getElementById('local-tile').classList.toggle('no-video', !state.isCameraOn);
    });

    // Analytics toggle
    document.getElementById('btn-toggle-analytics').addEventListener('click', toggleAnalytics);
    document.getElementById('btn-close-analytics').addEventListener('click', toggleAnalytics);

    // End call
    document.getElementById('btn-end-call').addEventListener('click', () => {
        state.socket.emit('leave_room_request', {});
    });

    // Copy room ID
    document.getElementById('btn-copy-room').addEventListener('click', () => {
        if (state.roomId) {
            navigator.clipboard.writeText(state.roomId).then(() => {
                showToast('Đã sao chép mã phòng!', 'success');
            });
        }
    });
}


function toggleAnalytics() {
    state.isAnalyticsOpen = !state.isAnalyticsOpen;
    const panel = document.getElementById('analytics-panel');
    panel.classList.toggle('hidden', !state.isAnalyticsOpen);

    const btn = document.getElementById('btn-toggle-analytics');
    btn.classList.toggle('muted', !state.isAnalyticsOpen);
}


// ============================================================
// UTILITIES
// ============================================================
function generateId() {
    return Math.random().toString(36).substring(2, 10);
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateConnectionStatus(connected) {
    const badge = document.getElementById('status-badge');
    const text = document.getElementById('status-text');

    if (badge && text) {
        badge.className = `status-badge ${connected ? 'connected' : 'disconnected'}`;
        text.textContent = connected ? 'Đã kết nối' : 'Mất kết nối';
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toast-out 0.3s forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
