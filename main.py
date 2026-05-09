from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib
import yt_dlp
import cv2
import numpy as np
import os
import tempfile
from database import init_db, create_user, get_user, email_exists, save_result, get_history, get_stats

app = FastAPI(title="VideoTruth API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ===== MODELS =====

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class VideoRequest(BaseModel):
    url: str
    username: str = None

# ===== HELPER =====

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ===== AUTH ROUTES =====

@app.post("/register")
def register(req: RegisterRequest):
    if len(req.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="Please enter a valid email.")

    if get_user(req.username):
        raise HTTPException(status_code=400, detail="This username is already taken. Please try another.")
    if email_exists(req.email):
        raise HTTPException(status_code=400, detail="This email is already registered.")

    success = create_user(req.username, req.email, hash_password(req.password))
    if not success:
        raise HTTPException(status_code=400, detail="Account could not be created. Please try again.")

    return {"message": "Account created successfully!", "username": req.username}

@app.post("/login")
def login(req: LoginRequest):
    user = get_user(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="Username not found. Please register first.")

    stored_hash = user[3]
    if stored_hash != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Incorrect password. Please try again.")

    return {"message": "Login successful!", "username": user[1]}

# ===== VIDEO ANALYSIS =====

def download_video(url: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "video.mp4")
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'mp4/best[height<=480]',
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'retries': 3,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def extract_frames(video_path: str, num_frames: int = 10) -> list:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total == 0:
        cap.release()
        return frames
    step = max(1, total // num_frames)
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) >= num_frames:
            break
    cap.release()
    return frames


# ===== REAL DETECTION FUNCTIONS =====

def check_noise_uniformity(frame) -> float:
    """
    AI generated frames tend to have very uniform noise.
    Real videos contain natural random noise.
    Score: 0 (real-like) to 1 (AI-like)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate local standard deviation
    mean, std = cv2.meanStdDev(gray)
    # Measure noise using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_std = laplacian.std()

    # AI videos tend to have very low or very high noise std
    if noise_std < 5.0:
        return 0.8  # Too smooth = AI
    elif noise_std > 80.0:
        return 0.3  # Very noisy = likely real
    else:
        # Normal range
        normalized = (noise_std - 5.0) / 75.0
        return max(0.1, 0.7 - normalized * 0.5)


def check_edge_consistency(frame) -> float:
    """
    AI videos tend to have very sharp and perfect edges.
    Real videos have slightly blurry/natural edges.
    Score: 0 (real) to 1 (AI)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Calculate gradient using Sobelx and Sobely
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_mean = gradient_magnitude.mean()

    # AI tends to have very consistent gradients
    gradient_std = gradient_magnitude.std()
    consistency_ratio = gradient_mean / (gradient_std + 1e-6)

    if consistency_ratio > 3.0:
        return 0.75  # Too consistent = AI
    elif consistency_ratio > 2.0:
        return 0.55
    else:
        return 0.25


def check_color_distribution(frame) -> float:
    """
    AI videos tend to have very balanced/perfect color distribution.
    Real videos show natural variations in color histograms.
    Score: 0 (real) to 1 (AI)
    """
    scores = []
    for channel in range(3):  # BGR channels
        hist = cv2.calcHist([frame], [channel], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize

        # Calculate entropy
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

        # AI videos tend to have very high entropy (too perfect distribution)
        if entropy > 7.5:
            scores.append(0.7)
        elif entropy > 6.5:
            scores.append(0.5)
        elif entropy < 4.0:
            scores.append(0.6)  # Too low entropy is also suspicious
        else:
            scores.append(0.2)

    return np.mean(scores)


def check_face_artifacts(frame) -> tuple:
    """
    Detect faces and check for artifacts around them.
    Returns: (score, face_found)
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

    if len(faces) == 0:
        return 0.5, False  # No face found, neutral score

    face_scores = []
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        # Check for artifacts around the face boundary
        # Take a slightly larger region
        margin = int(w * 0.15)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
        expanded_roi = frame[y1:y2, x1:x2]

        if expanded_roi.size == 0:
            continue

        # Frequency analysis on face region
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray_face.astype(float))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)

        # High frequency components check karo
        h_f, w_f = magnitude.shape
        center_h, center_w = h_f // 2, w_f // 2
        high_freq_region = magnitude.copy()
        high_freq_region[center_h-10:center_h+10, center_w-10:center_w+10] = 0
        high_freq_ratio = high_freq_region.mean() / (magnitude.mean() + 1e-6)

        if high_freq_ratio > 0.85:
            face_scores.append(0.7)  # Unusual high frequency = AI artifact
        elif high_freq_ratio > 0.75:
            face_scores.append(0.5)
        else:
            face_scores.append(0.25)

    if not face_scores:
        return 0.5, True

    return np.mean(face_scores), True


def check_temporal_consistency(frames) -> float:
    """
    Check the difference between consecutive frames.
    AI videos tend to have very smooth or unnatural frame transitions.
    Score: 0 (real) to 1 (AI)
    """
    if len(frames) < 2:
        return 0.5

    diffs = []
    for i in range(1, len(frames)):
        f1 = cv2.resize(frames[i-1], (160, 90))
        f2 = cv2.resize(frames[i], (160, 90))
        diff = cv2.absdiff(f1, f2)
        diffs.append(diff.mean())

    diffs = np.array(diffs)
    diff_mean = diffs.mean()
    diff_std = diffs.std()

    # AI videos tend to have very consistent frame differences (low std)
    variation_coefficient = diff_std / (diff_mean + 1e-6)

    if variation_coefficient < 0.2:
        return 0.75  # Too consistent = AI
    elif variation_coefficient < 0.4:
        return 0.55
    elif variation_coefficient > 1.5:
        return 0.3  # High variation = more natural
    else:
        return 0.35


def analyze_frames(frames: list) -> dict:
    """
    Real CV-based detection using multiple signals.
    """
    if not frames:
        return {"verdict": "Uncertain", "confidence": 50, "signals": []}

    # Collect per-frame scores
    noise_scores = []
    edge_scores = []
    color_scores = []
    face_scores = []
    face_found_any = False

    for frame in frames:
        noise_scores.append(check_noise_uniformity(frame))
        edge_scores.append(check_edge_consistency(frame))
        color_scores.append(check_color_distribution(frame))
        f_score, f_found = check_face_artifacts(frame)
        face_scores.append(f_score)
        if f_found:
            face_found_any = True

    # Temporal consistency (all frames together)
    temporal_score = check_temporal_consistency(frames)

    # Average scores
    avg_noise = np.mean(noise_scores)
    avg_edge = np.mean(edge_scores)
    avg_color = np.mean(color_scores)
    avg_face = np.mean(face_scores)

    # Weighted final score (0 = real, 1 = AI)
    if face_found_any:
        # Face found, give more weight to face analysis
        final_score = (
            avg_noise * 0.20 +
            avg_edge * 0.20 +
            avg_color * 0.15 +
            avg_face * 0.30 +
            temporal_score * 0.15
        )
    else:
        # No face found, other signals are more important
        final_score = (
            avg_noise * 0.30 +
            avg_edge * 0.25 +
            avg_color * 0.25 +
            temporal_score * 0.20
        )

    ai_percentage = int(final_score * 100)

    # Push score toward the extremes — avoid getting stuck in the middle
    if ai_percentage > 50:
        ai_percentage = 50 + int((ai_percentage - 50) * 1.4)
    elif ai_percentage < 50:
        ai_percentage = 50 - int((50 - ai_percentage) * 1.4)
    ai_percentage = max(0, min(100, ai_percentage))

    # Verdict decide karo — Uncertain sirf 48-52 range mein
    if ai_percentage >= 53:
        verdict = "AI Generated"
        confidence = min(ai_percentage, 95)

        signals = []
        if avg_noise > 0.55:
            signals.append({"label": "Unnatural noise uniformity detected", "level": "high"})
        if avg_edge > 0.55:
            signals.append({"label": "Edge consistency too perfect", "level": "high"})
        if avg_color > 0.55:
            signals.append({"label": "Color distribution anomaly", "level": "high"})
        if face_found_any and avg_face > 0.55:
            signals.append({"label": "Facial frequency artifacts found", "level": "high"})
        if temporal_score > 0.55:
            signals.append({"label": "Unnatural frame transitions", "level": "high"})
        if not signals:
            signals.append({"label": "Multiple weak AI signals combined", "level": "mid"})

    elif ai_percentage <= 47:
        verdict = "Real"
        confidence = min(100 - ai_percentage, 95)

        signals = []
        if avg_noise < 0.45:
            signals.append({"label": "Natural noise pattern detected", "level": "low"})
        if avg_edge < 0.45:
            signals.append({"label": "Natural edge variations present", "level": "low"})
        if avg_color < 0.45:
            signals.append({"label": "Normal color distribution", "level": "low"})
        if face_found_any and avg_face < 0.45:
            signals.append({"label": "No facial artifacts found", "level": "low"})
        if temporal_score < 0.45:
            signals.append({"label": "Natural frame transitions", "level": "low"})
        if avg_noise >= 0.45:
            signals.append({"label": "Minor compression artifacts (normal)", "level": "mid"})
        if not signals:
            signals.append({"label": "Overall natural video patterns", "level": "low"})

    else:
        # Only Uncertain in the 48-52 range — will be very rare
        verdict = "Uncertain"
        confidence = 50
        signals = [
            {"label": "Signals equally balanced", "level": "mid"},
            {"label": "Video quality analysis inconclusive", "level": "mid"},
        ]

    return {"verdict": verdict, "confidence": confidence, "signals": signals}


def cleanup(video_path: str):
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
        os.rmdir(os.path.dirname(video_path))
    except:
        pass

@app.get("/")
def root():
    return {"message": "VideoTruth API is running!"}

@app.post("/analyze")
async def analyze_video(req: VideoRequest):
    url = req.url.strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Please enter a valid URL.")

    video_path = None
    try:
        print(f"[*] Downloading: {url}")
        video_path = download_video(url)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise HTTPException(status_code=400, detail="Video file could not be downloaded or is empty. Please try again.")

        print(f"[*] Extracting frames...")
        frames = extract_frames(video_path)
        if not frames:
            raise HTTPException(status_code=400, detail="No frames found. Video format may not be supported or file is corrupt.")

        print(f"[*] Analyzing {len(frames)} frames...")
        result = analyze_frames(frames)
        save_result(req.username, url, result["verdict"], result["confidence"], result["signals"])
        print(f"[✓] {result['verdict']} ({result['confidence']}%)")
        return result

    except HTTPException:
        raise
    except yt_dlp.utils.DownloadError as e:
        err = str(e)
        if "getaddrinfo failed" in err or "Failed to resolve" in err:
            raise HTTPException(status_code=400, detail="Server could not connect to the internet. DNS error — please try again later.")
        elif "timed out" in err.lower():
            raise HTTPException(status_code=400, detail="Video download timed out. Instagram may be slow, please try again.")
        elif "Private" in err or "login" in err.lower():
            raise HTTPException(status_code=400, detail="This video is private. Please provide a public video URL.")
        else:
            raise HTTPException(status_code=400, detail=f"Video could not be downloaded: {err[:120]}")
    except Exception as e:
        print(f"[!] Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)[:150]}")
    finally:
        if video_path:
            cleanup(video_path)

@app.get("/history")
def history(username: str = None):
    rows = get_history(username)
    return [{"id": r[0], "username": r[1], "url": r[2], "verdict": r[3], "confidence": r[4], "created_at": r[6]} for r in rows]

@app.get("/stats")
def stats(username: str = None):
    return get_stats(username)
