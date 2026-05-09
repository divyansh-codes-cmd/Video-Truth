# 🎬 VideoTruth — AI Video Detector

A deepfake detection web app that analyzes any video URL to determine whether the content is **real or AI-generated**, using computer vision techniques.

---

## 🚀 Features

- 🔍 Detects AI-generated / deepfake videos from any public URL
- 📊 Returns verdict with confidence score (0–95%)
- 🧠 Analyzes 5 computer vision signals per frame
- 👤 User registration & login system
- 📁 Analysis history saved per user
- 🌐 Clean web-based frontend (no installation needed for users)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Computer Vision | OpenCV, NumPy |
| Video Download | yt-dlp |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Auth | SHA-256 password hashing |

---

## 🔬 How It Works

1. User pastes a video URL (YouTube, Instagram, etc.)
2. Video is downloaded using **yt-dlp**
3. **10 frames** are extracted from the video
4. Each frame is analyzed using 5 signals:
   - 🔊 **Noise Uniformity** — AI videos are too smooth
   - 📐 **Edge Consistency** — AI edges are too perfect
   - 🎨 **Color Distribution** — AI colors are too balanced
   - 👁️ **Face Artifacts** — frequency anomalies in facial regions
   - 🎞️ **Temporal Consistency** — unnatural frame transitions
5. Scores are combined into a final verdict:
   - ✅ **Real**
   - ⚠️ **AI Generated**
   - ❓ **Uncertain**

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/divyansh-codes-cmd/videotruth.git
cd videotruth

# Install dependencies
pip install fastapi uvicorn opencv-python numpy yt-dlp

# Run the server
uvicorn main:app --reload
```

Then open `index.html` in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/register` | Create a new account |
| POST | `/login` | Login to existing account |
| POST | `/analyze` | Analyze a video URL |
| GET | `/history` | Get analysis history |
| GET | `/stats` | Get analysis statistics |

---

## 📸 Output Example

```json
{
  "verdict": "AI Generated",
  "confidence": 78,
  "signals": [
    { "label": "Facial frequency artifacts found", "level": "high" },
    { "label": "Unnatural frame transitions", "level": "high" }
  ]
}
```

---

## ⚠️ Limitations

- Works only on **public** video URLs
- Uses traditional computer vision (not deep learning)
- Accuracy may vary on heavily compressed videos

---

## 👨‍💻 Author

**Divyansh** — [@divyansh-codes-cmd](https://github.com/divyansh-codes-cmd)

---

## 📄 License

This project is for educational purposes.
