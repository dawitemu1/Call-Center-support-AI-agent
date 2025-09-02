# 🎙️ Amharic Speech-to-Text & Tone Detection for Call Centers

This project provides **real-time Amharic speech transcription** and **tone detection** for **direct call center support**.  
Built with **FastAPI (Python)** for the backend and **React** for the frontend.  

---

## 🚀 Features
- 🎤 **Real-Time Speech-to-Text (STT)** — Optimized for **Amharic language**.
- 🎯 **Tone Detection** — Detects sentiment/tone during calls:
  - 🟢 Positive  
  - 🟡 Neutral  
  - 🔴 Negative
- 📡 **WebSocket Integration** — Enables **live captions** while recording or during a call.
- 🎧 **File Upload Support** — Supports audio formats:
  - `WAV`, `MP3`, `MP4`, `M4A`, `OGG`, `FLAC`, `AAC`, `WMA`
- 💾 **High-Quality Recording** — Call recordings are saved in `.wav`.
- 🏢 **Call Center Use Case** — Agents get **instant transcription + tone insight**.

---

## 🛠️ Tech Stack
- **Backend** → [FastAPI](https://fastapi.tiangolo.com/) (Python)  
- **Frontend** → [React](https://reactjs.org/)  
- **Speech-to-Text Engine** → (Whisper / Wav2Vec / Custom Amharic model)  
- **Tone Detection** → Transformer-based sentiment model  
- **Database (optional)** → PostgreSQL / MongoDB for storing transcripts  

---

## 📂 Project Structure
```bash
amharic-speech-tone/
│── backend/              # FastAPI server
│   ├── main.py           # API endpoints & WebSocket
│   ├── stt_model.py      # Speech-to-Text model
│   ├── tone_model.py     # Tone detection model
│   └── requirements.txt  # Backend dependencies
│
│── frontend/             # React app
│   ├── src/
│   │   ├── components/   # UI components
│   │   ├── pages/        # Main pages
│   │   └── App.js        # Main React app
│   └── package.json      # Frontend dependencies
│
│── README.md             # Project documentation

⚡ Installation
🔹 Backend (FastAPI)
cd backend
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
uvicorn main:app --reload


Demo (UI Preview)


[🎤 Start Recording]
[✅ Analyze] → "ደንበኛው እጅግ ደስ ብሎታል" → 🟢 Positive

