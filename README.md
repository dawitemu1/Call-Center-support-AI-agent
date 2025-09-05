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
- 🎧 **File Upload Support** — Supports audio and video formats:
  - **Audio**: `WAV`, `MP3`, `MP4`, `M4A`, `OGG`, `FLAC`, `AAC`, `WMA`
  - **Video**: `MP4`, `AVI`, `MOV`, `MKV`, `WMV`, `FLV`, `WebM`, `M4V`, `3GP`, `OGV`, `TS`, `MTS`, `M2TS`
- 💾 **High-Quality Recording** — Call recordings are saved in `.wav`.
- 🏢 **Call Center Use Case** — Agents get **instant transcription + tone insight**.
- ⚡ **Enhanced Real-Time Processing** — Improved handling of fragmented speech for better accuracy.

## 🎯 Call Center Use Cases  
- **Customer Support** → Detect frustrated customers early and alert supervisors  
- **Quality Monitoring** → Analyze tone & emotion across calls to measure satisfaction  
- **Training Agents** → Improve handling skills by reviewing tone + transcripts  
- **Compliance** → Store transcripts for auditing while analyzing customer emotions  

---

## 📸 Demo Screenshot  

<img width="2448" height="1139" alt="image" src="https://github.com/user-attachments/assets/388df45b-9274-47ea-acb4-95bb2cbfc1e4" />

<img width="1853" height="1085" alt="image" src="https://github.com/user-attachments/assets/0a55aa1b-3af5-4efb-a95b-74d63338bfde" />


*(Replace with a screenshot of live call transcription + tone indicator)*  

---

## 🛠️ Installation  

```bash
# Clone repo
git clone https://github.com/your-username/amharic-callcenter-support.git
cd amharic-callcenter-support

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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
```

## ⚡ Installation
### 🔧 Backend (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
uvicorn main:app --reload
```

### 🌐 Frontend (React)
```bash
cd frontend
npm install
npm start
```

## 🎯 Enhanced Real-Time Amharic Processing

This update includes significant improvements to real-time Amharic speech processing:

### 🔧 Key Enhancements:
- **Improved Fragmented Speech Handling**: Better reconstruction of fragmented speech chunks
- **Enhanced Audio Preprocessing**: Optimized for Amharic phoneme clarity
- **Real-Time Tone Detection**: Instant sentiment analysis during speech
- **WebSocket Performance**: Faster and more reliable real-time transcription
- **Accumulated Transcription**: Continuous building of conversation context

### 🚀 Real-Time Features:
1. **Live Transcription**: See speech converted to text as it's spoken
2. **Tone Indicators**: Visual sentiment analysis (Positive/Neutral/Negative)
3. **Transcription History**: Track conversation flow with timestamped entries
4. **Performance Metrics**: Monitor processing speed and accuracy

### 📊 Processing Improvements:
- **Faster Chunk Processing**: Optimized for sub-second response times
- **Better Noise Reduction**: Enhanced audio preprocessing for clearer transcription
- **Improved Word Boundary Detection**: Better reconstruction of fragmented words
- **Enhanced Confidence Scoring**: More accurate assessment of transcription quality

## 📹 Video Processing
The system now supports processing Amharic video files:
- Upload any supported video format through the file input
- The system automatically extracts audio and processes it
- Tone detection and transcription work the same as with audio files
- See [VIDEO_PROCESSING.md](VIDEO_PROCESSING.md) for detailed technical information

## Demo (UI Preview)

[🎤 Start Recording]
[✅ Analyze] → "ደንበኛው እጅግ ደስ ብሎታል" → 🟢 Positive

## 👨‍💻 Author

Dawit Shibabaw
🚀 Data Scientist | AI Researcher | Call Center AI Solutions

---

👉 Do you also want me to include **deployment steps** (e.g., Docker + Nginx for production) in this README so it's ready for real-world call center integration?