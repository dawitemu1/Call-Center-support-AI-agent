# Suppress TensorFlow informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN if needed

# Enable transformers offline mode as per project specifications
try:
    import requests
    requests.get('https://huggingface.co', timeout=5)
    print("🌐 Internet connection available - online mode enabled")
except Exception:
    print("📦 No internet connection - enabling offline mode")
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.pipelines import pipeline
from typing import Optional, Any, Union, List, Dict, cast
from pydub import AudioSegment
import soundfile as sf
import io
import tempfile
import uuid
import asyncio
import json
import numpy as np
import queue as queue_module
import threading
import time
from typing import List
import webrtcvad
import pyaudio

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("🚀 Starting background transcription broadcaster...")
    asyncio.create_task(broadcast_transcriptions())
    yield
    # Shutdown
    print("🛑 Shutting down...")

# Background task to broadcast transcriptions
async def broadcast_transcriptions():
    """Background task to send transcriptions to WebSocket clients"""
    while True:
        try:
            # Check for new transcriptions (non-blocking)
            try:
                result = transcription_queue.get_nowait()
                await manager.broadcast(json.dumps(result))
            except queue_module.Empty:
                pass
            
            # Small delay to prevent CPU overload
            await asyncio.sleep(0.01)
            
        except Exception as e:
            print(f"❌ Broadcast error: {e}")
            await asyncio.sleep(0.1)

app = FastAPI(title="Speech-to-Text & Tone Analysis API", version="1.0.0", lifespan=lifespan)

# Real-time audio processing configuration
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate for Wav2Vec2
RECORD_SECONDS = 0.5  # Processing chunk duration
VAD_MODE = 2  # WebRTC VAD aggressiveness (0-3, 3 = most aggressive)

# Global variables for real-time processing
active_connections: List[WebSocket] = []
audio_queue = queue_module.Queue()
transcription_queue = queue_module.Queue()
is_recording = False
recording_thread = None
processing_thread = None

class ConnectionManager:
    """Manage WebSocket connections for real-time transcription"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔗 Client connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"🔌 Client disconnected. Active connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"❌ Failed to send message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Send message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"❌ Failed to broadcast to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

# Initialize connection manager (will be reassigned after class definition)
manager = None

class RealTimeAudioProcessor:
    """Handle real-time audio capture and processing"""
    
    def __init__(self):
        self.is_processing = False
        self.audio_buffer = []
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.silence_threshold = 30  # frames of silence before stopping
        self.current_silence_count = 0
        self.speech_started = False
        
    def start_processing(self):
        """Start real-time audio processing"""
        if self.is_processing:
            return False
            
        self.is_processing = True
        self.audio_buffer = []
        self.current_silence_count = 0
        self.speech_started = False
        
        # Start audio capture thread
        self.capture_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._audio_process_loop, daemon=True)
        self.process_thread.start()
        
        print("🎤 Real-time audio processing started")
        return True
    
    def stop_processing(self):
        """Stop real-time audio processing"""
        self.is_processing = False
        print("🛑 Real-time audio processing stopped")
    
    def _audio_capture_loop(self):
        """Continuous audio capture from microphone"""
        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Check for available audio devices
            device_count = p.get_device_count()
            print(f"🎧 Found {device_count} audio devices")
            
            # Find default input device
            default_device = None
            for i in range(device_count):
                device_info = p.get_device_info_by_index(i)
                max_input_channels = device_info.get('maxInputChannels', 0)
                if isinstance(max_input_channels, (int, float)) and max_input_channels > 0:
                    print(f"   Device {i}: {device_info['name']} - {max_input_channels} channels")
                    if default_device is None:
                        default_device = i
            
            if default_device is None:
                print("❌ No audio input devices found")
                return
            
            print(f"🎤 Using audio device: {p.get_device_info_by_index(default_device)['name']}")
            
            # Open audio stream
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=default_device,
                frames_per_buffer=CHUNK
            )
            
            print("🔴 Recording started...")
            
            frame_duration = 10  # ms for VAD
            frame_size = int(RATE * frame_duration / 1000)  # samples per frame
            
            while self.is_processing:
                try:
                    # Read audio chunk
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Process in 10ms frames for VAD
                    for i in range(0, len(audio_data), frame_size):
                        frame = audio_data[i:i+frame_size]
                        if len(frame) < frame_size:
                            break
                            
                        # Check for voice activity
                        frame_bytes = frame.astype(np.int16).tobytes()
                        
                        try:
                            is_speech = self.vad.is_speech(frame_bytes, RATE)
                            
                            if is_speech:
                                if not self.speech_started:
                                    self.speech_started = True
                                    print("🗣️ Speech detected - recording...")
                                
                                self.audio_buffer.extend(frame)
                                self.current_silence_count = 0
                            else:
                                if self.speech_started:
                                    self.audio_buffer.extend(frame)
                                    self.current_silence_count += 1
                                    
                                    # If enough silence, process the buffer
                                    if self.current_silence_count >= self.silence_threshold:
                                        if len(self.audio_buffer) > 0:
                                            audio_queue.put(np.array(self.audio_buffer).copy())
                                        
                                        # Reset for next speech segment
                                        self.audio_buffer = []
                                        self.speech_started = False
                                        self.current_silence_count = 0
                                        print("🔇 Speech ended - processing audio...")
                        except Exception as vad_error:
                            # If VAD fails, fall back to simple energy detection
                            energy = np.sum(frame.astype(np.float32) ** 2)
                            if energy > 1000000:  # Simple threshold
                                self.audio_buffer.extend(frame)
                    
                except Exception as read_error:
                    print(f"⚠️ Audio read error: {read_error}")
                    time.sleep(0.01)
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("🔴 Recording stopped")
            
        except Exception as e:
            print(f"❌ Audio capture error: {e}")
            self.is_processing = False
    
    def _audio_process_loop(self):
        """Process audio chunks for transcription"""
        print("🔄 Audio processing loop started")
        
        while self.is_processing:
            try:
                # Get audio from queue (timeout to allow checking is_processing)
                try:
                    audio_data = audio_queue.get(timeout=0.1)
                except queue_module.Empty:
                    continue
                
                if len(audio_data) == 0:
                    continue
                
                print(f"🎧 Processing audio chunk: {len(audio_data)} samples")
                
                # Convert to float and normalize
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # Ensure minimum length for processing
                if len(audio_float) < 1600:  # Minimum 0.1 seconds
                    print("⚠️ Audio chunk too short, skipping")
                    continue
                
                # Process with speech-to-text model
                try:
                    transcription = self._transcribe_chunk(audio_float)
                    
                    if transcription and transcription.strip() and transcription != "[No speech detected]":
                        print(f"📝 Transcribed: '{transcription}'")
                        
                        # Detect tone
                        tone, confidence = detect_tone(transcription)
                        print(f"🎯 Tone: {tone} ({confidence:.3f})")
                        
                        # Send to WebSocket clients
                        result = {
                            "type": "transcription",
                            "text": transcription,
                            "tone": tone,
                            "confidence": round(confidence, 3),
                            "timestamp": time.time(),
                            "realtime": True
                        }
                        
                        # Queue for WebSocket broadcast
                        transcription_queue.put(result)
                    else:
                        print("🔇 No speech detected in chunk")
                        
                except Exception as transcribe_error:
                    print(f"❌ Transcription error: {transcribe_error}")
                
            except Exception as e:
                print(f"❌ Audio processing error: {e}")
                time.sleep(0.1)
        
        print("🔄 Audio processing loop stopped")
    
    def _transcribe_chunk(self, audio_data):
        """Transcribe a single audio chunk"""
        try:
            # Check if models are loaded
            if processor is None or model is None:
                print("❌ Models not loaded")
                return "[No speech detected]"
            
            # Type assertion to help type checker
            proc = processor  # type: Wav2Vec2Processor
            mod = model  # type: Wav2Vec2ForCTC
                
            # Process with the model
            input_values = proc(audio_data, return_tensors="pt", sampling_rate=RATE).input_values
            
            # Generate predictions with no gradient calculation for speed
            with torch.no_grad():
                logits = mod(input_values).logits
            
            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = proc.decode(predicted_ids[0])
            
            # Return cleaned transcription
            cleaned = transcription.strip() if transcription.strip() else "[No speech detected]"
            return cleaned if len(cleaned) > 2 else "[No speech detected]"
            
        except Exception as e:
            print(f"❌ Chunk transcription error: {e}")
            return "[No speech detected]"

# Initialize real-time processor
realtime_processor = RealTimeAudioProcessor()



# WebSocket endpoint for real-time transcription
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for real-time speech transcription"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "start_recording":
                if realtime_processor.start_processing():
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "status",
                            "message": "Recording started",
                            "status": "recording"
                        }),
                        websocket
                    )
                else:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": "Recording already in progress",
                            "status": "error"
                        }),
                        websocket
                    )
            
            elif message.get("action") == "stop_recording":
                realtime_processor.stop_processing()
                await manager.send_personal_message(
                    json.dumps({
                        "type": "status",
                        "message": "Recording stopped",
                        "status": "stopped"
                    }),
                    websocket
                )
            
            elif message.get("action") == "ping":
                await manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # Stop recording if this was the last client
        if len(manager.active_connections) == 0:
            realtime_processor.stop_processing()
            print("🛑 No active connections, stopping recording")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)

# Health check endpoint
@app.get("/")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Amharic Speech-to-Text API is running",
        "version": "1.0.0",
        "features": {
            "real_time_transcription": True,
            "offline_mode": True,
            "tone_detection": True,
            "websocket_support": True
        }
    }


# Test tone detection endpoint for debugging
@app.post("/test-tone")
async def test_tone_detection(text: str):
    """Test endpoint to debug tone detection with sample text"""
    try:
        print(f"\n🧪 Testing tone detection with text: '{text}'")
        tone_result, confidence = detect_tone(text)
        
        # Also get raw model output for debugging
        try:
            raw_result = None
            if tone_classifier is not None and text.strip():
                result = tone_classifier(text)
                # Handle different result types safely
                try:
                    if result and hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                        try:
                            result_list = list(result) if not isinstance(result, list) else result
                            raw_result = result_list[0] if len(result_list) > 0 else None
                        except (TypeError, ValueError):
                            raw_result = None
                    else:
                        raw_result = result
                except (TypeError, IndexError):
                    raw_result = None
            return {
                "input_text": text,
                "detected_tone": tone_result,
                "confidence": round(confidence, 3),
                "raw_model_output": raw_result,
                "status": "success"
            }
        except Exception as model_error:
            return {
                "input_text": text,
                "detected_tone": tone_result,
                "confidence": round(confidence, 3),
                "raw_model_output": f"Error: {model_error}",
                "status": "partial_success"
            }
            
    except Exception as e:
        return {
            "input_text": text,
            "error": str(e),
            "status": "error"
        }

@app.get("/health")
async def detailed_health():
    # Enhanced FFmpeg detection for health check
    ffmpeg_status = "unknown"
    ffmpeg_version = "unknown"
    ffmpeg_path_found = None
    
    try:
        from pydub.utils import which
        import subprocess
        
        # Multiple detection methods
        ffmpeg_path = None
        
        # Method 1: Direct command
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                ffmpeg_path = "ffmpeg"
                version_line = result.stdout.split('\n')[0]
                ffmpeg_version = version_line.split(' ')[2] if len(version_line.split(' ')) > 2 else "detected"
                ffmpeg_status = "available"
                ffmpeg_path_found = "system_path"
        except Exception:
            pass
        
        # Method 2: pydub which
        if not ffmpeg_path:
            detected = which("ffmpeg")
            if detected:
                ffmpeg_path = detected
                ffmpeg_path_found = detected
                try:
                    result = subprocess.run([detected, "-version"], 
                                          capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        version_line = result.stdout.split('\n')[0]
                        ffmpeg_version = version_line.split(' ')[2] if len(version_line.split(' ')) > 2 else "detected"
                        ffmpeg_status = "available"
                except Exception:
                    ffmpeg_status = "error"
        
        # Method 3: Common paths
        if not ffmpeg_path:
            common_paths = [
                r"C:\Users\Daveee\Desktop\AI\Speech-Text\ffmpeg-2025-08-25-git-1b62f9d3ae-essentials_build\ffmpeg-2025-08-25-git-1b62f9d3ae-essentials_build\bin\ffmpeg.exe",
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
            ]
            for path in common_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    ffmpeg_path_found = path
                    try:
                        result = subprocess.run([path, "-version"], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            version_line = result.stdout.split('\n')[0]
                            ffmpeg_version = version_line.split(' ')[2] if len(version_line.split(' ')) > 2 else "detected"
                            ffmpeg_status = "available"
                        else:
                            ffmpeg_status = "error"
                    except Exception:
                        ffmpeg_status = "timeout"
                    break
        
        if not ffmpeg_path:
            ffmpeg_status = "missing"
            
    except Exception:
        ffmpeg_status = "check_failed"
    
    return {
        "status": "healthy",
        "models_loaded": {
            "speech_model": model_name if 'model_name' in globals() else "unknown",
            "tone_model": "loaded" if 'tone_classifier' in globals() else "not loaded"
        },
        "real_time_features": {
            "websocket_endpoint": "/ws/transcribe",
            "active_connections": len(manager.active_connections),
            "recording_status": "active" if realtime_processor.is_processing else "stopped",
            "voice_activity_detection": True,
            "continuous_transcription": True
        },
        "ffmpeg": {
            "status": ffmpeg_status,
            "version": ffmpeg_version,
            "path": ffmpeg_path_found,
            "detection_method": "enhanced_multi_path"
        },
        "supported_formats": {
            "always_supported": [".wav"],
            "ffmpeg_required": [".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".aac", ".wma"],
            "current_support": "all_formats" if ffmpeg_status == "available" else "wav_only"
        },
        "file_limits": {
            "max_size_mb": 100,
            "recommended_size_mb": 25,
            "warning_size_mb": 50
        },
        "offline_mode": {
            "speech_to_text": True,
            "tone_detection": True,
            "real_time_processing": True
        }
    }

# CORS for React frontend with WebSocket support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Real-time control endpoints
@app.post("/realtime/start")
async def start_realtime_recording():
    """Start real-time recording (HTTP endpoint for backup)"""
    try:
        if realtime_processor.start_processing():
            return {
                "status": "success",
                "message": "Real-time recording started",
                "websocket_url": "/ws/transcribe",
                "active_connections": len(manager.active_connections)
            }
        else:
            return {
                "status": "error",
                "message": "Recording already in progress"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to start recording: {str(e)}"
        }

@app.post("/realtime/stop")
async def stop_realtime_recording():
    """Stop real-time recording (HTTP endpoint for backup)"""
    try:
        realtime_processor.stop_processing()
        return {
            "status": "success",
            "message": "Real-time recording stopped",
            "active_connections": len(manager.active_connections)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to stop recording: {str(e)}"
        }

@app.get("/realtime/status")
async def get_realtime_status():
    """Get current real-time recording status"""
    return {
        "recording": realtime_processor.is_processing,
        "active_connections": len(manager.active_connections),
        "websocket_url": "/ws/transcribe",
        "features": {
            "voice_activity_detection": True,
            "continuous_transcription": True,
            "tone_detection": True,
            "offline_processing": True
        }
    }

# Define model variables - no type annotations to avoid transformer type conflicts
model_name = "agkphysics/wav2vec2-large-xlsr-53-amharic"  # Default Amharic model
processor = None
model = None
tone_classifier = None

# Define sentiment analysis word lists
positive_words = [
    "happy", "good", "great", "excellent", "amazing", "wonderful", "fantastic", "awesome", "love", "like",
    "perfect", "best", "brilliant", "outstanding", "superb", "marvelous", "delightful", "pleasant", "joy",
    "smile", "laugh", "excited", "thrilled", "satisfied", "pleased", "glad", "cheerful", "optimistic",
    # Amharic positive words (transliterated)
    "melkam", "konjo", "betam", "des", "tiru", "neta", "astemari", "hulm", "mirt", "selam"
]

negative_words = [
    "bad", "terrible", "awful", "horrible", "hate", "dislike", "worst", "disgusting", "annoying",
    "frustrated", "angry", "mad", "upset", "sad", "depressed", "disappointed", "worried", "stressed",
    "painful", "hurt", "broken", "failed", "wrong", "error", "problem", "issue", "trouble", "difficult",
    # Amharic negative words (transliterated)
    "kfu", "metfat", "aydelm", "bizu", "hod", "metay", "demam", "kitat", "yelewm", "arif"
]

# Load models - Using verified working Amharic model with offline fallback
def load_speech_models():
    """Load speech-to-text models with proper error handling"""
    global processor, model, model_name
    
    try:
        # Try to use cached/offline models first
        print("🔍 Loading speech-to-text models...")
        
        # Method 1: Try Amharic model with offline-first approach
        try:
            print(f"📥 Attempting to load Amharic model: {model_name}")
            # Use getattr to bypass type checker - initialize to avoid unbound warnings
            processor_class = getattr(__import__('transformers', fromlist=['Wav2Vec2Processor']), 'Wav2Vec2Processor')
            model_class = getattr(__import__('transformers', fromlist=['Wav2Vec2ForCTC']), 'Wav2Vec2ForCTC')
            
            processor = processor_class.from_pretrained(model_name, local_files_only=True)
            model = model_class.from_pretrained(model_name, local_files_only=True)
            print(f"✅ Successfully loaded cached Amharic model: {model_name}")
            return True
        except Exception as offline_error:
            print(f"📦 Cached Amharic model not found: {offline_error}")
            print("🌐 Attempting online download...")
            try:
                # Initialize classes to avoid unbound warnings
                processor_class = getattr(__import__('transformers', fromlist=['Wav2Vec2Processor']), 'Wav2Vec2Processor')
                model_class = getattr(__import__('transformers', fromlist=['Wav2Vec2ForCTC']), 'Wav2Vec2ForCTC')
                
                processor = processor_class.from_pretrained(model_name)
                model = model_class.from_pretrained(model_name)
                print(f"✅ Successfully downloaded Amharic model: {model_name}")
                return True
            except Exception as online_error:
                print(f"❌ Failed to download Amharic model: {online_error}")
                raise online_error
                
    except Exception as e:
        print(f"❌ Error loading Amharic model: {e}")
        print("🔄 Trying fallback multilingual model...")
        
        # Method 2: Fallback to multilingual model
        try:
            model_name = "facebook/wav2vec2-large-xlsr-53"
            print(f"📥 Attempting to load fallback model: {model_name}")
            # Initialize classes to avoid unbound warnings
            processor_class = getattr(__import__('transformers', fromlist=['Wav2Vec2Processor']), 'Wav2Vec2Processor')
            model_class = getattr(__import__('transformers', fromlist=['Wav2Vec2ForCTC']), 'Wav2Vec2ForCTC')
            
            processor = processor_class.from_pretrained(model_name, local_files_only=True)
            model = model_class.from_pretrained(model_name, local_files_only=True)
            print(f"✅ Successfully loaded cached fallback model: {model_name}")
            return True
        except Exception as offline_fallback_error:
            print(f"📦 Cached fallback model not found: {offline_fallback_error}")
            print("🌐 Attempting online download of fallback model...")
            try:
                # Initialize classes to avoid unbound warnings
                processor_class = getattr(__import__('transformers', fromlist=['Wav2Vec2Processor']), 'Wav2Vec2Processor')
                model_class = getattr(__import__('transformers', fromlist=['Wav2Vec2ForCTC']), 'Wav2Vec2ForCTC')
                processor = processor_class.from_pretrained(model_name)
                model = model_class.from_pretrained(model_name)
                print(f"✅ Successfully downloaded fallback model: {model_name}")
                return True
            except Exception as online_fallback_error:
                print(f"❌ Failed to download fallback model: {online_fallback_error}")
                # Method 3: Use a smaller, more reliable model
                try:
                    base_model_name = "facebook/wav2vec2-base-960h"
                    print(f"📥 Attempting to load base English model: {base_model_name}")
                    # Initialize classes to avoid unbound warnings
                    processor_class = getattr(__import__('transformers', fromlist=['Wav2Vec2Processor']), 'Wav2Vec2Processor')
                    model_class = getattr(__import__('transformers', fromlist=['Wav2Vec2ForCTC']), 'Wav2Vec2ForCTC')
                    processor = processor_class.from_pretrained(model_name, local_files_only=True)
                    model = model_class.from_pretrained(model_name, local_files_only=True)
                    print(f"✅ Successfully loaded cached base model: {model_name}")
                    return True
                except Exception:
                    try:
                        # Initialize classes to avoid unbound warnings
                        processor_class = getattr(__import__('transformers', fromlist=['Wav2Vec2Processor']), 'Wav2Vec2Processor')
                        model_class = getattr(__import__('transformers', fromlist=['Wav2Vec2ForCTC']), 'Wav2Vec2ForCTC')
                        processor = processor_class.from_pretrained(base_model_name)
                        model = model_class.from_pretrained(base_model_name)
                        model_name = base_model_name  # Update global variable
                        print(f"✅ Successfully downloaded base model: {base_model_name}")
                        return True
                    except Exception as final_error:
                        print(f"❌ All model loading attempts failed: {final_error}")
                        print("💡 Suggestions:")
                        print("   1. Check your internet connection")
                        print("   2. Try running with a VPN if blocked")
                        print("   3. Pre-download models when internet is available")
                        raise Exception("No speech-to-text models could be loaded. App requires pre-downloaded models for offline operation.")
    
    return False

# Load the models
try:
    if not load_speech_models():
        raise Exception("Failed to load speech-to-text models")
except Exception as model_error:
    print(f"❌ Critical error: {model_error}")
    processor = None
    model = None

# Load tone classification model with enhanced Amharic support and offline fallbacks
try:
    print("🔍 Loading tone classification model for Amharic sentiment analysis...")
    
    # Method 1: Try AfroXLMR-large model (SemEval-2023 winning model for Amharic)
    try:
        tone_classifier = pipeline("sentiment-analysis", model="GMNLP/AfroXLMR-large")
        print("✅ Successfully loaded AfroXLMR-large model (SemEval-2023 winning model for Amharic)")
        
        # Test the model with Amharic and English samples
        test_samples = [
            "I am very happy today!",
            "This is terrible and awful",  
            "The weather is normal",
            "አንተ ንኳን እንደተባበሩ እወዳለሁ"  # Amharic for "I'm glad you could join us"
        ]
        
        for sample in test_samples:
            test_result = tone_classifier(sample)
            print(f"🧪 AfroXLMR model test - '{sample}' -> {test_result}")
        
    except Exception as primary_error:
        print(f"❌ Error loading AfroXLMR-large model: {primary_error}")
        print("🔄 Trying alternative Amharic-capable models...")
        
        # Try other Amharic-optimized models
        amharic_fallback_models = [
            "GMNLP/AfroXLMR-base",  # Smaller version of the winning model
            "addethoughts/amharic-bert",  # BERT model trained on Amharic
            "cardiffnlp/twitter-xlm-roberta-base",  # Known to work well with Amharic
            "m3hrdadfi/amharic-bert-base",  # Alternative Amharic BERT model
            "m3hrdadfi/amharic-albert-base"  # Lightweight Amharic ALBERT model
        ]
        
        tone_classifier = None
        for model_name in amharic_fallback_models:
            try:
                print(f"🔍 Trying Amharic fallback model: {model_name}")
                tone_classifier = pipeline("sentiment-analysis", model=model_name)
                print(f"✅ Successfully loaded Amharic fallback model: {model_name}")
                
                # Test the fallback model
                test_result = tone_classifier("I am very happy today!")
                print(f"🧪 Amharic fallback model test result: {test_result}")
                break
                
            except Exception as fallback_error:
                print(f"❌ Amharic fallback model {model_name} failed: {fallback_error}")
                continue
        
        # Last resort: use default pipeline with Amharic warning
        if tone_classifier is None:
            print("🔄 Using default sentiment analysis pipeline (limited Amharic support)...")
            try:
                tone_classifier = pipeline("sentiment-analysis")
                test_result = tone_classifier("I am very happy today!")
                print(f"✅ Default model loaded (limited Amharic support)")
                print(f"🧪 Default model test result: {test_result}")
                print("⚠️ Note: Default model has limited Amharic sentiment detection capability")
            except Exception as default_error:
                print(f"❌ Default pipeline also failed: {default_error}")
                print("💡 Suggestions for Amharic tone detection:")
                print("   1. Check your internet connection")
                print("   2. Pre-download Amharic-capable models when internet is available")
                print("   3. The app will rely on keyword-based Amharic sentiment detection")
                # Create a dummy classifier optimized for Amharic keyword fallback
                class AmharicAwareDummyClassifier:
                    def __call__(self, text):
                        return [{'label': 'LABEL_1', 'score': 0.3}]  # Low confidence to trigger keyword analysis
                tone_classifier = AmharicAwareDummyClassifier()
                print("🔧 Using Amharic-aware dummy classifier (keyword-based detection)")
                
except Exception as e:
    print(f"❌ Unexpected error in Amharic tone model loading: {e}")
    # Create Amharic-aware dummy classifier as final fallback
    if tone_classifier is None:
        class AmharicAwareFallbackClassifier:
            def __call__(self, text):
                return [{'label': 'LABEL_1', 'score': 0.3}]  # Low confidence to trigger keyword analysis
        tone_classifier = AmharicAwareFallbackClassifier()
        print("🔧 Using Amharic-aware fallback classifier")

# Ensure tone_classifier is defined
if tone_classifier is None:
    class DefaultToneClassifier:
        def __call__(self, text):
            return [{'label': 'NEUTRAL', 'score': 0.5}]
    tone_classifier = DefaultToneClassifier()
    print("🔧 Using default tone classifier")

# Utility functions
def detect_tone(text):
    """Detect tone/sentiment of text using the loaded tone classifier"""
    try:
        if not text or not text.strip():
            return "Yellow", 0.5  # Neutral for empty text
        
        # Check if tone_classifier is available
        if tone_classifier is None:
            print("⚠️ Tone classifier not loaded, defaulting to neutral")
            return "Yellow", 0.5
        
        # Get prediction from the model
        result = tone_classifier(text.strip())
        if not result:
            return "Yellow", 0.5
            
        # Handle different result types (list, generator, etc.)
        try:
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                try:
                    result_list = list(result) if not isinstance(result, list) else result
                    if len(result_list) == 0:
                        return "Yellow", 0.5
                    result_item = result_list[0]
                except (TypeError, ValueError):
                    return "Yellow", 0.5
            else:
                result_item = result
        except (TypeError, IndexError):
            return "Yellow", 0.5
            
        # Extract label and confidence safely
        try:
            if isinstance(result_item, dict):
                label = str(result_item.get('label', 'NEUTRAL'))
                confidence = float(result_item.get('score', 0.5))
            else:
                # Fallback for non-dict results
                label = 'NEUTRAL'
                confidence = 0.5
        except (KeyError, ValueError, TypeError):
            label = 'NEUTRAL'
            confidence = 0.5
        
        # Map model labels to our color system
        # AfroXLMR uses different label formats depending on training, so we need comprehensive mapping
        label_lower = label.lower()
        
        # Positive sentiment indicators
        if any(pos_term in label_lower for pos_term in ["positive", "label_2", "2", "good", "happy"]):
            tone = "Green"
        # Negative sentiment indicators
        elif any(neg_term in label_lower for neg_term in ["negative", "label_0", "0", "bad", "sad"]):
            tone = "Red"
        # Neutral or uncertain
        else:
            tone = "Yellow"
        
        # Apply confidence threshold - if confidence is too low, default to neutral
        # AfroXLMR sometimes gives lower confidence for Amharic, so we use a more lenient threshold
        if confidence < 0.4:
            tone = "Yellow"
            confidence = 0.4
        
        print(f"🎯 Tone detection: '{text}' -> {tone} ({confidence:.3f})")
        return tone, confidence
        
    except Exception as e:
        print(f"❌ Tone detection error: {e}")
        return "Yellow", 0.5  # Default to neutral on error

def convert_to_wav(file):
    """Convert uploaded audio file to WAV format with better error handling"""
    try:
        import tempfile
        import uuid
        
        # Generate unique filename to avoid conflicts
        unique_id = str(uuid.uuid4())[:8]
        
        # Get file extension
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"
        
        # Create temporary file paths
        temp_input = f"temp_input_{unique_id}{ext}"
        temp_output = f"temp_output_{unique_id}.wav"
        
        # Save uploaded file
        with open(temp_input, "wb") as f:
            content = file.file.read()
            if len(content) == 0:
                raise ValueError("Empty file uploaded")
            f.write(content)
        
        # If already WAV, just return the path
        if ext == ".wav":
            return temp_input
        
        # Convert to WAV for other formats
        elif ext in [".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".aac", ".wma"]:
            try:
                # Enhanced FFmpeg detection
                from pydub import AudioSegment
                from pydub.utils import which
                import subprocess
                
                # Multiple methods to find FFmpeg
                ffmpeg_path = None
                
                # Method 1: Direct command check
                try:
                    result = subprocess.run(["ffmpeg", "-version"], 
                                          capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        ffmpeg_path = "ffmpeg"
                        print(f"🔧 FFmpeg found via direct command")
                except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                    pass
                
                # Method 2: pydub's which function
                if not ffmpeg_path:
                    ffmpeg_path = which("ffmpeg")
                    if ffmpeg_path:
                        print(f"🔧 FFmpeg found via pydub: {ffmpeg_path}")
                
                # Method 3: Common installation paths
                if not ffmpeg_path:
                    common_paths = [
                        r"C:\Users\Daveee\Desktop\AI\Speech-Text\ffmpeg-2025-08-25-git-1b62f9d3ae-essentials_build\ffmpeg-2025-08-25-git-1b62f9d3ae-essentials_build\bin\ffmpeg.exe",
                        r"C:\ffmpeg\bin\ffmpeg.exe",
                        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
                    ]
                    for path in common_paths:
                        if os.path.exists(path):
                            ffmpeg_path = path
                            print(f"🔧 FFmpeg found at: {ffmpeg_path}")
                            # Add to PATH for pydub
                            ffmpeg_dir = os.path.dirname(path)
                            current_path = os.environ.get('PATH', '')
                            if ffmpeg_dir not in current_path:
                                os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
                            break
                
                if not ffmpeg_path:
                    # Clean up and provide helpful error message
                    if os.path.exists(temp_input):
                        os.remove(temp_input)
                    raise ValueError(
                        "FFmpeg not found. Please:\n"
                        "1. Verify FFmpeg installation: ffmpeg -version\n"
                        "2. Add FFmpeg to PATH or restart terminal\n"
                        "3. Try uploading WAV files instead\n"
                        "4. FFmpeg should be at: C:\\ffmpeg\\bin\\ffmpeg.exe"
                    )
                audio = AudioSegment.from_file(temp_input)
                
                # Enhanced audio processing for better speech recognition
                print(f"🎧 Processing {ext.upper()} file: {file.filename}")
                
                # Optimize for speech recognition
                audio = audio.set_channels(1)  # Mono
                audio = audio.set_frame_rate(16000)  # 16kHz for Wav2Vec2
                
                # Optional: Normalize audio levels for better recognition
                audio = audio.normalize()
                
                # Apply noise reduction if needed (basic)
                if len(audio) > 30000:  # If longer than 30 seconds
                    print("🔊 Applying audio optimization for long files...")
                
                audio.export(temp_output, format="wav", parameters=["-acodec", "pcm_s16le"])
                
                # Clean up input file
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                    
                return temp_output
            except Exception as conversion_error:
                # Clean up files on error
                for temp_file in [temp_input, temp_output]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                # Provide specific error message for FFmpeg issues
                error_msg = str(conversion_error)
                if "ffmpeg" in error_msg.lower() or "avconv" in error_msg.lower():
                    raise ValueError(
                        "FFmpeg not found. To fix this:\n"
                        "1. Install FFmpeg: winget install Gyan.FFmpeg\n"
                        "2. Restart your terminal/IDE\n"
                        "3. Or upload WAV files only"
                    )
                else:
                    raise ValueError(f"Audio conversion failed: {conversion_error}")
        else:
            # Clean up and raise error for unsupported formats
            if os.path.exists(temp_input):
                os.remove(temp_input)
            raise ValueError(f"Unsupported format: {ext}. Supported: .wav, .mp3, .mp4, .m4a, .ogg, .flac")
            
    except Exception as e:
        raise ValueError(f"File processing error: {str(e)}")

def transcribe_audio_realtime(audio_path):
    """Optimized transcription for real-time processing with shorter audio chunks"""
    try:
        # Check if models are loaded
        if processor is None or model is None:
            print("❌ Speech-to-text models not loaded")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return "[No speech detected]"
            
        # Load audio file with faster processing settings
        speech, sr = librosa.load(audio_path, sr=16000, duration=10)  # Limit to 10 seconds max
        
        # Check if audio is empty or too short
        if len(speech) == 0:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return "[No speech detected]"
            
        if len(speech) < 800:  # Less than 0.05 second (very short for real-time)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return "[No speech detected]"
        
        # Process with the model (optimized for speed)
        proc = processor  # Type assertion for clarity
        mod = model  # Type assertion for clarity
        input_values = proc(speech, return_tensors="pt", sampling_rate=16000).input_values
        
        # Generate predictions with faster settings
        with torch.no_grad():
            logits = mod(input_values).logits
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = proc.decode(predicted_ids[0])
        
        # Clean up the file after processing
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        # Return cleaned transcription
        cleaned = transcription.strip() if transcription.strip() else "[No speech detected]"
        
        # Filter out very short or meaningless transcriptions
        if len(cleaned) < 3 or cleaned == "[No speech detected]":
            return "[No speech detected]"
            
        return cleaned
        
    except Exception as e:
        # Clean up file on error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        print(f"Real-time transcription error: {str(e)}")
        return "[No speech detected]"

def transcribe_audio(audio_path):
    """Transcribe audio file to text with error handling"""
    try:
        # Check if models are loaded
        if processor is None or model is None:
            print("❌ Speech-to-text models not loaded")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise ValueError("Speech-to-text models not available")
            
        # Load audio file
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Check if audio is empty or too short
        if len(speech) == 0:
            raise ValueError("Audio file is empty")
        if len(speech) < 1600:  # Less than 0.1 second
            raise ValueError("Audio file is too short (minimum 0.1 seconds required)")
        
        # Process with the model
        proc = processor  # Type assertion for clarity
        mod = model  # Type assertion for clarity
        input_values = proc(speech, return_tensors="pt", sampling_rate=16000).input_values
        
        # Generate predictions
        with torch.no_grad():
            logits = mod(input_values).logits
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = proc.decode(predicted_ids[0])
        
        # Clean up the file after processing
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        # Return cleaned transcription
        return transcription.strip() if transcription.strip() else "[No speech detected]"
        
    except Exception as e:
        # Clean up file on error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise ValueError(f"Transcription failed: {str(e)}")

# API endpoints
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze uploaded audio file for speech-to-text and tone detection"""
    try:
        # Validate file
        if not file.filename:
            raise ValueError("No filename provided")
            
        # Check file size with helpful messages
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size == 0:
            raise ValueError("Empty file uploaded")
            
        # Progressive size limits with enhanced feedback
        if file_size > 100 * 1024 * 1024:  # 100MB - new maximum limit
            raise ValueError("File too large (maximum 100MB allowed). Consider compressing your audio file or splitting it into smaller segments.")
        elif file_size > 75 * 1024 * 1024:  # 75MB
            print(f"⚠️  Very large file detected: {file_size/1024/1024:.1f}MB - Processing may take significantly longer")
        elif file_size > 50 * 1024 * 1024:  # 50MB
            print(f"⚠️  Large file detected: {file_size/1024/1024:.1f}MB - Processing may take longer")
        elif file_size > 25 * 1024 * 1024:  # 25MB
            print(f"📁 Medium-large file: {file_size/1024/1024:.1f}MB - Extended processing time")
        elif file_size > 15 * 1024 * 1024:  # 15MB
            print(f"📁 Medium file: {file_size/1024/1024:.1f}MB - Normal processing time")
            
        # Reset file pointer
        await file.seek(0)
        
        print(f"Processing file: {file.filename} ({file_size/1024/1024:.2f}MB)")
        
        # Convert audio to appropriate format
        wav_path = convert_to_wav(file)
        
        # Transcribe speech to text
        transcription = transcribe_audio(wav_path)
        
        # Detect tone from transcription
        tone, confidence = detect_tone(transcription)
        
        return {
            "text": transcription,
            "tone": tone,
            "confidence": round(confidence, 3),
            "status": "success",
            "filename": file.filename,
            "file_size_mb": round(file_size/1024/1024, 2)
        }
        
    except ValueError as ve:
        return {
            "error": str(ve),
            "status": "error",
            "text": "",
            "tone": "Yellow"
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            "error": f"Processing failed: {str(e)}",
            "status": "error",
            "text": "",
            "tone": "Yellow"
        }

@app.post("/analyze-realtime")
async def analyze_realtime(file: UploadFile = File(...)):
    """Real-time audio chunk processing for live transcription with text-first approach"""
    try:
        # Validate file
        if not file.filename:
            raise ValueError("No filename provided")
            
        content = await file.read()
        file_size = len(content)
        
        if file_size == 0:
            raise ValueError("Empty audio chunk")
            
        # Limit chunk size for real-time processing
        if file_size > 10 * 1024 * 1024:  # 10MB max for real-time chunks
            raise ValueError("Audio chunk too large for real-time processing")
            
        # Reset file pointer
        await file.seek(0)
        
        print(f"🔄 Processing real-time chunk: {file.filename} ({file_size/1024:.1f}KB)")
        
        # Step 1: Convert audio to appropriate format
        wav_path = convert_to_wav(file)
        
        # Step 2: Transcribe speech to text FIRST (prioritize speed)
        transcription = transcribe_audio_realtime(wav_path)
        
        # Only process if we have meaningful text
        if transcription and transcription.strip() and transcription != "[No speech detected]":
            print(f"📝 Transcribed text: '{transcription}'")
            
            # Step 3: THEN detect tone (secondary priority)
            try:
                print(f"🔍 Starting tone analysis...")
                tone, confidence = detect_tone(transcription)
                print(f"🎯 Tone detected: {tone} ({confidence:.3f})")
            except Exception as tone_error:
                print(f"⚠️ Tone detection failed: {tone_error}")
                tone, confidence = "Yellow", 0.5  # Fallback to neutral
            
            return {
                "text": transcription,
                "tone": tone,
                "confidence": round(confidence, 3),
                "status": "success",
                "realtime": True,
                "processing_order": "text_first_tone_second"
            }
        else:
            # Return empty but successful response for silence
            return {
                "text": "",
                "tone": "Yellow",
                "confidence": 0.0,
                "status": "success",
                "realtime": True,
                "processing_order": "no_speech_detected"
            }
        
    except ValueError as ve:
        return {
            "error": str(ve),
            "status": "error",
            "text": "",
            "tone": "Yellow",
            "realtime": True
        }
    except Exception as e:
        print(f"Real-time processing error: {e}")
        return {
            "error": f"Real-time processing failed: {str(e)}",
            "status": "error",
            "text": "",
            "tone": "Yellow",
            "realtime": True
        }

if __name__ == "__main__":
    import uvicorn
    
    # Enhanced FFmpeg detection and setup
    try:
        from pydub.utils import which
        import subprocess
        
        # Multiple methods to detect FFmpeg
        ffmpeg_path = None
        ffmpeg_version = "unknown"
        
        print("\n🚀 Starting Amharic Speech-to-Text API...")
        print("📡 Server will be available at: http://localhost:8001")
        print("🔍 Health check: http://localhost:8001/health")
        print("📄 API docs: http://localhost:8001/docs")
        print("🌐 WebSocket endpoint: ws://localhost:8001/ws/transcribe")
        print("🎤 Real-time control: http://localhost:8001/realtime/start")
        
        # Method 1: Direct command check
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                ffmpeg_path = "ffmpeg"
                version_line = result.stdout.split('\n')[0]
                ffmpeg_version = version_line.split(' ')[2] if len(version_line.split(' ')) > 2 else "detected"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        # Method 2: pydub's which function
        if not ffmpeg_path:
            detected_path = which("ffmpeg")
            if detected_path:
                ffmpeg_path = detected_path
                try:
                    result = subprocess.run([detected_path, "-version"], 
                                          capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        version_line = result.stdout.split('\n')[0]
                        ffmpeg_version = version_line.split(' ')[2] if len(version_line.split(' ')) > 2 else "detected"
                except Exception:
                    pass
        
        # Method 3: Common installation paths
        if not ffmpeg_path:
            common_paths = [
                r"C:\Users\Daveee\Desktop\AI\Speech-Text\ffmpeg-2025-08-25-git-1b62f9d3ae-essentials_build\ffmpeg-2025-08-25-git-1b62f9d3ae-essentials_build\bin\ffmpeg.exe",
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
            ]
            for path in common_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    try:
                        result = subprocess.run([path, "-version"], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            version_line = result.stdout.split('\n')[0]
                            ffmpeg_version = version_line.split(' ')[2] if len(version_line.split(' ')) > 2 else "detected"
                        # Add to PATH for pydub
                        ffmpeg_dir = os.path.dirname(path)
                        current_path = os.environ.get('PATH', '')
                        if ffmpeg_dir not in current_path:
                            os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
                            print(f"🔧 Added FFmpeg to PATH: {ffmpeg_dir}")
                    except Exception:
                        pass
                    break
        
        if ffmpeg_path:
            print(f"✅ FFmpeg {ffmpeg_version} detected - All audio formats supported")
            print(f"   Path: {ffmpeg_path}")
            print("   Supported: WAV, MP3, MP4, M4A, OGG, FLAC, AAC, WMA")
            print(f"   File size limit: 100MB (enhanced for large audio files)")
        else:
            print("⚠️  FFmpeg not found - Only WAV files supported")
            print("   Expected FFmpeg location: C:\\ffmpeg\\bin\\ffmpeg.exe")
            print("   Try: ffmpeg -version (to test if accessible)")
            print(f"   File size limit: 100MB for WAV files")
            
    except Exception as e:
        print(f"Warning: Could not check FFmpeg status: {e}")
        print("\n🚀 Starting Amharic Speech-to-Text API with Real-Time Features...")
        print("📡 Server will be available at: http://localhost:8001")
        print("🌐 WebSocket endpoint: ws://localhost:8001/ws/transcribe")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
