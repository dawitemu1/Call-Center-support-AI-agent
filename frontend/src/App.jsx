import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [microphoneStatus, setMicrophoneStatus] = useState('unknown');
  const [audioUrl, setAudioUrl] = useState(null);
  
  // Real-time transcription states
  const [wsConnected, setWsConnected] = useState(false);
  const [liveTranscription, setLiveTranscription] = useState('');
  const [transcriptionHistory, setTranscriptionHistory] = useState([]);
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const audioContextRef = useRef(null);
  const recordingChunksRef = useRef([]);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  // WebSocket connection for real-time transcription
  const connectWebSocket = () => {
    if (wsRef.current || connectionAttempts >= 3) return;
    
    try {
      console.log('🔗 Connecting to WebSocket...');
      setConnectionAttempts(prev => prev + 1);
      
      const ws = new WebSocket('ws://localhost:8001/ws/transcribe');
      wsRef.current = ws;
      
      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setWsConnected(true);
        setConnectionAttempts(0);
        
        // Send ping to test connection
        ws.send(JSON.stringify({ action: 'ping' }));
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'transcription') {
            setLiveTranscription(data.text);
            
            // Add to history with timestamp and tone
            const timestamp = new Date().toLocaleTimeString();
            setTranscriptionHistory(prev => [...prev, {
              id: Date.now(),
              text: data.text,
              tone: data.tone,
              confidence: data.confidence,
              timestamp: timestamp
            }]);
          } else if (data.type === 'status') {
            console.log('📊 Status:', data.message);
          } else if (data.type === 'pong') {
            console.log('🏓 Pong received');
          }
        } catch (e) {
          console.error('❌ Error parsing WebSocket message:', e);
        }
      };
      
      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        setWsConnected(false);
        wsRef.current = null;
        
        // Auto-reconnect if we're recording
        if (isRecording && connectionAttempts < 3) {
          console.log(`🔄 Reconnecting... (attempt ${connectionAttempts + 1}/3)`);
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, 2000);
        }
      };
      
      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setWsConnected(false);
      };
      
    } catch (error) {
      console.error('❌ Failed to create WebSocket connection:', error);
      setWsConnected(false);
    }
  };
  
  const disconnectWebSocket = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setWsConnected(false);
    setConnectionAttempts(0);
  };
  

  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnectWebSocket();
    };
  }, []);
  
  // Toggle fullscreen mode for live transcription
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  // Auto-fullscreen when recording starts
  const enterFullscreenRecording = () => {
    setIsFullscreen(true);
  };

  // Get tone color
  const getToneColor = (tone) => {
    if (tone === "Green") return "#4CAF50";
    if (tone === "Yellow") return "#FFA726";
    if (tone === "Red") return "#F44336";
    return "#666666";
  };

  // Test microphone access
  const testMicrophone = async () => {
    try {
      setMicrophoneStatus('unknown');
      console.log('🎤 Testing microphone access...');
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('✅ Microphone test successful');
      setMicrophoneStatus('available');
      
      // Stop the test stream
      stream.getTracks().forEach(track => track.stop());
      
      alert('✅ Microphone access successful! You can now start recording.');
    } catch (error) {
      console.error('❌ Microphone test failed:', error);
      setMicrophoneStatus('denied');
      
      let message = '❌ Microphone test failed\n\n';
      if (error.name === 'NotAllowedError') {
        message += '🔒 Permission denied. Please allow microphone access and try again.';
      } else if (error.name === 'NotFoundError') {
        message += '🎤 No microphone found. Please connect a microphone.';
      } else {
        message += `⚠️ Error: ${error.message}`;
      }
      
      alert(message);
    }
  };

  // Convert audio to WAV format
  const convertToWav = (audioBuffer) => {
    const numberOfChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numberOfChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = audioBuffer.length * blockAlign;
    const bufferSize = 44 + dataSize;
    
    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, bufferSize - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);
    
    // Convert audio data
    const channelData = audioBuffer.getChannelData(0);
    let offset = 44;
    for (let i = 0; i < channelData.length; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
    
    return arrayBuffer;
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    // Clear recording if file is selected
    setRecordedAudio(null);
    setAudioUrl(null);
  };

  const startRecording = async () => {
    try {
      // Check if we're running on HTTPS or localhost
      const isSecureContext = window.isSecureContext || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
      
      if (!isSecureContext) {
        alert('Microphone access requires HTTPS or localhost. Please use https:// or run on localhost.');
        return;
      }
      
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support microphone access. Please use Chrome, Firefox, or Edge.');
        return;
      }
      
      console.log('🎤 Requesting microphone access...');
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000 // Optimized for Wav2Vec2
        } 
      });
      
      console.log('✅ Microphone access granted');
      setMicrophoneStatus('available');
      streamRef.current = stream;
      
      // Start WebSocket connection for live transcription (Zoom-like)
      if (!wsRef.current) {
        connectWebSocket();
      }
      
      // Start real-time transcription
      setTimeout(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ action: 'start_recording' }));
          setLiveTranscription('🎤 Listening...');
        }
      }, 1000); // Wait for WebSocket to connect
      
      // Create audio context for WAV recording
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      
      recordingChunksRef.current = [];
      
      processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        recordingChunksRef.current.push(new Float32Array(inputData));
      };
      
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);
      
      setIsRecording(true);
      setRecordingTime(0);
      
      // Removed auto-fullscreen as requested
      
      // Start timer
      intervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
      console.log('✅ Recording started successfully with live transcription');
      
    } catch (error) {
      console.error('❌ Microphone access error:', error);
      setMicrophoneStatus('denied');
      
      // Provide specific error messages based on error type
      let errorMessage = 'Could not access microphone. ';
      
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        errorMessage += `

🔒 Permission Denied:
• Click the camera/microphone icon in your browser's address bar
• Select "Allow" for microphone access
• Refresh the page and try again

🌐 Browser Steps:
• Chrome: Click 🔒 or 🎤 icon next to the URL
• Firefox: Click 🔒 icon, then "Permissions"
• Edge: Click 🔒 icon, select "Allow" for microphone`;
      } else if (error.name === 'NotFoundError' || error.name === 'DeviceNotFoundError') {
        errorMessage += `

🎤 No Microphone Found:
• Check if a microphone is connected
• Try plugging in headphones with a microphone
• Check Windows sound settings (Windows key + R, type "mmsys.cpl")`;
      } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
        errorMessage += `

⚠️ Microphone In Use:
• Close other applications using the microphone (Zoom, Teams, Skype)
• Try refreshing the page
• Restart your browser`;
      } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
        errorMessage += `

🔧 Audio Settings Issue:
• Your microphone may not support the required settings
• Try using a different microphone
• Update your audio drivers`;
      } else {
        errorMessage += `

🔍 Troubleshooting Steps:
• Make sure you're using Chrome, Firefox, or Edge
• Try refreshing the page
• Check if microphone works in other apps
• Restart your browser

🛠️ Technical Details: ${error.message}`;
      }
      
      alert(errorMessage);
    }
  };
  
  const stopRecording = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    // Stop real-time transcription
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop_recording' }));
      setLiveTranscription('');
    }
    
    if (audioContextRef.current && recordingChunksRef.current.length > 0) {
      // Create audio buffer from recorded chunks
      const totalLength = recordingChunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
      const audioBuffer = audioContextRef.current.createBuffer(1, totalLength, 16000);
      const channelData = audioBuffer.getChannelData(0);
      
      let offset = 0;
      recordingChunksRef.current.forEach(chunk => {
        channelData.set(chunk, offset);
        offset += chunk.length;
      });
      
      // Convert to WAV
      const wavArrayBuffer = convertToWav(audioBuffer);
      const blob = new Blob([wavArrayBuffer], { type: 'audio/wav' });
      
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const audioFile = new File([blob], `recording_${timestamp}.wav`, { 
        type: 'audio/wav' 
      });
      
      setRecordedAudio(audioFile);
      setFile(audioFile); // Set as current file for analysis
      
      // Create audio URL for playback
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      
      // Clear file input since we have a recording
      const fileInput = document.querySelector('input[type="file"]');
      if (fileInput) fileInput.value = '';
      
      // Cleanup
      audioContextRef.current.close();
    }
    
    setIsRecording(false);
    
    // Removed auto-exit fullscreen since auto-fullscreen is disabled
  };
  
  const clearRecording = () => {
    setRecordedAudio(null);
    setFile(null);
    setAudioUrl(null);
    setRecordingTime(0);
    
    // Clear live transcription data
    setLiveTranscription('');
    setTranscriptionHistory([]);
    
    // Disconnect WebSocket
    disconnectWebSocket();
    
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
  };
  
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSubmit = async () => {
    if (!file) return;
    
    setResult(null); // Clear previous results
    setIsProcessing(true); // Show processing state
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:8001/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (error) {
      // Handle API errors gracefully
      if (error.response && error.response.data) {
        setResult({
          error: error.response.data.error || "Analysis failed",
          status: "error",
          text: "",
          tone: "Yellow"
        });
      } else {
        setResult({
          error: "Network error. Please check if the backend server is running.",
          status: "error", 
          text: "",
          tone: "Yellow"
        });
      }
    } finally {
      setIsProcessing(false); // Hide processing state
    }
  };

  const getToneShadow = (tone) => {
    if (tone === "Green") return "0 0 10px rgba(76, 175, 80, 0.5)";
    if (tone === "Yellow") return "0 0 10px rgba(255, 167, 38, 0.5)";
    if (tone === "Red") return "0 0 10px rgba(244, 67, 54, 0.5)";
    return "0 0 5px rgba(102, 102, 102, 0.3)";
  };

  return (
    <>
      {/* Fullscreen transcription view */}
      {isFullscreen && isRecording && (
        <div className="fullscreen-transcription">
          <button
            className="fullscreen-close"
            onClick={toggleFullscreen}
            title="Exit fullscreen"
          >
            ×
          </button>
          
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <h1 style={{ fontSize: '2.5rem', marginBottom: '40px', textAlign: 'center' }}>
              🎤 Live Transcription
            </h1>
            
            <div style={{
              color: '#f44336', 
              fontWeight: 'bold',
              fontSize: '1.5rem',
              textAlign: 'center',
              marginBottom: '30px'
            }}>
              🔴 Recording: {formatTime(recordingTime)} {wsConnected ? '🟢 Connected' : '🔴 Offline'}
            </div>
            
            {/* Live transcription display - enhanced for fullscreen */}
            {liveTranscription && (
              <div className="live-transcription" style={{
                fontSize: '2rem',
                minHeight: '120px',
                color: '#333',
                background: 'rgba(255, 255, 255, 0.95)',
              }}>
                🗨️ {liveTranscription}
              </div>
            )}
            
            {/* Transcription history - enhanced for fullscreen */}
            {transcriptionHistory.length > 0 && (
              <div className="transcription-history" style={{
                maxHeight: '40vh',
                fontSize: '1.1rem'
              }}>
                <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>Recent Transcriptions:</h3>
                {transcriptionHistory.slice(-5).map((item) => (
                  <div key={item.id} style={{
                    marginBottom: '10px',
                    padding: '12px',
                    backgroundColor: '#f9f9f9',
                    borderRadius: '8px',
                    borderLeft: `4px solid ${getToneColor(item.tone)}`
                  }}>
                    <div style={{ 
                      fontSize: '0.9rem', 
                      color: '#666', 
                      marginBottom: '5px',
                      display: 'flex',
                      justifyContent: 'space-between'
                    }}>
                      <span>{item.timestamp}</span>
                      <span style={{ 
                        color: getToneColor(item.tone),
                        fontWeight: 'bold'
                      }}>
                        {item.tone === 'Green' ? '😊 Positive' : 
                         item.tone === 'Red' ? '😞 Negative' : '😐 Neutral'}
                      </span>
                    </div>
                    <div style={{ 
                      color: getToneColor(item.tone),
                      fontWeight: '500'
                    }}>
                      {item.text}
                    </div>
                  </div>
                ))}
              </div>
            )}
            
            <div style={{
              marginTop: '30px',
              textAlign: 'center'
            }}>
              <button 
                onClick={stopRecording}
                style={{
                  backgroundColor: '#f44336',
                  color: 'white',
                  border: 'none',
                  padding: '15px 30px',
                  borderRadius: '25px',
                  cursor: 'pointer',
                  fontSize: '1.2rem',
                  fontWeight: 'bold',
                  boxShadow: '0 4px 15px rgba(244, 67, 54, 0.3)'
                }}
              >
                ⏹️ Stop Recording
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Regular app view */}
      <div className="App" style={{ display: isFullscreen ? 'none' : 'flex' }}>
      <h1>Amharic Speech-to-Text & Tone Detection</h1>
      

      
      {/* File Upload Section */}
      <div className="audio-section">
        <input 
          type="file" 
          accept=".wav,.mp3,.mp4,.m4a,.ogg,.flac,.aac,.wma" 
          onChange={handleFileChange}
          disabled={isRecording}
          style={{ width: '100%', maxWidth: '400px' }}
        />
        <div style={{ fontSize: '12px', color: '#666', marginTop: '5px', textAlign: 'center' }}>
          Supported formats: WAV, MP3, MP4, M4A, OGG, FLAC, AAC, WMA<br/>
          Maximum file size: 100MB | Optimized for Amharic speech
        </div>
      </div>
      
      {/* Recording Section */}
      <div className="audio-section">
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
          {!isRecording ? (
            <>
              <button 
                onClick={startRecording}
                disabled={isProcessing}
                style={{
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  padding: '15px 25px',
                  borderRadius: '10px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  fontWeight: 'bold',
                  boxShadow: '0 4px 15px rgba(76, 175, 80, 0.3)',
                  transition: 'all 0.3s ease'
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = 'scale(1.05)';
                  e.target.style.backgroundColor = '#45a049';
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'scale(1)';
                  e.target.style.backgroundColor = '#4CAF50';
                }}
              >
                🎤 Start Recording (with Live Transcription)
              </button>
              
              {/* Manual fullscreen toggle */}
              {/* Removed fullscreen button as requested */}
            </>
          ) : (
            <>
              <button 
                onClick={stopRecording}
                style={{
                  backgroundColor: '#f44336',
                  color: 'white',
                  border: 'none',
                  padding: '15px 25px',
                  borderRadius: '10px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  fontWeight: 'bold',
                  boxShadow: '0 4px 15px rgba(244, 67, 54, 0.3)'
                }}
              >
                ⏹️ Stop Recording
              </button>
              
              {/* Fullscreen toggle during recording - Removed as requested */}
            </>
          )}
          
          {recordedAudio && (
            <button 
              onClick={clearRecording}
              disabled={isProcessing}
              style={{
                backgroundColor: '#ff9800',
                color: 'white',
                border: 'none',
                padding: '8px 16px',
                borderRadius: '5px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              🗑️ Clear
            </button>
          )}
          
          {isRecording && (
            <div style={{ color: '#f44336', fontWeight: 'bold' }}>
              🔴 Recording: {formatTime(recordingTime)} {wsConnected ? '🟢 Live' : '🔴 Offline'}
            </div>
          )}
          
          {/* Live Transcription Display - Enhanced Size and Sentence-based */}
          {isRecording && liveTranscription && (
            <div className="live-transcription" style={{
              marginTop: '20px',
              padding: '30px',
              backgroundColor: '#fff',
              border: '2px solid #2196F3',
              borderRadius: '15px',
              minHeight: '120px',
              fontSize: '22px',
              fontWeight: '500',
              color: '#333',
              textAlign: 'left',
              display: 'flex',
              alignItems: 'flex-start',
              justifyContent: 'flex-start',
              boxShadow: '0 6px 25px rgba(33, 150, 243, 0.2)',
              maxWidth: '95%',
              width: '100%',
              lineHeight: '1.6',
              wordWrap: 'break-word',
              whiteSpace: 'pre-wrap'
            }}>
              <div style={{ width: '100%' }}>
                🗨️ {liveTranscription}
              </div>
            </div>
          )}
          
          {/* Transcription History - Enhanced and No Color Names */}
          {isRecording && transcriptionHistory.length > 0 && (
            <div className="transcription-history" style={{
              marginTop: '20px',
              maxHeight: '300px',
              overflowY: 'auto',
              backgroundColor: '#fff',
              border: '1px solid #ddd',
              borderRadius: '10px',
              padding: '20px',
              maxWidth: '95%',
              width: '100%'
            }}>
              <h4 style={{ 
                margin: '0 0 15px 0', 
                fontSize: '16px', 
                color: '#666',
                textAlign: 'center'
              }}>Live Transcription History:</h4>
              {transcriptionHistory.slice(-5).map((item) => (
                <div key={item.id} style={{
                  marginBottom: '12px',
                  padding: '15px',
                  backgroundColor: getToneColor(item.tone) === '#4CAF50' ? 'rgba(76, 175, 80, 0.1)' : 
                                 getToneColor(item.tone) === '#F44336' ? 'rgba(244, 67, 54, 0.1)' : 
                                 'rgba(255, 167, 38, 0.1)',
                  borderRadius: '8px',
                  borderLeft: `4px solid ${getToneColor(item.tone)}`,
                  boxShadow: `0 2px 8px ${getToneColor(item.tone)}33`
                }}>
                  <div style={{ 
                    fontSize: '12px', 
                    color: '#888', 
                    marginBottom: '8px',
                    textAlign: 'right'
                  }}>
                    {item.timestamp}
                  </div>
                  <div style={{ 
                    color: getToneColor(item.tone),
                    fontWeight: '500',
                    fontSize: '15px',
                    lineHeight: '1.5',
                    wordWrap: 'break-word'
                  }}>
                    {item.text}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {recordedAudio && (
          <div style={{ 
            padding: '10px', 
            backgroundColor: '#e8f5e8', 
            borderRadius: '5px', 
            marginTop: '10px' 
          }}>
            <div style={{ fontSize: '14px', color: '#2e7d2e', marginBottom: '8px' }}>
              ✅ Recording ready: {recordedAudio.name} ({(recordedAudio.size / 1024 / 1024).toFixed(2)}MB)
            </div>
            {audioUrl && (
              <audio controls style={{ width: '100%', height: '40px' }}>
                <source src={audioUrl} type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            )}
          </div>
        )}
        
        <div style={{ fontSize: '12px', color: '#666', marginTop: '10px' }}>
          💡 Click "Start Recording" to record audio with live transcription like Zoom<br/>
          🎤 Records high-quality WAV files with real-time speech-to-text<br/>
          🎯 Live tone detection: Green (Positive), Yellow (Neutral), Red (Negative)<br/>
          🔗 Automatic WebSocket connection for live captions during recording
        </div>
      </div>
      <div style={{ textAlign: 'center', margin: '30px 0' }}>
        <button 
          onClick={handleSubmit} 
          disabled={!file || isProcessing || isRecording}
          style={{
            width: '100%',
            maxWidth: '600px',
            padding: '18px 30px',
            fontSize: '18px',
            fontWeight: 'bold',
            borderRadius: '12px',
            marginBottom: '20px',
            transition: 'all 0.3s ease',
            backgroundColor: !file || isProcessing || isRecording ? '#ccc' : '#1976d2',
            color: 'white',
            border: 'none',
            cursor: !file || isProcessing || isRecording ? 'not-allowed' : 'pointer',
            boxShadow: !file || isProcessing || isRecording ? 'none' : '0 4px 15px rgba(25, 118, 210, 0.3)'
          }}
        >
          {isProcessing ? "🔄 Processing..." : (!file ? "📁 Select a file or record audio first" : "🔍 Analyze")}
        </button>
      </div>
      
      {isProcessing && (
        <div style={{ 
          marginTop: '20px', 
          color: '#1976d2',
          fontSize: window.innerWidth > 768 ? '16px' : '14px',
          textAlign: 'center',
          padding: '0 20px'
        }}>
          🔄 Processing audio file... This may take a moment for large files.
        </div>
      )}
      
      {result && (
        <div
          className="tone-box"
          style={{ 
            backgroundColor: "white",
            border: `2px solid ${getToneColor(result.tone)}`,
            marginTop: "20px",
            position: "relative"
          }}
        >
          {/* Close Button */}
          <button
            onClick={() => setResult(null)}
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              background: 'none',
              border: 'none',
              fontSize: '18px',
              cursor: 'pointer',
              color: '#666',
              padding: '5px',
              borderRadius: '50%',
              width: '30px',
              height: '30px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease'
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#e0e0e0';
              e.target.style.color = '#333';
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = 'transparent';
              e.target.style.color = '#666';
            }}
            title="Close results"
          >
            ×
          </button>
          
          {result.error ? (
            <>
              <h2>Error:</h2>
              <p style={{ color: '#333', whiteSpace: 'pre-line', marginRight: '40px' }}>{result.error}</p>
              {result.error.includes('FFmpeg') && (
                <div style={{ marginTop: '10px', fontSize: '14px', color: '#555', marginRight: '40px' }}>
                  <strong>Quick Fix:</strong>
                  <br />• Upload WAV files instead, or
                  <br />• Install FFmpeg: <code>winget install Gyan.FFmpeg</code>
                </div>
              )}
            </>
          ) : (
            <>
              <p style={{
                color: getToneColor(result.tone),
                textShadow: getToneShadow(result.tone),
                cursor: "pointer",
                transition: "all 0.3s ease",
                fontSize: "16px",
                lineHeight: "1.5",
                marginRight: "40px"
              }}
              onMouseOver={(e) => {
                e.target.style.textShadow = `0 0 15px ${getToneColor(result.tone)}`;
                e.target.style.transform = "scale(1.02)";
              }}
              onMouseOut={(e) => {
                e.target.style.textShadow = getToneShadow(result.tone);
                e.target.style.transform = "scale(1)";
              }}>{result.text}</p>
            </>
          )}
        </div>
      )}
      </div>
    </>
  );
}

export default App;
