# Video Processing Feature

This document explains how to use the new video processing feature in the Amharic Speech-to-Text system.

## Overview

The system now supports processing Amharic video files by automatically extracting the audio content and performing speech-to-text transcription and tone analysis on the extracted audio.

## Supported Video Formats

The system supports the following video formats:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)
- M4V (.m4v)
- 3GP (.3gp)
- OGV (.ogv)
- TS (.ts)
- MTS (.mts)
- M2TS (.m2ts)

## How It Works

1. When a video file is uploaded, the system detects it as a video file
2. The video is temporarily saved to disk
3. FFmpeg is used to extract the audio from the video file
4. The extracted audio is converted to WAV format (16kHz, mono) for optimal speech recognition
5. The audio is processed through the Amharic speech-to-text pipeline
6. Tone analysis is performed on the transcribed text
7. Temporary files are cleaned up after processing

## Requirements

- FFmpeg must be installed and accessible in the system PATH
- The system looks for FFmpeg in the following locations:
  - System PATH
  - Project directory: `ffmpeg-2025-08-25-git-1b62f9d3ae-essentials_build\bin\ffmpeg.exe`
  - Common installation paths:
    - `C:\ffmpeg\bin\ffmpeg.exe`
    - `C:\Program Files\ffmpeg\bin\ffmpeg.exe`
    - `C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe`

## Backend Implementation

The video processing is implemented in the following functions in `backend/main.py`:

- `is_video_file(filename)`: Detects if a file is a video format
- `extract_audio_from_video(video_path, output_audio_path)`: Extracts audio from video using FFmpeg
- Modified `/analyze` and `/analyze-realtime` endpoints to handle video files

## Frontend Implementation

The frontend has been updated to:
- Accept video file formats in the file input
- Validate file types before upload
- Display appropriate messages for video processing

## API Response

When processing a video file, the API response includes a `file_type` field indicating whether the processed file was audio or video:

```json
{
  "text": "Transcribed Amharic text here",
  "tone": "Green",
  "confidence": 0.95,
  "status": "success",
  "filename": "example.mp4",
  "file_size_mb": 25.4,
  "file_type": "video"
}
```

## Error Handling

- Video files larger than 100MB will be rejected
- If FFmpeg is not found, the system will provide helpful error messages
- If audio extraction fails, the error will be propagated to the user
- Temporary files are cleaned up even in error conditions

## Testing

To test the video processing functionality:

1. Ensure FFmpeg is installed and accessible
2. Run the backend server
3. Upload a video file through the frontend interface
4. The system should automatically extract audio and process it

## Troubleshooting

### FFmpeg Not Found
If you encounter FFmpeg not found errors:
1. Install FFmpeg from https://ffmpeg.org/download.html
2. Add FFmpeg to your system PATH
3. Restart your terminal/IDE
4. Or place FFmpeg executable in one of the expected locations

### Large File Processing
For large video files:
- Processing may take longer
- Consider compressing videos before upload
- The system supports files up to 100MB

### No Audio Extracted
If no audio is extracted:
- Verify the video file has an audio track
- Check FFmpeg installation
- Try with a different video format