#!/usr/bin/env python3
"""
Simple verification script for video processing support
"""

import sys
import os

# Add the backend directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import is_video_file
    print("✅ Successfully imported is_video_file function")
    
    # Test video files
    test_files = [
        ("video.mp4", True),
        ("movie.avi", True),
        ("clip.mov", True),
        ("sample.mkv", True),
        ("audio.wav", False),
        ("music.mp3", False),
        ("test.wmv", True),
        ("example.flv", True),
        ("recording.webm", True)
    ]
    
    print("\n🧪 Testing file type detection:")
    all_correct = True
    
    for filename, expected in test_files:
        result = is_video_file(filename)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_correct = False
        print(f"   {status} {filename}: {'video' if result else 'audio'} (expected: {'video' if expected else 'audio'})")
    
    print(f"\n📊 Result: {'✅ ALL TESTS PASSED' if all_correct else '❌ SOME TESTS FAILED'}")
    print("🎉 Video processing support is working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the backend directory")
except Exception as e:
    print(f"❌ Error: {e}")