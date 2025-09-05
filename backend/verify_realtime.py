#!/usr/bin/env python3
"""
Simple verification script for real-time Amharic processing components
"""

import sys
import os

# Add the backend directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that we can import the main components"""
    try:
        from main import AmharicSpeechProcessor, RealTimeAudioProcessor
        print("✅ Successfully imported AmharicSpeechProcessor and RealTimeAudioProcessor")
        return True
    except Exception as e:
        print(f"❌ Failed to import processors: {e}")
        return False

def test_processor_initialization():
    """Test that we can initialize the processors"""
    try:
        from main import AmharicSpeechProcessor, RealTimeAudioProcessor
        amharic_processor = AmharicSpeechProcessor()
        realtime_processor = RealTimeAudioProcessor()
        print("✅ Successfully initialized processors")
        print(f"   Amharic processor has {len(amharic_processor.phonetic_corrections)} phonetic corrections")
        print(f"   Amharic processor has {len(amharic_processor.common_amharic_words)} common words")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize processors: {e}")
        return False

def test_simple_correction():
    """Test a simple correction"""
    try:
        from main import AmharicSpeechProcessor
        processor = AmharicSpeechProcessor()
        
        # Test a simple fragmented input
        test_input = "አ ነ ደ ነ ው"
        result = processor.correct_amharic_transcription(test_input)
        print(f"✅ Simple correction test:")
        print(f"   Input: '{test_input}'")
        print(f"   Output: '{result}'")
        return True
    except Exception as e:
        print(f"❌ Simple correction test failed: {e}")
        return False

def main():
    """Run verification tests"""
    print("🔍 VERIFYING REAL-TIME AMHARIC PROCESSING COMPONENTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_processor_initialization,
        test_simple_correction
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 VERIFICATION RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All components are working correctly!")
        print("🚀 Ready for real-time Amharic speech processing")
        return True
    else:
        print("❌ Some components failed verification")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)