#!/usr/bin/env python3
"""
Emergency test for severely corrupted Amharic transcriptions
Tests the new emergency cleanup system with the user's latest problematic input
"""

import sys
import os

# Add the backend directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import AmharicSpeechProcessor
except ImportError:
    print("❌ Could not import AmharicSpeechProcessor from main.py")
    exit(1)

def test_emergency_correction():
    """Test the emergency correction system with severely corrupted input"""
    
    print("🚨 TESTING EMERGENCY AMHARIC CORRUPTION CORRECTION")
    print("=" * 70)
    
    # Initialize the enhanced processor
    processor = AmharicSpeechProcessor()
    
    # User's severely corrupted transcription
    corrupted_input = """ኣረረደጥናዳልለረረደደበበጥዳጥታቶደበረቀልልጥየበበጥበረውኣበሸባሩተዛባረደርበድድፋውውድታውየቅበድየየበረበጨውደጥደቁበደደደረተተተበሬቶተባሪውውድየቸበርበጥበደበዛቸጥውውልረባሪቁበተተተቸጥእኝጀራ ውጥበበተቦደበዳልእግወታርሶጥውጥጥልሩርየፀረበበረጥበበርየተበበቆተዋረውየቆቸባገተሩየየጨጀጨሸውቁየረገድልረቶውሩወተሩቁበጥጥጡጥጥጥጡጥቶጥጥጥጥጥተተተየተረተሪ ተዋዛኣጫ አለ ኣቅጡበጥጥሬ የገረሸጥጥተቸ ሐያጥዋቸ ጥበጥጥጥረተዳ ጥቸተዋተራኣ ቅኣጥጥጥበደኛ ያጥጥበያለየጥጥነጥጥየበረዳኛተተተራ የቻየበተዋተተቶሽጥቶደጥራ የጥጥጥየበረዳኛ በኝየቶኝ እኝደኔበተተዋተ ኣሽጥተሸታርቦጥቶሩጥየበረተዳኣኛ ውቸርጥዘበመጥተዋተ ሽጥቆሽተዘጭጀቼጥጥጥየበተዳ ተታበተጥጥየተተዛተጥኳረዘድርሽጀጥጡረዳ ኣቅገጥተዋዛኣደተ ጥኣሽተደክተተጥረተደያኛ ስድስት ኣዲጫ ያኣጥጥበጥጥጠደያጥጥበረደጥደረታየተረሩጥበተተተደሬዳጣተጥበደተባጥጥጥጥበተሸተደኣበረጥኝበደባደተቸቶጥበደዋኣበደደጥበጥባጥጥጥጥዳበተበደዳጭጥደቸተባተደየጀጥየራገጥደጥየተርበተገበባበየጥጥተበጨዳተተጥዳተተናቸውበጥየተየበተጨተቸጥደተበቸጡጥጥጥጥቅጥጥቅጥጥጥጥጥጥጡጥጥጥበጥጥደጥጡጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥጥቅጥጥጥበጥጥጥጥ"""
    
    print("📥 SEVERELY CORRUPTED INPUT:")
    print(f"   Length: {len(corrupted_input)} characters")
    print(f"   Word count: {len(corrupted_input.split())} fragments")
    print(f"   Sample: '{corrupted_input[:200]}...'")
    print()
    
    # Test corruption assessment
    print("🔍 ASSESSING CORRUPTION LEVEL...")
    corruption_score = processor.assess_corruption_level(corrupted_input)
    print()
    
    # Test emergency cleanup if needed
    if corruption_score > 0.7:
        print("🚨 APPLYING EMERGENCY CLEANUP...")
        emergency_cleaned = processor.emergency_cleanup(corrupted_input)
        print(f"   Emergency result: '{emergency_cleaned[:100]}...'")
        print()
    
    # Test full correction pipeline
    print("🔧 APPLYING FULL CORRECTION PIPELINE...")
    corrected_output = processor.correct_amharic_transcription(corrupted_input)
    
    print("\n📤 EMERGENCY CORRECTED OUTPUT:")
    print(f"   '{corrected_output}'")
    print(f"   Length: {len(corrected_output)} characters")
    print(f"   Word count: {len(corrected_output.split())} words")
    print()
    
    # Analysis of improvements
    print("📊 EMERGENCY CORRECTION ANALYSIS:")
    original_words = len(corrupted_input.split())
    corrected_words = len(corrected_output.split())
    reduction_ratio = (1 - corrected_words / original_words) if original_words > 0 else 0
    
    print(f"   - Original fragments: {original_words}")
    print(f"   - Corrected words: {corrected_words}")
    print(f"   - Reduction ratio: {reduction_ratio:.1%}")
    print(f"   - Corruption score: {corruption_score:.1%}")
    print(f"   - Length change: {((len(corrected_output) - len(corrupted_input)) / len(corrupted_input)):.1%}")
    
    # Test specific corruption patterns
    print("\n🔍 TESTING SPECIFIC CORRUPTION PATTERNS:")
    
    test_corruption_patterns = [
        "ጥጥጥጥጥጥጥጥጥ",  # Excessive repetition
        "ኣረረደጥናዳልለረረደደ",  # Mixed repetition
        "የጥጥጥየበረዳኛተተተራ",  # Pattern corruption
        "ስድስት",  # Should be preserved (real word)
        "አለ",     # Should be preserved (real word)
    ]
    
    for pattern in test_corruption_patterns:
        if pattern in corrupted_input:
            corrected_pattern = processor.correct_amharic_transcription(pattern)
            print(f"   '{pattern}' -> '{corrected_pattern}'")
    
    print("\n✅ Emergency correction test completed!")
    
    # Recommendations based on results
    print(f"\n💡 RECOMMENDATIONS:")
    if corruption_score > 0.8:
        print("   🔴 CRITICAL: Audio quality is extremely poor")
        print("   • Check microphone setup and recording environment")
        print("   • Reduce background noise significantly") 
        print("   • Consider using a different recording device")
        print("   • Try speaking more slowly and clearly")
    elif corruption_score > 0.5:
        print("   🟡 MODERATE: Audio has significant issues")
        print("   • Improve recording environment (reduce noise)")
        print("   • Check audio format and sample rate")
        print("   • Consider shorter audio segments")
    else:
        print("   🟢 GOOD: Audio quality is acceptable")
        print("   • Current correction system should handle this well")
    
    return corrected_output

def main():
    """Run emergency correction tests"""
    
    try:
        # Test the emergency correction system
        corrected = test_emergency_correction()
        
        print(f"\n🎉 EMERGENCY TESTING COMPLETED!")
        print(f"   The emergency system has been activated to handle")
        print(f"   extremely corrupted transcriptions like yours.")
        
        # Save results for reference
        with open("emergency_test_results.txt", "w", encoding="utf-8") as f:
            f.write("Emergency Amharic Correction Test Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("Corrected Output:\n")
            f.write(corrected + "\n\n")
            f.write("This demonstrates the emergency correction capabilities\n")
            f.write("for extremely corrupted Amharic speech recognition.\n")
        
        print(f"   Results saved to: emergency_test_results.txt")
        
        return True
        
    except Exception as e:
        print(f"\n❌ EMERGENCY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)