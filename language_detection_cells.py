# ============================================================================
# LANGUAGE AUTO-DETECTION FROM VOICE INPUT
# ============================================================================
# User Story: "As a farmer, I want the app to auto-detect my language from 
# voice input so I don't have to configure settings."
#
# This module provides automatic language detection and speech-to-text
# functionality for the Crop Disease Detection system.
# ============================================================================

# %% [markdown]
# ## Language Auto-Detection from Voice Input
# 
# This section implements automatic language detection from farmer's voice input.
# The system can:
# 1. Accept voice input (audio files)
# 2. Automatically detect the spoken language
# 3. Transcribe the speech to text in the detected language
# 4. Generate responses in the farmer's language
#
# **Supported Languages:** English, Hindi, Telugu, Tamil, Kannada, Marathi, Bengali

# %% Cell 1: Install Required Dependencies
# ============================================================================
# CELL 1: INSTALL REQUIRED DEPENDENCIES
# ============================================================================
# These packages are required for language detection from voice:
# - SpeechRecognition: For converting speech to text
# - pydub: For audio file format conversion  
# - langdetect: For detecting language from text
# - gTTS: For generating text-to-speech responses

# Uncomment and run these if not installed:
# !pip install SpeechRecognition pydub langdetect gTTS

# %% Cell 2: Import Required Libraries
# ============================================================================
# CELL 2: IMPORT REQUIRED LIBRARIES
# ============================================================================

import speech_recognition as sr  # For speech-to-text conversion
from langdetect import detect, detect_langs  # For language detection from text
from langdetect.lang_detect_exception import LangDetectException
from gtts import gTTS  # For text-to-speech output
from pydub import AudioSegment  # For audio format conversion
import os
from pathlib import Path
import tempfile

print("✅ Language detection libraries loaded successfully!")

# %% Cell 3: Define Supported Languages Configuration
# ============================================================================
# CELL 3: LANGUAGE CONFIGURATION
# ============================================================================
# This dictionary maps language codes to their full names and configurations
# for speech recognition. These are common languages spoken by farmers in India.

SUPPORTED_LANGUAGES = {
    # Language Code: (Full Name, Speech Recognition Code, Google TTS Code)
    "en": {
        "name": "English",
        "sr_code": "en-IN",  # Indian English for speech recognition
        "tts_code": "en",    # TTS language code
        "greeting": "Hello! How can I help you with your crops?"
    },
    "hi": {
        "name": "Hindi (हिंदी)",
        "sr_code": "hi-IN",
        "tts_code": "hi",
        "greeting": "नमस्ते! आपकी फसलों में मैं कैसे मदद कर सकता हूं?"
    },
    "te": {
        "name": "Telugu (తెలుగు)",
        "sr_code": "te-IN",
        "tts_code": "te",
        "greeting": "నమస్కారం! మీ పంటలలో నేను ఎలా సహాయం చేయగలను?"
    },
    "ta": {
        "name": "Tamil (தமிழ்)",
        "sr_code": "ta-IN",
        "tts_code": "ta",
        "greeting": "வணக்கம்! உங்கள் பயிர்களில் நான் எவ்வாறு உதவ முடியும்?"
    },
    "kn": {
        "name": "Kannada (ಕನ್ನಡ)",
        "sr_code": "kn-IN",
        "tts_code": "kn",
        "greeting": "ನಮಸ್ಕಾರ! ನಿಮ್ಮ ಬೆಳೆಗಳಲ್ಲಿ ನಾನು ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
    },
    "mr": {
        "name": "Marathi (मराठी)",
        "sr_code": "mr-IN",
        "tts_code": "mr",
        "greeting": "नमस्कार! तुमच्या पिकांमध्ये मी कशी मदत करू शकतो?"
    },
    "bn": {
        "name": "Bengali (বাংলা)",
        "sr_code": "bn-IN",
        "tts_code": "bn",
        "greeting": "নমস্কার! আপনার ফসলে আমি কিভাবে সাহায্য করতে পারি?"
    },
    "gu": {
        "name": "Gujarati (ગુજરાતી)",
        "sr_code": "gu-IN",
        "tts_code": "gu",
        "greeting": "નમસ્તે! તમારા પાકમાં હું કેવી રીતે મદદ કરી શકું?"
    },
    "pa": {
        "name": "Punjabi (ਪੰਜਾਬੀ)",
        "sr_code": "pa-IN",
        "tts_code": "pa",
        "greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਤੁਹਾਡੀਆਂ ਫਸਲਾਂ ਵਿੱਚ ਮੈਂ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
    }
}

# Default fallback language if detection fails
DEFAULT_LANGUAGE = "en"

print(f"✅ {len(SUPPORTED_LANGUAGES)} languages configured for voice input")

# %% Cell 4: Audio Preprocessing Functions
# ============================================================================
# CELL 4: AUDIO PREPROCESSING FUNCTIONS
# ============================================================================
# These functions handle audio file conversion and preparation for
# speech recognition. Farmers might use different devices/apps to record,
# so we need to handle various audio formats.

def convert_audio_to_wav(audio_path: str) -> str:
    """
    Convert any audio file to WAV format for speech recognition.
    
    Why we need this:
    -----------------
    - Speech recognition works best with WAV format
    - Farmers might record audio in different formats (MP3, M4A, etc.)
    - This ensures consistent processing regardless of input format
    
    Parameters:
    -----------
    audio_path : str
        Path to the input audio file (can be MP3, WAV, M4A, etc.)
    
    Returns:
    --------
    str
        Path to the converted WAV file
    
    Example:
    --------
    >>> wav_path = convert_audio_to_wav("farmer_query.mp3")
    >>> print(wav_path)  # "farmer_query_converted.wav"
    """
    audio_path = Path(audio_path)
    
    # If already WAV, return as-is
    if audio_path.suffix.lower() == '.wav':
        return str(audio_path)
    
    # Convert to WAV using pydub
    # pydub automatically detects the format based on file extension
    audio = AudioSegment.from_file(str(audio_path))
    
    # Create output path with .wav extension
    output_path = audio_path.with_suffix('.wav')
    
    # Export as WAV with standard settings for speech recognition
    # 16kHz sample rate is optimal for speech
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(str(output_path), format="wav")
    
    print(f"✅ Audio converted: {audio_path.name} → {output_path.name}")
    return str(output_path)


def validate_audio_file(audio_path: str) -> dict:
    """
    Validate that an audio file exists and is readable.
    
    This function performs basic checks before processing:
    - File existence check
    - File size check (not empty)
    - Format support check
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file to validate
    
    Returns:
    --------
    dict
        Validation result with 'valid' boolean and 'message' string
    
    Example:
    --------
    >>> result = validate_audio_file("recording.wav")
    >>> if result['valid']:
    ...     print("Audio is ready for processing")
    """
    audio_path = Path(audio_path)
    
    # Check if file exists
    if not audio_path.exists():
        return {
            "valid": False,
            "message": f"Audio file not found: {audio_path}"
        }
    
    # Check if file is empty
    if audio_path.stat().st_size == 0:
        return {
            "valid": False,
            "message": "Audio file is empty. Please record again."
        }
    
    # Check supported formats
    supported_formats = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac']
    if audio_path.suffix.lower() not in supported_formats:
        return {
            "valid": False,
            "message": f"Unsupported audio format: {audio_path.suffix}. "
                      f"Supported: {', '.join(supported_formats)}"
        }
    
    return {
        "valid": True,
        "message": "Audio file is valid and ready for processing"
    }

print("✅ Audio preprocessing functions defined")

# %% Cell 5: Language Detection Core Functions  
# ============================================================================
# CELL 5: LANGUAGE DETECTION CORE FUNCTIONS
# ============================================================================
# These are the main functions that detect language from voice input.
# The approach is:
# 1. First, transcribe audio using Google's speech recognition (language-agnostic initial pass)
# 2. Then, detect the language from the transcribed text
# 3. Finally, re-transcribe with the correct language for accuracy

def detect_language_from_text(text: str) -> dict:
    """
    Detect the language of given text using langdetect library.
    
    How it works:
    -------------
    The langdetect library uses a character n-gram model trained on 
    Wikipedia data to identify languages. It can detect 55+ languages
    with high accuracy when given sufficient text.
    
    Parameters:
    -----------
    text : str
        The text to analyze for language detection
    
    Returns:
    --------
    dict
        Contains detected language code, confidence, and full name
        
    Example:
    --------
    >>> result = detect_language_from_text("मेरे टमाटर के पत्ते पीले हो रहे हैं")
    >>> print(result)
    {'language': 'hi', 'name': 'Hindi (हिंदी)', 'confidence': 0.99, 'success': True}
    """
    # Handle empty or very short text
    if not text or len(text.strip()) < 3:
        return {
            "success": False,
            "language": DEFAULT_LANGUAGE,
            "name": SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["name"],
            "confidence": 0.0,
            "message": "Text too short for reliable language detection"
        }
    
    try:
        # Get all detected languages with probabilities
        # detect_langs returns a list of Language objects with lang and prob attributes
        detected_languages = detect_langs(text)
        
        # Get the most probable language
        top_detection = detected_languages[0]
        detected_lang = top_detection.lang
        confidence = top_detection.prob
        
        # Check if detected language is in our supported list
        if detected_lang in SUPPORTED_LANGUAGES:
            return {
                "success": True,
                "language": detected_lang,
                "name": SUPPORTED_LANGUAGES[detected_lang]["name"],
                "confidence": round(confidence, 2),
                "message": f"Detected {SUPPORTED_LANGUAGES[detected_lang]['name']}"
            }
        else:
            # If detected language is not supported, fallback to English
            # but still report what was detected
            return {
                "success": True,
                "language": DEFAULT_LANGUAGE,
                "name": SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["name"],
                "confidence": round(confidence, 2),
                "message": f"Detected '{detected_lang}' (not supported), using English",
                "original_detected": detected_lang
            }
            
    except LangDetectException as e:
        # langdetect can fail on very short or ambiguous text
        return {
            "success": False,
            "language": DEFAULT_LANGUAGE,
            "name": SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["name"],
            "confidence": 0.0,
            "message": f"Language detection failed: {str(e)}"
        }


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """
    Convert speech in an audio file to text using Google Speech Recognition.
    
    How it works:
    -------------
    This function uses the SpeechRecognition library which interfaces with
    Google's free speech recognition API. For better accuracy with Indian
    languages, we use the appropriate language code (e.g., "hi-IN" for Hindi).
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file (WAV format preferred)
    language : str, optional
        Language code for recognition. If None, uses default (en-IN)
    
    Returns:
    --------
    dict
        Contains transcription, success status, and any error messages
        
    Example:
    --------
    >>> result = transcribe_audio("farmer_query.wav", language="hi")
    >>> print(result)
    {'success': True, 'transcription': 'मेरे टमाटर में बीमारी है', 'language': 'hi'}
    """
    # Initialize the speech recognizer
    recognizer = sr.Recognizer()
    
    # Determine the speech recognition language code
    if language and language in SUPPORTED_LANGUAGES:
        sr_language = SUPPORTED_LANGUAGES[language]["sr_code"]
    else:
        sr_language = SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["sr_code"]
    
    try:
        # Convert audio to WAV if needed
        wav_path = convert_audio_to_wav(audio_path)
        
        # Load the audio file
        with sr.AudioFile(wav_path) as source:
            # Adjust for ambient noise (helps with recordings in fields)
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Record the audio data
            audio_data = recognizer.record(source)
        
        # Use Google Speech Recognition API
        # This is free and works well for most use cases
        transcription = recognizer.recognize_google(
            audio_data, 
            language=sr_language
        )
        
        return {
            "success": True,
            "transcription": transcription,
            "language": language or DEFAULT_LANGUAGE,
            "message": "Audio transcribed successfully"
        }
        
    except sr.UnknownValueError:
        # Speech was not understood
        return {
            "success": False,
            "transcription": "",
            "language": language or DEFAULT_LANGUAGE,
            "message": "Could not understand the audio. Please speak clearly and try again."
        }
        
    except sr.RequestError as e:
        # API request failed (network issue)
        return {
            "success": False,
            "transcription": "",
            "language": language or DEFAULT_LANGUAGE,
            "message": f"Speech recognition service error: {str(e)}. Check internet connection."
        }
        
    except Exception as e:
        # Other errors
        return {
            "success": False,
            "transcription": "",
            "language": language or DEFAULT_LANGUAGE,
            "message": f"Error processing audio: {str(e)}"
        }

print("✅ Language detection core functions defined")

# %% Cell 6: Main Language Detection Pipeline
# ============================================================================
# CELL 6: MAIN LANGUAGE DETECTION PIPELINE
# ============================================================================
# This is the main function that the UI team will call. It combines all
# the above functions into a single, easy-to-use pipeline.

def detect_language_from_audio(audio_path: str) -> dict:
    """
    MAIN FUNCTION: Automatically detect language from voice input.
    
    This is the primary function for the user story:
    "As a farmer, I want the app to auto-detect my language from voice input 
    so I don't have to configure settings."
    
    How it works (2-pass approach):
    -------------------------------
    1. First Pass: Transcribe audio with English settings (works for all languages
       with Google's speech recognition, just may not be 100% accurate)
    2. Detect Language: Analyze the transcribed text to detect the actual language
    3. Second Pass: Re-transcribe with correct language settings for accuracy
    
    Parameters:
    -----------
    audio_path : str
        Path to the farmer's voice recording
    
    Returns:
    --------
    dict
        Complete result with detected language, transcription, and confidence
        
    Example (for UI team):
    ----------------------
    >>> # Farmer records a voice message about their crop problem
    >>> result = detect_language_from_audio("farmer_query.wav")
    >>> print(result)
    {
        'success': True,
        'language': 'hi',
        'language_name': 'Hindi (हिंदी)',
        'transcription': 'मेरे टमाटर के पत्ते पीले हो रहे हैं',
        'confidence': 0.95,
        'greeting': 'नमस्ते! आपकी फसलों में मैं कैसे मदद कर सकता हूं?'
    }
    """
    # Step 1: Validate the audio file
    validation = validate_audio_file(audio_path)
    if not validation["valid"]:
        return {
            "success": False,
            "language": DEFAULT_LANGUAGE,
            "language_name": SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["name"],
            "transcription": "",
            "confidence": 0.0,
            "message": validation["message"]
        }
    
    print(f"🎤 Processing audio: {audio_path}")
    
    # Step 2: First-pass transcription (using English as default)
    # Google's speech recognition is quite good at transcribing even when
    # the language setting doesn't match perfectly
    initial_transcription = transcribe_audio(audio_path, language=DEFAULT_LANGUAGE)
    
    if not initial_transcription["success"]:
        # Try with Hindi as many Indian farmers speak Hindi
        initial_transcription = transcribe_audio(audio_path, language="hi")
        
        if not initial_transcription["success"]:
            return {
                "success": False,
                "language": DEFAULT_LANGUAGE,
                "language_name": SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["name"],
                "transcription": "",
                "confidence": 0.0,
                "message": initial_transcription["message"]
            }
    
    transcribed_text = initial_transcription["transcription"]
    print(f"📝 Initial transcription: {transcribed_text}")
    
    # Step 3: Detect language from the transcribed text
    lang_detection = detect_language_from_text(transcribed_text)
    detected_lang = lang_detection["language"]
    
    print(f"🌐 Detected language: {lang_detection['name']} (confidence: {lang_detection['confidence']})")
    
    # Step 4: If detected language is different from default, re-transcribe
    # for better accuracy
    if detected_lang != DEFAULT_LANGUAGE and lang_detection["confidence"] > 0.7:
        print(f"🔄 Re-transcribing with {lang_detection['name']} settings...")
        final_transcription = transcribe_audio(audio_path, language=detected_lang)
        
        if final_transcription["success"]:
            transcribed_text = final_transcription["transcription"]
            print(f"📝 Final transcription: {transcribed_text}")
    
    # Step 5: Return the complete result
    return {
        "success": True,
        "language": detected_lang,
        "language_name": lang_detection["name"],
        "transcription": transcribed_text,
        "confidence": lang_detection["confidence"],
        "greeting": SUPPORTED_LANGUAGES[detected_lang]["greeting"],
        "message": f"Successfully detected {lang_detection['name']}"
    }

print("✅ Main language detection pipeline ready")

# %% Cell 7: Text-to-Speech Response Generation
# ============================================================================
# CELL 7: TEXT-TO-SPEECH RESPONSE GENERATION
# ============================================================================
# Generate audio responses in the farmer's detected language.
# This allows the app to speak back to farmers who may not read well.

def generate_audio_response(text: str, language: str = "en", output_path: str = None) -> str:
    """
    Generate an audio response in the farmer's language.
    
    Why this is important:
    ----------------------
    Many farmers may have limited literacy but excellent oral comprehension.
    By generating audio responses in their native language, we make the
    app more accessible and user-friendly.
    
    Parameters:
    -----------
    text : str
        The text to convert to speech
    language : str
        Language code (e.g., 'hi' for Hindi, 'te' for Telugu)
    output_path : str, optional
        Where to save the audio file. If None, uses 'response.mp3'
    
    Returns:
    --------
    str
        Path to the generated audio file
        
    Example:
    --------
    >>> # After detecting a disease, respond in farmer's language
    >>> audio_path = generate_audio_response(
    ...     "आपके टमाटर में अगेती झुलसा रोग है। नीम के तेल का छिड़काव करें।",
    ...     language="hi"
    ... )
    >>> print(f"Audio response saved to: {audio_path}")
    """
    # Get the TTS language code
    if language in SUPPORTED_LANGUAGES:
        tts_lang = SUPPORTED_LANGUAGES[language]["tts_code"]
    else:
        tts_lang = "en"
    
    # Set default output path
    if output_path is None:
        output_path = "response.mp3"
    
    try:
        # Create the text-to-speech object
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        
        # Save the audio file
        tts.save(output_path)
        
        print(f"🔊 Audio response generated: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error generating audio: {str(e)}")
        return None

print("✅ Text-to-speech response function ready")

# %% Cell 8: Complete Voice Interaction Pipeline
# ============================================================================
# CELL 8: COMPLETE VOICE INTERACTION PIPELINE
# ============================================================================
# This combines language detection with the crop disease system for a
# complete voice-based interaction.

def process_voice_query(audio_path: str) -> dict:
    """
    Complete pipeline for processing a farmer's voice query.
    
    This function:
    1. Detects the farmer's language automatically
    2. Transcribes their voice query
    3. Returns structured data for the UI/disease detection system
    
    Integration with Crop Disease Detection:
    ----------------------------------------
    After calling this function, the UI can:
    1. Display the transcription to confirm understanding
    2. If query is about crop disease, pass to disease detection model
    3. Generate response in the farmer's language
    
    Parameters:
    -----------
    audio_path : str
        Path to the farmer's voice recording
    
    Returns:
    --------
    dict
        Complete query information ready for processing
        
    Example (Complete flow):
    ------------------------
    >>> # Step 1: Process voice query
    >>> query_result = process_voice_query("farmer_question.wav")
    >>> 
    >>> if query_result['success']:
    ...     print(f"Farmer ({query_result['language_name']}) asked:")
    ...     print(f"'{query_result['transcription']}'")
    ...     
    ...     # Step 2: Process with disease detection (if applicable)
    ...     # disease_result = predict_disease(image_path)
    ...     
    ...     # Step 3: Respond in farmer's language
    ...     response_text = translate_response(disease_result, query_result['language'])
    ...     generate_audio_response(response_text, query_result['language'])
    """
    print("\n" + "="*60)
    print("🌾 CROP DISEASE ASSISTANT - VOICE QUERY PROCESSOR")
    print("="*60)
    
    # Process the audio and detect language
    result = detect_language_from_audio(audio_path)
    
    if result["success"]:
        print("\n📊 QUERY PROCESSING RESULT:")
        print(f"   Language: {result['language_name']}")
        print(f"   Confidence: {result['confidence'] * 100:.1f}%")
        print(f"   Query: {result['transcription']}")
        print(f"\n💬 Greeting: {result['greeting']}")
    else:
        print(f"\n❌ Error: {result['message']}")
    
    print("="*60 + "\n")
    
    return result


def get_supported_languages() -> list:
    """
    Get list of all supported languages for voice input.
    
    This is useful for:
    - Displaying language options in UI (if manual selection is wanted)
    - Showing users what languages are available
    
    Returns:
    --------
    list
        List of dictionaries with language code and name
        
    Example:
    --------
    >>> languages = get_supported_languages()
    >>> for lang in languages:
    ...     print(f"{lang['code']}: {lang['name']}")
    """
    return [
        {"code": code, "name": info["name"]}
        for code, info in SUPPORTED_LANGUAGES.items()
    ]

print("✅ Complete voice interaction pipeline ready")
print("\n📌 SUMMARY: The following functions are available for UI integration:")
print("   1. detect_language_from_audio(audio_path) - Main detection function")
print("   2. process_voice_query(audio_path) - Complete query pipeline")
print("   3. generate_audio_response(text, language) - Text-to-speech")
print("   4. get_supported_languages() - List available languages")

# %% Cell 9: Testing the Language Detection (Example)
# ============================================================================
# CELL 9: TESTING THE LANGUAGE DETECTION
# ============================================================================
# Test the language detection with sample text (no audio file needed)

def test_language_detection():
    """
    Test the language detection with sample texts in different languages.
    Run this to verify the language detection is working correctly.
    """
    print("\n" + "="*60)
    print("🧪 TESTING LANGUAGE DETECTION")
    print("="*60)
    
    # Test samples in different languages
    test_samples = [
        ("My tomato leaves are turning yellow", "English"),
        ("मेरे टमाटर के पत्ते पीले हो रहे हैं", "Hindi"),
        ("నా టమాటో ఆకులు పసుపు రంగుకు మారుతున్నాయి", "Telugu"),
        ("என் தக்காளி இலைகள் மஞ்சள் நிறமாக மாறுகின்றன", "Tamil"),
        ("माझ्या टोमॅटोच्या पानांचा रंग पिवळा होत आहे", "Marathi"),
    ]
    
    for text, expected_lang in test_samples:
        result = detect_language_from_text(text)
        status = "✅" if result["success"] else "❌"
        print(f"\n{status} Input: '{text[:50]}...'")
        print(f"   Expected: {expected_lang}")
        print(f"   Detected: {result['name']} (confidence: {result['confidence']:.2f})")
    
    print("\n" + "="*60)
    print("🧪 LANGUAGE DETECTION TEST COMPLETE")
    print("="*60)

# Run the test
test_language_detection()

# %% Cell 10: Example Usage with Audio File
# ============================================================================
# CELL 10: EXAMPLE USAGE (for testing with actual audio)
# ============================================================================
# Uncomment the code below to test with an actual audio file

"""
# Example: Process a farmer's voice query

# Path to the audio file (change this to your test audio)
TEST_AUDIO_PATH = "farmer_query.wav"

if Path(TEST_AUDIO_PATH).exists():
    # Process the voice query
    result = process_voice_query(TEST_AUDIO_PATH)
    
    if result['success']:
        print(f"\\nFarmer's language: {result['language_name']}")
        print(f"Farmer asked: {result['transcription']}")
        
        # Generate a response in the farmer's language
        response = f"We detected your query in {result['language_name']}. " \
                   f"Please upload a photo of your affected crop."
        
        # Generate audio response
        audio_response_path = generate_audio_response(
            SUPPORTED_LANGUAGES[result['language']]['greeting'],
            language=result['language']
        )
        print(f"\\nAudio response saved to: {audio_response_path}")
else:
    print(f"Test audio file not found: {TEST_AUDIO_PATH}")
    print("To test, record an audio file and update the path above.")
"""

print("\n✅ Language detection module fully loaded and ready!")
print("📖 Check the comments in each cell for detailed explanations.")
