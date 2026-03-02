#!/usr/bin/env python3
"""Test voice integration with Sarvam AI or Bhashini.

This script tests the voice STT/TTS integration using real API calls.
Prefers Sarvam AI (SARVAM_API_KEY), falls back to Bhashini if configured.

Sarvam AI Models:
- STT: Saaras v3 (saaras:v3) - 23 Indian languages, transcribe mode
- TTS: Bulbul v3 (bulbul:v3) - 30+ natural voices

Usage:
    python scripts/test_voice_integration.py

Requirements:
    - SARVAM_API_KEY or BHASHINI_API_KEY set in environment or .env
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_voice_connection():
    """Test basic voice API connectivity."""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("VOICE INTEGRATION TEST (Sarvam AI / Bhashini)")
    print("=" * 60)

    sarvam_key = os.environ.get("SARVAM_API_KEY", "")
    bhashini_key = os.environ.get("BHASHINI_API_KEY", "")

    if sarvam_key:
        print(f"\n[OK] SARVAM_API_KEY configured (length: {len(sarvam_key)})")
        print("     Using Sarvam AI for voice services")
        from src.integrations.sarvam import configure_sarvam_client
        client = configure_sarvam_client(api_key=sarvam_key)
        provider = "Sarvam AI"
    elif bhashini_key:
        print(f"\n[OK] BHASHINI_API_KEY configured (length: {len(bhashini_key)})")
        print("     Using Bhashini for voice services")
        from src.integrations.bhashini import configure_bhashini_client
        client = configure_bhashini_client(api_key=bhashini_key)
        provider = "Bhashini"
    else:
        print("\n[!] No voice API key configured")
        print("    Voice features will use fallback (text-only) mode")
        print("\n    To enable voice with Sarvam AI (recommended):")
        print("    1. Sign up at https://console.sarvam.ai/")
        print("    2. Get 1000 free credits on signup")
        print("    3. Add to .env: SARVAM_API_KEY=your-key-here")
        print("\n    Alternative: Bhashini (requires paid subscription)")
        print("    1. Register at https://bhashini.gov.in/ulca")
        print("    2. Add to .env: BHASHINI_API_KEY=your-key-here")
        print("\n[SKIP] Voice integration tests skipped")
        return None, None

    return client, provider


async def test_text_to_speech(client, provider):
    """Test TTS functionality."""
    print("\n" + "-" * 60)
    print(f"TEST: Text-to-Speech (TTS) via {provider}")
    print("-" * 60)

    test_text = "नमस्ते, दिल्ली स्कीम साथी में आपका स्वागत है।"
    print(f"Input text: {test_text}")

    try:
        result = await client.text_to_speech(
            text=test_text,
            target_lang="hi",
            voice="female",
        )

        if result.audio_bytes:
            print(f"[OK] TTS successful!")
            print(f"     Audio size: {len(result.audio_bytes)} bytes")
            print(f"     Content type: {result.content_type}")

            # Save audio for manual verification
            output_file = "/tmp/voice_tts_test.wav"
            with open(output_file, "wb") as f:
                f.write(result.audio_bytes)
            print(f"     Saved to: {output_file}")
            return True
        else:
            print("[FAIL] TTS returned empty audio")
            return False

    except Exception as e:
        print(f"[FAIL] TTS error: {e}")
        return False


async def test_speech_to_text(client, provider, audio_file: str = None):
    """Test STT functionality."""
    print("\n" + "-" * 60)
    print(f"TEST: Speech-to-Text (STT) via {provider}")
    print("-" * 60)

    # Use provided audio file or generate one via TTS
    if audio_file and os.path.exists(audio_file):
        print(f"Using audio file: {audio_file}")
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_format = "ogg" if audio_file.endswith(".ogg") else "wav"
    else:
        # Generate test audio via TTS first
        print("No audio file provided, generating via TTS...")
        test_text = "मुझे विधवा पेंशन योजना के बारे में जानकारी चाहिए"
        tts_result = await client.text_to_speech(text=test_text, target_lang="hi")

        if not tts_result.audio_bytes:
            print("[SKIP] Cannot test STT without audio (TTS failed)")
            return None

        audio_bytes = tts_result.audio_bytes
        audio_format = "wav"
        print(f"Generated TTS audio: {len(audio_bytes)} bytes")

    try:
        result = await client.speech_to_text(
            audio_bytes=audio_bytes,
            source_lang="hi",
            audio_format=audio_format,
        )

        if result.text:
            print(f"[OK] STT successful!")
            print(f"     Transcription: {result.text}")
            print(f"     Confidence: {result.confidence:.2f}")
            print(f"     Language: {result.language}")
            return True
        else:
            print("[FAIL] STT returned empty transcription")
            print(f"       Confidence: {result.confidence}")
            return False

    except Exception as e:
        print(f"[FAIL] STT error: {e}")
        return False


async def test_language_detection(client):
    """Test language detection."""
    print("\n" + "-" * 60)
    print("TEST: Language Detection")
    print("-" * 60)

    test_cases = [
        ("नमस्ते, मुझे पेंशन चाहिए", "hi"),
        ("Hello, I need pension information", "en"),
        ("123 456 789", "hi"),  # numbers default to Hindi
    ]

    all_passed = True
    for text, expected in test_cases:
        detected = await client.detect_language(text)
        status = "[OK]" if detected == expected else "[FAIL]"
        if detected != expected:
            all_passed = False
        print(f'{status} "{text[:30]}..." -> {detected} (expected: {expected})')

    return all_passed


async def main():
    """Run all voice integration tests."""
    client, provider = await test_voice_connection()

    if not client:
        print("\n" + "=" * 60)
        print("RESULT: Tests skipped (no API key)")
        print("=" * 60)
        return

    results = {}

    # Test language detection (doesn't need API)
    results["language_detection"] = await test_language_detection(client)

    # Test TTS
    results["tts"] = await test_text_to_speech(client, provider)

    # Test STT (uses TTS output)
    results["stt"] = await test_speech_to_text(client, provider)

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Provider: {provider}")

    for test_name, passed in results.items():
        status = "[PASS]" if passed else ("[SKIP]" if passed is None else "[FAIL]")
        print(f"  {status} {test_name}")

    all_passed = all(v is True for v in results.values() if v is not None)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    # Close client
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
