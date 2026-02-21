# IMPORTANT: Install necessary libraries:

# pip install openai-whisper
# pip install pydub (still good to have for general audio handling, though whisper handles MP3 directly)
# Ensure you have FFmpeg installed on your system and added to your PATH (download from https://ffmpeg.org/download.html)
# For faster inference with GPU, you might need:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 (for CUDA 11.8, adjust as needed)


# II. Import the libraries
import os
import sys
import whisper # Import the whisper library
# import pydub # Still useful if you need to handle different audio formats or check duration

# III. Use the Whisper Model by OpenAI
# A. Create the function that transcribes the audio to text

def transcribe_mp3_to_text_whisper(mp3_file_path, model_name="base"):
    """
    Transcribes an MP3 audio file into text using the Whisper ASR model.

    Args:
        mp3_file_path (str): The path to the MP3 audio file.
        model_name (str): The name of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
                          Larger models are more accurate but slower and require more memory.

    Returns:
        str: The transcribed text, or None if an error occurred.
    """
    if not os.path.exists(mp3_file_path):
        print(f"Error: MP3 file not found at '{mp3_file_path}'")
        return None

    try:
        print(f"Loading Whisper model '{model_name}' (this may take some time the first time)...")
        model = whisper.load_model(model_name)
        print("Whisper model loaded.")

        print(f"Transcribing audio file: {mp3_file_path} using Whisper...")
        # Whisper can directly process various audio formats, including MP3.
        # It handles the internal audio loading and processing.
        result = model.transcribe(mp3_file_path)

        return result["text"]

    except Exception as e:
        print(f"An error occurred during Whisper transcription: {e}")
        print("Please ensure you have whisper installed: pip install openai-whisper")
        print("Also, ensure you have FFmpeg installed on your system and added to your PATH.")
        return None

# B. Ask the user for input, then call the transcription function to generate the output

def generate_audio(mp3_file_path, whisper_model = "medium"):
    if len(mp3_file_path) < 2:
        print("mp3 file missing")
        sys.exit(1)


    # Optional: Allow specifying the Whisper model name as a second argument
    # whisper_model = "base" # Default model
    # supported_models = ["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en"]

    print(f"\nAttempting to transcribe: {mp3_file_path} using Whisper model: {whisper_model}")
    transcribed_text = transcribe_mp3_to_text_whisper(mp3_file_path, model_name=whisper_model)

    if transcribed_text:
        print("\n--- Transcription Result (Whisper) ---")
        print(transcribed_text)
    else:
        print("\nTranscription failed.")

# Get the directory where your script is currently saved
current_dir = os.path.dirname(os.path.abspath(__file__))

# Combine that directory with your filename
file_path = os.path.join(current_dir, "malunggay.mp3")

print("File path to transcribe:", file_path)
generate_audio(file_path)