import os
import platform
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

def text_to_speech_doctor(text, output_filepath="doctor_voice.mp3", play_audio=True):
    """
    Synthesizes the doctor's text response using Deepgram TTS, optionally plays it locally, and returns the path.
    """
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        raise ValueError("DEEPGRAM_API_KEY is not set in your .env file")
        
    deepgram = DeepgramClient(api_key=deepgram_api_key)
    audio = deepgram.speak.v1.audio.generate(
        text=text,
        model="aura-2-thalia-en",
        encoding="mp3",
    )
    
    # Resolve absolute path to save the output file
    audio_path = Path(output_filepath).resolve()
    with audio_path.open("wb") as file:
        for chunk in audio:
            file.write(chunk)
            
    # Play the file automatically using the system default player if requested
    if play_audio:
        if platform.system() == "Windows":
            os.startfile(audio_path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", audio_path])
        else:
            subprocess.call(["xdg-open", audio_path])
            
    return str(audio_path)