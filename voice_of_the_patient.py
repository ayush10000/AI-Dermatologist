import logging
import os
import io
import wave
import numpy as np
import sounddevice as sd

# Configure FFmpeg binary path before importing pydub to prevent warning logs
ffmpeg_bin_dir = r"C:\Users\Shavaik\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin"
if os.path.exists(ffmpeg_bin_dir):
    os.environ["PATH"] += os.pathsep + ffmpeg_bin_dir

from pydub import AudioSegment
from dotenv import load_dotenv
from groq import Groq

if os.path.exists(ffmpeg_bin_dir):
    AudioSegment.converter = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

import queue
import sys

def record_audio(file_path, sample_rate=16000):
    """
    Record audio from the microphone using sounddevice and save it as an MP3 file.
    The recording runs until the user presses ENTER.
    
    Args:
    file_path (str): Path to save the recorded audio file.
    sample_rate (int): Sample rate of the recording.
    """
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    logging.info("Start speaking now... Press ENTER to stop recording.")
    
    # Start the recording stream
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=callback)
    with stream:
        input()  # Wait for the user to press Enter
        
    logging.info("Recording complete. Processing audio...")
    
    # Collect all the recorded chunks from the queue
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())
        
    if not audio_data:
        logging.error("No audio was recorded.")
        return
        
    recording = np.concatenate(audio_data, axis=0)
    
    # Write to a WAV buffer in memory
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    
    wav_io.seek(0)
    
    # Convert WAV to MP3 using pydub
    audio_segment = AudioSegment.from_wav(wav_io)
    audio_segment.export(file_path, format="mp3", bitrate="128k")
    logging.info(f"Audio saved to {file_path}")

def transcribe_patient_voice(audio_filepath):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in your .env file")

    client = Groq(api_key=groq_api_key)
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=os.environ.get("WHISPER_MODEL", "whisper-large-v3"),
        )

    return transcription.text

if __name__ == "__main__":
    audio_filepath = "patient_voice_test.mp3"
    
    # 1. Record audio (runs until user presses ENTER)
    record_audio(audio_filepath)
    
    # 2. Transcribe audio
    logging.info("Transcribing audio...")
    try:
        text = transcribe_patient_voice(audio_filepath)
        logging.info(f"Transcription result: {text}")
    except Exception as e:
        logging.error(f"Failed to transcribe: {e}")