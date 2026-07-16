import logging
import os
from dotenv import load_dotenv

# Import our customized modules
from voice_of_the_patient import record_audio, transcribe_patient_voice
from Brain_Of_The_Docter import brain_of_the_doctor
from voice_of_the_docter import text_to_speech_doctor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

def main():
    # Setup files
    patient_audio_path = "patient_voice.mp3"
    doctor_audio_path = "doctor_response.mp3"
    
    # We will use the sample skin image we generated in the directory
    image_path = "acne_sample.png"
    
    if not os.path.exists(image_path):
        logging.error(f"Image file {image_path} not found. Please place an image in the project root.")
        return
        
    logging.info("===== AI MEDICAL DOCTOR ASSISTANT =====")
    
    # Step 1: Record patient's voice
    logging.info("Step 1: Recording patient's explanation...")
    record_audio(patient_audio_path)
    
    # Step 2: Transcribe patient's voice
    logging.info("Step 2: Transcribing patient's explanation...")
    try:
        patient_text = transcribe_patient_voice(patient_audio_path)
        logging.info(f"Patient said: \"{patient_text}\"")
    except Exception as e:
        logging.error(f"Failed to transcribe patient voice: {e}")
        return

    # Step 3: Send transcription and image to Groq Vision
    logging.info("Step 3: Analyzing text and image with Doctor Brain...")
    try:
        doctor_response = brain_of_the_doctor(
            patient_text=patient_text,
            image_filepath=image_path
        )
        logging.info(f"Doctor response: \"{doctor_response}\"")
    except Exception as e:
        logging.error(f"Failed to generate doctor response: {e}")
        return

    # Step 4: Play Doctor's Response using TTS
    logging.info("Step 4: Synthesizing and playing doctor's voice response...")
    try:
        text_to_speech_doctor(doctor_response, output_filepath=doctor_audio_path)
        logging.info("Pipeline complete! Playing doctor response...")
    except Exception as e:
        logging.error(f"Failed to synthesize doctor voice: {e}")

if __name__ == "__main__":
    main()
