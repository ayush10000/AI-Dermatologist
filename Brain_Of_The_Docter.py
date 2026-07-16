import base64
import os
from io import BytesIO
from dotenv import load_dotenv
from groq import Groq
from PIL import Image

load_dotenv()

def encode_image_for_groq(filepath):
    """
    Resizes and base64 encodes an image for Groq Vision models.
    """
    image = Image.open(filepath)
    image.thumbnail((1024, 1024))

    buffer = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def brain_of_the_doctor(patient_text, image_filepath=None, messages_history=None):
    """
    Main brain function to analyze the patient text (with history context) and image,
    and return the doctor's diagnosis.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env or environment")

    # Build the messages payload
    messages = []
    
    # 1. System instruction
    system_prompt = (
        "You are a confident, natural doctor specializing in skin care. Speak with the reassurance, clarity, and authority of a real doctor. "
        "Limit your response to two or three sentences maximum. "
        "Do not use any special characters, symbols, asterisks, or markdown formatting in your response because it will be converted directly to audio."
    )
    messages.append({"role": "system", "content": system_prompt})
    
    # 2. Append history if exists (filtering out system messages to avoid duplicates)
    if messages_history:
        for msg in messages_history:
            if msg["role"] != "system":
                # Ensure the history content is plain text (not list/dict format)
                messages.append({"role": msg["role"], "content": msg["content"]})
                
    # 3. Add the latest user message
    if image_filepath:
        image_data = encode_image_for_groq(image_filepath)
        user_content = [
            {"type": "text", "text": patient_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                },
            },
        ]
    else:
        user_content = patient_text
        
    messages.append({"role": "user", "content": user_content})

    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    return chat_completion.choices[0].message.content
