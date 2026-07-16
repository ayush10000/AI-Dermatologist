import logging
import os
import gradio as gr
from dotenv import load_dotenv

# Hugging Face ZeroGPU compatibility
try:
    import spaces
except ImportError:
    # Dummy fallback decorator for local development
    class spaces:
        @staticmethod
        def GPU(func):
            return func

@spaces.GPU
def dummy_gpu_trigger():
    """Dummy function to satisfy Hugging Face ZeroGPU startup check."""
    pass

# Import functions from our modules
from voice_of_the_patient import transcribe_patient_voice
from Brain_Of_The_Docter import brain_of_the_doctor
from voice_of_the_docter import text_to_speech_doctor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Predefined greeting matching the UI reference
GREETING_TEXT = (
    "Hello. I am the DermaAI Diagnostic Assistant. Please upload a clear photo of "
    "your skin concern, or describe what you're noticing. I am here to provide an "
    "empathetic and clinical preliminary analysis."
)

def get_initial_state():
    return {
        "history": [{"role": "assistant", "content": GREETING_TEXT}],
        "image_path": None
    }

def on_image_upload(file, state):
    """
    Saves the uploaded image in the session context and returns a visible preview.
    """
    if file:
        state["image_path"] = file
        logging.info(f"Image uploaded to session state: {file}")
        return gr.update(value=file, visible=True), state
    return gr.update(visible=False, value=None), state

def handle_user_message(user_text, user_audio, state):
    """
    Processes the user's text/voice message and image in the conversation.
    """
    if not state or not state.get("history"):
        state = get_initial_state()

    patient_text = ""

    # 1. Transcribe voice message if recorded
    if user_audio:
        logging.info(f"Transcribing audio message: {user_audio}")
        try:
            patient_text = transcribe_patient_voice(user_audio)
            logging.info(f"Transcribed patient text: {patient_text}")
        except Exception as e:
            logging.error(f"Voice transcription failed: {e}")
            return state["history"], None, "", None, gr.update(visible=False, value=None), state

    # 2. Combine with text notes if typed
    if user_text:
        if patient_text:
            patient_text += f"\nAdditional note: {user_text}"
        else:
            patient_text = user_text

    # 3. Handle image from session context
    uploaded_image = state.get("image_path")
    if uploaded_image:
        logging.info(f"Using attached session image for diagnosis: {uploaded_image}")
        # Insert image into history as a separate visual bubble for the chatbot
        state["history"].append({"role": "user", "content": {"path": uploaded_image}})

    # 4. Check if we have any text prompt to run
    if not patient_text.strip():
        if uploaded_image:
            patient_text = "I have uploaded an image of my skin condition. Please analyze it."
        else:
            return state["history"], None, "", None, gr.update(visible=False, value=None), state

    # 5. Append user text message to history
    state["history"].append({"role": "user", "content": patient_text})

    # 6. Convert history for doctor brain (filtering file dictionaries so it only gets text prompts)
    history_dicts = []
    for msg in state["history"]:
        if isinstance(msg["content"], str):
            history_dicts.append(msg)

    # 7. Run Doctor Brain with history context and image
    logging.info("Generating doctor diagnosis...")
    try:
        doctor_text = brain_of_the_doctor(
            patient_text=patient_text,
            image_filepath=uploaded_image,
            messages_history=history_dicts
        )
        logging.info(f"Doctor response: {doctor_text}")
    except Exception as e:
        logging.error(f"Doctor brain analysis failed: {e}")
        return state["history"], None, "", None, gr.update(visible=False, value=None), state

    # 8. Append assistant response to history
    state["history"].append({"role": "assistant", "content": doctor_text})

    # 9. Synthesize audio response
    try:
        doctor_audio_path = text_to_speech_doctor(
            text=doctor_text,
            output_filepath="doctor_web_response.mp3",
            play_audio=False
        )
    except Exception as e:
        logging.error(f"TTS synthesis failed: {e}")
        doctor_audio_path = None

    # Once sent, clear the active image from the context
    state["image_path"] = None

    return state["history"], doctor_audio_path, "", None, gr.update(visible=False, value=None), state

# Custom medical theme styling (Outfit Google Font)
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

# Custom CSS matching DermaAI reference
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');

/* Main layout wrapper */
body, html {
    font-family: 'Outfit', sans-serif !important;
    background-color: #f8fafc !important;
}

.container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

/* Top Navbar style */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 25px;
}
.navbar-logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e40af;
}
.navbar-links {
    display: flex;
    gap: 30px;
}
.navbar-link {
    font-size: 1rem;
    font-weight: 500;
    color: #64748b;
    text-decoration: none;
    padding-bottom: 4px;
}
.navbar-link.active {
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
}
.navbar-profile {
    color: #64748b;
    cursor: pointer;
}

/* Diagnostic Chat Header */
.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.chat-title-wrapper {
    display: flex;
    align-items: center;
    gap: 10px;
}
.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
}
.chat-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0;
}
.analysis-badge {
    background-color: #d1fae5;
    color: #065f46;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 20px;
}

/* Chatbot component styling */
.gradio-container .chatbot {
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

/* Override Chatbot bubbles to match UI */
.message.assistant {
    background-color: #eff6ff !important;
    border-radius: 16px !important;
    color: #1e293b !important;
    padding: 16px !important;
    max-width: 85% !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
}
.message.user {
    background-color: #2563eb !important;
    border-radius: 16px !important;
    color: white !important;
    padding: 14px 18px !important;
    max-width: 80% !important;
}

/* Pill Input Bar */
.input-row {
    background-color: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 50px !important;
    padding: 6px 16px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03) !important;
    align-items: center !important;
    gap: 12px !important;
    margin-top: 20px !important;
}

.input-row > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.input-row textarea {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    resize: none !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    color: #1e293b !important;
    padding: 8px 0 !important;
}

/* Icons styling */
.icon-btn button {
    background: transparent !important;
    border: none !important;
    font-size: 22px !important;
    cursor: pointer !important;
    padding: 4px !important;
    box-shadow: none !important;
    transition: opacity 0.2s !important;
}
.icon-btn button:hover {
    opacity: 0.7 !important;
}

/* Clean audio recorder container styling */
.audio-mic {
    min-width: 160px !important;
}

/* Send Button style */
.send-btn button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    padding: 0 !important;
    border: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2) !important;
    transition: background-color 0.2s !important;
}
.send-btn button:hover {
    background-color: #1d4ed8 !important;
}

/* Disclaimer and Footer */
.disclaimer {
    text-align: center;
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 12px;
}
.footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
    padding-top: 20px;
    color: #64748b;
    font-size: 0.85rem;
}
.footer-links {
    display: flex;
    gap: 20px;
}
.footer-links a {
    color: #64748b;
    text-decoration: none;
}
.footer-links a:hover {
    color: #2563eb;
}
"""

with gr.Blocks() as demo:
    # Maintain user history and active context state
    session_state = gr.State(get_initial_state)

    with gr.Column(elem_classes="container"):
        
        # 1. Custom Header Navbar
        gr.HTML("""
        <div class="navbar">
            <div class="navbar-logo">DermaAI</div>
            <div class="navbar-links">
                <a href="#" class="navbar-link active">Consultation</a>
            </div>
        </div>
        """)

        # 2. Main Chat Panel Header
        gr.HTML("""
        <div class="chat-header">
            <div class="chat-title-wrapper">
                <span class="status-dot"></span>
                <h2 class="chat-title">Diagnostic Chat</h2>
            </div>
            <span class="analysis-badge">Active AI Analysis</span>
        </div>
        """)

        # 3. Conversational Chatbot Panel
        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": GREETING_TEXT}],
            label=None,
            show_label=False,
            min_height=400
        )

        # 4. Doctor voice response audio player
        doctor_audio = gr.Audio(
            label="Doctor's Voice Response",
            type="filepath",
            autoplay=True,
            visible=True
        )

        # Image Preview Block (hidden by default)
        image_preview = gr.Image(
            label="Attached Image",
            visible=False,
            height=100,
            width=100,
            interactive=False,
            container=False,
            show_label=False
        )

        # 5. Customized Pill Input Bar
        with gr.Row(elem_classes="input-row"):
            # Image upload button (Camera Icon)
            image_btn = gr.UploadButton(
                "📷", 
                file_types=["image"], 
                elem_classes="icon-btn"
            )
            
            # Voice recording button (Microphone Icon)
            voice_btn = gr.Audio(
                sources=["microphone"], 
                type="filepath",
                label=None, 
                show_label=False,
                container=False,
                elem_classes="audio-mic"
            )
            
            # Message input text bar
            text_msg = gr.Textbox(
                placeholder="Describe symptoms or ask a question...",
                label=None,
                show_label=False,
                container=False,
                scale=5
            )
            
            # Send message action button
            submit_btn = gr.Button("➔", elem_classes="send-btn")

        # 6. Bottom Disclaimer
        gr.HTML("""
        <div class="disclaimer">
            DermaAI is an AI tool and does not provide a definitive medical diagnosis. Please consult a board-certified dermatologist for clinical confirmation.
        </div>
        """)

        # 7. Page Footer
        gr.HTML("""
        <div class="footer" style="justify-content: center;">
            <div class="footer-links">
                <a href="#">Privacy Policy</a>
                <a href="#">Medical Disclaimer</a>
            </div>
        </div>
        """)

        # Wire handlers
        # Image upload trigger
        image_btn.upload(
            fn=on_image_upload,
            inputs=[image_btn, session_state],
            outputs=[image_preview, session_state]
        )

        # Submit trigger
        submit_btn.click(
            fn=handle_user_message,
            inputs=[text_msg, voice_btn, session_state],
            outputs=[chatbot, doctor_audio, text_msg, voice_btn, image_preview, session_state]
        )
        
        # Also wire textbox Enter submit
        text_msg.submit(
            fn=handle_user_message,
            inputs=[text_msg, voice_btn, session_state],
            outputs=[chatbot, doctor_audio, text_msg, voice_btn, image_preview, session_state]
        )

if __name__ == "__main__":
    demo.launch(share=False, theme=custom_theme, css=custom_css)
