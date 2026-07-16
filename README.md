# AI Medical Doctor Backend

A modular Python backend for the AI Medical Doctor assistant, initialized and managed using the `uv` tool.

## Requirements

- Python 3.14+
- `uv` package manager (executable via `python -m uv`)

## Setup

1. **Install dependencies and sync environment**:
   ```powershell
   python -m uv sync
   ```
   This will automatically set up the virtual environment (`.venv`) and install all packages.

2. **Add more dependencies**:
   ```powershell
   python -m uv add <package-name>
   ```

## Running the Application

### Option A: Gradio Web UI (Recommended)
To launch the interactive web interface:
```powershell
python -m uv run app.py
```
Then, open the provided local link (e.g., `http://127.0.0.1:7860`) in your web browser. You will be able to upload skin images, record/upload voice symptom descriptions, and view/hear the doctor's response.

### Option B: Terminal CLI (Voice-to-Voice Loop)
To run the automated console pipeline:
```powershell
python -m uv run main.py
```
This records from your microphone for 5 seconds (using `acne_sample.png` as the default test image), transcribes, queries the doctor LLM, and plays the doctor's audio response locally.
