import whisper
import sounddevice as sd
import numpy as np
import subprocess
import queue
import threading

MODEL = "base"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000

print("Loading Whisper model...")
model = whisper.load_model(MODEL)
print("Model loaded. Start speaking — text will appear in VS Code as you talk.")

audio_queue = queue.Queue()

def get_vscode_window():
    result = subprocess.run(['xdotool', 'search', '--name', 'Visual Studio Code'], capture_output=True, text=True)
    window_id = result.stdout.strip().split('\n')[0]
    return window_id

def type_to_vscode(text):
    text = text.strip()
    if not text:
        return
    window_id = get_vscode_window()
    if window_id:
        subprocess.run(['xdotool', 'windowfocus', '--sync', window_id])
        subprocess.run(['xdotool', 'type', '--window', window_id, '--clearmodifiers', text + ' '])

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def transcribe_loop():
    buffer = []
    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)
        if len(buffer) >= 8:
            audio_data = np.concatenate(buffer, axis=0).flatten().astype(np.float32)
            buffer = []
            result = model.transcribe(audio_data, fp16=False, language='en')
            text = result["text"].strip()
            if text:
                print(f">> {text}")
                type_to_vscode(text)

t = threading.Thread(target=transcribe_loop, daemon=True)
t.start()

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                    blocksize=BLOCK_SIZE, callback=audio_callback):
    print("Listening... (Ctrl+C to stop)")
    threading.Event().wait()
