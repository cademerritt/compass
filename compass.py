import whisper
import sounddevice as sd
import numpy as np
import subprocess
import queue
import threading

MODEL = "small"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000

print("Loading Whisper model...")
model = whisper.load_model(MODEL)
print("Model loaded. Listening... (Ctrl+C to stop)")

audio_queue = queue.Queue()

def type_to_cursor(text):
    text = text.strip()
    if not text:
        return
    subprocess.run(['xdotool', 'type', '--clearmodifiers', text + ' '])

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

SILENCE_THRESHOLD = 0.002

def transcribe_loop():
    import os
    buffer = []
    stop_count = 0
    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)
        if len(buffer) >= 8:
            audio_data = np.concatenate(buffer, axis=0).flatten().astype(np.float32)
            buffer = []
            if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
                stop_count = 0
                continue
            result = model.transcribe(audio_data, fp16=False, language='en')
            text = result["text"].strip()
            if text:
                print(f">> {text}")
                stop_count += text.lower().count("stop")
                if stop_count >= 3:
                    print("Stopping COMPASS.")
                    os.kill(os.getpid(), 9)
                type_to_cursor(text)

t = threading.Thread(target=transcribe_loop, daemon=True)
t.start()

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                    blocksize=BLOCK_SIZE, callback=audio_callback):
    threading.Event().wait()
