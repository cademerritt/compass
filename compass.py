import whisper
import sounddevice as sd
import numpy as np
import subprocess
import queue
import threading
import os
import time
from pynput import keyboard

MODEL = "small"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000
SILENCE_THRESHOLD = 0.008

# Screen centers (from xrandr)
SCREENS = {
    1: (960,  806),   # DP-2
    2: (2880, 600),   # DP-4
    3: (4800, 797),   # HDMI-0
}

print("Loading Whisper model...")
model = whisper.load_model(MODEL)
print("Model loaded. Listening... (Ctrl+C to stop)")
print("Press F8 then speak a command. It executes after the next transcription.")

audio_queue = queue.Queue()
next_is_command = False
last_f8 = 0

# --- Ripple effect ---
def show_ripple(x, y):
    script = f"""
import tkinter as tk
root = tk.Tk()
root.overrideredirect(True)
root.attributes('-topmost', True)
root.configure(bg='black')
root.geometry('200x200+{x-100}+{y-100}')
canvas = tk.Canvas(root, width=200, height=200, bg='black', highlightthickness=0)
canvas.pack()
step = [0]
def animate():
    canvas.delete('all')
    for i in range(3):
        r = (step[0] * 12 + i * 35) % 110
        if r > 0:
            canvas.create_oval(100-r, 100-r, 100+r, 100+r, outline='cyan', width=2)
    step[0] += 1
    if step[0] < 12:
        root.after(50, animate)
    else:
        root.destroy()
animate()
root.mainloop()
"""
    subprocess.Popen(['python3', '-c', script])

# --- Screen movement ---
def move_to_screen(n):
    if n in SCREENS:
        cx, cy = SCREENS[n]
        subprocess.run(['xdotool', 'mousemove', str(cx), str(cy)])
        show_ripple(cx, cy)
        print(f"Moved to screen {n}")

# --- Command parsing ---
def parse_command(text):
    t = text.lower()
    if any(w in t for w in ['screen one', 'screen 1', 'green one', 'screen won', 'green 1']):
        move_to_screen(1)
    elif any(w in t for w in ['screen two', 'screen 2', 'screen too', 'screen to', 'green two', 'green too', 'green 2']):
        move_to_screen(2)
    elif any(w in t for w in ['screen three', 'screen 3', 'screen free', 'screen tree', 'green three', 'green free', 'green 3']):
        move_to_screen(3)
    else:
        print(f"[CMD] No match for: {text}")

# --- F8 key listener (with debounce) ---
def on_press(key):
    global next_is_command, last_f8
    if key == keyboard.Key.f8:
        now = time.time()
        if now - last_f8 > 0.5:
            last_f8 = now
            next_is_command = True
            print("[CMD] Ready — speak your command now.")

listener = keyboard.Listener(on_press=on_press)
listener.start()

# --- Audio callback ---
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# --- Main transcription loop ---
def transcribe_loop():
    global next_is_command
    buffer = []
    stop_count = 0
    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)
        if len(buffer) >= 4:
            audio_data = np.concatenate(buffer, axis=0).flatten().astype(np.float32)
            buffer = []
            if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
                stop_count = 0
                continue
            result = model.transcribe(audio_data, fp16=False, language='en')
            text = result["text"].strip()
            if text:
                if next_is_command:
                    next_is_command = False
                    print(f"[CMD] >> {text}")
                    parse_command(text)
                else:
                    print(f">> {text}")
                    stop_count += text.lower().count("stop")
                    if stop_count >= 3:
                        print("Stopping COMPASS.")
                        os.kill(os.getpid(), 9)
                    type_to_cursor(text)

def type_to_cursor(text):
    text = text.strip()
    if not text:
        return
    subprocess.run(['xdotool', 'type', '--clearmodifiers', text + ' '])

t = threading.Thread(target=transcribe_loop, daemon=True)
t.start()

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                    blocksize=BLOCK_SIZE, callback=audio_callback):
    threading.Event().wait()
