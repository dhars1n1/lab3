"""
Q1 (a) - Record the utterance:  "My name is <YOUR NAME>"
          (spoken continuously, with a short pause after each word)

Requirements: fs = 16000 Hz, bit-depth = 16, mono, raw .wav

Run this on your OWN machine with a microphone attached
(it will not work inside a sandboxed / headless environment).

    pip install sounddevice soundfile numpy
"""

import sounddevice as sd
import soundfile as sf

FS = 16000          # sampling frequency required by the assignment
DURATION = 4.0       # seconds - adjust to comfortably fit your name
CHANNELS = 1
BIT_DEPTH_SUBTYPE = "PCM_16"   # 16-bit PCM  -> raw .wav
OUT_PATH = "my_name_is_utterance.wav"


def record_and_save(out_path=OUT_PATH, duration=DURATION, fs=FS):
    print(f"Recording for {duration:.1f} s at {fs} Hz ... speak now:")
    print('  "My name is ... <pause> ... <your name>"')
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=CHANNELS, dtype="int16")
    sd.wait()
    sf.write(out_path, audio, fs, subtype=BIT_DEPTH_SUBTYPE)
    print(f"Saved -> {out_path}  (fs={fs} Hz, 16-bit PCM, mono)")


if __name__ == "__main__":
    record_and_save()
