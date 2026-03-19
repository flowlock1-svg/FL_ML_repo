import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import csv

# ==============================
# LOAD MODEL
# ==============================

print("Loading YAMNet...")
model = hub.load("https://tfhub.dev/google/yamnet/1")
print("Model Loaded.\n")

# ==============================
# LOAD CLASS NAMES
# ==============================

class_map_path = model.class_map_path().numpy()

class_names = []
with open(class_map_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])

# ==============================
# SETTINGS
# ==============================

SAMPLE_RATE = 16000
DURATION = 1.0

DB_THRESHOLD = -50        # Adjust if needed
MIN_CONFIDENCE = 0.07

previous_event = None

# ==============================
# FUNCTION
# ==============================

def classify_audio():
    global previous_event

    # Record audio
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)
    sd.wait()

    audio = np.squeeze(audio)

    # ------------------------------
    # Compute dBFS
    # ------------------------------
    rms = np.sqrt(np.mean(audio**2))
    db = 20 * np.log10(rms + 1e-10)

    print(f"\nCurrent Level: {db:.2f} dBFS")

    # ------------------------------
    # Run YAMNet
    # ------------------------------
    scores, embeddings, spectrogram = model(audio)
    scores = scores.numpy()
    mean_scores = np.mean(scores, axis=0)

    top_index = np.argmax(mean_scores)
    top_label = class_names[top_index]
    top_conf = mean_scores[top_index]

    classified = top_conf > MIN_CONFIDENCE

    # ==============================
    # YOUR REQUIRED LOGIC
    # ==============================

    if db > DB_THRESHOLD:

        if classified:
            print("\n🚨 High Noise Detected!")
            print(f"⚠ Sound Classification: {top_label} ({top_conf*100:.2f}%)")
        else:
            print("\n🚨 High Noise Detected!")
            print("⚠ Noise detected but could not classify the sound.")

    else:
        # If noise is low, optionally print classification only
        if classified and top_label != previous_event:
            print(f"Detected Sound: {top_label} ({top_conf*100:.2f}%)")
            previous_event = top_label


# ==============================
# MAIN LOOP
# ==============================

print("Noise Monitor Running...")
print("Press CTRL+C to stop.\n")

try:
    while True:
        classify_audio()

except KeyboardInterrupt:
    print("\nStopped by user.")