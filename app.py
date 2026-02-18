import os
import json
import wave
import logging

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from pydub.utils import which

import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter/threading errors
import matplotlib.pyplot as plt

import torch
from model.model import SNNModel
from spikingjelly.activation_based import functional

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'flac', 'mp3', 'ogg'}
MODEL_PATH = 'model/snn_model.pth'
VOSK_MODEL_PATH = os.path.abspath("model/vosk-model-small-en-us-0.15")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set ffmpeg path for pydub
ffmpeg_bin_path = "C:\\SEM 5\\sangeetha\\NM_01\\ffmpeg-7.1.1-essentials_build\\ffmpeg-7.1.1-essentials_build\\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_bin_path
AudioSegment.converter = which("ffmpeg") or os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
AudioSegment.ffprobe = which("ffprobe") or os.path.join(ffmpeg_bin_path, "ffprobe.exe")

# -----------------------------------------------------------
# Initialize Flask app and models
# -----------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load Vosk model (only once)
vosk_model = Model(VOSK_MODEL_PATH)

# Load SNN model
snn_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
snn_model.eval()

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_mfcc(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T
    return mfccs

def plot_mfcc(mfccs, filename):
    plot_path = os.path.join('static', secure_filename(filename) + '_mfcc.png')
    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs.T, aspect='auto', origin='lower')
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def recognize_speech(file_path):
    try:
        wf = wave.open(file_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            return "Audio file must be WAV format mono PCM."

        recognizer = KaldiRecognizer(vosk_model, wf.getframerate())

        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                results.append(result.get("text", ""))

        final_result = json.loads(recognizer.FinalResult())
        results.append(final_result.get("text", ""))

        return " ".join(results).strip() or "No speech detected."

    except Exception as e:
        logging.error("Error in recognize_speech: %s", str(e))
        return "An error occurred while processing the audio."

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in request.'

        file = request.files['file']
        if file.filename == '':
            return 'No file selected.'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Convert non-wav to wav
            if filename.lower().endswith(('.flac', '.mp3', '.ogg')):
                sound = AudioSegment.from_file(filepath)
                filepath_wav = filepath.rsplit('.', 1)[0] + '.wav'
                sound.export(filepath_wav, format='wav')
                filepath = filepath_wav

            # Extract MFCC and save plot
            mfcc_features = extract_mfcc(filepath)
            plot_mfcc(mfcc_features, filename)

            # Recognize speech
            recognized_text = recognize_speech(filepath)

            # Prepare MFCC input for SNN model
            mfcc_padded = np.zeros((100, 13))
            mfcc_trimmed = mfcc_features[:100]
            mfcc_padded[:mfcc_trimmed.shape[0], :] = mfcc_trimmed
            input_tensor = torch.tensor(mfcc_padded, dtype=torch.float32).unsqueeze(0)

            # Predict speaker
            functional.reset_net(snn_model)
            with torch.no_grad():
                output = snn_model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            return render_template('index.html',
                                   mfcc=mfcc_features.tolist(),
                                   img_filename=filename + '_mfcc.png',
                                   text=recognized_text,
                                   prediction=f"Predicted Speaker ID: {predicted_class}")

        return 'Invalid file format.'

    return render_template('index.html')

# -----------------------------------------------------------
# Run app
# -----------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
