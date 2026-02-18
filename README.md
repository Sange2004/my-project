# my-project
This project is about recognizing speech from audio files and identifying the speaker using Spiking Neural Networks. It uses MFCC features and a Flask web application for user interaction.

# Speech Recognition and Speaker Identification System Using Spiking Neural Networks (SNN)

## Project Overview
This project is designed to recognize speech from audio files and identify the speaker.  
The system takes an audio file as input, converts speech into text, and predicts the speaker using a Spiking Neural Network (SNN).

A Flask web application is used to upload audio files and display the results.

---

## Objectives
- To recognize speech from audio files
- To extract MFCC features from speech signals
- To identify the speaker using a Spiking Neural Network
- To visualize MFCC features
- To provide a simple web interface using Flask

---

## Technologies Used
- Python  
- Flask  
- Vosk (Speech Recognition)  
- Librosa (MFCC Feature Extraction)  
- PyTorch  
- SpikingJelly  
- Pydub and FFmpeg  
- HTML and CSS  

## Project Structure 
my-project/
│
├── app.py                     # Main Flask application
│
├── model/                     # Trained models
│   └── snn_model.pth           # Saved trained SNN model
│
├── templates/                 # HTML files
│   └── index.html
│
├── static/                    # MFCC images and static files
├── uploads/                   # Uploaded audio files
│
├── train_ann_model.py          # ANN training script
├── train_rnn_model.py          # RNN training script
├── train_lstm_model.py         # LSTM training script
├── train_snn_model.py          # SNN training script (ADD THIS)


## Input
- Audio files in WAV, MP3, FLAC, or OGG format

## Output
- Recognized speech text
- MFCC feature visualization
- Predicted speaker ID

## System Workflow
1. User uploads an audio file
2. Audio is converted to WAV format if required
3. MFCC features are extracted from the audio
4. MFCC features are visualized as an image
5. Speech is recognized using the Vosk model
6. MFCC features are passed to the SNN model
7. The system predicts the speaker ID
8. Results are displayed on the web page

## How to Run the Project
1. Install Python (version 3.8 or above)
2. Install required libraries:
3. Make sure FFmpeg is installed and configured
4. Run the application:
5. Open the browser and go to:  http://127.0.0.1:5000/


## Key Features
- Offline speech recognition
- Speaker identification using Spiking Neural Networks
- MFCC feature extraction and visualization
- Simple Flask-based user interface


## Academic Purpose
This project is developed for academic and learning purposes to demonstrate the use of Spiking Neural Networks in speech recognition and speaker identification.

## Author
Sangeetha S

## License
This project is intended for educational use only.



