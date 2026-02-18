import os
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import neuron, functional

# -------- PARAMETERS --------
LIBRISPEECH_PATH = "C:\\nm_p1\\uploads\\LibriSpeech"
MAX_FILES = 300
MFCC_LENGTH = 100
N_MFCC = 13
BATCH_SIZE = 16
EPOCHS = 500
LEARNING_RATE = 1e-3
MODEL_PATH = 'snn_model.pth'
LOG_PATH = 'snn_training_log.json'
USE_SNN = True  # Set to False to use ReLU for debugging

# -------- FEATURE EXTRACTION --------
def extract_features_from_directory(base_dir, max_files=100):
    features = []
    labels = []
    count = 0
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.flac'):
                path = os.path.join(root, f)
                y, sr = librosa.load(path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
                features.append(mfcc)
                labels.append(os.path.basename(root))  # label = folder (speaker ID)
                count += 1
                if count >= max_files:
                    return features, labels
    return features, labels

def pad_features(x, length=MFCC_LENGTH):
    padded = np.zeros((length, N_MFCC))
    padded[:min(length, x.shape[0])] = x[:length]
    return padded

# -------- SNN MODEL --------
class SNNModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, output_size=10, use_snn=True):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = neuron.LIFNode() if use_snn else nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act2 = neuron.LIFNode() if use_snn else nn.ReLU()

    def forward(self, x):
        x = x.mean(dim=1)  # Temporal mean pooling
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x

# -------- TRAINING --------
def main():
    features, labels = extract_features_from_directory(LIBRISPEECH_PATH, MAX_FILES)
    
    if not features or len(set(labels)) < 2:
        return []

    X_padded = np.array([pad_features(f) for f in features])
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    X_tensor = torch.tensor(X_padded, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SNNModel(input_size=N_MFCC, output_size=len(le.classes_), use_snn=USE_SNN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    training_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in dataloader:
            if USE_SNN:
                functional.reset_net(model)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        training_history.append({
            "epoch": epoch + 1,
            "loss": round(total_loss, 4),
            "accuracy": round(accuracy, 2)
        })

    torch.save(model, MODEL_PATH)

    # -------- Save training log as JSON --------
    with open(LOG_PATH, 'w') as f:
        json.dump(training_history, f, indent=4)

    return training_history

if __name__ == '__main__':
    training_log = main()
