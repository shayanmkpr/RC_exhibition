import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import IPython.display as ipd
import pandas as pd


def process_ESC(ESC_path , ESC_CSV , n_fft = 1024 , s_r = 20e3):   #the sammpling rate must be chagned due to the larger file and wider classes
    mfcc_features = []
    mel_spectrograms = []
    audio_files = []
    labels = []

    csv_path = ESC_CSV
    data_path = ESC_path
    
    metadata = pd.read_csv(csv_path)
    
    for index, row in metadata.iterrows():
        file = row['filename']
        label = row['target']
        file_path = os.path.join(data_path, file)
        
        if file.endswith('.wav'):
            audio, sample_rate = librosa.load(file_path, sr=s_r)
            audio_files.append(audio)

            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=12, n_fft=n_fft)
            mfcc_features.append(mfcc.T)
            
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft)
            mel_spectrograms.append(mel_spec.T) 
            labels.append(label)
    
    return mfcc_features, mel_spectrograms, audio_files, labels, sample_rate


def audio_to_tensor(audio_files):
    return [torch.tensor(audio, dtype=torch.float32) for audio in audio_files]

def process_audio(data_path, n_fft=1024 , s_r = 20e3):
    mfcc_features = []
    mel_spectrograms = []
    audio_files = []
    labels = []
    for file in os.listdir(data_path):
        if file.endswith('.wav'):
            label = int(file.split('_')[0])
            file_path = os.path.join(data_path, file)
            audio, sample_rate = librosa.load(file_path, sr=s_r)
            audio_files.append(audio)

            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=12, n_fft=n_fft)
            mfcc_features.append(mfcc.T)
            
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft)
            mel_spectrograms.append(mel_spec.T) 
            labels.append(label)
        else:
            audio = librosa.load(file_path)
            mel_spec = librosa.feature.melspectrogram(audio)
            print("testing?")
            return mel_spec
    return mfcc_features, mel_spectrograms, audio_files, labels, sample_rate


def mel_to_audio(mel_spectrogram, sr):

    mel_inverted = librosa.feature.inverse.mel_to_stft(mel_spectrogram.t(), sr=sr)

    audio_reconstructed = librosa.griffinlim(mel_inverted, n_iter=512)

    return audio_reconstructed

def load_data(data_folder , save_path = "" , ratio=0.2 , n_fft=1024 , s_r = 20e3 , ESC = False):

    #ESC default sampling rate is 40k

    if ESC != True:
        data_path= os.path.join(data_folder ,'\audio')
        mfcc_features, mel_spectrograms, amplitudes, labels, sample_rate = process_audio(data_path , n_fft , s_r)

    else:
        data_path = os.path.join(data_folder ,'\audio')
        mfcc_features, mel_spectrograms, amplitudes, labels, sample_rate = process_ESC(data_path,
             ESC_CSV= os.path.join(data_folder ,'\meta\esc50.csv'), n_fft = n_fft , s_r = s_r)



    combined_data = list(zip(mfcc_features, mel_spectrograms, amplitudes, labels))
    random.shuffle(combined_data)

    mfcc_features, mel_spectrograms, amplitudes, labels = zip(*combined_data)

    amp_tensors = audio_to_tensor(amplitudes)
    mfcc_tensors = audio_to_tensor(mfcc_features)
    mel_tensors = audio_to_tensor(mel_spectrograms)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    mfcc_train, mfcc_test, mel_train, mel_test, amp_train, amp_test, labels_train, labels_test = train_test_split(
        mfcc_tensors, mel_tensors, amp_tensors, labels_tensor, test_size= ratio, random_state=42)
    
    
    if ESC != True:
        name = f'prcessed_data_{s_r/1000}_{ratio}.pt'
    else:
        name = f'prcessed_data_{s_r/1000}_{ratio}_{ESC}.pt'

    save_path = os.path.join(save_path, name)

    torch.save({
        'mfcc_train': mfcc_train,
        'mfcc_test': mfcc_test,
        'mel_train': mel_train,
        'mel_test': mel_test,
        'amp_train': amp_train,
        'amp_test': amp_test,
        'labels_train': labels_train,
        'labels_test': labels_test,

        'mel_tensors':mel_tensors,
        'mfcc_tensors':mfcc_tensors,
        'amp_tensors':amp_tensors,
        'labels_tensor':labels_tensor
    }, save_path)

    print(f"Data ESD = {ESC} with sample rate of {s_r} and train test ratio of {ratio} was saved in {save_path}")