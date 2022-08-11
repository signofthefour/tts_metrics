## perform loading dataset
import os


import pyworld
import pysptk
import librosa
import numpy as np

def get_mfccs(filepath=None,
              y=None,
              sample_rate=22050,
              S=None,
              n_mfcc=14, # 13 + 1 overall 
              dct_type=2,
              norm='ortho',
              lifter=0,
              hop_length=1024):
    if y == None:
        y, sr = librosa.load(filepath)
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate,\
            S=S, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter, hop_length=hop_length) # [B, n_mfccs, t]
        return mfccs
    else:
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate,\
            S=S, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter, hop_length=hop_length) # [B, n_mfccs, t]
        return mfccs
    
def  wav_to_mcep(wavfile, target_directory, alpha=0.65, fft_size=512, mcep_size=24, SAMPLING_RATE=22050, FRAME_PERIOD=5.0):
    # make relevant directories
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    loaded_wav, _ = librosa.load(wavfile, sr=SAMPLING_RATE)

    # Use WORLD vocoder to spectral envelope
    _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=SAMPLING_RATE,
                                   frame_period=FRAME_PERIOD, fft_size=fft_size)

    # Extract MCEP features
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                           etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    fname = os.path.basename(wavfile).split('.')[0] + '-mcep'
    np.save(os.path.join(target_directory, fname + '.npy'),mgc, allow_pickle=False)
    return mgc