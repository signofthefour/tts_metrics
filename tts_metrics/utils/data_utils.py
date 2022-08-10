## perform loading dataset
from random import sample
import librosa

def get_mfccs(filepath,
              sample_rate=22050,
              S=None,
              n_mfcc=14, # 13 + 1 overall 
              dct_type=2,
              norm='ortho',
              lifter=0,
              hop_length=1024):
    y, sr = librosa.load(filepath)
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate,\
        S=S, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter, hop_length=hop_length) # [B, n_mfccs, t]
    print(f"[!] mfccs max: {mfccs.max()}, mfccs min: {mfccs.min()}")
    return mfccs
    