from dis import dis
import os
from dataclasses import dataclass
from turtle import distance

from fastdtw import fastdtw
import numpy as np
import librosa

from tts_metrics.base import BasedMetric
from tts_metrics.utils.data_utils import get_mfccs

_logdb_const = 10.0 / np.log(10.0) * np.log(2.0)

@dataclass
class MCDConfig:
    use_dtw: bool = True

class MCD(BasedMetric):
    """Model to calculate MCD
    """
    def __init__(self, dataroot, data_mapper_path, metric_config, name=None, mfccs_outdir='mfccs'):
        super().__init__(dataroot, data_mapper_path, metric_config, name)
        self.config = metric_config
        self.use_dtw = metric_config.use_dtw # Incase you don't have a ensurement show that ref and syn have same length, you should enable this flag
        self.data_pairs = self.data_mapper
        self.mfccs_outdir = mfccs_outdir

        self.mfccs_pairs = self.generate_mfcc() 
            
    def generate_mfcc(self):
        """Generate and store file to folder
            for further works, such as visualization
        """
        os.makedirs(os.path.join(self.dataroot, self.mfccs_outdir), exist_ok=True)
        
        mfccs_pairs = {}
        for ref_file in self.data_pairs:
            ref_filename = os.path.splitext(os.path.basename(ref_file))[0] + '-mfcc.npy'
            np.save(os.path.join(self.dataroot, self.mfccs_outdir, ref_filename),
                    get_mfccs(os.path.join(self.dataroot, ref_file)))
            
            syn_filename = os.path.splitext(os.path.basename(self.data_pairs[ref_file]))[0] + '-mfcc.npy'
            np.save(os.path.join(self.dataroot, self.mfccs_outdir, syn_filename),
                    get_mfccs(os.path.join(self.dataroot,self.data_pairs[ref_file])))
            
            mfccs_pairs.update({ref_filename: syn_filename})
        
        return mfccs_pairs
    
    def mcd(self, mel_1, mel_2):
        """calculate mcd between two input mel cepstral (MFCC is same)

        Args:
            mel_1 (np.ndarray):
            mel_2 (np.ndarray): 
        """
        total_frame = 0
        if self.use_dtw:
            distance, path = fastdtw(mel_1.T, mel_2.T)
            mel_1 = np.take(mel_1, [l[0] for l in path], axis=1)
            mel_2 = np.take(mel_2, [l[1] for l in path], axis=1)
            total_frame = len(path)
        
        distance = mel_1 - mel_2
        distance = distance[1:, ...] * distance[1:, ...]
        print(distance.shape[0])
        distance = np.sum(distance, axis=0, keepdims=False)
        distance = np.sqrt(distance)
        distance = np.sum(distance, axis=0, keepdims=False) / total_frame
        print(distance)
        return distance
                   
    def compute_mcd(self):
        if self.use_dtw:
            pass
        for ref_mel_file, syn_mel_file in self.mfccs_pairs.items():
            mel_1 = np.load(os.path.join(self.dataroot, self.mfccs_outdir, ref_mel_file))
            mel_2 = np.load(os.path.join(self.dataroot, self.mfccs_outdir, syn_mel_file))
            self.mcd(mel_1, mel_2)
        return None
    
    def compute(self):
        return self.compute_mcd()