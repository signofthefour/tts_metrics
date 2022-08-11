""" Gross Pitch Error (GPE)
See: Nakatani, Tomohiro, Amano, Shigeaki, Irino, Toshio, Ishizuka, Kentaro, and Kondo, Tadahisa.
A method for fundamental frequency estimation and voicing decision: Application to infant utterances recorded in real acoustical environments.
Speech Communication, 50(3):203-214, 2008.
"""
import os
from dataclasses import dataclass

import json
import librosa
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from tts_metrics.base import BaseMetric

@dataclass
class GPEConfig:
    use_dtw: bool = True
    method: str = 'pyin'
    fmin: float = librosa.note_to_hz('C2')
    fmax: float = librosa.note_to_hz('C7')
    frame_length: int = 1024
    sampling_rate: int = 22050

class GPE(BaseMetric):
    def __init__(self, dataroot, data_mapper_path, metric_config:GPEConfig, name):
        super().__init__(dataroot, data_mapper_path, metric_config, name)
        self.metric_config = metric_config
        self.dataroot = dataroot
        self.pairs = json.load(open(data_mapper_path))

    
    def estimate_pitch(self, wavefile, method='pyin', normalize_mean=None,
                   normalize_std=None, n_formants=1):

        if self.metric_config.method == 'pyin':
            filepath = os.path.join(self.dataroot, wavefile)
            snd, sr = librosa.load(filepath, sr=self.metric_config.sampling_rate)
            (pitch, 
             voiced_flag,
             voiced_probs) = librosa.pyin(snd,
                                          fmin=self.metric_config.fmin,
                                          fmax=self.metric_config.fmax,
                                          frame_length=self.metric_config.frame_length)

            pitch = np.where(np.isnan(pitch), 0.0, pitch)
            
        else:
            print("[!] Current version do not support another method"
                  "\nPlease request the methods you want (your contribution are also welcome), many thanks!")
            raise ValueError

        return pitch, voiced_flag
    
    def compute_gpe(self, wavefile1, wavefile2, is_logging=False):
        # Compute pitch and voice flag of wavefiles
        p1, v1 = self.estimate_pitch(wavefile1)
        p2, v2 = self.estimate_pitch(wavefile2)
        
        if self.metric_config.use_dtw:
            distance, path = fastdtw(p1, p2, dist=euclidean)
            p1 = np.take(p1, [l[0] for l in path], axis=0)
            v1 = np.take(v1, [l[0] for l in path], axis=0)
            
            p2 = np.take(p2, [l[1] for l in path], axis=0)
            v2 = np.take(v2, [l[1] for l in path], axis=0)
        else:
            if len(p1) > len(p2):
                p1 = p1[: len(p2)]
                v1 = v1[: len(p2)]
            else:
                v2 = v2[: len(p1)]

        diff = np.abs(p1 - p2)
        diff = diff > 0.2 * p1
        
        both_voiced = v1.astype(np.int32) * v2.astype(np.int32)
        F0_err = diff.astype(np.int32) * both_voiced
        
        GPE = np.sum(F0_err) / np.sum(both_voiced)
        VDE = np.sum(np.not_equal(v1, v2).astype(np.int32)) / v1.shape[0]
        FFE = ((np.sum(F0_err) + np.sum(np.not_equal(v1, v2).astype(np.int32))) / v1.shape[0])

        if is_logging:
            print(f"[INFO] GPE rate between {wavefile1} & {wavefile2} is: {GPE}")
            print(f"[INFO] VDE rate between {wavefile1} & {wavefile2} is: {VDE}")
            print(f"[INFO] FFE rate between {wavefile1} & {wavefile2} is: {FFE}")
        return GPE, VDE, FFE, v1.shape[0]
        
    def compute(self):
        GPEs, VDEs, FFEs, total_frames = [], [], [], 0
        for ref_file, syn_file in self.pairs.items():
            GPE, VDE, FFE, num_frame = self.compute_gpe(ref_file, syn_file)
            GPEs.append(GPE)
            VDEs.append(VDE)
            FFEs.append(FFE)
            total_frames += num_frame
        
        GPE = sum(GPEs) / len(GPEs) * 100
        VDE = sum(VDEs) / len(VDEs) * 100
        FFE = sum(FFEs) / len(FFEs) * 100
        print(f"[\t\t\t---MEASUREMENT RESULT OVER {total_frames} frames---\t\t\t]")
        print(f"[INFO] average Gross Pitch Error (GPE): {GPE} %")
        print(f"[INFO] average Voicing Decision Error (VDE): {VDE} %")
        print(f"[INFO] average F0 Frame Error (FFE): {FFE} %")
        return bool