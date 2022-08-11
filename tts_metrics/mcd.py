import os
import math
from dataclasses import dataclass

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
from tqdm import tqdm

from tts_metrics.base import BaseMetric
from tts_metrics.utils.data_utils import get_mfccs, wav_to_mcep

log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)

def log_spec_dB_dist(x, y):
    diff = x - y
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))


@dataclass
class MCDConfig:
    use_dtw: bool = True
    K: int = 14
    sampling_rate: int = 22050

class MCD(BasedMetric):
    """Model to calculate MCD
    """
    def __init__(self, dataroot, data_mapper_path, metric_config, use_mfcc=True, name=None, mfccs_outdir='mfccs', mceps_outdir='mceps'):
        super().__init__(dataroot, data_mapper_path, metric_config, name)
        self.config = metric_config
        self.use_dtw = metric_config.use_dtw # Incase you don't have a ensurement show that ref and syn have same length, you should enable this flag
        self.data_pairs = self.data_mapper
        self.mfccs_outdir = mfccs_outdir
        self.mceps_outdir = mceps_outdir
        self.K = metric_config.K
        self.sampling_rate = metric_config.sampling_rate
        self.use_mfcc = use_mfcc
        
        if self.use_mfcc:
            self.mfccs_pairs = self.generate_mfcc() 
            
    def generate_mfcc(self):
        """Generate and store file to folder
            for further works, such as visualization
        """
        os.makedirs(os.path.join(self.dataroot, self.mfccs_outdir), exist_ok=True)
        
        mfccs_pairs = {}
        for ref_file in self.data_pairs:
            ref_filename = os.path.splitext(os.path.basename(ref_file))[0] + '-mfcc.npy'
            mfccs =  get_mfccs(os.path.join(self.dataroot, ref_file), n_mfcc=self.K)
            np.save(os.path.join(self.dataroot, self.mfccs_outdir, ref_filename), mfccs)
            
            syn_filename = os.path.splitext(os.path.basename(self.data_pairs[ref_file]))[0] + '-mfcc.npy'
            mfccs = get_mfccs(os.path.join(self.dataroot,self.data_pairs[ref_file]), n_mfcc=self.K)
            np.save(os.path.join(self.dataroot, self.mfccs_outdir, syn_filename), mfccs)
            
            mfccs_pairs.update({ref_filename: syn_filename})
        
        return mfccs_pairs
    
    def mcd_mfccs(self, mfcc_1, mfcc_2, info=None):
        """calculate mcd between two input mel cepstral (MFCC is same)
        Actually, There something not work well there, you should use the mcep install util this comment is removed

        Args:
            mfcc_1 (np.ndarray):
            mel_2 (np.ndarray): 
        """
        num_frame = 0
        if self.use_dtw:
            distance, path = fastdtw(mfcc_1.T[:, 1:], mfcc_2.T[:, 1:], dist=log_spec_dB_dist)
            mfcc_1 = np.take(mfcc_1, [l[0] for l in path], axis=1)
            mfcc_2 = np.take(mfcc_2, [l[1] for l in path], axis=1)
            num_frame = len(path)
        else:
            if mfcc_1.shape[1] < mfcc_2.shape[1]:
                mfcc_2 = mfcc_2[:, :mfcc_1.shape[1]]
            else:
                mfcc_1 = mfcc_1[:, :mfcc_2.shape[1]]
            num_frame = mfcc_1.shape[1]
            distance = mfcc_1 - mfcc_2
            distance = np.inner(distance[1:, ...], distance[1:, ...])
            distance = np.sum(distance, axis=0, keepdims=False)
        distance = distance / num_frame
        return distance, num_frame
    
    def mcd_mcep(self, mcep_1, mcep_2, info=None):
        """Actually, There something not work well there, you should use the mcep install util this comment is removed

        Args:
            mcep_1 (ndarray): mel cepstral
            mcep_2 (ndarrat): synthesized cepstral
            info (str, optional): info to printout when. Defaults to None.

        Returns:
            _type_: _description_
        """
        if self.use_dtw:
            distance, path = fastdtw(mcep_1[:, 1:], mcep_2[:, 1:], dist=log_spec_dB_dist)
            mcep_1 = np.take(mcep_1[:, 1:], [l[0] for l in path], axis=0)
            mcep_2 = np.take(mcep_2[:, 1:], [l[1] for l in path], axis=0)
            num_frames = len(path)
        else:
            if mcep_1.shape[0] > mcep_2.shape[0]:
                mcep_1 = mcep_1[:mcep_2.shape[0], 1:]
            else:
                mcep_2 = mcep_2[:mcep_1.shape[0], 1:]
            num_frames = mcep_1.shape[0]
            distance = mcep_1 - mcep_2
            distance = np.inner(distance, distance)
            distance = np.sum(distance, axis=1, keepdims=False)
            distance = np.sum(distance, axis=0, keepdims=False)
        distance = distance / num_frames
        return distance, num_frames
                   
    def compute_mcd(self):
        if self.use_mfcc:
            distances = []
            total_frames = 0
            for ref_mel_file, syn_mel_file in tqdm(self.mfccs_pairs.items()):
                mel_1 = np.load(os.path.join(self.dataroot, self.mfccs_outdir, ref_mel_file))
                mel_2 = np.load(os.path.join(self.dataroot, self.mfccs_outdir, syn_mel_file))
                distance, num_frames = self.mcd_mfccs(mel_1, mel_2, f"MCD between {ref_mel_file} & {syn_mel_file} is")
                distances.append(distance)
                total_frames += num_frames  
            print(f"[INFO] MCD using MFCC return average value {sum(distances)/len(distances)} over {total_frames} frames")
            return distance, total_frames
        else:
            distances = []
            total_frames = 0
            for ref_file, syn_file in tqdm(self.data_pairs.items()):
                mcep_1 = wav_to_mcep(wavfile=os.path.join(self.dataroot, ref_file), 
                                     target_directory=os.path.join(self.dataroot, self.mceps_outdir))
                mcep_2 = wav_to_mcep(wavfile=os.path.join(self.dataroot, syn_file),
                                     target_directory=os.path.join(self.dataroot, self.mceps_outdir))
                distance, num_frames = self.mcd_mcep(mcep_1, mcep_2)
                distances.append(distance)
                total_frames += num_frames
                print(f'[INFOR] MCD between {ref_file} and {syn_file} is: {distance}')
            print(f"[INFO] MCD using MCEP return average value {sum(distances)/len(distances)} over {total_frames} frames")
            print(f"[INFO] MCD using MCEP return per pair value is {distances} over {total_frames} frames")
        return None
    
    def compute(self):
        return self.compute_mcd()