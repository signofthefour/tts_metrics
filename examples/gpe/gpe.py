import argparse
import logging
import os
import yaml

from tts_metrics.gpe import GPE, GPEConfig
import librosa

def parse_and_config():
    """Parse arguments and set confirguration parameters.
    """
    parser = argparse.ArgumentParser(
        description="Mel ceptral distortion measurement"
        "(see detail in metrics/mcd.py)"
    )
    
    parser.add_argument(
        "--dataroot",
        default="./data/",
        help="Root dir of pairs exist in JSON",
    )
    parser.add_argument(
        "--datapairs",
        default="./data/data_pairs.json",
        help="path to JSON file contains information of pairs",
    )
    
    parser.add_argument(
        "--use_dtw",
        type=int,
        default=0,
        help="Flag to add Dynamic time warping phase before calculating MCD. 1 to turn on and 0 for vice versa"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default='pyin',
        help="Number of MFCCs coefficient to be extracted"
    )
    
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sampling rate of wavefile"
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=librosa.note_to_hz('C2'),
        help="Minimum frequency to be estimated"
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=librosa.note_to_hz('C7'),
        help="Maximum frequency to be estimated"
    )
    parser.add_argument(
        "--frame_length",
        type=float,
        default=1024,
        help="Window size in another word"
    )
    
    
    return parser.parse_args()

def compute_GPE(args, config):
    gpe = GPE(dataroot=args.dataroot,
                         data_mapper_path=args.datapairs,
                         metric_config=config,
                         name="DTW-GPE")
    gpe.compute()
    
if __name__ == "__main__":
    args = parse_and_config()
    config = GPEConfig(use_dtw=bool(args.use_dtw),
                           sampling_rate=args.sr,
                           fmin=args.fmin,
                           fmax=args.fmax,
                           frame_length=args.frame_length)
    print(f'[!] GPE using DTW is set to {config.use_dtw}')
    compute_GPE(args, config)