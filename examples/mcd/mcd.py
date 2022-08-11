import argparse
import logging
import os
import yaml

from tts_metrics.mcd import MCD, MCDConfig

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
        "--K",
        type=int,
        default=24,
        help="Number of MFCCs coefficient to be extracted"
    )
    
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sampling rate of wavefile"
    )
    parser.add_argument(
        "--use_mfcc",
        type=int,
        default=0,
        help="Flag toggle using MFCCs or mel ceptral"
    )
    
    return parser.parse_args()

def compute_MCD(args, config):
    mcd_calculator = MCD(dataroot=args.dataroot,
                         data_mapper_path=args.datapairs,
                         metric_config=mcd_config,
                         use_mfcc=bool(args.use_mfcc),
                         name="DTW-MCD")
    mcd_calculator.compute()
    
if __name__ == "__main__":
    args = parse_and_config()
    mcd_config = MCDConfig(use_dtw=bool(args.use_dtw), K = args.K)
    print(f'[!] MCD using DTW is set to {mcd_config.use_dtw}')
    compute_MCD(args, mcd_config)