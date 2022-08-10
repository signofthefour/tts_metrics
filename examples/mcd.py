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
        type=bool,
        default=False,
        help="Flag to add Dynamic time warping phase before calculating MCD"
    )
    
    return parser.parse_args()

def compute_MCD(args, config):
    mcd_calculator = MCD(dataroot=args.dataroot, data_mapper_path=args.datapairs, metric_config=mcd_config, name="DTW-MCD")
    mcd_calculator.compute()
    
if __name__ == "__main__":
    args = parse_and_config()
    mcd_config = MCDConfig()
    compute_MCD(args, mcd_config)