import argparse

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="ECG Model Trainer")
    parser.add_argument("--model", type=str, choices=["autoencoder", "lstm"], default="autoencoder", help="Type of model to train (autoencoder or lstm)")
    args = parser.parse_args()
    return args

