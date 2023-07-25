
import argparse


def para_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=60, type=int)

    args = parser.parse_args()

    return args

