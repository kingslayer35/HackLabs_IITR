import argparse
from termmap.common.io import load_config
from termmap.pretraining.trainer import fit
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true")
    fit(load_config(),ap.parse_args().smoke)


if __name__=="__main__":main()
