import argparse
from termmap.common.io import load_config
from termmap.posttraining.trainer import train_reranker

def main():
    p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");p.add_argument("--final",action="store_true")
    a=p.parse_args();train_reranker(load_config(),a.smoke,a.final)

if __name__=="__main__":main()
