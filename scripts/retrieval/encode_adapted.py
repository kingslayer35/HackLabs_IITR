"""Inference only: consume a finalized v2 manifest without touching training."""
import argparse
from termmap.common.io import load_config
from termmap.retrieval.asymmetric import encode_frozen_v2,retrieve_v2


def main():
    p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");p.add_argument("--retrieve",action="store_true")
    p.add_argument("--profile",choices=["eval","eval_test","final"],default="eval");p.add_argument("--run",default="C2_v2");a=p.parse_args();cfg=load_config()
    if a.retrieve:retrieve_v2(cfg,a.run,a.profile,{"eval_test":("test",),"final":("all","alias"),"eval":("dev","probe")}[a.profile])
    else:encode_frozen_v2(cfg,a.run,a.profile,a.smoke)


if __name__=="__main__":main()
