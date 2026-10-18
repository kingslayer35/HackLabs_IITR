import argparse
from termmap.common.io import load_config,verify_frozen
from termmap.retrieval.cache import prepare_queries,build_lexical
ap=argparse.ArgumentParser();ap.add_argument("--profile",default="eval");ap.add_argument("--test",action="store_true")
args=ap.parse_args();cfg=load_config()
if args.test:verify_frozen(cfg)
prepare_queries(cfg,args.profile,args.test)
build_lexical(cfg,"eval_test" if args.test else args.profile)
