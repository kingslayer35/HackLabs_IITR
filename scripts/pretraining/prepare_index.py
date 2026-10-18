"""Build original SapBERT inputs for the restored pinned training generation."""
from termmap.common.io import load_config
from termmap.pretraining.trainer import pinned
from termmap.retrieval.cache import build_lexical, encode_dense
from termmap.retrieval.asymmetric import base_model
from scripts.retrieval.baseline import retrieve_datasets


def main():
    cfg=pinned(load_config())
    build_lexical(cfg)
    encode_dense(cfg,base_model(cfg),"C1")
    retrieve_datasets(cfg,"C1",datasets=("dev","probe"))


if __name__=="__main__":main()
