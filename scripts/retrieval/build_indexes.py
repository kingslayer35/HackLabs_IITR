"""Rebuild disposable indexes from the portable assets; never train a model."""
import argparse
from termmap.common.io import load_config, path_for, seed_everything
from termmap.retrieval.cache import prepare_queries, build_lexical, encode_dense
from termmap.retrieval.asymmetric import base_model, encode_generation, encode_frozen_v2, retrieve_v2
from termmap.retrieval.dense import Encoder, require_allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--training-pools", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    require_allocation()
    seed_everything(13, True)
    if args.smoke:
        encoder = Encoder(base_model(cfg), 40, 8)
        assert encoder.encode(["hypertension", "chronic kidney disease"]).shape[0] == 2
        print("Original-index encoding smoke passed")
        return
    prepare_queries(cfg)
    build_lexical(cfg)
    encode_dense(cfg, base_model(cfg), "anchor")
    encode_generation(cfg, "C1")
    encode_frozen_v2(cfg, "C2_v2")
    if args.training_pools:
        retrieve_v2(cfg, "C1", datasets=("train", "alias", "dev", "probe"))
    else:
        retrieve_v2(cfg, "C1")
    retrieve_v2(cfg, "C2_v2")
    print("Indexes rebuilt with original aliases and per-system adapted queries; no model training")


if __name__ == "__main__":
    main()
