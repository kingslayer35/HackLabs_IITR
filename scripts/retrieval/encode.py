import argparse
from termmap.common.io import load_config
from termmap.retrieval.asymmetric import encode_generation, encode_frozen_v2, retrieve_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--profile", choices=["eval", "final"], default="eval")
    args = parser.parse_args()
    cfg = load_config()
    if args.retrieve:
        if args.profile == "final":
            # All-label LoRA still trains on original-SapBERT candidate pools.
            retrieve_v2(cfg, "final_base", "final", ("all", "alias"))
        else:
            for run in ["C1", "C2_v2"]:
                retrieve_v2(cfg, run)
    else:
        base, adapted = (("final_base", "final_adapted") if args.profile == "final" else ("C1", "C2_v2"))
        encode_generation(cfg, base, args.profile, smoke=args.smoke)
        encode_frozen_v2(cfg, adapted, args.profile, smoke=args.smoke)


if __name__ == "__main__":
    main()
