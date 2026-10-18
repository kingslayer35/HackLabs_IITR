from termmap.common.io import load_config
from termmap.retrieval.confirmation import confirm

if __name__ == "__main__":
    confirm(load_config())
