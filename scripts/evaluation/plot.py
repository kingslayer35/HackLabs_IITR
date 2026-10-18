"""Standalone aggregate risk/coverage figures, without clinical text."""
import json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"]="termmap13"
import matplotlib.pyplot as plt
from termmap.common.io import load_config,path_for

def main():
    cfg=load_config();runs=path_for(cfg,"run_dir");reports=path_for(cfg,"report_dir")
    for dataset in ["dev","test"]:
        fig,ax=plt.subplots(figsize=(7,4.5),layout="constrained");found=False
        for flow,color in zip(["F00","F10","F01","F11"],["#315a90","#b16d22","#408064","#87528c"]):
            path=runs/flow/f"{dataset}_metrics.json"
            if not path.exists():continue
            found=True;curve=json.loads(path.read_text())["risk_coverage"]
            ax.plot([p["coverage"]*100 for p in curve],[p["risk"]*100 for p in curve],label=flow,color=color,marker="o",markersize=4)
        if found:
            ax.set(xlabel="Rows attempted (%)",ylabel="Error rate on attempted rows (%)",title="Grouped out-of-fold dev" if dataset=="dev" else "Frozen sealed-test inference",xlim=(5,102),ylim=(0,100))
            ax.grid(alpha=.2);ax.legend(frameon=False,ncol=4)
            fig.savefig(reports/f"risk_coverage_{dataset}.png",dpi=180)
            fig.savefig(reports/f"risk_coverage_{dataset}.svg",metadata={"Date":None})
        plt.close(fig)

if __name__=="__main__":main()
