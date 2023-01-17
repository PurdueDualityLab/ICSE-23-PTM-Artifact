import json
import os
from pprint import pprint

import matplotlib.pyplot as plt


def detection(ax):
    """manual compare detection models"""

    results = "/Users/matthewhyatt/cs/.datasets/COCOdataset2017/results"
    files = [f for f in os.listdir(results) if ".txt" in f]
    data = {}

    claims = {
        "facebook_detr-resnet-50": 0.420,
        "hustvl_yolos-tiny": 0.287,
        "facebook_detr-resnet-101": 0.435,
        "hustvl_yolos-small": 0.361,
        "hustvl_yolos-base": 0.420,
        "facebook_detr-resnet-101-dc5": 0.449,
        "nickmuchi_yolos-small-finetuned-masks": None,  # None
        "facebook_detr-resnet-50-dc5": 0.433,
        "hustvl_yolos-small-300": 0.361,  # None
        "hustvl_yolos-small-dwr": 0.376,
        "SamMorgan_yolo_v4_tflite": None,  # None
    }

    for f in files:

        f = os.path.join(results, f)
        with open(f, "r") as file:
            ap = file.readlines()[0].strip().split(" ")[-1]

        f = f.split("/")[-1].split(".")[0]
        data[f] = ap

    x, y = [], []
    for k, v in data.items():
        print(k)
        print(v, claims[k], "\n")

        x.append(k)
        y.append(100 * abs(float(v) - float(claims[k])))

    bars = ax.barh(x, y)
    ax.set_xticks([2,4],["2%","4%"])

    for yi,bar  in zip([round(yi,2) for yi in y],bars):
        h, w = bar.get_height(), bar.get_width()
        # w = 0-plt.gcf().get_figwidth()
        x,y = bar.get_x(), bar.get_y()
        ax.text(x+w, y+h/2, f'{yi}%', ha="left", va="center", fontsize=12)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(xlabel="Discrepancy")

    return ax
    # pprint(data)


def classification(ax):
    """manual compare classification models"""

    results_path = "/Users/matthewhyatt/cs/.datasets/IMAGENET/results"
    distil = "deit-base-distilled-patch16-224"
    models = [m for m in os.listdir(results_path) if "deit" in m and not distil in m]

    accuracy = []
    for m in models:
        with open(os.path.join(results_path, m), "r") as file:
            a = json.load(file)["accuracy"]
            accuracy.append(a)

    models = [m.split(".")[0].replace("_", "/") for m in models]

    results = {k: v for k, v in zip(models, accuracy)}
    results = {k: v["top1"] for k, v in results.items()}

    claims = {
        "facebook/deit-tiny-patch16-224": {"top1": 0.722, "top5": 0.911},
        "facebook/deit-small-patch16-224": {"top1": 0.755, "top5": 0.950},
        "facebook/deit-base-patch16-224": {"top1": 0.818, "top5": 0.956},
        "facebook/deit-small-distilled-patch16-224": {"top1": 0.812, "top5": 0.954},
    }
    claims = {k: v["top1"] for k, v in claims.items()}

    errs = {k: 100 * abs(v - claims[k]) for k, v in results.items()}

    bars = ax.barh([e.split("/")[1] for e in errs.keys()], list(errs.values()))

    for err,bar  in zip([round(e,2) for k,e in errs.items()],bars):
        h, w = bar.get_height(), bar.get_width()
        # w = 0-plt.gcf().get_figwidth()
        x,y = bar.get_x(), bar.get_y()
        ax.text(x+w, y+h/2, f'{err}%', ha="left", va="center", fontsize=12)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(xlabel="Discrepancy")

    ax.set_xticks([2,4],["2%","4%"])
    return ax


def sentiment(ax):
    """manual compare sentiment analysis models"""

    with open("sentiment_err.json", "r") as file:
        data = json.load(file)
        data = [d*100 for d in data]

    data.remove(max(data)) # 91% err
    data.remove(max(data)) # 20% err
    data.remove(max(data)) # 10% err

    ax.hist(data)
    # ax.set_yscale("log")
    ax.set_xticks([1],["1%"])
    ax.set(ylabel="Frequency (log)")
    ax.set(xlabel="Discrepancy")

    return ax


def main():

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    detection(ax=axs[0])
    classification(ax=axs[1])
    sentiment(ax=axs[2])

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
