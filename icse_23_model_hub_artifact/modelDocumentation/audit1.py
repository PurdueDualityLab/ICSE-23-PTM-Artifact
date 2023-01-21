"""motivate sudy by showing model discrepancies"""
from pprint import pprint

import os
import matplotlib.pyplot as plt
import json

results_path = "/Users/matthewhyatt/cs/.datasets/IMAGENET/results"
results = [
    r
    for r in os.listdir(results_path)
    if "deit" in r and not "deit-base-distilled-patch16-224" in r
]
accuracy = []

for r in results:
    with open(os.path.join(results_path, r), "r") as file:
        a = json.load(file)["accuracy"]
        accuracy.append(a)

results = [r.split(".")[0].replace("_", "/") for r in results]

data = {k: v for k, v in zip(results, accuracy)}
pprint(data)

val_top1 = [a["top1"] for a in accuracy]
val_top5 = [a["top5"] for a in accuracy]


# print(val_top1)
# print(val_top5)
# quit()

# deit only
claim_top1 = [72.2, 79.9, 81.8, 81.2]
claim_top5 = [91.1, 95.0, 95.6, 95.4]

val_top1 = [0.66462, 0.77058, 0.80116, 0.81006]
val_top5 = [0.87716, 0.92958, 0.93628, 0.95418]
val_top1 = [100 * i for i in val_top1]
val_top5 = [100 * i for i in val_top5]


_ = {
    "facebook/deit-tiny-patch16-224": {"top1": 0.66462, "top5": 0.87716},
    "facebook/deit-small-patch16-224": {"top1": 0.77058, "top5": 0.92958},
    "facebook/deit-base-patch16-224": {"top1": 0.80116, "top5": 0.93628},
    "facebook/deit-small-distilled-patch16-224": {"top1": 0.81006, "top5": 0.95418},
}

width = 0.3

x = [0, 1, 2, 3]
plt.bar([2 * i - 0.6 for i in x], claim_top1, width=width, label="claimed top1")
plt.bar([2 * i - 0.2 for i in x], val_top1, width=width, label="validated top1")
# plt.bar([2 * i + 0.2 for i in x], claim_top5, width=width, label="claimed top5")
# plt.bar([2 * i + 0.6 for i in x], val_top5, width=width, label="validated top5")

plt.xticks()

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylim(65,85)
plt.legend(loc="upper left")
plt.title("Accuracy Claims of Facebook DEIT Models")
plt.ylabel("Accuracy %")

# bar(claim_top1)
# bar(claim_top5)
# bar(val_top1)
# bar(val_top5)

plt.show()
