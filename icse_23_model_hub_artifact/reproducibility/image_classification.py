"image classification"

from argparse import ArgumentParser as AP
import os
from pprint import pprint
import arguments

from PIL import Image
from datasets import list_datasets, list_metrics, load_dataset, load_metric
import matplotlib.pyplot as plt
import requests
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, pipeline
import uutils as UU


def predict():

    args = arguments.get_args()
    task = __file__.split("/")[-1].split(".")[0]

    if args.input:
        with open(args.input, "r") as file:
            args.models = json.load(file)

    try:
        with open(f"results/{task}.json", "r") as file:
            data = json.load(file)
    except:
        data = {"task": task, "results": {}, 'ignore':[]}

    dataset = "imagenet-1k" # could change w argument
    dataset = load_dataset(dataset, split="test")

    print(dataset[0])
    quit()

    done = [k for k in data["results"].keys()] + data['ignore']
    models = [m for m in args.models if m not in done]

    root = args.root if args.root else [print("need root"), quit()]
    imagenet_path = os.path.join(root, "IMAGENET")

    # paths = [p for p in os.listdir(os.path.join(imagenet_path, "val"))]
    # dataset = [os.path.join(imagenet_path, "val", p) for p in paths]

    for model in models:
        print(model)

        try:
            pipe = pipeline("image-classification", model=model, return_all_scores=True)

            synset_map = UU.imagenet.get_synset_map(path=imagenet_path)
            id2l = lambda id: synset_map["id2l"][id]
            l2id = lambda l: synset_map["l2id"][l]

            Y = []
            for x in tqdm(dataset):

                y = pipe(x)
                y.sort(key=lambda x: x["score"], reverse=True)
                y = [l2id(i["label"]) for i in y[:5]]
                y = {"top1": y[0], "top5": y}
                Y.append(y)

                prediction = pipe(x["text"])
                y = np.array([i["score"] for i in prediction[0]]).argmax()
                Y.append(y)

            Yh = [y["label"] for y in dataset]
            f1 = evaluate.load("f1")
            eval = lambda a: f1.compute(references=Yh, predictions=Y, average=a)["f1"]
            result = {f"f1-{a}": eval(a) for a in ["macro", "micro", "weighted"]}
            data["results"][model] = result
        
        except Exception as ex:
            quit() if ex is KeyboardInterrupt else print(ex)
            data['ignore'] = data['ignore'] + [model]

        # inside loop so you dont lose intermediate progress
        with open(f"results/{task}.json", "w") as file:
            json.dump(data, file)


    quit()
    '''
    paths = [p.split(".")[0] for p in paths]

    dt = {p: d for p, d in zip(paths, dt)}
    UU.imagenet.write_eval(dt=dt, name=model.replace("/", "_"), path=imagenet_path)


    dt = UU.imagenet.read_eval(name=model.replace("/", "_"), path=imagenet_path)["dt"]
    paths = [k for k, v in dt.items()]
    dt = [v for k, v in dt.items()]

    top1 = [d["top1"] for d in dt]
    top5 = [d["top5"] for d in dt]

    solution_map = UU.imagenet.get_solution_map(path=imagenet_path)
    gt = [solution_map[p]["labels"][0] for p in paths]

    top1 = sum([int(d == g) for d, g in zip(top1, gt)]) / len(gt)
    top5 = sum([int(g in d) for d, g in zip(top5, gt)]) / len(gt)
    acc = {"top1": top1, "top5": top5}

    dt = {p: d for p, d in zip(paths, dt)}
    UU.imagenet.write_eval(dt=dt, name=model.replace("/", "_"), acc=acc, path=imagenet_path)
    '''

def main():
    pass

if __name__ == '__main__': 
    predict()
