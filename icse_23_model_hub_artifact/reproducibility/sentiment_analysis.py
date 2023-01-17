"sentiment analysis"

from argparse import ArgumentParser as AP
import json
import os
from pprint import pprint

from PIL import Image
from datasets import list_datasets, list_metrics, load_dataset, load_metric
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, pipeline
import uutils as UU

import arguments


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

    dataset = load_dataset("emotion", split="test")
    done = [k for k in data["results"].keys()] + data['ignore']
    models = [m for m in args.models] #  if m not in done]

    for model in models:
            print(model)

        # try:
            pipe = pipeline("text-classification", model=model, return_all_scores=True)

            Y = []
            for x in tqdm(dataset):

                prediction = pipe(x["text"])
                y = np.array([i["score"] for i in prediction[0]]).argmax()
                Y.append(y)

            Yh = [y["label"] for y in dataset]
            f1 = evaluate.load("f1")
            eval = lambda a: f1.compute(references=Yh, predictions=Y, average=a)["f1"]
            result = {f"f1-{a}": eval(a) for a in ["macro", "micro", "weighted"]}
            data["results"][model] = result
        
        # except Exception as ex:
            # quit() if ex is KeyboardInterrupt else print(ex)
            # data['ignore'] = data['ignore'] + [model]

        # inside loop so you dont lose intermediate progress
        # with open(f"results/{task}.json", "w") as file:
            # json.dump(data, file)

            print(result)

# def main():
    # pass
    # predict()


if __name__ == "__main__":
    predict()
