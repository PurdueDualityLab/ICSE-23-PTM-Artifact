'reproduce fill mask'


from argparse import ArgumentParser as AP
import os
from pprint import pprint

from PIL import Image
from datasets import list_datasets, list_metrics, load_dataset, load_metric
from evaluate import load
import matplotlib.pyplot as plt
import requests
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, pipeline
import uutils as UU


def predict(model, args):

    print(model)

    metrics = [load('glue', mode) for mode in ('mrpc', 'stsb','cola')]  # 'mrpc' or 'qqp'
    references = [0, 1]
    predictions = [0, 1]
    results = [m.compute(predictions=predictions, references=references) for m in metrics]
    print(results)

    pipe = pipeline("fill-mask", model=model, device=-1)
    root = args.root if args.root else [print('need root'),quit()]

    # imagenet_path = os.path.join(root, 'IMAGENET')
    # paths = [p for p in os.listdir(os.path.join(imagenet_path,'val'))]
    # dataset = [os.path.join(imagenet_path,'val',p) for p in paths]
    dataset = metrics[0] 

    if args.inference:
        print('inference')

        # synset_map = UU.imagenet.get_synset_map(path=imagenet_path)
        # id2l = lambda id: synset_map['id2l'][id]
        # l2id = lambda l: synset_map['l2id'][l]

        dt = []

        for item in tqdm(dataset):
            d = pipe(item)
            print(d)
            quit()

            d.sort(key=lambda x: x['score'],  reverse=True)
            d = [l2id(i['label']) for i in d[:5]]
            d = {'top1': d[0], 'top5': d}
            dt.append(d)
        
        paths = [p.split('.')[0] for p in paths]

        dt = {p:d for p,d in zip(paths,dt)}
        UU.imagenet.write_eval(dt=dt,name=model.replace('/','_'),path=imagenet_path)

    if args.eval:

        dt = UU.imagenet.read_eval(name=model.replace('/','_'), path=imagenet_path)['dt']
        paths = [k for k,v in dt.items()]
        dt = [v for k,v in dt.items()]

        top1 = [d['top1'] for d in dt]
        top5 = [d['top5'] for d in dt]

        solution_map = UU.imagenet.get_solution_map(path=imagenet_path)
        gt = [solution_map[p]['labels'][0] for p in paths]

        top1 = sum([int(d==g) for d,g in zip(top1,gt)]) / len(gt)
        top5 = sum([int(g in d) for d,g in zip(top5,gt)]) / len(gt)
        acc = {'top1':top1, 'top5':top5}

        dt = {p:d for p,d in zip(paths,dt)}
        UU.imagenet.write_eval(dt=dt,name=model.replace('/','_'), acc=acc, path=imagenet_path)

def main(model, args):
 
    # imagenet = os.path.join(args.root, 'IMAGENET')
    # results = UU.imagenet.has_results(name=model,path=imagenet) 
    # ignore = UU.imagenet.has_ignore(name=model, path=imagenet)

    # if not (results or ignore): 
        try:
            predict(model, args)
        except Exception as ex:
            print(ex)
            quit()
            # UU.imagenet.ignore(name=model,path=os.path.join(args.root,'IMAGENET'))

if __name__ == '__main__': 

    from args import get_args
    args = get_args()
    main(args.models[0],args)
