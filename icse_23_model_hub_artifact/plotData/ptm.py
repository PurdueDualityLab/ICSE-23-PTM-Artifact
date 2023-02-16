from argparse import ArgumentParser as AP
from concurrent.futures import ThreadPoolExecutor
import functools
from functools import reduce
import json
import os
from pprint import pprint
import re
import time

import datasets
from huggingface_hub import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
import yaml
import ndjson

from loguru import logger

def reduce_li(li):
    """reduces a list of lists to a list of items"""

    return reduce(lambda x, y: x + y, li)


def keyquit(func):
    """on keyboard interrupt, quit"""

    def deco(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):

            try:
                result = func(*args, **kwargs)
                return result
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                quit()

        return wrap

    return deco if func is None else deco(func)


def safe_get(d, k):
    """retrieve a value from a dict safely"""

    try:
        return d[k]
    except:
        return None


class Metric:
    """a metric result"""

    def __init__(self, *, info, dataset, task, ptms):

        self.name = info["name"]
        self.type = info["type"]
        self.value = float(info["value"])
        self.verified = info["verified"] if "verified" in info else None

        self.ds_name = dataset["name"] if "name" in dataset else None
        self.ds_type = dataset["type"] if "type" in dataset else None
        self.ds_config = dataset["config"] if "config" in dataset else None
        self.ds_args = dataset["args"] if "args" in dataset else None
        self.ds_split = dataset["split"] if "split" in dataset else None

        self.dataset = dataset

        self.task = task
        self.task_type = task["type"] if "type" in task else None

        self.ptms = ptms

    def __repr__(self):

        return f"<METRIC {self.dataset['name']}: {round(self.value,4)} {self.name}>"

    def comparable(self, other):
        """are two metrics comparable"""

        if isinstance(other, Metric):
            return all(
                [self.dataset == other.dataset, self.task == other.task, self.name == other.name]
            )
        return False

    def __eq__(self, other):
        """docstring"""

        if self.comparable(other):
            return self.value == other.value
        return False

    def __sub__(self, other):
        """the discrepancy between two metrics ... always positive value"""

        if self.comparable(other):
            return abs(self.value - other.value)
        else:
            raise Exception("uncomparable")


class PTM:
    """a pretrained model repository on huggingface"""
    with open("../../data/modelinfo.ndjson", "r") as file:
    # with open("./reproducibility/results.json", "r") as file:
        data = ndjson.load(file)
    data = {m["id"]: m for m in data}
    print("loaded local data")

    config_path = "config.json"
    try:
        with open(config_path) as file:
            config = json.load(file)
        print("loaded config")
    except:
        config = {"ignore": [], "path": config_path}

    api = HfApi()
    ptms = []
    approved = {"datasets": ["imagenet-1k", "cifar10", "cifar100", "beans"]}

    @keyquit
    def __init__(self, id):
        self.id = id

        # modelinfo
        try:
            self.modelinfo = PTM.data[id]  #  else PTM.api.model_info(id).__dict__
            self.downloads = self.modelinfo["downloads"] if "downloads" in self.modelinfo else None
            self.task = self.modelinfo["pipeline_tag"]
            # self.dataset = [ t.replace("dataset:", "") for t in self.modelinfo["tags"] if "dataset:" in t ]
            self.license = [t for t in self.modelinfo["tags"] if "license:" in t]
            self.carddata = self.modelinfo["cardData"] if "cardData" in self.modelinfo else None

            # just yaml claims and metadata right now
            self.claims = self.get_claims()
            PTM.ptms.append(self)

        except RepositoryNotFoundError as ex:
            self.ignore()
            print(f"restricted repository: {self.id}")
            attrs = ["modelinfo", "downloads", "task", "license", "carddata", "claims"]
            for a in attrs:
                setattr(self, a, None)

        except KeyError:  # don't have the model locally
            pass

    def __contains__(self, metric):
        """return True if repo contains a certain metric"""

        return (
            metric in reduce_li([[m.name, m.type] for m in self.claims]) if self.claims else False
        )

    def ignore(self):
        """add self to config.ignore and write to file"""

        PTM.config["ignore"] += [self.id]
        with open(PTM.config["path"], "w") as file:
            json.dump(PTM.config, file)

    @classmethod
    def show_tasks(cls, ptms=None, *, mode=None):
        """show distribution of tasks"""

        ptms = cls.ptms if ptms is None else ptms
        tasks = [p.task for p in cls.ptms]
        hist = {t: 0 for t in set(tasks)}
        for t in tasks:
            hist[t] += 1
        pprint(hist)

        if None in hist:
            hist["None"] = hist[None]
            del hist[None]

        # if mode == "visual":
        plt.barh(list(hist.keys()), list(hist.values()))
        plt.tight_layout()
        plt.show()

    @classmethod
    def show_nclaims(cls, ptms=None, *, mode=None):
        """show distribution of number of claims made"""

        ptms = cls.ptms if ptms is None else ptms
        nclaims = [len(p.claims) for p in ptms]

        plt.hist(nclaims, bins=100)
        plt.yscale("log")
        plt.show()

    @classmethod
    def show_claims(cls, ptms=None, *, mode=None):
        """show distribution of claims"""

        ptms = cls.ptms if ptms is None else ptms
        names = reduce_li([[m.type for m in p.claims] for p in ptms])
        names = [n.lower() for n in names]
        # logger.debug(names)

        hist = {n: 0 for n in set(names)}
        hist["None"] = len([p for p in ptms if not p.claims])
        for n in names:
            hist[n] += 1

        if len(hist) > 10:
            top10 = sorted(hist.values(), reverse=True)[9]
            other = 0
            remove = []
            for k, v in hist.items():
                if v < top10:
                    other += v
                    remove.append(k)

            for k in remove:
                del hist[k]
            hist["other"] = other

        keys, values = [x for x in zip(*sorted(hist.items(), key=lambda x: x[1]))]
        pprint(hist)

        plt.barh(keys, values)
        plt.tight_layout()
        plt.show()

    @classmethod
    def show_claims_ds(cls, ptms=None, *, mode=None):
        """show distribution of datasets claims are made on"""

        ptms = cls.ptms if ptms is None else ptms
        claims = reduce_li([p.claims for p in ptms])
        ds = [c.dataset["type"] for c in claims if "type" in c.dataset]

        hist = {d: 0 for d in set(ds)}
        hist["None"] = len([p for p in ptms if not p.claims])
        for d in ds:
            hist[d] += 1

        if len(hist) > 10:
            top10 = sorted(hist.values(), reverse=True)[9]
            other = 0
            remove = []
            for k, v in hist.items():
                if v < top10:
                    other += v
                    remove.append(k)

            for k in remove:
                del hist[k]
            hist["other"] = other

        keys, values = [x for x in zip(*sorted(hist.items(), key=lambda x: x[1]))]
        # remove author from datasets
        keys = [x.split("/")[1] if "/" in x else x for x in keys]
        pprint(hist)

        plt.barh(keys, values)
        plt.tight_layout()
        plt.show()

    def get_claims(self, claim=None):
        """returns claims of type <claim> else all claims from yaml"""

        try:  # ptms makes claims in carddata
            results = self.carddata["model-index"][0]["results"]

            metrics = []
            for r in results:
                task = r["task"]
                dataset = r["dataset"]
                assert type(dataset) is dict

                metrics += [
                    Metric(info=m, dataset=dataset, task=task, ptms=self) for m in r["metrics"]
                ]

            return metrics

        except:
            return []

    def validate(self):
        """returns discrepancies between claims"""

        ''' suggestion from autoeval.py
        parse task
        if no task then quit

        clone model
        if makes no claims then quit

        parse dataset
        download dataset
        if no dataset then quit

        eval model
        compare results
        '''

        # look for metric in appropriate results/<task>.json file

        # else
        """
        download dataset
        inference on dataset
        compute important metrics (determined by claims?)
        store in appropriate results/<task>.json file

        call self._validate(metric)
        """

        print(self)

        for c in [c for c in self.claims if "tatoeba_mt" in c.ds_type]:

            try:
                args = [a for a in [c.ds_type, c.ds_config, c.ds_args] if a]
                args = [a for a in ["Helsinki-NLP/tatoeba_mt", c.ds_config, c.ds_args] if a]
                kwargs = {"split": c.ds_split} if c.ds_split else {}
                data = datasets.load_dataset(*args, **kwargs)

            except Exception as ex:
                return
                ds = [d for d in datasets.list_datasets() if c.ds_type in d]
                print(ds)

            try:
                print(c.task)
                quit()
                pipe = transformers.pipeline(c.task, model=self.id)
                # pipe = transformers.pipeline(task, model=self.id, return_all_scores=True)

                # Yhat = pipe(dataset)

                Yhat = []
                for x in tqdm(data):
                    Yhat.append(pipe(x))

                print("done")
                # print(len(Yhat))
                quit()

            except Exception as ex:
                raise ex
                pass
                # from datasets import list_metrics

        # comparable = [c for c in self.claims if c.comparable(metric)]
        # if len(comparable) > 1:
        # raise Exception("why is there more than 1 claim?")

        # return [c - metric for c in comparable]

    @classmethod
    @keyquit
    def load(cls, ids=None, file=None):
        """loads available ptms repos or repos in ids"""

        assert not (ids and file), 'it doesnt do both'

        print("loading ptms...")
        ids = list(cls.data.keys()) if not ids else ids
        ids = [x for x in ids if x not in cls.config["ignore"]]

        with ThreadPoolExecutor() as ex:
            ptms = list(tqdm(ex.map(PTM, ids), total=len(ids)))
            logger.debug(f"after ThreadPoolExecutor{len(ptms)}")
            ptms = [p for p in ptms if p]
            logger.debug(f"Final {len(ptms)}")

        print(f"loaded {len(ptms)} ptms")
        return ptms

    def __repr__(self):

        keys = ["task", "claims", "downloads"]
        values = [self.__dict__[k] for k in keys]
        print(f"<PTM> {self.id}")
        pprint({k: v for k, v in zip(keys, values)})
        return ""

def plot_fig6():
    """generates figure 6"""

    # PTM.load([m.id for m in PTM.api.list_models()])
    PTM.load()
    
    tasks = set([p.task for p in PTM.ptms])

    hist = {t: {"claimed": 0, "claimless": 0} for t in tasks}

    for p in PTM.ptms:
        hist[p.task]["claimed" if p.claims else "claimless"] += 1

    if None in hist:
        hist["None"] = hist[None]
        del hist[None]

    for k, v in hist.items():
        ratio = v["claimed"] / (v["claimed"] + v["claimless"])
        total = v["claimed"] + v["claimless"]
        hist[k] = [ratio, total]

    others, claims, delete = [], [], []
    for k, v in hist.items():
        if v[0] < 0.001:
            claims.append(v[0] * v[1])
            others.append(v[1])
            delete.append(k)
    hist["other"] = [sum(claims) / sum(others), sum(others)]
    for d in delete:
        del hist[d]

    keys, values = [x for x in zip(*sorted(hist.items(), key=lambda x: x[1][0]))]
    totals = [v[1] for v in values]
    values = [v[0] for v in values]

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bars = plt.barh(keys, values)
    for rect, ratio, total in zip(bars, values, totals):
        h, w = rect.get_height(), rect.get_width()
        # w = 0-plt.gcf().get_figwidth()
        x, y = rect.get_x(), rect.get_y()
        plt.text(x + w, y, f"{int(ratio*total)}/{total}", ha="left", va="bottom", fontsize=11)

    plt.xticks([0.05, 0.1, 0.15, 0.2], ["5%", "10%", "15%", "20%"])
    plt.tight_layout()

    plt.show()

def main():
    
    # PTM.load()
    # PTM.ptms = [p for p in PTM.ptms if p.id in models]
    plot_fig6()
    # PTM.show_claims()

    fontsize = 12

    ptms = PTM.ptms 
    fig, ax = plt.subplots()
    # fig, axs = plt.subplots(1,2)

    claims = reduce_li([p.claims for p in ptms])
    ds = [c.dataset["type"] for c in claims if "type" in c.dataset]

    hist = {d: 0 for d in set(ds)}
    hist["None"] = len([p for p in ptms if not p.claims])
    for d in ds:
        hist[d] += 1

    if len(hist) > 10:
        top10 = sorted(hist.values(), reverse=True)[9]
        other = 0
        remove = []
        for k, v in hist.items():
            if v < top10:
                other += v
                remove.append(k)

        for k in remove:
            del hist[k]
        hist["other"] = other

    keys, values = [x for x in zip(*sorted(hist.items(), key=lambda x: x[1]))]
    values = [100*v/sum(values) for v in values]
    # remove author from datasets
    keys = [x.split("/")[1] if "/" in x else x for x in keys]
    # keys = [x.replace("_","\n") for x in keys]
    pprint(hist)
    ticks = keys

    # axs[0].barh(keys, values)

    ax.barh([i*2 for i in range(len(values))], values, label='dataset')
    for i, v in enumerate(values):
        ax.text(v, 2*i, f'{v:.1f}', verticalalignment='center', size=fontsize)

    mtype = lambda p: p.modelinfo['config']['model_type'] if p.modelinfo and p.modelinfo['config'] and 'model_type' in p.modelinfo['config'] else "None"

    names = [mtype(p) for p in ptms]

    hist = {d: 0 for d in set(names)}
    for n in names:
        hist[n] += 1

    if len(hist) > 10:
        top10 = sorted(hist.values(), reverse=True)[9]
        other = 0
        remove = []
        for k, v in hist.items():
            if v < top10:
                other += v
                remove.append(k)

        for k in remove:
            del hist[k]
        hist["other"] = other

    keys, values = [x for x in zip(*sorted(hist.items(), key=lambda x: x[1]))]
    values = [100*v/sum(values) for v in values]
    # remove author from datasets
    keys = [x.split("/")[1] if "/" in x else x for x in keys]
    pprint(hist)

    ticks = [[b,a] for a,b in zip(ticks,keys)]
    ticks = [i for j in ticks for i in j]

    # axs[1].barh(keys, values)

    ax.barh([i*2-1 for i in range(len(values))], values, label='architecture')
    for i, v in enumerate(values):
        ax.text(v, 2*i-1, f'{v:.1f}', verticalalignment='center', size=fontsize)

    ax.set_xscale('log')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.yticks([i for i in range(-1,len(ticks)-1)],ticks, fontsize=fontsize)

    xticks = [1,10,100]
    plt.xticks(xticks, [f'{x}%' for x in xticks], fontsize=fontsize)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.pause(4)

if __name__ == '__main__': 
    main()
