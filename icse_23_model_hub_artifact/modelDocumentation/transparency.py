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


def prelim():
    """preliminary studies and graphs with n readme lines / downloads"""

    with open("readme_len.txt", "r") as file:
        info = file.read().split("\n")

    info = [i.strip().split(" ") for i in info]
    info = [i for i in info if i]
    info = info[:-2]

    readme = [i[0] for i in info]
    names = [
        i[1].strip(" ").replace("/README.md", "").replace("mnt/hyattm/cs/hf_repos", "").strip("/")
        for i in info
    ]
    api = HfApi()

    # write decorator to cache results in a file
    # with ThreadPoolExecutor() as ex:
    # downloads = list(tqdm(ex.map(lambda n: api.model_info(n).downloads, names), total=len(names)))

    with open("data-backup.json", "r") as file:
        downloads = json.load(file)
        downloads = [(d["id"], d["downloads"]) for d in downloads]
        downloads = [d for d in downloads if d[0] in names]

    readme, names = [
        l for l in zip(*[(r, n) for r, n in zip(readme, names) if n in [d[0] for d in downloads]])
    ]

    models = [
        {"name": n, "readme": int(r), "downloads": d[1]}
        for n, r, d in zip(names, readme, downloads)
    ]
    models.sort(key=lambda x: x["downloads"], reverse=True)
    idx = [i for i in range(len(models))]

    # print(len(models),len(names),len(readme),len(downloads))
    # print([n for n in names if n not in [d[0] for d in downloads]])
    # quit()

    with open("readme_stats.json", "w") as file:
        json.dump(models, file)
    quit()

    plt.hist([x["readme"] if x["readme"] < 200 else 200 for x in models], bins=200)
    plt.show()
    quit()

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(idx, [x["downloads"] for x in models], label="downloads")
    axs[1].hist2d(idx, [x["readme"] for x in models], label="readme", bins=[20, 300], cmap="Blues")

    # _ = [ax.set_yscale("log") for ax in axs]
    axs[0].set_yscale("log")

    # axs[1].set_ylim([0,300])
    # axs[1].set_yticks(np.arange(0,m:=max([x['readme'] for x in models]),m/10))

    # axs[0].ylim([0,1.1*max([x['downloads'] for x in models])])

    _ = [ax.set(xlabel="idx", ylabel=l) for ax, l in zip(axs, ["downloads", "readme"])]
    plt.suptitle("PTNN transparency")
    plt.tight_layout()

    plt.show()


def safe_get(d, k):
    """retrieve a value from a dict safely"""

    try:
        return d[k]
    except:
        return None


class Metric:
    """a metric result"""

    def __init__(self, *, info, dataset, task, ptnn):

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

        self.ptnn = ptnn

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


class PTNN:
    """a pretrained nn repository on huggingface"""

    with open("data-backup.json", "r") as file:
        data = json.load(file)  # data from rohan
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
    ptnns = []
    approved = {"datasets": ["imagenet-1k", "cifar10", "cifar100", "beans"]}

    @keyquit
    def __init__(self, id):
        self.id = id

        # modelinfo
        try:
            self.modelinfo = PTNN.data[id]  #  else PTNN.api.model_info(id).__dict__
            self.downloads = self.modelinfo["downloads"] if "downloads" in self.modelinfo else None
            self.task = self.modelinfo["pipeline_tag"]
            # self.dataset = [ t.replace("dataset:", "") for t in self.modelinfo["tags"] if "dataset:" in t ]
            self.license = [t for t in self.modelinfo["tags"] if "license:" in t]
            self.carddata = self.modelinfo["cardData"] if "cardData" in self.modelinfo else None

            # just yaml claims and metadata right now
            self.claims = self.get_claims()
            PTNN.ptnns.append(self)

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

        PTNN.config["ignore"] += [self.id]
        with open(PTNN.config["path"], "w") as file:
            json.dump(PTNN.config, file)

    @classmethod
    def show_tasks(cls, ptnns=None, *, mode=None):
        """show distribution of tasks"""

        ptnns = cls.ptnns if ptnns is None else ptnns
        tasks = [p.task for p in cls.ptnns]
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
    def show_nclaims(cls, ptnns=None, *, mode=None):
        """show distribution of number of claims made"""

        ptnns = cls.ptnns if ptnns is None else ptnns
        nclaims = [len(p.claims) for p in ptnns]

        plt.hist(nclaims, bins=100)
        plt.yscale("log")
        plt.show()

    @classmethod
    def show_claims(cls, ptnns=None, *, mode=None):
        """show distribution of claims"""

        ptnns = cls.ptnns if ptnns is None else ptnns
        names = reduce_li([[m.type for m in p.claims] for p in ptnns])
        names = [n.lower() for n in names]

        hist = {n: 0 for n in set(names)}
        hist["None"] = len([p for p in ptnns if not p.claims])
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
    def show_claims_ds(cls, ptnns=None, *, mode=None):
        """show distribution of datasets claims are made on"""

        ptnns = cls.ptnns if ptnns is None else ptnns
        claims = reduce_li([p.claims for p in ptnns])
        ds = [c.dataset["type"] for c in claims if "type" in c.dataset]

        hist = {d: 0 for d in set(ds)}
        hist["None"] = len([p for p in ptnns if not p.claims])
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

        try:  # ptnn makes claims in carddata
            results = self.carddata["model-index"][0]["results"]

            metrics = []
            for r in results:
                task = r["task"]
                dataset = r["dataset"]
                assert type(dataset) is dict

                metrics += [
                    Metric(info=m, dataset=dataset, task=task, ptnn=self) for m in r["metrics"]
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
    def load(cls, ids=None):
        """loads available ptnn repos or repos in ids"""

        print("loading ptnns...")
        ids = list(cls.data.keys()) if not ids else ids
        ids = [x for x in ids if x not in cls.config["ignore"]]

        with ThreadPoolExecutor() as ex:
            ptnns = list(tqdm(ex.map(PTNN, ids), total=len(ids)))
            ptnns = [p for p in ptnns if p]

        print(f"loaded {len(ptnns)} ptnns")
        return ptnns

    def __repr__(self):

        keys = ["task", "claims", "downloads"]
        values = [self.__dict__[k] for k in keys]
        print(f"<PTNN> {self.id}")
        pprint({k: v for k, v in zip(keys, values)})
        return ""


def main():
    """docstring"""

    tasks = ["sentiment_analysis", "image_classification"]
    task = tasks[-1]

    with open(f"../reproducibility/results/{task}.json", "r") as file:
        results = json.load(file)["results"]
        tempids = list(results.keys())

    """TODO
    make one results.json with all results
    id: [*tasks]
    """

    tempids = [m.id for m in PTNN.api.list_models()]

    # tempids = [m.id for m in PTNN.api.list_models(filter="image-classification")]

    # temp2 = set([m.id for m in PTNN.api.list_models(filter="imagenet")])
    # tempids = [t for t in tempids if t in temp2]

    # tempids = [m.id for m in PTNN.api.list_models(filter="object-detection")]

    # tasks = ["automatic-speech-recognition", "reinforcement-learning", "text-classification", "fill-mask", "text-generation"]

    PTNN.load(tempids)
    # PTNN.show_tasks()

    tasks = set([p.task for p in PTNN.ptnns])
    hist = {t: {"claimed": 0, "claimless": 0} for t in tasks}

    for p in PTNN.ptnns:
        hist[p.task]["claimed" if p.claims else "claimless"] += 1

    if None in hist:
        hist["None"] = hist[None]
        del hist[None]

    for k, v in hist.items():
        ratio  = v["claimed"] / (v["claimed"] + v["claimless"])
        total = v['claimed'] + v['claimless']
        hist[k] = [ratio,total]

    keys, values = [x for x in zip(*sorted(hist.items(), key=lambda x: x[1][0]))]
    totals = [v[1] for v in values]
    values = [v[0] for v in values]

    bars = plt.barh(keys, values)
    for rect,ratio,total in zip(bars,values,totals):
        h, w = rect.get_height(), rect.get_width()
        x,y = rect.get_x(), rect.get_y()
        plt.text(x+w, y, f'{int(ratio*total)}/{total}', ha="left", va="bottom")

    plt.tight_layout()
    plt.xticks([0.05,0.1,0.15,0.2],["5%","10%","15%","20%"])
    plt.show()
    quit()

    PTNN.ptnns = [p for p in PTNN.ptnns if p.claims]
    PTNN.ptnns = [
        p for p in PTNN.ptnns if any([(c.value > 1 and c.type == "accuracy") for c in p.claims])
    ]
    print(len(PTNN.ptnns))
    print(PTNN.ptnns)
    quit()

    # PTNN.show_claims_ds()
    # PTNN.show_claims()

    _ = [p.validate() for p in tqdm(PTNN.ptnns, total=len(PTNN.ptnns))]

    quit()

    i = 0
    for p in ptnns:
        metric = "accuracy"
        metrics = p.readme.parse(f"{metric} *?(?:.|\w*)? *?(\d*\.\d*)")

        if metrics:
            i += 1

            metrics = [float(i) for i in metrics]
            metrics = [i / 100 if i > 1 else i for i in metrics]

            if len(metrics) > 1:
                raise Exception("len>1")
            else:
                metrics = metrics[0]

            print(metrics, p.id)
            # quit()
    print(i)
    datasets = reduce(lambda x, y: x + y, [p.dataset for p in ptnns])
    quit()

    ptnns = [p for p in ptnns if p.checkable()]
    print(len(ptnns))

    datasets = set(reduce(lambda x, y: x + y, [p.dataset for p in ptnns]))
    datasets = reduce(lambda x, y: x + y, [p.dataset for p in ptnns])
    print(len(datasets))

    # with open("datasets.json", "w") as file:
    # json.dump(datasets, file)

    """
    TODO
    find the most used datasets ... or tags at least
    """

    quit()

    hasgit = [p for p in ptnns if ".gitattributes" in p]
    print(f"{len(hasgit)} .gitattributes")

    hasreadme = [p for p in ptnns if "README.md" in p]
    print(f"{len(hasreadme)} READMEs")

    def print_mention(s):
        """print how many ptnns mention s"""

        print(f'mentions of "{s}"')
        print(sum([int(s in p.readme) for p in hasreadme]))
        print(sum([int(s in p.readme) for p in hasreadme]) / len(hasreadme))
        print()

    print_mention("eval")
    print_mention("accuracy")

    claims = [(r.id, r.readme.get_claim("accuracy")) for r in hasreadme]
    claims = [c for c in claims if c[1]]

    print(len(claims))

    val = list(map(lambda c: int(results[c[0]]["f1-micro"] == c[1]["value"]), claims))
    err = list(map(lambda c: abs(results[c[0]]["f1-micro"] - c[1]["value"]), claims))
    print(sum(val) / len(val))
    print(sum(err) / len(err))

    print([(c[0], e) for c, e in zip(claims, err) if e > 90])

    print([e for e in err if e > 1])
    # with open('err.json','w') as file:
    # json.dump(err,file)


if __name__ == "__main__":
    main()
