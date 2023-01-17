from argparse import ArgumentParser as AP
from concurrent.futures import ThreadPoolExecutor
import os
import time

from huggingface_hub import HfApi
from tqdm import tqdm
import uutils as UU

import transparency

ap = AP()
ap.add_argument("-m","--models", nargs='+', type=str)
# ap.add_argument("-o", "--output", type=str)
args = ap.parse_args()

# @UU.deco.sleep(time=1)
# make a decorator that makes a method silent
# @UU.deco.stdout('clone.log')
def clone(id):
    """clones a huggingface repo"""

    output = "/mnt/hyattm/cs/hf_repos/"
    url = f"https://huggingface.co/{id}"
    os.system(f"mkdir -p {output}/{id}")
    os.system(f"git clone --quiet {url} {output}/{id}")
    time.sleep(25)


def main():
    """main"""

    try:
        ids = args.models
    except:
        api = HfApi()
        models = api.list_models()
        ids = [m.id for m in models]


        ptnns = transparency.load_ptnns()
        ptids = set([p.id for p in ptnns])

        # the ids you havent downloaded yet
        ids = [i for i in ids if i not in ptids]


    with ThreadPoolExecutor() as ex:
        _ = list(tqdm(ex.map(clone, ids), total=len(ids)))


if __name__ == "__main__":
    main()
