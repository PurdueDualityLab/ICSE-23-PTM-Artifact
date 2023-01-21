"""motivation data"""
import uutils as UU
from argparse import ArgumentParser as AP
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi
import matplotlib.pyplot as plt
from requests_html import HTMLSession

# import uutils as UU
from tqdm import tqdm
import yaml

def get_args():

    ap = AP()
    ap.add_argument("-m", "--models", type=str, nargs="+")
    ap.add_argument("-e", "--eval", action="store_true")
    ap.add_argument("-i", "--inference", action="store_true")
    ap.add_argument('-r','--root',type=str) # the root dir for .datasets (in progress)

    ap.add_argument('--classification',action='store_true')
    ap.add_argument('--detection',action='store_true')

    args = ap.parse_args()
    return args


def filter_models(models, tag):
    """filter models by a keyword in the tag"""
    return [m for m in models if any([tag in t for t in m.tags])]


# @UU.deco.sleep(rate=0.1)
def dependent(model, tag):
    """return true if an <tag> is mentioned in the <model> README"""

    try:
        sess = HTMLSession()
        url = f"https://huggingface.co/{model}"
        r = sess.get(url)
        return model if tag.lower() in r.text.lower() else ""
    except Exception as ex:
        print(ex)
        return ""


def main():

    args = get_args()
    api = HfApi()
    # tags = api.get_model_tags._tag_dictionary
    # item = api.model_info(item.id)

    if args.inference: 
        with ThreadPoolExecutor() as ex:

            models = [m.id for m in api.list_models()]
            def get_info(id):
                """try except for api.model_info"""

                try:
                    return api.model_info(id)
                except:
                    return id

            models = list(tqdm(ex.map(get_info, models), total=len(models)))

            with open("hf.yaml", "w") as file:
                yaml.dump(models, file)

    if args.eval:
            
        @UU.deco.timer
        def load_hf():
            """docstring"""

            with open("hf.yaml", "r") as file:
                models = yaml.load(file, Loader=yaml.Loader)
                num_models = len(models)
                models = [m for m in models if not isinstance(m,str)] # because of try except...
                num_accessible = len(models)
            return models, num_models, num_accessible

        models, num_models, num_accessible = load_hf()
        
        def has_ds(model):
            """checks if a model has dataset tags"""

            try:
                return model.cardData['datasets']
            except:
                return False

        datasets = {}
        for m in tqdm(models):
            ds = has_ds(m)
            if not ds:
                ds = ['<NO_TAGS>']
            if type(ds) is str:
                ds = [ds]
            for d in ds:
                try:
                    i = datasets[d]
                    datasets[d]  = i+1
                except:
                    datasets[d] = 1

        'something about other catagory'
        other, thresh  = 0, 10
        todel = []
        for k,v in datasets.items():
            if v < thresh:
                other += v
                todel.append(k)
        for k in todel:
            del datasets[k]
        datasets['other'] = other

        fig,ax = plt.subplots()
        # ax.barh([i for i in range(len(datasets))], list(datasets.values()))
        # ax.set_yticklabels(list(datasets.keys()))

        width = 0.5
        ax.bar(1,num_models, width=width, label='total number of models')
        ax.bar(1, num_accessible,width=width, label='model has acessible info')
        ax.bar(1, datasets['<NO_TAGS>'], width=width, label='<NO_TAGS>')
        plt.legend()

        plt.yscale('log')
        plt.title('Frequency of Dataset Tags in Model Cards')
        # plt.xlabel('Tag Name')
        plt.ylabel('Frequency')
        plt.show()
        print(len(datasets))
        print(datasets)
        # models = [m for m in tqdm(models, total=len(models)) if not has_ds(m)]
        # print(len(models))
        quit()

        criterion = lambda c: [ m for m in list( tqdm(ex.map(lambda m: dependent(m, c), models), total=len(models))) if m ]

        imagenet = criterion("imagenet")

        print(len(imagenet))
        quit()


if __name__ == "__main__":
    main()
