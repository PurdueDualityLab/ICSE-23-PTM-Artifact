"import stuff"
from argparse import ArgumentParser as AP
from huggingface_hub import HfApi
import classification
import detection
"might have to refactor to multiple files"


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

def filter_models(models,tag):
    '''filter models by a keyword in the tag'''
    return [m for m in models if any([tag in t for t in m.tags])]

def main():

    """
    find out what kind of model it is from the tags on the model card
    classification, detection, etc...
    """
    
    args = get_args()
    api = HfApi()

    if args.classification:
        repos = api.list_models(filter="image-classification")
        repos = filter_models(repos,'imagenet')

        models = [r.modelId for r in repos]
        tasks = ["image-classification" for m in models]

        for model, task in zip(models, tasks):
            do(model, task)

    if args.detection:
        repos = api.list_models(filter='object-detection')
        repos = filter_models(repos,'coco')

        models = [r.modelId for r in repos]
        tasks = ["object-detection" for m in models]

        for model, task in zip(models, tasks):
            do(model, task)

def do(model, task):
    """
    does whatever the model is intended to do
    only supports classification and obj detection rn
    """

    options = {
        "image-classification": classification.main,
        "object-detection": detection.main,
    }

    func = options[task]
    func(model, get_args())  # not *args ... args object


if __name__ == "__main__":
    main()
