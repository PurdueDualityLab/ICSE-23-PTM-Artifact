from tqdm import tqdm
from pprint import pprint
import re

from huggingface_hub import HfApi
from requests_html import HTMLSession


def has(model, word):
    'whether or not the README of the model card has a word'

    raw = lambda model : f'https://huggingface.co/{model}/raw/main/README.md'
    url = raw(model)
    session = HTMLSession()

    r = session.get(url)
    word = f'[{word[0].upper()}{word[0].lower()}]{word[1:]}'
    return bool(re.findall(f'.*{word}.*',r.text))

def main():
    api = HfApi()
    repos = api.list_models(filter='image-classification')
    
    words = ['eval','acc','top']
    filter = lambda m: m if any([has(m,w) for w in words]) else None

    repos = [r for r in tqdm(repos,total=len(repos)) if filter(r)]
    ids = [r.id for r in repos]

    print(ids)

main()
