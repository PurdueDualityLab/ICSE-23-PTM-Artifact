import json
import os
import time

from huggingface_hub import HfApi
from tqdm import tqdm


def main():
    """docstring"""

    with open("data.json", "r") as file:
        data = json.load(file)

    try:
        with open("ignore.json", "r") as file:
            ignore = json.load(file)
    except:
        ignore = []

    api = HfApi()
    oldids = set([m["id"] for m in data] + ignore)
    newids = [m.id for m in api.list_models() if m.id not in oldids]
    print(len(newids))

    for id in tqdm(newids):

        try:
            info = api.model_info(id).__dict__
            del info["siblings"]
            data.append(info)

            with open("data.json", "w") as file:
                json.dump(data, file)

        except KeyboardInterrupt:
            quit()
        except Exception as ex:
            print(ex)

            ignore.append(id)
            with open("ignore.json", "w") as file:
                json.dump(ignore, file)

        time.sleep(5)

        # print(info)

        # quit()


if __name__ == "__main__":
    main()
