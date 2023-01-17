from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import List

from huggingface_hub.hf_api import HfApi, ModelInfo
from pandas import DataFrame
from progress.bar import Bar
from requests.exceptions import ConnectionError, HTTPError

_defaultValue = lambda: None


def parseArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog=f"Hugging Face Model Information Downloader",
        usage="Download Hugging Face model information to a CSV file",
    )
    parser.add_argument(
        "-a",
        "--access-token",
        type=str,
        required=True,
        help="Hugging Face REST API access token",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="huggingfaceModelInformation.csv",
        help="File to output data to",
    )
    return parser.parse_args()


def getModelIDs(api: HfApi) -> list:
    print("Getting the complete list of models on HuggingFace.co...")
    models: List[ModelInfo] = api.list_models()
    return [model.modelId for model in models]


def getInformation(api: HfApi, modelIDs: list, accessToken: str) -> list:
    data: list = []
    with Bar("Getting detailed model information... ", max=len(modelIDs)) as bar:
        id: str
        for id in modelIDs:
            try:
                modelInfo: dict = api.model_info(repo_id=id, token=accessToken).__dict__
            except HTTPError:
                bar.max -= 1
                bar.update()
                continue
            except ConnectionError:
                bar.max -= 1
                bar.update()
                continue

            modelInfo: defaultdict = defaultdict(_defaultValue, modelInfo)

            try:
                modelCardData: defaultdict = defaultdict(
                    _defaultValue, modelInfo["cardData"]
                )
            except TypeError:
                modelCardData: defaultdict = defaultdict(_defaultValue)
            except ValueError:
                bar.max -= 1
                bar.update()
                continue

            try:
                modelConfig: defaultdict = defaultdict(
                    _defaultValue, modelInfo["config"]
                )
            except TypeError:
                modelConfig: defaultdict = defaultdict(_defaultValue)

            temp: defaultdict = defaultdict(_defaultValue)
            temp["modelId"]: str = modelInfo["modelId"]
            temp["author"]: str = modelInfo["author"]
            temp["datasets"]: str = modelCardData["datasets"]
            temp["language"]: str | list = modelCardData["language"]
            temp["license"]: str = modelCardData["license"]
            temp["tags"]: list = modelCardData["tags"]
            temp["downloads"]: int = modelInfo["downloads"]
            temp["likes"]: int = modelInfo["likes"]

            temp["architecture"]: str = modelConfig["model_type"]

            try:
                temp["tags"].extend(modelInfo["tags"])
            except AttributeError:
                temp["tags"] = modelInfo["tags"]

            if temp["author"] == None:
                temp["author"] = "HuggingFace"

            if type(temp["language"]) is str:
                temp["language"] = [temp["language"]]

            data.append(dict(temp))
            bar.next()

    return data


def main() -> None:
    args: Namespace = parseArgs()

    api: HfApi = HfApi()
    api.set_access_token(args.access_token)

    modelIDs: list = getModelIDs(api)
    data: list = getInformation(api, modelIDs, accessToken=args.access_token)

    df: DataFrame = DataFrame(data)
    df = df.fillna(value={"architecture": "None", "downloads": 0, "likes": 0})
    df["downloads"] = df["downloads"].fillna(0)

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
