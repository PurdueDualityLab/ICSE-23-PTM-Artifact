import json
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub.hf_api import HfApi
from tqdm import tqdm


def parseArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="Hugging Face Model Download Extractor",
        usage="Extract the downloads of all models from the Hugging Face REST API to a JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="modelDownloads.json",
        help="Filename of the output JSON file",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="Hugging Face REST API token",
    )
    return parser.parse_args()


def main() -> None:
    args: Namespace = parseArgs()

    api: HfApi = HfApi()
    repos = api.list_models()
    repo_ids = [repo.id for repo in tqdm(repos)]

    with ThreadPoolExecutor() as executor:

        def get_downloads(name):
            try:
                return api.model_info(name, token=args.token).downloads
            except:
                return -1

        downloads = list(
            tqdm(executor.map(get_downloads, repo_ids), total=len(repo_ids))
        )

    with open(args.output, "w") as file:
        json.dump(downloads, file)
        file.close()


if __name__ == "__main__":
    main()
