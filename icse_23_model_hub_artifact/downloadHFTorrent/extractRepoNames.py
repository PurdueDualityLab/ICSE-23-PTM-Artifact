from argparse import ArgumentParser, Namespace

from huggingface_hub.hf_api import HfApi, ModelInfo


def parseArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="Hugging Face Repo Name Extractor",
        usage="Tool to extract the names of repositories from the Hugging Face REST API to a file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="repoNames.txt",
        required=False,
        help="Filename to store the list of repo names",
    )
    return parser.parse_args()


def main() -> None:
    data: list = []
    args: Namespace = parseArgs()

    api: HfApi = HfApi()
    models: list = api.list_models()
    # TODO: Add code for getting a list of datasets and spaces

    model: ModelInfo
    for model in models:
        data.append(f"{model.modelId}\n")

    with open(args.output, "w") as file:
        file.writelines(data)
        file.close()


if __name__ == "__main__":
    main()
