from argparse import Namespace, ArgumentParser
from io import TextIOWrapper
from time import sleep
from bs4 import BeautifulSoup, ResultSet
from progress.bar import Bar
from requests import get
from requests.models import Response

def findCompaniesArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="Hugging Face Measure Verified Organizations",
        usage="Scrape Hugging Face organization webpages for a verification tag",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input .txt file of HuggingFace.co model URLS",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="verifiedOrganizations.txt",
        help="Output .txt file to store verified organization URLs",
    )
    return parser.parse_args()

def scrapePageForCompany(request: Response) -> bool:
    soup: BeautifulSoup = BeautifulSoup(markup=request.text, features="lxml")
    potentialTags: ResultSet = soup.find_all(name="span", attrs={"class": "capitalize"})
    return bool(potentialTags)


def scrapePageForVerification(request: Response) -> bool:
    soup: BeautifulSoup = BeautifulSoup(markup=request.text, features="lxml")
    potentialTags: ResultSet = soup.find_all(
        name="div", attrs={"title": "Verified organization"}
    )
    return bool(potentialTags)


def main() -> None:
    args: Namespace = findCompaniesArgs()

    inputFile: TextIOWrapper
    with open(args.input, "r") as inputFile:
        urls: list = inputFile.readlines()
        inputFile.close()

    urls = [url.strip() for url in urls]
    urls: set = set(urls)

    with Bar(
        "Scraping webpages for company and verification tags... ", max=len(urls)
    ) as bar:
        data: list = []
        count: int = 1
        url: str
        for url in urls:
            if count % 500 == 0:
                sleep(600)

            request: Response = get(url)
            # if scrapePageForCompany(request=request) and scrapePageForVerification(
            #     request=request
            # ):
            if scrapePageForVerification(request=request):
                data.append(f"{url}\n")
            count += 1
            bar.next()

    outputFile: TextIOWrapper
    with open(args.output, "w") as outputFile:
        outputFile.writelines(data)
        outputFile.close()


if __name__ == "__main__":
    main()
