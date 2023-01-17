from bs4 import BeautifulSoup, ResultSet 
from bs4.element import Tag
from progress.bar import Bar
from requests import Response, get
from argparse import ArgumentParser, Namespace

def parseArgs() ->  Namespace:
    parser: ArgumentParser = ArgumentParser(prog="Hugging Face Download Organization List", usage="Tool to download the list of organizations from Hugging Face")
    return parser.parse_args()

def generateURL(maxPages: int = 125) -> None:
    page: int
    for page in range(maxPages):
        yield f"https://huggingface.co/organizations?p={page}"


def getSoup(url: str) -> BeautifulSoup:
    data: bytes = get(url).content
    return BeautifulSoup(data, features="lxml")


def findOrganizations(soup: BeautifulSoup) -> list:
    urls: list = []

    raw: ResultSet = soup.find_all(
        name="article", attrs={"class": "overview-card-wrapper"}
    )
    
    data: Tag
    for data in raw:
        orgURL: str = f"https://huggingface.co{(data.find(name='a').get('href'))}\n"
        urls.append(orgURL)

    return urls


def main() -> None:
    parseArgs()
    data: list = []
    maxPages: int = 125
    urlGenerator = generateURL(maxPages)

    with Bar("Scraping pages for organizations... ", max=maxPages) as bar:
        while True:
            try:
                soup: BeautifulSoup = getSoup(next(urlGenerator))
                data.extend(findOrganizations(soup))
                bar.next()
            except StopIteration:
                break

    data = list(set(data))
    with open("organizations.txt", "w") as output:
        output.writelines(data)


if __name__ == "__main__":
    main()
