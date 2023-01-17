import os
from argparse import ArgumentParser, Namespace

from progress.bar import Bar


def main() -> None:
    parser: ArgumentParser = ArgumentParser(
        prog="Hugging Face Measure Repositories with Signed Commits",
        usage="This tool is to be ran in directory containing HFTorrent or similar dataset",
    )
    parser.parse_args()

    reposWithSignedCommitsCount: int = 0
    reposWithSignedCommits: list = []
    dirs: list = next(os.walk("repos"))[1]

    with Bar("Running Commands", max=len(dirs)) as bar:
        dir: str
        for dir in dirs:

            command: str = f"git -C repos/{dir} log --pretty=format:%GK"
            with os.popen(command) as log:
                signed: str = "".join([l.strip() for l in log])
                log.close()

            if len(signed) > 0:
                reposWithSignedCommitsCount += 1
                reposWithSignedCommits.append(dir)
            bar.next()

    with open("verifiedOrganizations.txt", "r") as vo:
        data: list = vo.readlines()
        data = [
            d.strip().replace("-", " ").replace("_", " ").split("/")[-1] for d in data
        ]
        vo.close()

    d: str
    r: str
    voCount: int = 0
    for d in data:
        for r in reposWithSignedCommits:
            test: str = r.strip().replace("_", " ").replace("-", " ")
            if test.find(d) > -1:
                print(d)
                voCount += 1
                break

    print(voCount)


if __name__ == "__main__":
    main()
