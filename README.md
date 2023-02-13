# ICSE'23 Artifact

> This is the artifact for the ICSE'23 paper: An Empirical Study of Pre-Trained Model Reuse in the Hugging Face Deep Learning Model Registry

## Update 2/9/2023

We now provide a VirtualBox 7 VM to install and run this software. The VM's image can be downloaded at https://doi.org/10.5281/zenodo.7623127

## About

This repository contains all of the elements of the paper.

### Paper-view

Our artifact includes the following (ordered by appearance in the paper):

| Item | Description | Corresponding content in the paper | Raw data| Scripts/Protocol | Analysis/files to generate figures |Scientific interest | Relation to prior work |
|------|-------------|---------------------|------|-------|----------|-------------|------------------------|
| Motivation | Monthly downloads of the 100 most-downloaded packages from NPM, PyPi, and Hugging Face | Section I, especially Figure 1 |data/transformed packages.csv| icse_23_model_hub_artifact/modelDownloadSaver | icse_23_model_hub_artifact/modelDownloads/downloadList.xlsx| | Compare the popularity of deep learning model registries to traditional package registries|
| Interview_study | Interview protocol, demographic data, transcripts, framework analysis, and saturation calculation | Section IV.A | interview_study/Interview Transcripts with Memos, interview_study/Demographic_Data.csv| interview_study/Interview Protocol - PTNN.docx | interview_study/Interview Transcripts with Memos/Memos.xlsx, interview_study/InterviewSaturation.ipynb || First interview study on Pre-trained model (PTM) reuse |
| Risk measurement | Measurements of potential risks in Hugging Face | Section VIII, especially Figure 6 and 7 | data/modelInformation_8-15-2022.csv, data/fullStratifiedSampleOfModels_8-10-2022.csv |icse_23_model_hub_artifact/downloadHTML, icse_23_model_hub_artifact/downloadModelInformation, icse_23_model_hub_artifact/measureRepositoriesWithSignedCommits, icse_23_model_hub_artifact/measureVerifiedOrganizations, icse_23_model_hub_artifact/modelDocumentation| icse_23_model_hub_artifact/plotData| | First measurements on potential risks in PTM registries|
| HFTorrent Dataset | A dataset for 63,182 open-source PTM packages from Hugging Face model hub | Section IX |data/hftorrent_8-15-2022|icse_23_model_hub_artifact/downloadHFTorrent|| First snapshot of PTM packages | |

We provide all *code* to collect data, analyze data, and generate charts for our paper in the [icse_23_model_hub_artifact](icse_23_model_hub_artifact/) folder. 
<!-- We have provided an installation package for our `Python` code made availible in our repository's release section. -->
Helper bash scripts are also provided to assist with executing our code for batch jobs.

For instructions on how to build our code from source or to download and install the precompiled package, see the [INSTALL](INSTALL) file for more details.
We discuss how to reproduce our results in the [Reproduce Results](#reproduce-results) section of this document.

We provide all *data* (both collected for analysis and used for graphing data) for our paper in the [data](data/) folder.
Additionally, we provide the [HFTorrent Dataset](#hftorrent-dataset) within this repository from August 15th under [this](data/hftorrent_8-15-2022/HFTorrent) subdirectory of the [data](data/) folder.

  For information on how to utilize it locally, see the [INSTALL](INSTALL) file for more details.

### Repo-view

The next table describes the "table of contents" for this repository.

| Top-level folder | Description |
|------------------|-------------|
| `data/`            | Collected data, including motivation data, HFTorrent, and Hugging Face model informations.         |
| `icse_23_model_hub_artifact/` | Scripts that download the data and generate the figures in our paper. |
| `interview_study/` | Interview protocol, transcripts, and analysis |

There are READMEs within each top-level folder for further information.

## Interview Protocol and Data
 This [folder](interview_study) includes:
 - The interview protocol, including interview questions.
 - All the interviews are semi-structured.
 - The demographic data of 12 participants.
 - The transcripts of 12 interviews.
 - The memos we created for analysis.
 - The transcription was performed automatically by service from [Rev](https://www.rev.com/).

## HFTorrent Dataset

The HFTorrent dataset contains the repository histories of the 63,182 PTM packages available on Hugging Face as of August 2022.
They are provided as bare git clones (`git clone --bare`) to reduce space, resulting in a compressed footprint of ~20 GB. 
Each PTM package can be reconstructed to its most recent version, including the model card, architecture, weights, and other elements provided by the maintainers.
Information about how to download the dataset and how to extract it can be found in the [INSTALL](INSTALL) document.

## Reproduce Results

We are *not* applying for ACM's "Reproducible" badge, but we have made an effort to promote reproducibility.

We provided the data used to generate charts for the HuggingFace-related results reported in our paper.
To independently collect this data yourself, you will need to utilize [Git LFS](https://git-lfs.github.com/) to clone the repository.
This is because we host a ~20 GB dataset within this repository.

**Well, not anymore because it costs money. We moved the 20 GB dataset to Zenodo: https://doi.org/10.5281/zenodo.7556031. To access the raw data, download and place in directory [data](data/), then our scripts should work.**

Additionally, you will need to follow the instructions within the [INSTALL](INSTALL) file to install the tools from source or through a precompiled package.
We provide the help details of each script below.

Figures can be generated from the data in the [data](data/) folder manually or through the usage of a script if supported.

### Included Scripts

To get a list of current models and their metadata from the Hugging Face REST API:

- `icse-download-model-information`
```
usage: Download Hugging Face model information to a CSV file

options:
  -h, --help            show this help message and exit
  -a ACCESS_TOKEN, --access-token ACCESS_TOKEN
                        Hugging Face REST API access token
  -o OUTPUT, --output OUTPUT
                        File to output data t
```

To extract a list of current model names from the Hugging Face REST API:

- `icse-extract-model-names`
```
usage: Tool to extract the names of repositories from the Hugging Face REST API to a file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Filename to store the list of repo names
```

To extract the the number of downloads per model from the Hugging Face REST API:

- `icse-extract-model-downloads`
```
usage: Extract the downloads of all models from the Hugging Face REST API to a JSON file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Filename of the output JSON file
  -t TOKEN, --token TOKEN
                        Hugging Face REST API token
```

To plot the output of `icse-extract-model-downloads`:

- `icse-plot-model-downloads`
```
usage: Plot model download data on a logarithmic scale

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        A JSON file containing the number of downloads per model
                        generated by Hugging Face Model Download Extractor
  -o OUTPUT, --output OUTPUT
                        Filename to save plot to
  --no-negatives        Flag to not plot models with -1 downloads (e.g. prohibited
                        access models)
```

To download a model repositories HTML file tree for malware analysis:

- `icse-download-malware-html`
```
usage: Get model information into a usable format

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Model author/repo string to download html
```

To download a list of Hugging Face organizations:

- `icse-download-organization-list`
```
usage: Tool to download the list of organizations from Hugging Face

options:
  -h, --help  show this help message and exit
```

To download a directory of HTML files from `icse-download-malware-html` for malware:

- `icse-scan-for-malware`
```
usage: Scan a directory of Hugging Face model repository HTML file tree files for malware tags and print out the amount of malware found

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory of HTML files to analyze
```

To count the number of repositories that have signed commits from the HFTorrent dataset:

- `icse-measure-repos-with-signed-commits`
```
usage: This tool is to be ran in directory containing HFTorrent or similar dataset

options:
  -h, --help  show this help message and exit
```

To scrape Hugging Face organization webpages (from `icse-download-organization-list`) for verified organizations:

- `icse-measure-verified-organizations`
```
usage: Scrape Hugging Face organization webpages for a verification tag

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input .txt file of HuggingFace.co model URLS
  -o OUTPUT, --output OUTPUT
                        Output .txt file to store verified organization URLs
```

## Code we relied on

- GNU Parallel
```
@software{tange_2022_6682930,
      author       = {Tange, Ole},
      title        = {GNU Parallel 20220622 ('Bongbong')},
      month        = Jun,
      year         = 2022,
      note         = {{GNU Parallel is a general parallelizer to run
                       multiple serial command line programs in parallel
                       without changing them.}},
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.6682930},
      url          = {https://doi.org/10.5281/zenodo.6682930}
}
```
