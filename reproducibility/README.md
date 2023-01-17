# Reproducibility

├── results
│   ├── image_classification.json
│   └── sentiment_analysis.json
├── README.md
├── arguments.py
├── audit.py
├── classification.py
├── detection.py
├── emotion-models.json
├── image_classification.py
├── main.py
├── manual-compare.py
├── sentiment_analysis.py
├── sentiment_err.json
├── text_classification.py
└── workflow.py

## Results 

contains the results of models after they have been evaluated on their respective dataset
- image classification
- sentiment analysis

## Audit.py

intended to link evaluation files (detection.py etc) together so that a user could specify a model via URL and audit.py would figure out what sort of model it is and fwd it to the proper evaluation pipeline.

## Arguments.py 

basic arguments for audit.py

## Classification.py

intended to peform generic text classification with a model URL on any dataset.
I chose an easier problem (only emotion dataset ... see sentiment_analysis.py) as a test case but could not complete this file in time.
no data is generated from this file but would be useful in future work

## Detection.py

performs object detection with a model URL from COCO2017
requires local COCO dataset

## image_classification.py

performs image classification with a model URL from imagenet
requires local imagenet dataset downloaded

## manual-compare.py

generates validation graphs by comparing recieved valudation results to claimed validation results
- detection
- classification

## sentiment_analysis.py

performs sentiment analysis with a model URL from emotion dataset
