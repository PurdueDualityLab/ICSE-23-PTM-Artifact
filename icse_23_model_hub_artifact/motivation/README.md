# Motivation

## clone.py

used to clone all 67K repos

## data.json, data-backup.json

contains model metadata for 67K+ repos.  I have been iteratively updating when new models are released
used in transparency.py

## ignore.json

some models require a sign in to get access and others are restricted (for various reasons ie: gpt-4chan)
we ignore those

## main.py

Generate graphs.

## modelinfo.py

updates data in the data.json to contain most up to date info on 67K repos.  

## readme_stats.json

## transparency.py

this file does most of my analysis.  It loads in all the models and their associated metadata as PTM objects and makes it easy to find trends among tasks, downloads, claims, etc
the main() function is sort of a mess.

everything else in the file is pretty organized
