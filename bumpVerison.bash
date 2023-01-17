#!/bin/bash

if [ -z "$1" ]
then
    echo "No version supplied"
    exit 1
fi

poetry version $1
poetry version -s | xargs git tag
