#!/bin/bash

if [ -z "$1" ]
then
    echo "No repo names file supplied"
    exit 1
fi

split -l 1000 $1
