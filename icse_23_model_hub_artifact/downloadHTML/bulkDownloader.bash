#!/bin/bash

for OUTPUT in $(ls x*)
do
    echo "Downloading malware HTML from file ${OUTPUT}"
    cat $OUTPUT | parallel "python downloadHTML.py -i {}"
    echo "Finished downloading HTML from file ${OUTPUT}"
    echo "Now sleeping for 600 seconds..."
    sleep 600
done
