#!/bin/bash

username=$1
password=$2
splitReposPath=$3
hftorrentFolder="HFTorrent"

if [ -z "$1" ]
then
    echo "No username supplied"
    exit 1
fi

if [ -z "$2" ]
then
    echo "No password supplied"
    exit 2
fi

if [ -z "$3" ]
then
    splitReposPath="."
fi


files=`ls ${splitReposPath}/x*`

mkdir $hftorrentFolder

for file in $files
do
    echo "Downloading part ${file} of the HFTorrent dataset"
    cat $file | parallel "git clone --bare https://${username}:${password}!@huggingface.co/{} ${hftorrentFolder}/{= \$_=~ s/\//-/g; =}"
    echo "Finished downloading part ${file} of the HFTorrent dataset"
    echo "Now sleeping for 600 seconds to stay under Hugging Face's rate limits"
    sleep 600
done
