#!/bin/bash

bareRepoPath=$1
newRepoPath=$2

if [ -z "$1" ]
then
    echo "No git directory supplied"
    exit 1
fi

if [ -z "$2" ]
then
    echo "No new git directory supplied"
    exit 2
fi

git clone -s $bareRepoPath $newRepoPath
