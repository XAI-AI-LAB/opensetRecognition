#!/bin/bash

data_path="$(pwd)/data"

if [ -d "$data_path" ]
then
    echo "$data_path exists."
else
    mkdir data
    echo "data created"
fi

cd data

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

rm tiny-imagenet-200.zip

