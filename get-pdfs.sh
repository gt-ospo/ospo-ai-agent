#!/usr/bin/env bash

set -xeuo pipefail

OUT_DIR=$(pwd)/docs
mkdir $OUT_DIR
cd $(mktemp -d)
wget https://github.com/docugami/KG-RAG-datasets/archive/547f132d3c2bcc2b9976ca1d8413b99d8ba45aa1.zip -O datasets.zip
unzip datasets.zip
cp ./KG-RAG-datasets-*/sec-10-q/data/v1/docs/* $OUT_DIR
