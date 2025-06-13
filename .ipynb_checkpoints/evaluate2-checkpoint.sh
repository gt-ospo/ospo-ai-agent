#!/usr/bin/env bash

./get.sh $1
2&>1 time MODEL=$1 python evaluate.py < docs/qna_data.csv > results/full_$1
