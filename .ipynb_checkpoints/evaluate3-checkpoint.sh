#!/usr/bin/env bash

set -eux

./get.sh $1
time 2>&1 MODEL=$1 python evaluate.py < docs/qna_data.csv > results3/full3_$1
