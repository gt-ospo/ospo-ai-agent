#!/usr/bin/env bash

time MODEL=$1 python evaluate.py < docs/qna_data_mini.csv > results/mini_$1
