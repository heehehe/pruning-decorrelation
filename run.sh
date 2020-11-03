#!/bin/bash

python modeling_default.py > result/default_.txt &&
python modeling_pruning.py > result/pruning_.txt &&
python modeling_decorrelation.py > result/decorrelation_.txt &&
python modeling_pruning+decorrelation.py > result/pruning+decorrelation.txt