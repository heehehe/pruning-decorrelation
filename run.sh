#!/bin/bash
RESULT_DIR=result_201104

if [ ! -d $RESULT_DIR ]; then
    mkdir $RESULT_DIR
fi

#python modeling_default.py > $RESULT_DIR/default.txt #&
python modeling_pruning.py > $RESULT_DIR/pruning_prune90.txt &
python modeling_decorrelation.py > $RESULT_DIR/decorrelation_lambda1.txt &
python modeling_pruning+decorrelation.py > $RESULT_DIR/pruning+decorrelation_lambda1+prune90.txt
