#!/bin/bash
RESULT_DIR=result_201203

if [ ! -d $RESULT_DIR ]; then
    mkdir $RESULT_DIR
fi

#python modeling_default.py > $RESULT_DIR/default.txt #&
#python modeling_pruning.py > $RESULT_DIR/pruning_prune90.txt &
#python modeling_decorrelation.py > $RESULT_DIR/decorrelation_lambda1.txt &
#python modeling_pruning+decorrelation.py > $RESULT_DIR/pruning+decorrelation_lambda1+prune90.txt

#python modeling.py --prune_type structured --prune_rate 0.5 > $RESULT_DIR/prune_05.txt
#python modeling.py --prune_type structured --prune_rate 0.6 > $RESULT_DIR/prune_06.txt
#python modeling.py --prune_type structured --prune_rate 0.8 > $RESULT_DIR/prune_08.txt &
#python modeling.py --prune_type structured --prune_rate 0.7 > $RESULT_DIR/prune_07.txt

#python modeling.py --reg reg_cov --odecay 0.9 > $RESULT_DIR/reg_9.txt
#python modeling.py --reg reg_cov --odecay 0.8 > $RESULT_DIR/reg_8.txt
#python modeling.py --reg reg_cov --odecay 0.7 > $RESULT_DIR/reg_7.txt
#python modeling.py --reg reg_cov --odecay 0.6 > $RESULT_DIR/reg_6.txt
#python modeling.py --reg reg_cov --odecay 0.5 > $RESULT_DIR/reg_5.txt

#python modeling.py --prune_type structured --prune_rate 0.5 --reg reg_cov --odecay 0.7 > $RESULT_DIR/prune_05_reg_07.txt &
#python modeling.py --prune_type structured --prune_rate 0.5 --reg reg_cov --odecay 0.8 > $RESULT_DIR/prune_05_reg_08.txt &
#python modeling.py --prune_type structured --prune_rate 0.5 --reg reg_cov --odecay 0.9 > $RESULT_DIR/prune_05_reg_09.txt &
#python modeling.py --prune_type structured --prune_rate 0.6 --reg reg_cov --odecay 0.7 > $RESULT_DIR/prune_06_reg_07.txt &
#python modeling.py --prune_type structured --prune_rate 0.6 --reg reg_cov --odecay 0.8 > $RESULT_DIR/prune_06_reg_08.txt &
python modeling.py --prune_type structured --prune_rate 0.6 --reg reg_cov --odecay 0.9 > $RESULT_DIR/prune_06_reg_09.txt 

