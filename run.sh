#!/bin/bash
RESULT_DIR=result

if [ ! -d $RESULT_DIR ]; then
    mkdir $RESULT_DIR
fi


#python modeling.py --prune_type structured --prune_rate 0.5 > $RESULT_DIR/prune_50.txt
#python modeling.py --prune_type structured --prune_rate 0.6 > $RESULT_DIR/prune_60.txt
#python modeling.py --prune_type structured --prune_rate 0.7 > $RESULT_DIR/prune_70.txt
#python modeling.py --prune_type structured --prune_rate 0.8 > $RESULT_DIR/prune_80.txt

#python modeling.py --reg reg_cov --odecay 0.5 > $RESULT_DIR/reg_05.txt
#python modeling.py --reg reg_cov --odecay 0.6 > $RESULT_DIR/reg_06.txt
#python modeling.py --reg reg_cov --odecay 0.7 > $RESULT_DIR/reg_07.txt
#python modeling.py --reg reg_cov --odecay 0.8 > $RESULT_DIR/reg_08.txt
#python modeling.py --reg reg_cov --odecay 0.9 > $RESULT_DIR/reg_09.txt


#python modeling.py --prune_type structured --prune_rate 0.5 --reg reg_cov --odecay 0.7 > $RESULT_DIR/prune_50_reg_07.txt
#python modeling.py --prune_type structured --prune_rate 0.5 --reg reg_cov --odecay 0.8 > $RESULT_DIR/prune_50_reg_08.txt
#python modeling.py --prune_type structured --prune_rate 0.5 --reg reg_cov --odecay 0.9 > $RESULT_DIR/prune_50_reg_09.txt
python modeling.py --prune_type structured --prune_rate 0.6 --reg reg_cov --odecay 0.7 > $RESULT_DIR/prune_60_reg_07.txt
#python modeling.py --prune_type structured --prune_rate 0.6 --reg reg_cov --odecay 0.8 > $RESULT_DIR/prune_60_reg_08.txt
#python modeling.py --prune_type structured --prune_rate 0.6 --reg reg_cov --odecay 0.9 > $RESULT_DIR/prune_60_reg_09.txt 
