#!/bin/bash

#Bert on imdb
# Unzip imdb data in data/Imdb_2senternce

n_instance=3
exp_dir=exp/computation_eff/nlp
mkdir -p $exp_dir
data_path=$exp_dir/imdb_2_sentence_$n_instance
#python3 generate_perturbation.py -gpu 2 -dataset imdb \
#				-generate_data_path $data_path -n_test_data $n_instance -min_length_limit 15 \
#				-max_length_limit 15 -gen_method banzhaf_all && exit 1
#exit 1

python3 -u computation_compare_nlp.py -seed 2022 -exp_dir $exp_dir -l 2 -n_trials 20 -n_intn $n_instance \
							-data_path $data_path\.train -min_d 15 -max_d 15 \
							-cpt_start 1000 -cpt_diff 400 -cpt_end 4200 -lasso 1e-4 | tee $exp_dir/lasso1e-4_instance$n_instance.log  
exit 1

#bank
exp_dir=exp/computation_eff/bank
mkdir -p $exp_dir
data_path=$exp_dir/bank_50instance

python3 generate_perturbation.py -gpu 1 -dataset bank -n_test_data 50 \
	-generate_data_path $data_path -gen_method banzhaf_all && exit 1


exp_dir=exp/bank/l2/
mkdir -p $exp_dir
python3 -u computation_compare_bank.py -seed 2022 -exp_dir $exp_dir -l 2 -n_trials 20 -n_intn 50  \
									-data_path $data_path\.train \
									-cpt_start 1000 -cpt_diff 700 -cpt_end 8001 -lasso 1e-6  
