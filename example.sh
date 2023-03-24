#!/bin/bash
n_instance=5
exp_dir=exp/example
mkdir -p $exp_dir

data_path=$exp_dir/faith_shap_pert
python3 generate_perturbation.py -gpu 1 -dataset imdb \
	-generate_data_path $data_path -n_test_data $n_instance -n_train_perturbations 4000 \
	-n_valid_perturbations 0 -max_length_limit 20 -gen_method faith_shap 

python3 -u example.py -method faith_shap -dataset imdb -T 2 -train_data_path $data_path.train \
		 -lasso_alpha 0.01  | tee $output_txt




