# Faith-Shap: The Faithful Shapley Interaction Index 
This is the code for our JMLR paper "Faith-Shap: The Faithful Shapley Interaction Index". 

## Compatability

This code is tested on Python3.7 with the following packages:

* ktrain == 0.25.1
* tensorflow == 2.1.0
* transformers == 3.1.0
* xgboost == 1.6.2
* h5py == 2.10.0

## Instructions
* To replicate the computational efficiency experiment, run `bash run_exp1.sh`.
* `solver.py` contains our implementation of Faithful Shapley, shapley Taylor, and Shapley interaction indices. 
* To compute Faith-Shap, we first sample each coalition $S \subseteq [d]$ with probability $\propto \frac{d-1}{{d \choose |S|}|S|(d-|S|)}$ in `generate_perturbation.py`, and solve the constrained regression.
* `example.sh` contains an example of applying faith-Shap to generate explanations on Bert trained on IMDB data.


## Citation

If you find this code useful, please cite the following paper:

```
@article{tsai2022faith,
  title={Faith-shap: The faithful shapley shapley interaction index},
  author={Tsai, Che-Ping and Yeh, Chih-Kuan and Ravikumar, Pradeep},
  journal={arXiv preprint arXiv:2203.00870},
  year={2022}
}
```
