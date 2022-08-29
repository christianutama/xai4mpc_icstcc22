This is the repository containing the code used to derive the results of the following paper: 
"Explainable artificial intelligence for deep learning-based model predictive controllers", presented at the 26th International Conference on System Theory, Control and Computing (ICSTCC).

The scripts should be run in the following order:
- Generate training data for NNs by running gen_data.py
- Train a full-sized NN-MPC by running train_network.py
- Calculate the SHAP values distribution for the trained NN-MPC by running calculate_shap_values_all_nn.py
- Train reduced NN-MPCs (XAI NN and PCA NN) by running train_network_xai.py and train_network_pca.py
- Evaluate all controllers by running:
    - main_mpc.py
    - main_nn.py
    - main_nn_xai.py
    - main_nn_pca.py
- Generate a comparison of MPC's and NN-MPC's SHAP values distributions by running:
    - shap_comparison_mpc.py
    - shap_comparison_nn.py
    
Once all the scripts have been run and all the results files have been gathered, run all the scripts in the Jupyter notebook results.ipynb to summarize the results.