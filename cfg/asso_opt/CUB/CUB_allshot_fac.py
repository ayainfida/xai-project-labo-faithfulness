_base_ = 'CUB_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/CUB/CUB_allshot_fac'
lr = 5e-5
bs = 512

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]

concept_json = 'class2concepts_gemini.json'