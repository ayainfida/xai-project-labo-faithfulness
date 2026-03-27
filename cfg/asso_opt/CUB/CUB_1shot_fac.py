_base_ = 'CUB_base.py'
n_shots = 1
data_root = 'exp/asso_opt/CUB/CUB_1shot_fac'
lr = 5e-5
bs = 64
# on_gpu = False

concept_type = "all_submodular"
concept_json = 'class2concepts_gemini.json'
