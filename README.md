# Evaluating Visual Faithfulness in LLM-Based Concept Bottleneck Models

This project is based on the LaBo codebase from the CVPR 2023 paper ["Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification"](https://arxiv.org/abs/2211.11158).

## Set up environment
We recommend Python 3.9.x and a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
You need to modify the source code of Apricot to run the submodular optimization. See details [here](https://github.com/YueYANG1996/LaBo/issues/1).

## Directories
* `cfg/` saves the config files for all experiments, including linear probe (`cfg/linear_probe`) and LaBo (`cfg/asso_opt`). You can modify the config files to change the system arguments.
* `datasets/` stores the dataset-specific data, including `images`, `splits`, and `concepts`. Please check `datasets/DATASET.md` for details. 

	**Note**: the images of each dataset are not provided in this repo; you need to download them and store in the corresponding folder: `datasets/{dataset name}/images/`. Check `datasets/DATASET.md` for instructions on downloading all datasets.
	
* `exp/` is the work directories of the experiments. The config files and model checkpoints will be saved in this folder.
* `models/` saves the models:
	* Linear Probe: `models/linear_prob/linear_prob.py`
	* LaBo: `models/asso_opt/asso_opt.py`
	* concept selection functions: `models/select_concept/select_algo.py`
* `output/`: text performance logs (e.g., accuracies) saved as `.txt` files.
* `results/`: JSON faithfulness evaluation files (per-image and `_summary.json`) produced by `llava_score.py` and related analysis scripts.
* Other files: 
	* `data.py` and `data_lp.py` are the dataloaders for LaBo and Linear Probe, respectively.
	* `main.py` is the interface to run all experiments, and `utils.py` contains the preprocess and feature extraction functions.
	* `linear_probe.sh` is the bash script to run the linear probe. `labo_train.sh` and `labo_test.sh` are the bash scripts to train and test LaBo.

## Linear Probe
To get the linear probe performance, just run:

```
sh linear_probe.sh {DATASET} {SHOTS} {CLIP SIZE}
```
For example, for flower dataset 1-shot with ViT-L/14 image encoder, the command is:

```
sh linear_probe.sh flower 1 ViT-L/14
```

The code will automatically encode the images and run a hyperparameter search on the L2 regularization using the dev set. The best validation and test performance will be saved in the `output/linear_probe/{DATASET}.txt`.

## LaBo Training
To train the LaBo, run the following command:

```
sh labo_train.sh {SHOTS} {DATASET}
```
The training logs will be uploaded to the `wandb`. You may need to set up your `wandb` account locally. After reaching the maximum epochs, the checkpoint with the highest validation accuracy and the corresponding config file will be saved to `exp/asso_opt/{DATASET}/{DATASET}_{SHOT}shot_fac/`.

## LaBo Testing
To get the test performance, use the model checkpoint and corresponding configs saved in `exp/asso_opt/{DATASET}/{DATASET}_{SHOT}shot_fac/` and run:

```bash
sh labo_test.sh {CONFIG_PATH} {CHECKPOINT_PATH}
```
The test accuracy will be printed to `output/asso_opt/{DATASET}.txt`.

## LLaVA-based faithfulness scoring
After training LaBo, you can run the LLaVA-based faithfulness audit with:

```bash
python llava_score.py \
  --ckpt path/to/checkpoint.ckpt \
  --split_file path/to/split.pkl \
  --image_dir datasets/{DATASET}/images \
  --save_path results/{DATASET}_faith.json \
  --n_per_class 2 \
  --top_k 5 \
  --seed 42
```

This will create a JSON file with per-image faithfulness scores and a corresponding `{DATASET}_faith_summary.json` file with aggregate statistics in `results/`.

## Computing FAITH@k
Given the JSON produced by `llava_score.py`, you can compute **FAITH@k** statistics with:

```bash
python faith.py --input_json results/{DATASET}_faith.json
```

This prints **FAITH@k** scores for k = 1, 2, 3, 4, 5.