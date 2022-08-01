# EPTX: a Faithful Explainer for STory-form Algebra

This is a repository for the paper [EPTX (TBA)](https://aclanthology.org/2022.acl-long.305/).

## Prerequisites

- You need Python 3.8+ to run this repository.

- Install libraries using: `pip install -r requirements.txt`

## How to run training/evaluation

- Download dataset files from [here](https://github.com/snucclab/pen), and extract it.
- Modify dataset path specified in the './run.sh' and execute the file. 
- Or, run python script files
  
### Training & Hyperparameter searching
```text
usage: train_model.py [-h] --name NAME --dataset DATASET [--seed SEED]
                      --experiment-dir EXPERIMENT_DIR [--beam-expl BEAM_EXPL]
                      [--beam-expr BEAM_EXPR]
                      [--window-size WINDOW_SIZE [WINDOW_SIZE ...]]
                      [--max-iter MAX_ITER]
                      [--stop-conditions [STOP_CONDITIONS [STOP_CONDITIONS ...]]]
                      [--model {EPT,EPTX,SWAN_A,SWAN_B,SWAN_P1} [{EPT,EPTX,SWAN_A,SWAN_B,SWAN_P1} ...]]
                      [--encoder ENCODER] [--equation-hidden EQUATION_HIDDEN]
                      [--equation-intermediate EQUATION_INTERMEDIATE]
                      [--equation-layer EQUATION_LAYER [EQUATION_LAYER ...]]
                      [--equation-head EQUATION_HEAD]
                      [--explanation-shuffle EXPLANATION_SHUFFLE]
                      [--log-path LOG_PATH] [--num-cpu NUM_CPU]
                      [--num-gpu NUM_GPU] [--opt-lr OPT_LR [OPT_LR ...]]
                      [--opt-beta1 OPT_BETA1] [--opt-beta2 OPT_BETA2]
                      [--opt-eps OPT_EPS] [--opt-grad-clip OPT_GRAD_CLIP]
                      [--opt-warmup OPT_WARMUP [OPT_WARMUP ...]]
                      [--batch-size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit

Dataset & Evaluation:
  --name NAME, -name NAME
  --dataset DATASET, -data DATASET
  --seed SEED, -seed SEED
  --experiment-dir EXPERIMENT_DIR, -exp EXPERIMENT_DIR
  --beam-expl BEAM_EXPL, -beamX BEAM_EXPL
  --beam-expr BEAM_EXPR, -beamQ BEAM_EXPR
  --window-size WINDOW_SIZE [WINDOW_SIZE ...], -win WINDOW_SIZE [WINDOW_SIZE ...]
  --max-iter MAX_ITER, -iter MAX_ITER
  --stop-conditions [STOP_CONDITIONS [STOP_CONDITIONS ...]], -stop [STOP_CONDITIONS [STOP_CONDITIONS ...]]

Model:
  --model {EPT,EPTX,EPTX_U,EPTX_F,EPTX_P1} [{EPT,EPTX,EPTX_U,EPTX_F,EPTX_P1} ...], -model {EPT,EPTX,EPTX_U,EPTX_F,EPTX_P1} [{EPT,EPTX,EPTX_U,EPTX_F,EPTX_P1} ...]
  --encoder ENCODER, -enc ENCODER
  --equation-hidden EQUATION_HIDDEN, -eqnH EQUATION_HIDDEN
  --equation-intermediate EQUATION_INTERMEDIATE, -eqnI EQUATION_INTERMEDIATE
  --equation-layer EQUATION_LAYER [EQUATION_LAYER ...], -eqnL EQUATION_LAYER [EQUATION_LAYER ...]
  --equation-head EQUATION_HEAD, -eqnA EQUATION_HEAD
  --explanation-shuffle EXPLANATION_SHUFFLE, -expS EXPLANATION_SHUFFLE

Logger setup:
  --log-path LOG_PATH, -log LOG_PATH

Worker setup:
  --num-cpu NUM_CPU, -cpu NUM_CPU
  --num-gpu NUM_GPU, -gpu NUM_GPU

Optimization setup:
  --opt-lr OPT_LR [OPT_LR ...], -lr OPT_LR [OPT_LR ...]
  --opt-beta1 OPT_BETA1, -beta1 OPT_BETA1
  --opt-beta2 OPT_BETA2, -beta2 OPT_BETA2
  --opt-eps OPT_EPS, -eps OPT_EPS
  --opt-grad-clip OPT_GRAD_CLIP, -clip OPT_GRAD_CLIP
  --opt-warmup OPT_WARMUP [OPT_WARMUP ...], -warmup OPT_WARMUP [OPT_WARMUP ...]
  --batch-size BATCH_SIZE, -bsz BATCH_SIZE
```

### Training folds based on hyper-parameter search results
```text
usage: train_fold.py [-h] --name NAME --experiment-dir EXPERIMENT_DIR
                     [EXPERIMENT_DIR ...] --model-config MODEL_CONFIG
                     [MODEL_CONFIG ...] [--log-path LOG_PATH]

optional arguments:
  -h, --help            show this help message and exit

Dataset & Evaluation:
  --name NAME, -name NAME
  --experiment-dir EXPERIMENT_DIR [EXPERIMENT_DIR ...], -exp EXPERIMENT_DIR [EXPERIMENT_DIR ...]
  --model-config MODEL_CONFIG [MODEL_CONFIG ...], -model MODEL_CONFIG [MODEL_CONFIG ...]

Logger setup:
  --log-path LOG_PATH, -log LOG_PATH
```

### Evaluating models
```text
usage: test_model.py [-h] --model MODEL [MODEL ...] --dataset DATASET
                     --experiment-dir EXPERIMENT_DIR [--seed SEED]
                     [--faithfulness {sufficiency,comprehensiveness,error_propagation_corr,error_propagation_tree} [{sufficiency,comprehensiveness,error_propagation_corr,error_propagation_tree} ...]]
                     [--bootstrap-trials BOOTSTRAP_TRIALS]
                     [--sample-size SAMPLE_SIZE]
                     [--repeating-counts REPEATING_COUNTS [REPEATING_COUNTS ...]]
                     [--digressing-counts DIGRESSING_COUNTS [DIGRESSING_COUNTS ...]]
                     [--perturbation-samples PERTURBATION_SAMPLES]
                     [--num-cpu NUM_CPU] [--num-gpu NUM_GPU]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL [MODEL ...], -model MODEL [MODEL ...]
  --dataset DATASET, -data DATASET
  --experiment-dir EXPERIMENT_DIR, -exp EXPERIMENT_DIR
  --seed SEED, -seed SEED
  --faithfulness {sufficiency,comprehensiveness,error_propagation_corr,error_propagation_tree} [{sufficiency,comprehensiveness,error_propagation_corr,error_propagation_tree} ...], -faith {sufficiency,comprehensiveness,error_propagation_corr,error_propagation_tree} [{sufficiency,comprehensiveness,error_propagation_corr,error_propagation_tree} ...]
  --bootstrap-trials BOOTSTRAP_TRIALS, -ntr BOOTSTRAP_TRIALS
  --sample-size SAMPLE_SIZE, -smp SAMPLE_SIZE
  --repeating-counts REPEATING_COUNTS [REPEATING_COUNTS ...], -rep REPEATING_COUNTS [REPEATING_COUNTS ...]
  --digressing-counts DIGRESSING_COUNTS [DIGRESSING_COUNTS ...], -dig DIGRESSING_COUNTS [DIGRESSING_COUNTS ...]
  --perturbation-samples PERTURBATION_SAMPLES, -ptr PERTURBATION_SAMPLES
  --num-cpu NUM_CPU, -cpu NUM_CPU
  --num-gpu NUM_GPU, -gpu NUM_GPU
```

- To use BLEURT, please download BLEURT-D6-20.zip from the BLEURT repository and unzip it under 'resource/bleurt'. 

## File structure (expected)

- `/train_model.py` Python script file for hyperparameter searching or training model
- `/train_model_debug.py` Python script file for debug purpose
- `/train_fold.py` Python script file for training cross-validation folds based on the result of `train_model.py`
- `/test_model.py` Python script file for evaluating model
- `/run.sh` Shell script for training models across datasets

- `/common` python package for constants or common functionalities
    - `/common/const` python package for constants
        - `/common/const/model.py` model-specific key names
        - `/common/const/operand.py` definition of constant operands, index of operands, and how to represent it in the text.
        - `/common/const/operator.py` definition of operators, its computational procedure, and how to represent it in the text.
        - `/common/const/pad.py` defines PAD_ID and *_INF values
    - `/common/data` python package for data classes
        - `/common/data/__init__.py` Example-level data class and its information data class
        - `/common/data/base.py` Base abstract classes for data representation
        - `/common/data/explanation.py` Explanation-related data classes
        - `/common/data/encoded.py` Data class for context vectors
        - `/common/data/equation.py` Equation-related data classes
        - `/common/data/label.py` Data class for target labels
        - `/common/data/prediction.py` Data class for predicted result
        - `/common/data/text.py` Data class for text input labels
    - `/common/seed` python package for evaluating SEED dataset
        - `/common/seed/metric.py` Metric for explanation plausibility
        - `/common/seed/parse.py` Parser function for infix equation
        - `/common/seed/pattern.py` Frequently used regex patterns, like numbers or fractions.
        - `/common/seed/solve.py` Solver class that evaluates the output equation, and related functions/constants.
    - `/common/torch` python package for adopting PyTorch
        - `/common/torch/loss.py` Defines Cross-entropy loss with label smoothing
        - `/common/torch/util.py` Various utility function for models
    - `/common/dataset.py` Class for build minibatch and other dataset-specific things 
    - `/common/tester.py` Class for evaluating correctness & plausibility 
    - `/common/trial.py` Utility for building trial directory names (for Ray)

- `/experiment` python package for evaluation experiments
    - `/experiment/__init__.py` Defines experimet loader
    - `/experiment/base.py` Abstrat class for evaluation experiment
    - `/experiment/endtask.py` Classes for error propagation analyses
    - `/experiment/faithfulness.py` Classes for faithfulness experiment

- `/model` python package for model classes
    - `/model/__init__.py` Defines model loader
    - `/model/base` python package for defining base classes
        - `/model/base/beamsearch.py` Utility for beam search
        - `/model/base/chkpt.py` Abstract class for checkpointing a model
        - `/model/base/layernorm.py` Class for LayerNorm with pad masked
        - `/model/base/posenc.py` Defines position encoding
        - `/model/base/util.py` Defines utility function for defining EPT
    - `/model/ept` python package for defining EPT (EMNLP 2020)
        - `/model/ept/__init__.py` Defines EPT model
        - `/model/ept/attention.py` Implements Multi-head Attention and Transformer Decoder layer
        - `/model/ept/decoder.py` Defines decoder part of EPT, except operand-context pointer
        - `/model/ept/encoder.py` Defines encoder part of EPT
    - `/model/eptx` python package for defining EPTX (the proposed model)
        - `/model/eptx/__init__.py` Defines EPTX model (based on EPT)
        - `/model/eptx/explain.py` Defines explanation decoder part of EPTX, except pointer-generator head.
        - `/model/eptx/pg_head.py` Defines pointer-generator head for explanation decoder.

- `/learner` python package for defining learning algorithm
    - `/learner/__init__.py` Defines Supervised Trainer class
    - `/learner/const.py` Defines constants for learner-specific options
    - `/learner/scheduler.py` Defines various warm-up scheduling methods
    - `/learner/util.py` Defines utility for defining Trainer class
    
- `/resource` Directory for storing resources: Please download this folder from [here](https://github.com/snucclab/pen).
    - `/resource/dataset/pen.json` the PEN dataset
    - `/resource/experiments` Directory containing split specifications
        - `/resource/experiments/pen` Directory containing splits of PEN dataset
        - `/resource/experiments/draw` Directory containing splits of DRAW-1K dataset
        - `/resource/experiments/alg514-fold*` Directory containing splits of ALG514-fold* dataset
        - `/resource/experiments/mawps-fold*` Directory containing splits of MAWPS-fold* dataset
        - Each directory has train, test, and/or dev file.

## Citation

Whenever you use this code for any academic purpose, please cite the paper below.

```text
@inproceedings{kim-etal-2022-ept,
    title = "{EPT}-{X}: An Expression-Pointer Transformer model that generates e{X}planations for numbers",
    author = "Kim, Bugeun  and
      Ki, Kyung Seo  and
      Rhim, Sangkyu  and
      Gweon, Gahgene",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.305",
    doi = "10.18653/v1/2022.acl-long.305",
    pages = "4442--4458",
    abstract = "In this paper, we propose a neural model EPT-X (Expression-Pointer Transformer with Explanations), which utilizes natural language explanations to solve an algebraic word problem. To enhance the explainability of the encoding process of a neural model, EPT-X adopts the concepts of plausibility and faithfulness which are drawn from math word problem solving strategies by humans. A plausible explanation is one that includes contextual information for the numbers and variables that appear in a given math word problem. A faithful explanation is one that accurately represents the reasoning process behind the model{'}s solution equation. The EPT-X model yields an average baseline performance of 69.59{\%} on our PEN dataset and produces explanations with quality that is comparable to human output. The contribution of this work is two-fold. (1) EPT-X model: An explainable neural model that sets a baseline for algebraic word problem solving task, in terms of model{'}s correctness, plausibility, and faithfulness. (2) New dataset: We release a novel dataset PEN (Problems with Explanations for Numbers), which expands the existing datasets by attaching explanations to each number/variable.",
}
```

