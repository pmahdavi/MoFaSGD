# MoFaSGD: Low-rank Momentum Factorization for Memory Efficient Training

This repository contains the official implementation for the paper **"Low-rank Momentum Factorization for Memory Efficient Training"** (TMLR, 2025), which introduces **Momentum Factorized SGD (MoFaSGD)**.

MoFaSGD is a memory-efficient optimizer that enables full-parameter model updates with a memory footprint comparable to parameter-efficient fine-tuning (PEFT) methods like LoRA. It achieves this by maintaining and dynamically updating a low-rank factorization of the first-order momentum at each training step.

This codebase is built upon the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repository and is designed to run the key experiments from the paper.

## Installation

1.  **Clone the Repository**

    Clone the repository and initialize the `GaLore` submodule, which is used as a baseline in the paper.
    ```bash
    git clone --recurse-submodules https://github.com/pmahdavi/MoFaSGD.git
    cd MoFaSGD
    ```
    If you have already cloned the repository without the submodules, you can initialize them with:
    ```bash
    git submodule update --init --recursive
    ```

2.  **Create and Activate the Conda Environment**

    The provided `environment.yml` file contains all the necessary dependencies, including the specific PyTorch nightly build required for a consistent experimental setup.
    ```bash
    conda env create -f environment.yml
    conda activate mofasgd
    ```
    This command will create a new conda environment named `mofasgd` and install all packages, including the local `GaLore` submodule.

## Running Paper Experiments

This repository allows for the execution of the **NanoGPT pre-training experiments** (Section 5.1) from the paper.

### How it Works

The `run.py` script automatically loads the base configuration for the chosen optimizer from the corresponding YAML file in `configs/optimizers/`. You can then override any of these settings using the `--config` argument with a JSON string. The base configurations are set for rank 32 runs.

### Example Commands

#### MoFaSGD (Ours)

This command runs the `MoFaSGD` experiment with rank 16 on 4 GPUs. It loads the base settings from `configs/optimizers/mfsgd.yaml` and applies the specific hyperparameters from the paper as overrides.

```bash
python run.py --optimizer mfsgd --num-gpus 4 \
  --config '{"lr": 0.0009, "rank": 16, "beta": 0.95, "eta1": 0.25, "eta2": 0, "use_current_projection": true, "use_ones_for_nonzero_s": false, "nesterov": false, "eps": 1e-6, "max_value": 1000, "warmup_steps": 300, "beta_start": 0.75, "beta_end": 0.95, "cooldown_frac": 0.4}'
```

#### GaLore (Baseline)

This command runs the `GaLore` baseline experiment with rank 16. It loads base settings from `configs/optimizers/galore.yaml` and applies the specific overrides.

```bash
python run.py --optimizer galore --num-gpus 4 \
  --config '{"lr": 0.008, "group_params": {"rank": 16, "update_proj_gap": 150, "scale": 0.25, "proj_type": "std"}}'
```


## LLaMA-Factory Implementation

For the instruction-tuning experiments on LLaMA-3.1, as detailed in the paper, please see our other repository: [pmahdavi/llama-factory-mfsgd](https://github.com/pmahdavi/llama-factory-mfsgd). That repository contains the implementation of MoFaSGD within the LLaMA-Factory framework.


## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{mahdavinia2025mofasgd,
  title={Low-rank Momentum Factorization for Memory Efficient Training},
  author={Mahdavinia, Pouria and Mahdavi, Mehrdad},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=W3D3TVo9a3}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. 