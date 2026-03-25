# Uboreshaji Modeli

## Project Overview

"Uboreshaji Modeli" is the Kiswahili term for model finetuning.

This code implements a modular and extensible framework for fine-tuning object
detection models like OWL-v2. It is designed to be a robust foundation for
various models and tasks, with a focus on configuration-driven experiments and a
clean separation of concerns.

## Key Features

*   **Configuration-Driven**: Experiments are defined using `ml_collections`.
    Key paths (model, data, output) can be overridden via command-line flags to
    ensure portability and simplify experimentation.
*   **Robust Data Pipeline**: Features a two-step class mapping strategy that
    handles dynamic exclusion of classes (e.g., "background", "discard") while
    guaranteeing a stable label-to-ID mapping for the model during training. It
    also includes data loading and preprocessing utilities tailored for OWL-v2,
    such as specialized bounding box normalization.
*   **Decoupled Trainer and Loss**:
    *   Provides a generic, model-agnostic `CustomTrainer` that extends the
        Hugging Face Trainer.
    *   Model-specific logic, like input tensor reshaping, is correctly isolated
        in the data collator (`collate_fn`), making the trainer reusable for
        future models.
    *   Implements a DETR-style loss via a `SetCriterion` class that uses a
        `HungarianMatcher` for bipartite matching.
*   **Scalable Experimentation**: Includes a `main.py` script for local runs and
    support for launching scalable experiments.

## Quick Start

You can verify the installation and run the test suite by executing the provided
`run.sh` script from the parent directory. This script will setup a virtual
environment and install dependencies:

```bash
./Uboreshaji_Modeli/run.sh
```

## Installation

To install the dependencies for running the `main.py` script, it is recommended
to set up a Python virtual environment:

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r Uboreshaji_Modeli/requirements.txt
```

## Usage

The main entry point for running fine-tuning experiments is `main.py`. You can
configure your experiment using a Python config file or JSON string, and
override specific settings via command-line flags.

**Flags:**

*   `--config`: Path to Python config file. (Required if `--config_json` is not
    provided)
*   `--config_json`: JSON string of the experiment configuration. (Required if
    `--config` is not provided)
*   `--model_id`: Path or ID of the pretrained model. Overrides the value in the
    config.
*   `--dataset_path`: Path to the dataset. Overrides the value in the config.
*   `--output_dir`: Directory to save outputs. Overrides the value in the
    config.

**Example:**

```bash
python3 -m Uboreshaji_Modeli.main \
    --config=Uboreshaji_Modeli/common/config.py \
    --model_id="google/owlv2-base-patch16-ensemble" \
    --dataset_path="/path/to/your/dataset" \
    --output_dir="/tmp/outputs"
```

## License

This project is licensed under the Apache 2.0 License.
