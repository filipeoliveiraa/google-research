### Disclaimer: This is not an officially supported Google product.

This code accompanies a paper: 
**Featurising Pixels from Dynamic 3D Scenes with Linear In-Context Learners**
Nikita Araslanov, Martin Sundermeyer, Hidenobu Matsuki, David Joseph Tan, Federico Tombari

# Training & Evaluation

This guide provides instructions for configuring training runs, accessing pre-trained models, and evaluating tasks.

---

## Training

To start a training run, you need to add a configuration entry to the `./launch/train_many.sh` script.

### Configuration Format
Clone `RAFT` repository into `raft/` directory (e.g. you should have `raft/raft.py`).
Edit the `RUNS` array in the script using the following syntax:

```bash
RUNS=(
    "<dataset> <model> <run-id> <optional-args>"
)
```
#### Parameters

* **`<dataset>`**: `ytvos` or `davis`
* **`<model>`**: The architecture to use (e.g., `dinov2_vitb14`)
* **`<run-id>`**: A unique identifier for the run
* **`<optional-args>`**: Overrides for configuration defaults (optional)

### Examples
1. *Basic Run Train* a DINOv2 model with a ViT-B backbone and default parameters on YouTube-VOS:
```bash
RUNS=(
    "ytvos dinov2_vitb14 baseline"
)
```
2. *Custom Batch Size* Train the same model but override the batch size to 32:
```bash
RUNS=(
    "ytvos dinov2_vitb14 baseline train_loader.batch_size=32"
)
```
**Note:** For a full list of configurable options, refer to `configs/train_ytvos.yaml`.

## Evaluation
To evaluate any task, the workflow is standard across models.

1. Configure the Job
Add the task to launch/eval_many by modifying the JOBS array:

```bash
JOBS=(
    # ... other jobs
    "<model-config-name> <task>"
)
```
Parameters:

* `<model-config-name>`: Specifies the model to evaluate. The script automatically infers the architecture, backbone, and parameter sizes.
* `<task>`: Specifies the evaluation mode.

Task Options:

* `vos, vosknn, seg, normals, openseg`: Evaluates two variants (encoder features alone AND joint encoder-decoder).
* `vos:<spec>`: Evaluates a specific setup.
* `vos:enc` (Encoder-only)
* `vos:full` (Encoder-decoder)

2. Run the Script
Execute the launch script (no arguments needed):

```bash
./launch/eval_many.sh
```

3. View Results
Check Tensorboard for the evaluation metrics and results.
