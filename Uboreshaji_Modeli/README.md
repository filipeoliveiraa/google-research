# Uboreshaji Modeli – A Comprehensive Brief

## What is This Project?

**Uboreshaji Modeli** Kiswahili for ("model fine-tuning") is a modular,
production-ready framework from Google Research for fine-tuning object
detection models like OWL-v2. It's built to be extensible across different
models while maintaining clean separation of concerns and configuration-driven
workflows.

**Why Use Uboreshaji Modeli?**

This framework aims to:

-   Streamline and accelerate the fine-tuning process for object detection
    tasks.
-   Promote reproducible experiments through versioned configurations.
-   Simplify the adoption of new models and datasets with minimal code changes.
-   Provide a robust and tested pipeline for research and production use cases.

**Target Audience:** ML Researchers and Engineers working on object detection.

## 📁 Folder Structure

```
Uboreshaji_Modeli/
├── README.md                    # Complete project documentation
├── main.py                      # Entry point for fine-tuning experiments
├── example_config.py            # Sample configuration file
├── requirements.txt             # Python dependencies (Python 3.11+)
├── run.sh                       # Quick start & test suite script
├── workflow.ipynb               # Interactive Jupyter notebook walkthrough
├── common/                      # Core utilities & infrastructure
│   ├── augmentations.py         # Augmentation schemas
│   ├── config.py                # Configuration schemas
│   ├── config_utils.py          # Config loading helpers
│   ├── data.py                  # Data pipeline & preprocessing
│   ├── metrics.py               # Evaluation metrics
│   └── trainer.py               # Custom training loop (Hugging Face-based)
└── engines/                     # Model-specific implementations
    ├── factory.py               # Factory pattern for model loading
    └── [model-specific engines] # OWL-v2 and other model handlers
```

## 🎯 Key Features

1. **Configuration-Driven**: Use Python or JSON configs to define experiments.
Override paths via CLI flags (`--model_id`, `--dataset_path`, `--output_dir`)
2. **Robust Data Pipeline**:
   - Two-step class mapping strategy
   - Dynamic class exclusion (e.g., background, discard classes)
   - Stable label-to-ID mappings during training
3. **Decoupled Architecture**:
   - Model-agnostic `CustomTrainer` (extends HF Trainer)
   - Model-specific logic isolated in data collators
   - DETR-style loss with Hungarian Matching
4. **Scalable**: Built for both local experimentation and distributed training

## 🚀 How to Get Started

### 0. **Get the code onto your device**
```bash
git clone --depth=1 https://github.com/google-research/google-research.git
```

NOTE: This clones all projects in the Google Research repository. Our
fine tuning code is just one folder. It has no dependencies on the other
content, so you can just retain ours, or you can see what else Google Research
has been up to!

### 1. **Quick Setup** (Fastest way)
```bash
cd google-research
./Uboreshaji_Modeli/run.sh
```
This automatically sets up a virtual environment, installs dependencies, and
runs the test suite.

### 2. **Manual Setup (with venv)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r Uboreshaji_Modeli/requirements.txt
```

### 3. **Run a Fine-tuning Experiment**
```bash
python3 -m Uboreshaji_Modeli.main \
  --config=Uboreshaji_Modeli/example_config.py \
  --model_id="google/owlv2-base-patch16-ensemble" \
  --dataset_path="/path/to/your/huggingface_dataset" \
  --output_dir="/tmp/outputs"
```

**Evaluation-Only Example:**

To run evaluation on a saved checkpoint without training, update your config
file to set `config.eval.run_eval_only = True` and then run:

```bash
python3 -m Uboreshaji_Modeli.main \
    --config=Uboreshaji_Modeli/common/config.py \
    --model_id="/path/to/your/saved/checkpoint" \
    --output_dir="/tmp/evaluation_outputs"
```

*Note: Evaluation defaults to the `"test"` split if it exists, falling back to
validation if unavailable.*

## 💻 For New Users: Engagement Path

### **Step 1: Understand the Project** (15 min)

- Read the `README.md` for full context
- Skim `example_config.py` to understand configuration structure
- Review the key features above

### **Step 2: Explore the Code** (30 min)

- **`main.py`**: The orchestration logic—shows the complete fine-tuning workflow
- **`common/data.py`**: Data loading, class mapping, preprocessing pipeline
- **`common/trainer.py`**: Custom training loop with loss computation
- **`engines/factory.py`**: How different models are loaded and configured

### **Step 3: Try It Out** (1 hour)

- Run `./run.sh` to verify installation
- Examine `workflow.ipynb` for a guided, step-by-step walkthrough
- Copy `example_config.py` and modify it with your own dataset

### **Step 4: Customize for Your Use Case** (2+ hours)

- Prepare your dataset in the expected format (see data pipeline in `main.py`)
- Create your config file specifying:
  - Model ID (e.g., OWL-v2 variant)
  - Dataset path and class names
  - Training parameters (batch size, learning rate, epochs)
  - Output directory
- Run training: `python -m Uboreshaji_Modeli.main --config=your_config.py`
- Monitor training via TensorBoard logs in the output directory

### **Step 5: Extend & Contribute** (Advanced)

- Add model support in `engines/` if using a different model
- Customize data preprocessing in `common/data.py`
- Enhance metrics in `common/metrics.py`
- Implement custom callbacks in `CustomTrainer`

## 📋 Requirements

- **Python 3.11+** (enforced at runtime)
- **PyTorch 2.0+**, **Transformers 4.36+**, **Datasets 2.14+**
- **GPU recommended** (CPU supported but slow)
- CUDA-enabled device for BF16 mixed precision (optional)

## 📊 What You Get After Training

- **Checkpoints** in `output_dir/run_YYYYMMDD_HHMMSS/`
- **evaluation.json**: Metrics (mAP, per-class performance)
- **config.json**: Your exact experiment configuration
- **TensorBoard logs**: Real-time training curves

## 🔗 License
Apache 2.0 (see file header for details)