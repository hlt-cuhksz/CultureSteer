# CultureSteer: From Word to World - Evaluate and Mitigate Culture Bias in LLMs via Word Association Test

[![arXiv](https://img.shields.io/badge/arXiv-2505.18562-b31b1b.svg)](https://arxiv.org/abs/2505.18562v2)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## 📖 Overview

CultureSteer is a comprehensive framework for evaluating and mitigating cultural bias in Large Language Models (LLMs) through word association tests. This project addresses the critical issue of cultural bias in AI systems by providing tools to assess and steer language models towards more culturally-aware outputs.

## 🚀 Installation
### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CultureSteer.git
cd CultureSteer
```

2. Install dependencies:
```bash
pip install torch transformers numpy pandas matplotlib seaborn
```

3. Set up environment variables:
```bash
export PYTHONPATH=/path/to/CultureSteer:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"  # Specify GPU device
```

## 📁 Project Structure

```
CultureSteer/
├── culturesteer/           # Core model components
│   ├── model_base.py      # Base model wrapper
│   ├── model_lora_steer.py # LoRA steering implementation
│   ├── steers.py          # Steering mechanisms
│   └── utils.py           # Utility functions
├── dataset/               # Cultural datasets
│   ├── CN_steer.json     # Chinese cultural data
│   ├── USA_steer.json    # US cultural data
│   ├── UK_steer.json     # UK cultural data
│   └── OC_steer.json     # Other cultures data
├── src/                   # Source code
│   ├── steer_trainer.py  # Training pipeline
│   ├── config.py         # Configuration
│   └── result_output/    # Analysis tools
└── script/               # Training scripts
    ├── trainer.sh        # Main training script
    └── cal_pwk.sh        # Evaluation scripts
```

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{dai2025wordworldevaluatemitigate,
  title={From Word to World: Evaluate and Mitigate Culture Bias in LLMs via Word Association Test},
  author={Dai, Xunlian and Zhou, Li and Wang, Benyou and Li, Haizhou},
  journal={arXiv preprint arXiv:2505.18562},
  year={2024}
}
```