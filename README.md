# CultureSteer: From Word to World - Evaluate and Mitigate Culture Bias in LLMs via Word Association Test

[![arXiv](https://img.shields.io/badge/arXiv-2505.18562-b31b1b.svg)](https://arxiv.org/abs/2505.18562v2)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## 📖 Overview

CultureSteer is a comprehensive framework for evaluating and mitigating cultural bias in Large Language Models (LLMs) through word association tests. This project addresses the critical issue of cultural bias in AI systems by providing tools to assess and steer language models towards more culturally-aware outputs.

## 🎯 Key Features

- **Cultural Bias Evaluation**: Systematic assessment of cultural bias across different regions (USA, UK, OC, CN)
- **Steering Mechanisms**: Multiple adaptor classes (multiply, add, offset) for model steering
- **Multi-Model Support**: Compatible with Llama and Qwen model families
- **Comprehensive Datasets**: Word association datasets covering 17 cultural domains
- **LoRA Integration**: Efficient fine-tuning with Low-Rank Adaptation
- **Cross-Cultural Analysis**: Tools for analyzing cultural differences in model outputs

## 🏗️ Architecture

The framework consists of several key components:

- **Base Model**: `LMSteerBase` - Core model wrapper with steering capabilities
- **Steering Module**: `Projected_Adaptor` - Implements different steering mechanisms
- **Training Pipeline**: End-to-end training with cultural bias mitigation
- **Evaluation Tools**: Comprehensive metrics and visualization tools

## 📊 Supported Cultural Domains

The framework evaluates cultural bias across 17 semantic domains:

1. Physical world
2. Kinship
3. Animals
4. The body
5. Food and drink
6. Clothing and grooming
7. The house
8. Agricultural and vegetation
9. Action and technology
10. Motion
11. Possession
12. Spatial relations
13. Quantity
14. Time
15. Sense perception
16. Emotion and values
17. Cognition

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (recommended for GPU acceleration)

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

## 🔧 Usage

### Training a Cultural Steering Model

1. **Basic Training**:
```bash
cd script
bash trainer.sh
```

2. **Custom Training**:
```python
from culturesteer.get_model import get_model
from src.steer_trainer import main

# Load model and tokenizer
model, tokenizer = get_model("Llama")  # or "Qwen"

# Train with cultural steering
main(args)
```

### Evaluating Cultural Bias

```python
from culturesteer.model_base import LMSteerBase

# Load trained model
model = LMSteerBase.load_from_checkpoint("path/to/checkpoint")

# Generate culturally-steered text
text, scores = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    steer_values=[0, 0, 0, 1],  # Steer towards Chinese culture
    seed=42
)
```

### Cross-Cultural Analysis

```bash
# Calculate cross-cultural steering performance
bash script/cal_pwk_cross_steer.sh

# Generate evaluation tables
python src/result_output/table_main.py

# Create visualization plots
python src/result_output/plot_main.py
```

## 📈 Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **Top-K Accuracy**: Performance at different K values (3, 5, 10, 20)
- **Cultural Alignment**: How well outputs align with target cultures
- **Cross-Cultural Transfer**: Performance across different cultural domains
- **Steering Effectiveness**: Impact of steering mechanisms

## 🎛️ Configuration

Key configuration options in `src/config.py`:

```python
MODEL_PATH = {
    'Llama': "/path/to/Llama-3.1-8B",
    "Qwen": "/path/to/Qwen2.5-7B",
    # Add your model paths
}
```

## 📊 Results

The framework generates detailed results including:

- Performance tables across cultural domains
- Visualization plots for cultural bias analysis
- Cross-cultural steering effectiveness metrics
- Comparative analysis between different models

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{dai2024word,
  title={From Word to World: Evaluate and Mitigate Culture Bias in LLMs via Word Association Test},
  author={Dai, Xunlian and Zhou, Li and Wang, Benyou and Li, Haizhou},
  journal={arXiv preprint arXiv:2505.18562},
  year={2024}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to the open-source community for the foundational models
- Special thanks to contributors who helped with dataset collection and validation
- Appreciation for the cultural experts who provided domain knowledge

## 📞 Contact

For questions and support:
- Email: [your-email@domain.com]
- Issues: [GitHub Issues](https://github.com/yourusername/CultureSteer/issues)

---

**Note**: This project is part of ongoing research in cultural bias mitigation for large language models. Please refer to the paper for detailed methodology and results.