# TransMLA: Multi-Head Latent Attention Converter

This project implements the TransMLA approach described in the paper ["TransMLA: Multi-Head Latent Attention Is All You Need"](https://github.com/fxmeng/TransMLA) by Fanxu Meng, Zengwei Yao, and Muhan Zhang. 

The implementation provides tools to convert Group Query Attention (GQA) based models to Multi-Head Latent Attention (MLA) based models to enhance performance while maintaining the same KV cache size.

## Overview

Modern large language models (LLMs) often face communication bottlenecks rather than purely computational limitations. The TransMLA project addresses this issue by converting existing GQA-based models to use MLA, which offers greater expressive power with the same memory requirements.

**Note**: This implementation currently supports LLaMA architecture models. For other model architectures, please refer to the original [TransMLA repository](https://github.com/fxmeng/TransMLA).

### Key Features

- **Model Conversion**: Convert GQA-based models (e.g., LLaMA, Qwen, Mistral) to MLA models
- **Performance Testing**: Tools to benchmark and compare original vs. MLA models
- **KV Cache Reduction**: Maintain the same KV cache size while improving model expressiveness
- **SVD Initialization**: Specialized initialization using Singular Value Decomposition (SVD)

## Installation

```bash
# Clone the repository
git clone https://github.com/bet0x/transmla-converter.git
cd transmla-converter

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Converting a Model

```bash
python transmla_converter.py --model "meta-llama/Llama-3-8B" --output "llama-mla-model" --test
```

### Testing Model Performance

```bash
python transmla_tester.py --model "llama-mla-model" --original "meta-llama/Llama-3-8B" --tokens 100
```

### Fine-tuning a Converted Model

```bash
# Example of fine-tuning a converted model
python transmla_finetune.py --model "llama-mla-model" --dataset "your_dataset.jsonl" --output "fine-tuned-mla"
```

Or finetune using Unsloth via https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo

## How It Works

TransMLA works through several key steps:

1. **Matrix Decomposition**: The key and value projection matrices from GQA are decomposed using SVD.
2. **Low-Rank Factorization**: The matrices are factorized into smaller components (Wa and Wb).
3. **Enhanced Expressiveness**: The decomposition allows for more expressive representation in the same memory footprint.

### Technical Details

The MLA approach introduces compression and decompression matrices:
- `k_compress` and `v_compress`: Map from hidden dimension to latent dimension
- `k_decompress` and `v_decompress`: Map from latent dimension to full attention dimension

This design allows:
- Reduced KV cache size (same as GQA)
- Better expressiveness (theoretically proven superior to GQA)
- Improved performance on downstream tasks

## Results

Our experiments show that TransMLA models consistently outperform their GQA counterparts:

- Faster convergence during fine-tuning
- Higher accuracy on benchmark tasks
- Better performance on coding and mathematical reasoning tasks

## Limitations

- Slight increase in computation during inference
- Minor increase in parameter count (typically <2%)
- Requires fine-tuning to fully realize performance benefits

## Citation

Please cite both the original paper and this implementation:

```bibtex
@article{meng2025transmla,
  title={TransMLA: Multi-Head Latent Attention Is All You Need},
  author={Meng, Fanxu and Yao, Zengwei and Zhang, Muhan},
  journal={arXiv preprint arXiv:2502.07864v2},
  year={2025}
}

@software{ferrer2025transmlaimplementation,
  author = {Ferrer, Alberto},
  title = {TransMLA Converter: Implementation of Multi-Head Latent Attention for LLMs},
  url = {https://github.com/bet0x/transmla-converter},
  year = {2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original TransMLA repository: https://github.com/fxmeng/TransMLA
- HuggingFace for the transformer library
