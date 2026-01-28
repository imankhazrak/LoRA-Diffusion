#!/bin/bash
# Script to pre-download models and datasets before running jobs
# Run this on a login node (not compute node) to avoid network timeouts

set -e

echo "=========================================="
echo "Pre-downloading Models and Datasets"
echo "=========================================="

# Activate environment
cd ~/LoRA-Diffusion
source venv/bin/activate

# Set cache directories
export HF_HOME=$HOME/LoRA-Diffusion/data/hf_cache
export TRANSFORMERS_CACHE=$HOME/LoRA-Diffusion/data/transformers_cache
export HF_DATASETS_CACHE=$HOME/LoRA-Diffusion/data/datasets_cache
export TORCH_HOME=$HOME/LoRA-Diffusion/data/torch_cache
export HUGGINGFACE_HUB_CACHE=$HOME/LoRA-Diffusion/data/hub_cache

# Create cache directories
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $TORCH_HOME $HUGGINGFACE_HUB_CACHE

echo "Downloading BERT tokenizer..."
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"

echo "Downloading SST-2 dataset..."
python -c "from datasets import load_dataset; load_dataset('glue', 'sst2', split='train[:100]')"

echo "Downloading SQuAD dataset..."
python -c "from datasets import load_dataset; load_dataset('squad', split='train[:100]')"

echo "Downloading XSum dataset..."
python -c "from datasets import load_dataset; load_dataset('xsum', split='train[:100]')"

echo ""
echo "=========================================="
echo "Pre-download Complete!"
echo "=========================================="
echo "Cache locations:"
echo "  Tokenizer: $TRANSFORMERS_CACHE"
echo "  Datasets: $HF_DATASETS_CACHE"
echo ""
echo "You can now submit jobs without network timeouts."
