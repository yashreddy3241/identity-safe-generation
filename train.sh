#!/bin/bash
set -e

# Setup Env
export PYTHONPATH=$PYTHONPATH:.

# 1. Check Data
if [ ! -d "data/train/real" ] || [ -z "$(ls -A data/train/real)" ]; then
    echo "ERROR: No training data found in data/train/real!"
    echo "Please place your StarGAN dataset images in:"
    echo "  - data/train/real/"
    echo "  - data/train/fake/"
    echo "  - data/test/real/"
    echo "  - data/test/fake/"
    exit 1
fi

# 2. Run Training
echo "Starting SOTA Training (Dual-Stream EfficientNetV2)..."
python3 src/trainer.py --config configs/config.yaml

echo "Training Complete. Model saved to results/checkpoints/best_model.pth"
