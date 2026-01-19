#!/bin/bash
set -e

# Setup Env
export PYTHONPATH=$PYTHONPATH:.

# 1. Check Data (If empty, generate dummy data for prototype run)
if [ ! -d "data/train/real" ] || [ -z "$(ls -A data/train/real)" ]; then
    echo "No training data found. Generating Dummy SOTA Prototype Data..."
    python3 -c "
import os, cv2, numpy as np
for split in ['train', 'test']:
    for cls in ['real', 'fake']:
        os.makedirs(f'data/{split}/{cls}', exist_ok=True)
        for i in range(10): # 10 dummy images per class
            # Create a random image (256x256)
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            # If fake, add some grid artifacts to simulate GAN
            if cls == 'fake':
                img[::4, ::4] = 255 
            cv2.imwrite(f'data/{split}/{cls}/{i:04d}.jpg', img)
"
fi

# 2. Run Training
echo "Starting SOTA Training (Dual-Stream EfficientNetV2)..."
python3 src/trainer.py --config configs/config.yaml

echo "Training Complete. Model saved to results/checkpoints/best_model.pth"
