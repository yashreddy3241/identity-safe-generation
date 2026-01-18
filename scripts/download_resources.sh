#!/bin/bash
set -e

echo "Downloading SOTA StyleGAN2 Weights (FFHQ)..."
mkdir -p results/models

# Official NVIDIA Mirror
URL="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
OUTPUT="results/models/ffhq.pkl"

if [ -f "$OUTPUT" ]; then
    echo "File $OUTPUT already exists. Skipping download."
else
    curl -L --progress-bar "$URL" -o "$OUTPUT"
    echo "Download complete: $OUTPUT"
fi

echo "Setup complete. You can now run the demo with Real Generation!"
