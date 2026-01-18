#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:.

echo "Step 1: Generate Dummy Data (Prototype)"
python3 scripts/generate_dummy_data.py --output_dir data/raw/dummy_faces --num 20

echo "Step 2: Factor Extraction"
python3 scripts/stage1_extract.py --input_dir data/raw/dummy_faces --output_path data/processed/factors_dummy.pt

echo "Step 3: Train Factor Encoders"
python3 scripts/stage2_train_encoders.py --factors_path data/processed/factors_dummy.pt --save_path results/models/encoders.pth --epochs 10

echo "Step 4: Run Demo (Generation + Safety + Explainability)"
python3 run_demo.py --num_samples 1 --tau 100.0 --output_dir results/final_demo

echo "Done. Check results/final_demo/report.html"
