#!/usr/bin/env bash
# ==============================================================================
# Script: run_training.sh
# Purpose: Train the Genomic Sequence Classification Model on the Augmented Dataset
# Framework: PyTorch + Genomic_Framework Engine
# Output:
#   - Checkpoints: checkpoints_augmented/best_model.pt
#   - Loss/Acc Curves: checkpoints_augmented/training_curves.png
#   - Epoch History JSON: checkpoints_augmented/training_summary.json
#   - Training Log: training.log
# ==============================================================================

set -e

# Configuration
DATASET_DIR="dataset_augmented"
CHECKPOINT_DIR="checkpoints_augmented"
EPOCHS=50
MIN_EPOCHS=25
BATCH_SIZE=16
LEARNING_RATE="1e-4"
PATIENCE=25
SEED=42

echo "=============================================================================="
echo " Starting Model Training with Augmented Dataset"
echo " Dataset Root   : ${DATASET_DIR}"
echo " Checkpoint Dir : ${CHECKPOINT_DIR}"
echo " Epochs (Max)   : ${EPOCHS}"
echo " Min Epochs     : ${MIN_EPOCHS}"
echo " Batch Size     : ${BATCH_SIZE}"
echo " Learning Rate  : ${LEARNING_RATE}"
echo " Patience       : ${PATIENCE}"
echo "=============================================================================="

# Create checkpoint directory if not existing
mkdir -p "${CHECKPOINT_DIR}"

# Execute PyTorch Training Pipeline
python train.py \
    --dataset-root "${DATASET_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --epochs "${EPOCHS}" \
    --min-epochs "${MIN_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --patience "${PATIENCE}" \
    --seed "${SEED}"

echo ""
echo "=============================================================================="
echo " Training Completed Successfully!"
echo " Results Summary:"
echo "   - Training History File: ${CHECKPOINT_DIR}/training_summary.json"
echo "   - Training Curve Plot  : ${CHECKPOINT_DIR}/training_curves.png"
echo "   - Best Model Weights   : ${CHECKPOINT_DIR}/best_model.pt"
echo "=============================================================================="
