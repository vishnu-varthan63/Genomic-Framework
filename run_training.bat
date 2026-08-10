@echo off
REM ==============================================================================
REM Script: run_training.bat
REM Purpose: Train the Genomic Sequence Classification Model on Augmented Dataset (Windows)
REM ==============================================================================

set DATASET_DIR=dataset_augmented
set CHECKPOINT_DIR=checkpoints_augmented
set EPOCHS=50
set MIN_EPOCHS=25
set BATCH_SIZE=16
set LEARNING_RATE=1e-4
set PATIENCE=25
set SEED=42

echo ==============================================================================
echo  Starting Model Training with Augmented Dataset
echo  Dataset Root   : %DATASET_DIR%
echo  Checkpoint Dir : %CHECKPOINT_DIR%
echo  Epochs (Max)   : %EPOCHS%
echo  Min Epochs     : %MIN_EPOCHS%
echo  Batch Size     : %BATCH_SIZE%
echo  Learning Rate  : %LEARNING_RATE%
echo  Patience       : %PATIENCE%
echo ==============================================================================

if not exist %CHECKPOINT_DIR% mkdir %CHECKPOINT_DIR%

python train.py ^
    --dataset-root %DATASET_DIR% ^
    --checkpoint-dir %CHECKPOINT_DIR% ^
    --epochs %EPOCHS% ^
    --min-epochs %MIN_EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --lr %LEARNING_RATE% ^
    --patience %PATIENCE% ^
    --seed %SEED%

echo.
echo ==============================================================================
echo  Training Completed Successfully!
echo  Results Summary:
echo    - Training History File: %CHECKPOINT_DIR%\training_summary.json
echo    - Training Curve Plot  : %CHECKPOINT_DIR%\training_curves.png
echo    - Best Model Weights   : %CHECKPOINT_DIR%\best_model.pt
echo ==============================================================================
