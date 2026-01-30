#!/bin/bash
# Design C - Training Launcher

set -e

echo "=== Design C: Training Launcher ==="

# Configuration
DATASET_TYPE="${1:-facescape}"
VOLUME_DIR="${2:-data/FaceScape/voxelized_shapes}"
IMAGE_DIR="${3:-}"
EPOCHS="${4:-100}"
BATCH_SIZE="${5:-2}"

echo "Dataset type: $DATASET_TYPE"
echo "Volume dir: $VOLUME_DIR"
echo "Image dir: $IMAGE_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo

# Check if volume directory exists
if [ ! -d "$VOLUME_DIR" ]; then
    echo "Error: Volume directory not found: $VOLUME_DIR"
    exit 1
fi

# Build command
CMD="python3 designC/train.py \
    --dataset-type $DATASET_TYPE \
    --volume-dir $VOLUME_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --loss-type bce \
    --lr 1e-4 \
    --checkpoint-dir designC/checkpoints \
    --log-dir designC/logs"

# Add image dir if specified
if [ -n "$IMAGE_DIR" ]; then
    CMD="$CMD --image-dir $IMAGE_DIR"
fi

echo "Running command:"
echo "$CMD"
echo

# Run training
eval $CMD

echo
echo "✓ Training complete"
echo "Checkpoints: designC/checkpoints/"
echo "Logs: designC/logs/"
echo
echo "View TensorBoard:"
echo "  tensorboard --logdir designC/logs"
