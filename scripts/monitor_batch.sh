#!/bin/bash
# Real-time batch processing monitor

WORK_DIR="/home/safa-jsk/Documents/VRN"
LOG_FILE="$WORK_DIR/data/out/designA/batch_process.log"
TIME_LOG="$WORK_DIR/data/out/designA/time.log"

echo "================================"
echo "VRN Batch Processing Monitor"
echo "================================"
echo ""

# Check if batch is running
if pgrep -f "batch_process_aflw2000" > /dev/null; then
    echo "Status: ✓ RUNNING"
else
    echo "Status: ✗ NOT RUNNING"
fi

echo ""
echo "--- Progress ---"
if [ -f "$LOG_FILE" ]; then
    PROCESSED=$(grep -c "^Processing:" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "Images processed: $PROCESSED"
    
    # Show latest processed image
    LATEST=$(grep "^Processing:" "$LOG_FILE" | tail -1 | sed 's/.*Processing: //' | sed 's/=.*//')
    if [ -n "$LATEST" ]; then
        echo "Latest image: $LATEST"
    fi
fi

echo ""
echo "--- Recent Log Output ---"
if [ -f "$LOG_FILE" ]; then
    tail -10 "$LOG_FILE"
fi

echo ""
echo "--- Timing Statistics ---"
if [ -f "$TIME_LOG" ]; then
    echo "Total timing entries: $(wc -l < $TIME_LOG)"
    echo "Recent timings:"
    tail -5 "$TIME_LOG"
fi

echo ""
echo "--- Output Files Generated ---"
OBJ_COUNT=$(find "$WORK_DIR/data/out/designA" -name "*.obj" -type f 2>/dev/null | wc -l)
NPY_COUNT=$(find "$WORK_DIR/data/out/designA" -name "*.npy" -type f 2>/dev/null | wc -l)
echo "OBJ meshes: $OBJ_COUNT"
echo "NPY volumes: $NPY_COUNT"

echo ""
echo "================================"
