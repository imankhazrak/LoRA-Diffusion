#!/bin/bash
# Quick script to verify all results are saved after training

OUTPUT_DIR=$1
CHECKPOINT_DIR=$2

if [ -z "$OUTPUT_DIR" ] || [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <output_dir> <checkpoint_dir>"
    echo "Example: $0 /fs/scratch/users/\$USER/lora-diffusion/outputs/sst2_lora_diffusion_12345 /fs/scratch/users/\$USER/lora-diffusion/checkpoints/sst2_lora_diffusion_12345"
    exit 1
fi

echo "=========================================="
echo "Verifying Results Saving"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Check output directory files
echo "Checking output directory files:"
[ -f "$OUTPUT_DIR/training_history.json" ] && echo "  ✓ training_history.json exists" || echo "  ✗ training_history.json MISSING"
[ -f "$OUTPUT_DIR/evaluation_history.json" ] && echo "  ✓ evaluation_history.json exists" || echo "  ✗ evaluation_history.json MISSING"
[ -f "$OUTPUT_DIR/training_summary.json" ] && echo "  ✓ training_summary.json exists" || echo "  ✗ training_summary.json MISSING"
echo ""

# Check checkpoint directories
echo "Checking checkpoint directories:"
[ -d "$CHECKPOINT_DIR/best_model" ] && echo "  ✓ best_model checkpoint exists" || echo "  ✗ best_model MISSING"
[ -d "$CHECKPOINT_DIR/final_model" ] && echo "  ✓ final_model checkpoint exists" || echo "  ✗ final_model MISSING"
echo ""

# Check checkpoint contents
if [ -d "$CHECKPOINT_DIR/best_model" ]; then
    echo "Checking best_model checkpoint contents:"
    [ -f "$CHECKPOINT_DIR/best_model/config.json" ] && echo "  ✓ config.json exists" || echo "  ✗ config.json MISSING"
    [ -f "$CHECKPOINT_DIR/best_model/lora_module.pt" ] && echo "  ✓ lora_module.pt exists" || echo "  ✗ lora_module.pt MISSING"
    [ -f "$CHECKPOINT_DIR/best_model/optimizer.pt" ] && echo "  ✓ optimizer.pt exists" || echo "  ✗ optimizer.pt MISSING"
    [ -f "$CHECKPOINT_DIR/best_model/checkpoint_metadata.json" ] && echo "  ✓ checkpoint_metadata.json exists" || echo "  ✗ checkpoint_metadata.json MISSING"
    echo ""
fi

# Check file sizes
echo "File sizes:"
if [ -f "$OUTPUT_DIR/training_history.json" ]; then
    SIZE=$(du -h "$OUTPUT_DIR/training_history.json" | cut -f1)
    LINES=$(wc -l < "$OUTPUT_DIR/training_history.json" 2>/dev/null || echo "0")
    echo "  training_history.json: $SIZE ($LINES lines)"
fi

if [ -f "$OUTPUT_DIR/evaluation_history.json" ]; then
    SIZE=$(du -h "$OUTPUT_DIR/evaluation_history.json" | cut -f1)
    LINES=$(wc -l < "$OUTPUT_DIR/evaluation_history.json" 2>/dev/null || echo "0")
    echo "  evaluation_history.json: $SIZE ($LINES lines)"
fi

if [ -f "$CHECKPOINT_DIR/best_model/lora_module.pt" ]; then
    SIZE=$(du -h "$CHECKPOINT_DIR/best_model/lora_module.pt" | cut -f1)
    echo "  lora_module.pt: $SIZE"
fi

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
