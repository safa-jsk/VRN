# Design C – FaceScape Extension (Planned)

| Item | Value |
|---|---|
| Dataset | FaceScape |
| Status | **Skeleton only** – awaiting dataset access |
| CAMFM stages | Same as Design B |

## Planned CLIs

```bash
# Inference on FaceScape data
python3 -m src.designC.infer_facescape \
    --facescape_root /path/to/facescape \
    --splits_csv    splits/test.csv \
    --output_dir    artifacts/meshes_facescape/

# Evaluation
python3 -m src.designC.eval_facescape \
    --pred_dir artifacts/meshes_facescape/ \
    --gt_dir   /path/to/facescape/ground_truth/ \
    --output   artifacts/eval_facescape/metrics.json
```

## Notes

Source stubs are in `src/designC/`.  Each script prints a clear
"FaceScape data not found" message if the dataset is unavailable.
