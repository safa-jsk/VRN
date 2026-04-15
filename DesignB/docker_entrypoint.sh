#!/bin/bash
# DesignB Docker entrypoint
# Copies the pre-built CUDA extension .so into the mounted project tree
# if it isn't already present (avoids rebuilding on every run).
set -e

EXT_DIR="/workspace/external/marching_cubes_cuda_ext"
BUILT_SO=$(ls /opt/mc_ext/*.so 2>/dev/null | head -1)

if [ -n "$BUILT_SO" ] && [ -d "$EXT_DIR" ]; then
    SO_NAME=$(basename "$BUILT_SO")
    if [ ! -f "${EXT_DIR}/${SO_NAME}" ]; then
        cp "$BUILT_SO" "${EXT_DIR}/"
        echo "[entrypoint] Copied ${SO_NAME} → ${EXT_DIR}/"
    fi
fi

exec "$@"
