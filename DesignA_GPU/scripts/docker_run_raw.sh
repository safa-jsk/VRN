#!/bin/bash
# Modified VRN runner for DesignA_GPU
# Runs inside asjackson/vrn:latest container
# Produces .raw volume file instead of .obj mesh (GPU marching cubes done on host)
#
# Same argument convention as /runner/run.sh:
#   $1 = this script path (entrypoint prefix)
#   $2 = INPUT_WDIR (full path to input image inside container, e.g. /data/300W_LP/AFW/img.jpg)
# Outputs:
#   ${INPUT_WDIR}.raw   – VRN volume (200×192×192 uint8)
#   ${INPUT_WDIR}.crop.jpg – cropped face JPEG

source /etc/profile
source /root/usr/local/torch/install/bin/torch-activate

cd /runner

export TERM=dumb
TMPDIR=/tmp/
INPUT_WDIR=$2
INPUT=$(basename "$INPUT_WDIR")

echo "Please wait. Your image is being processed."

convert -auto-orient "$INPUT_WDIR" "$TMPDIR/$INPUT"

pushd face-alignment > /dev/null
th main.lua -model 2D-FAN-300W.t7 \
   -input "$TMPDIR/$INPUT" \
   -detectFaces true \
   -mode generate \
   -output "$TMPDIR/$INPUT.txt" \
   -device cpu \
   -outputFormat txt

exit_code=$?
popd > /dev/null

if [ ! -f "$TMPDIR/$INPUT.txt" ]; then
    rm -f "$TMPDIR/$INPUT"
    echo "The face detector failed to find your face."
    exit 1
fi

if [ $exit_code -ne 0 ]; then
    echo "Error occurred while running the face detector"
    exit 1
fi

awk -F, 'BEGIN {
              minX=100000;
              maxX=0;
              minY=100000;
              maxY=0;
            }
            $1 > maxX { maxX=$1 }
            $1 < minX { minX=$1 }
            $2 > maxY { maxY=$2 }
            $2 < minY { minY=$2 }
            END {
              scale=90/sqrt((minX-maxX)*(minY-maxY));
              width=maxX-minX;
              height=maxY-minY;
              cenX=width/2;
              cenY=height/2;
              printf "%s %s %s\n",
                (minX-cenX)*scale,
                (minY-cenY)*scale,
                (scale)*100
   }' "$TMPDIR/$INPUT.txt" > "$TMPDIR/$INPUT.crop"

cat "$TMPDIR/$INPUT.crop" | \
    while read x y scale; do
    convert "$TMPDIR/$INPUT" \
        -scale $scale% \
        -crop 192x192+$x+$y \
        -background white \
        -gravity center \
        -extent 192x192 \
        "$TMPDIR/$INPUT"

    if [ $? -ne 0 ]; then
        echo "Error occurred while cropping the image."
        exit 1
    fi

    echo "Cropped and scaled $INPUT"
done

rm -f "$TMPDIR/$INPUT.crop"

# VRN inference – Docker image lacks cunn, use cpu device for Torch7
# GPU acceleration is in the host-side CUDA marching cubes (DesignA_GPU)
th process.lua \
   --model vrn-unguided.t7 \
   --input "$TMPDIR/$INPUT" \
   --output "$TMPDIR/$INPUT.raw" \
   --device cpu

if [ $? -ne 0 ]; then
    echo "Error occurred while regressing the 3D volume."
    exit 1
fi

# Save crop JPEG (same as original runner)
cp "$TMPDIR/$INPUT" "${INPUT_WDIR}.crop.jpg"

# Output .raw volume (GPU marching cubes runs on host, not here)
cp "$TMPDIR/$INPUT.raw" "${INPUT_WDIR}.raw"

# Cleanup temp files
rm -f "$TMPDIR/$INPUT.txt" "$TMPDIR/$INPUT.raw" "$TMPDIR/$INPUT"

echo "Volume saved: ${INPUT_WDIR}.raw"
