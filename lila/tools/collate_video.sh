#!/bin/bash
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



INPUT_ROOT="$1"
OUTPUT_ROOT="$2"

mkdir -p "$OUTPUT_ROOT"

# 3. Iterate through each subdirectory in the input folder
for seq_dir in "$INPUT_ROOT"/*/; do
    # Check if the glob actually found directories
    [ -d "$seq_dir" ] || continue

    # Extract the sequence name (removes trailing slash and path)
    seq_name=$(basename "$seq_dir")
    
    echo "------------------------------------------------"
    echo "Processing sequence: $seq_name"


	TARGET_DIR=$1

    ffmpeg \
    -framerate 24 -i $seq_dir/%04d_frame.png \
    -framerate 24 -i $seq_dir/%04d_ground.png \
    -framerate 24 -i $seq_dir/%04d_feats.png \
    -framerate 24 -i $seq_dir/%04d_pred.png \
    -filter_complex \
    "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \
    -map "[v]" \
    -c:v libx264 -pix_fmt yuv420p \
    "$OUTPUT_ROOT/$seq_name.mp4"

    if [ $? -eq 0 ]; then
        echo "Success: Created $OUTPUT_ROOT/$seq_name.mp4"
    else
        echo "Error: Failed to process $seq_name"
    fi
done
