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



ROOT_DIR="/usr/local/google/home/araslanov/gsfuse/araslanov/logs/codelab_evals"

DIR_ENC_ROOT="${ROOT_DIR}/$1"
DIR_DEC_ROOT="${ROOT_DIR}/$2"
OUTPUT_ROOT="$3"

SEQ_LIST="bear blackswan breakdance-flare bus camel paragliding dog drift-turn flamingo mallard-water rhino snowboard"
#SEQ_LIST="bear blackswan breakdance-flare bus camel dog drift-turn flamingo goat"

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# 2. Loop through the Decoder directory (assuming it dictates the sequences)
#for dec_seq_path in "$DIR_DEC_ROOT"/*/; do
for seq_name in $SEQ_LIST; do

    dec_seq_path=${DIR_DEC_ROOT}/${seq_name}

    # Safety check: directory exists
    [ -d "$dec_seq_path" ] || continue

    # Get the sequence name (e.g., "bear")
    seq_name=$(basename "$dec_seq_path")

    # Construct the corresponding Encoder path
    enc_seq_path="$DIR_ENC_ROOT/$seq_name"

    # 3. Check if the matching sequence exists in the Encoder folder
    if [ ! -d "$enc_seq_path" ]; then
        echo "Warning: Sequence '$seq_name' found in Decoder dir but NOT in Encoder dir. Skipping."
        continue
    fi

    echo "------------------------------------------------"
    echo "Processing sequence: $seq_name"

    # 4. FFmpeg Command
    # We use -y to overwrite, -v error to reduce noise
    # We apply 'drawtext' to each stream individually before stacking

    ffmpeg -y -hide_banner -v error \
        -framerate 24 -i "$enc_seq_path/%04d_feats_enc.png" \
        -framerate 24 -i "$enc_seq_path/%04d_pred.png" \
        -framerate 24 -i "$dec_seq_path/%04d_feats.png" \
        -framerate 24 -i "$dec_seq_path/%04d_pred.png" \
        -filter_complex \
        "[0:v]drawtext=text='DINOv2 B14 (PCA)':fontcolor=white:fontsize=24:x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5[v0]; \
         [1:v]drawtext=text='DINOv2 B14 (VOS)':fontcolor=white:fontsize=24:x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5[v1]; \
         [2:v]drawtext=text='LILA (PCA)':fontcolor=white:fontsize=24:x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5[v2]; \
         [3:v]drawtext=text='LILA (VOS)':fontcolor=white:fontsize=24:x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5[v3]; \
         [v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \
        -map "[v]" \
        -c:v libx264 -pix_fmt yuv420p \
        "$OUTPUT_ROOT/$seq_name.mp4"

    if [ $? -eq 0 ]; then
        echo "Success: Created $OUTPUT_ROOT/$seq_name.mp4"
    else
        echo "Error: Failed to process $seq_name"
    fi
done
