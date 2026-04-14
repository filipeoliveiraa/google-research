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



if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <video_dir> <title_text> <subtitle_text>"
    echo "Example: ./create_summary.sh ./results"
    exit 1
fi

VIDEO_DIR="$1"
SEQ_LIST="bear blackswan breakdance-flare bus camel paragliding dog drift-turn flamingo mallard-water rhino snowboard"
#SEQ_LIST="bear blackswan bike-packing bmx-bumps bmx-trees"
TITLE1="Featurising Pixels from Dynamic 3D Scenes"
TITLE2="with Linear In-Context Learners"
SUBTITLE="Qualitative Results"

OUTPUT_FILE="${VIDEO_DIR}/final.mp4"

# --- CONFIGURATION ---
# Match this to your 2x2 grid resolution!
# If input images are 640x480, grid is 1280x960.
WIDTH=1708
HEIGHT=960
DURATION=4  # Duration of intro in seconds

echo "------------------------------------------------"
echo "Phase 1: Generating Intro Video..."

# We create a temporary intro.mp4
# -f lavfi -i color: Creates a black background
# drawtext 1: The Title (Centered, slightly shifted up)
# drawtext 2: The Subtitle (Centered, slightly shifted down)
#
ffmpeg -y -hide_banner -v error \
    -f lavfi -i color=c=white:s=${WIDTH}x${HEIGHT}:d=$DURATION \
    -vf "drawtext=text='$TITLE1':fontcolor=black:fontsize=60:x=(w-text_w)/2:y=(h-text_h)/2-80, \
         drawtext=text='$TITLE2':fontcolor=black:fontsize=60:x=(w-text_w)/2:y=(h-text_h)/2-10, \
         drawtext=text='$SUBTITLE':fontcolor=black:fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2+60" \
    -c:v libx264 -pix_fmt yuv420p -r 24 \
    ${VIDEO_DIR}/intro_temp.mp4

echo "Intro created."

echo "------------------------------------------------"
echo "Phase 2: Building File List..."

# Create a text file for the concat demuxer
# 1. Add the intro
echo "file '${VIDEO_DIR}/intro_temp.mp4'" > ${VIDEO_DIR}/vidlist.txt

# 2. Add all mp4 files from the target directory (sorted alphabetically)
# Loop through the space-separated list provided in Argument 2
for seq_name in $SEQ_LIST; do

    # Construct the expected filename
    # We assume the input is just "bear", so we add ".mp4"
    full_path="$VIDEO_DIR/$seq_name.mp4"

    # Check if the file actually exists
    if [ -f "$full_path" ]; then
        # Append absolute path to the list
        # Using Realpath or PWD ensures safe file referencing
        echo "file '$(realpath "$full_path")'" >> ${VIDEO_DIR}/vidlist.txt
        echo "Added: $seq_name"
    else
        echo "WARNING: Could not find sequence '$seq_name' in $VIDEO_DIR. Skipping."
    fi
done

echo "File list built with $(wc -l < ${VIDEO_DIR}/vidlist.txt) entries."

echo "------------------------------------------------"
echo "Phase 3: Concatenating..."

# -f concat: Uses the list we just made
# -safe 0: Allows reading absolute paths if necessary
# -c copy: Copies streams without re-encoding (FAST), implies all videos match resolution/codec exactly.
# If concatenation fails, change "-c copy" to "-c:v libx264" to force re-encoding.

ffmpeg -y -hide_banner -v error \
    -f concat -safe 0 -i ${VIDEO_DIR}/vidlist.txt \
    -c copy \
    "$OUTPUT_FILE"

echo "------------------------------------------------"
# Cleanup
rm ${VIDEO_DIR}/intro_temp.mp4 ${VIDEO_DIR}/vidlist.txt

if [ $? -eq 0 ]; then
    echo "Done! Output saved to: $OUTPUT_FILE"
else
    echo "Error during concatenation."
fi
