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



DS=$1
MODEL=$2
RUN_NAME=$3

shift 3

# Assign all remaining arguments ($@) into a single string variable named ARGS.
OVERRIDES="$@"

####
ARGS="--config-name=train_${DS}"
RUN_NAME="${DS}_${MODEL}_${RUN_NAME}"

echo "---------------------------------------------------------"

RUN_ID=$RUN_NAME #$(python version.py $ARGS runtime.version=$RUN_NAME 2> /dev/null)
PY_STATUS=$?

# Check if version.py executed successfully.
if [ "$PY_STATUS" -ne 0 ]; then
  echo "Error: Failed to execute version.py (exit code $PY_STATUS)."
  exit 1
fi

echo "LAUNCHING *** $RUN_ID ***"

shift 1  # Remove model and run_id from the argument list.
#AUX_ARGS=("$@") # capture all remaining arguments in an array. crucial for handling spaces correctly.

# Create a directory for the run.
RUN_DIR="${TB_DIR}_v2/$RUN_ID"
mkdir -p "$RUN_DIR" || { echo "Failed to create run directory: $RUN_DIR"; exit 1; }
echo "RUNTIME DIR: $RUN_DIR"

# Get the current Git commit ID.
COMMIT_ID=$(git rev-parse HEAD)

# Save the Git commit ID to a file.
echo "Commit ID: $COMMIT_ID" > "$RUN_DIR/commit_id.txt"

# Save the Git diff to a file.
git diff > "$RUN_DIR/diff.patch" || { echo "Failed to save git diff"; exit 1; }


# Check if the script exists.  Important for preventing silent failures.
CMD="python train_lila.py $ARGS runtime.name=$RUN_ID model.encoder=${MODEL} $OVERRIDES"
echo $CMD
echo $CMD > "$RUN_DIR/cmd.bash" || { echo "Failed to save command"; exit 1; }

#Run the script, redirect output.
$CMD
#> "$RUN_DIR/train.log"
#echo "Started process $!"
#echo "---------------------------------------------------------"

#sleep 5
#tail -f "$RUN_DIR/train.log"

exit 0
