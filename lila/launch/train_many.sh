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


# -*- coding: utf-8 -*-

# --- Configuration ---

# Set a dataset name (constant for all runs).
DATASET_NAME="ytvos"

# Define your runs as an associative array (dictionary).
# KEY: The second argument ('name') for the worker script.
# VALUE: A string of all optional arguments. Leave empty for none.
RUNS=(
    "ytvos dinov2_vitb14 baseline train_loader.batch_size=28 model.w_edge=1.0"
)

# --- Script Logic ---

# Path to the script you want to run.
WORKER_SCRIPT="./launch/train.sh"

echo "🚀 Starting sequential bash script runs..."

# Loop through the keys (the run names) of the array.
for run_opts in "${RUNS[@]}"; do
  # Get the corresponding string of optional arguments.

  echo "--------------------------------------------------"
  echo "▶️  Launching run: '${run_opts}'"

  # Call the worker script.
  # We pass the required arguments first, quoted to handle spaces.
  # Then we pass ${optional_args} *without quotes*.
  # This allows the shell to split the string into separate arguments.
  "${WORKER_SCRIPT}" ${run_opts}
  #echo "${WORKER_SCRIPT}" "${DATASET_NAME}" "${run_name}" ${optional_args}

done

echo "✅ All runs completed."
