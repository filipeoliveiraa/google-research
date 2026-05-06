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


set -euo pipefail

DS=$1
MODEL=$2
RUN_NAME=$3
shift 3

RUN_ID="${DS}_${MODEL}_${RUN_NAME}"
RUN_ROOT="${TB_DIR:-./runs}/train"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"

mkdir -p "$RUN_DIR"
echo "Launching ${RUN_ID}"
echo "Runtime dir: ${RUN_DIR}"

git rev-parse HEAD > "${RUN_DIR}/commit_id.txt"
git diff > "${RUN_DIR}/diff.patch"

CMD=(
  python
  train_lila.py
  "--config-name=train_${DS}"
  "runtime.root=${RUN_ROOT}"
  "runtime.name=${RUN_ID}"
  "model.encoder=${MODEL}"
  "$@"
)

printf '%q ' "${CMD[@]}" | tee "${RUN_DIR}/cmd.bash"
echo

"${CMD[@]}"
