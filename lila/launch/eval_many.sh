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



# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================

# Global Version Tag (Applies to all runs in this batch)
VERSION="v22"

# Default Mode: "enc" (Encoder Only), "full" (With Decoder/LILA), or "both"
DEFAULT_MODE="both"

# Default Tasks to run if not specified in the JOB string
DEFAULT_TASKS=("vosknn" "seg") 

# --- THE JOB LIST ---
# Format: "CONFIG_NAME [OPTIONAL_OVERRIDE_TASKS]"
# If you don't list tasks, it uses DEFAULT_TASKS.
# <snapshot> <task>:<mode>
JOBS=(
    #"ytvos_vits14_ast70_68_v03_batch48_noWD"
    #"ytvos_vits14_ast70_68_v03_batch48_lr3e-4 vosknn:enc"
    #"ytvos_vitb14_ast70_68_v01_dim192_pamrS005_scaleFeat025"
    #"ytvos_vitl14_ast70_68_v01_dim256_pamrS005_scaleFeat025"
    #"kts_vitl14_h100x8_arx09_07_vitl_bs32"
    #"kts_vitg14_h100x8_arx10_08_vitg_bs32_actual"
    
    # ablations
    #"ytvos_vits14_ast71_79_ablation_noEdge"
    #"ytvos_vits14_ast71_79_ablation_noPAMR"
 
    # newer models (no artefacts)   
    #"ytvos_dinov2_vits14_ast75_74_wFullConf_v4_veryLongB48"
    #"ytvos_dinov2_vitb14_ast75_74_wFullConf_v4_veryLong"
    #"ytvos_dinov2_vitl14_ast75_74_wFullConf_v4"
)

# ==============================================================================
# 2. CONSTANTS & SPECS
# ==============================================================================

declare -A MODEL_SPECS
MODEL_SPECS["vits"]="384 128"
MODEL_SPECS["vitb"]="768 192"
MODEL_SPECS["vitl"]="1024 256"
MODEL_SPECS["vitg"]="1536 384"

# Common Configs
VOS_CFG="eval.resize_input_size=476"
VOSKNN_CFG=""
NORML_CFG="eval.num_epochs=30 eval.lr=0.0003 eval.batch_size=16 eval.enc_layer_idx=-2"
SEG_CFG="eval.num_epochs=30 eval.batch_size=64 eval.lr=0.0003"
OPENSEG_CFG="eval.num_epochs=5 eval.batch_size=32 eval.lr=0.0003 eval.weight_decay=0.01"
    
# ==============================================================================
# 3. TASK REGISTRY (Define your settings here)
# ==============================================================================

# Returns: SUFFIX | ARGS | ENC_ALPHA | DEC_ALPHA
get_task_settings() {
    local task=$1
    local mode=$2 # 'enc' or 'full'

    case "$task" in
        "vos")
            if [ "$mode" == "enc" ]; then
                echo "encOnly probe.alpha_enc=1.0 probe.alpha_dec=0.0 $VOS_CFG"
            else
                echo "wLILA probe.alpha_enc=1.0 probe.alpha_dec=1.0 $VOS_CFG"
            fi
            ;;
        "vosknn")
            # VOS KNN usually implies encoder features, but if you have a 'full' mode:
            if [ "$mode" == "enc" ]; then
                 echo "encOnly probe.alpha_enc=1.0 probe.alpha_dec=0.0 $VOSKNN_CFG"
            else
                 echo "wLILA probe.alpha_enc=0.0 probe.alpha_dec=1.0 $VOSKNN_CFG"
            fi
            ;;
        "norml")
            if [ "$mode" == "enc" ]; then
                echo "encOnly $NORML_CFG probe.alpha_enc=1.0 probe.alpha_dec=0.0"
            else
                echo "wLILA $NORML_CFG probe.alpha_enc=1.0 probe.alpha_dec=1.0"
            fi
            ;;
        "seg")
            if [ "$mode" == "enc" ]; then
                echo "encOnly $SEG_CFG probe.alpha_enc=1.0 probe.alpha_dec=0.0"
            else
                echo "wLILA $SEG_CFG probe.alpha_enc=1.0 probe.alpha_dec=1.0"
            fi
            ;;
        "openseg")
            if [ "$mode" == "enc" ]; then
                echo "encOnly $OPENSEG_CFG probe.alpha_enc=1.0 probe.alpha_dec=0.0"
            else
                echo "wLILA $OPENSEG_CFG probe.alpha_enc=1.0 probe.alpha_dec=1.0"
            fi
            ;;
    esac
}

# ==============================================================================
# 4. PARSING LOGIC
# ==============================================================================

RUNS=()
add_run() { RUNS+=("$1"); }

parse_model_name() {
    echo "$1" | awk -F'_' '{
        if ($2 ~ /^vit/) { m="dinov2"; a=$2; } else { m=$2; a=$3; }
        print m, a, substr(a, 1, 4);
    }'
}

# Allow CLI override for MODE
MODE_OVERRIDE=""
while getopts "m:" opt; do
  case $opt in
    m) MODE_OVERRIDE="$OPTARG" ;;
  esac
done

FINAL_MODE="${MODE_OVERRIDE:-$DEFAULT_MODE}"

# ==============================================================================
# 5. MAIN EXECUTION LOOP
# ==============================================================================

echo "--- Setup ---"
echo "Version Tag: $VERSION"
echo "Global Mode: $FINAL_MODE (used as fallback)"
echo "-------------"

for JOB in "${JOBS[@]}"; do
    # 1. Split Job String into Config and the Task Definition String
    read -r CONFIG_NAME TASK_DEF_STR <<< "$JOB"

    # 2. Model Parsing (Extract Arch/Enc for arguments)
    read -r MODEL_TYPE ARCH ENC <<< $(parse_model_name "$CONFIG_NAME")
    MODEL_NAME="${MODEL_TYPE}_${ARCH}"
    
    specs="${MODEL_SPECS[$ENC]}"
    if [ -z "$specs" ]; then echo "Skipping unknown spec: $ENC"; continue; fi
    read -r ENC_DIM DEC_DIM <<< "$specs"
    BASE_ARGS="model.encoder=${MODEL_NAME} probe.enc_fdim=${ENC_DIM} probe.dec_fdim=${DEC_DIM}"

    # 3. Normalize Task List
    # We want to convert "all:enc" or empty string into a standard array of "task:mode" items
    
    ITEMS_TO_PROCESS=()

    if [[ -z "$TASK_DEF_STR" ]]; then
        # Case A: Empty -> Use Defaults
        for t in "${DEFAULT_TASKS[@]}"; do ITEMS_TO_PROCESS+=("$t:$FINAL_MODE"); done
    
    elif [[ "$TASK_DEF_STR" == "all"* ]]; then
        # Case B: "all" or "all:mode"
        # Extract mode if present (e.g. "all:full" -> mode="full")
        if [[ "$TASK_DEF_STR" == *":"* ]]; then
             batch_mode=${TASK_DEF_STR##*:}
        else
             batch_mode=$FINAL_MODE
        fi
        
        # Expand "all" to every known task with that mode
        ALL_KNOWN_TASKS=("vos" "vosknn" "norml" "seg" "openseg")
        for t in "${ALL_KNOWN_TASKS[@]}"; do ITEMS_TO_PROCESS+=("$t:$batch_mode"); done

    else
        # Case C: Comma separated list "vos:enc,seg:full"
        IFS=',' read -r -a RAW_ITEMS <<< "$TASK_DEF_STR"
        for item in "${RAW_ITEMS[@]}"; do
            if [[ "$item" == *":"* ]]; then
                # Already has mode
                ITEMS_TO_PROCESS+=("$item")
            else
                # Append default global mode
                ITEMS_TO_PROCESS+=("$item:$FINAL_MODE")
            fi
        done
    fi

    # 4. Execute the Process List
    for item in "${ITEMS_TO_PROCESS[@]}"; do
        
        # Split "task:mode"
        task=${item%%:*}
        mode=${item##*:}

        # Determine variants to run based on the specific mode
        VARIANTS=()
        if [[ "$mode" == "enc" || "$mode" == "both" ]]; then VARIANTS+=("enc"); fi
        if [[ "$mode" == "full" || "$mode" == "both" ]]; then VARIANTS+=("full"); fi

        for variant in "${VARIANTS[@]}"; do
            # Retrieve settings
            SETTINGS=$(get_task_settings "$task" "$variant")
            
            if [ -z "$SETTINGS" ]; then 
                # echo "   [!] Task $task does not support variant $variant"
                continue
            fi 

            SUFFIX=$(echo "$SETTINGS" | awk '{print $1}')
            ARGS=$(echo "$SETTINGS" | cut -d' ' -f2-)

            # Construct Run ID
            RUN_ID="${VERSION}_${task}_${SUFFIX}"
            
            CMD="${task} ${CONFIG_NAME} ${RUN_ID} ${ARGS} ${BASE_ARGS}"
            add_run "$CMD"
        done
    done
done


# --- Script Logic ---
# Path to the script you want to run.
WORKER_SCRIPT="./launch/eval.sh"

echo "🚀 Starting sequential bash script runs..."

# Loop through the keys (the run names) of the array.
for opts in "${RUNS[@]}"; do

  echo "--------------------------------------------------"
  echo "▶️  Launching run: '${opts}'"

  # Call the worker script.
  # We pass the required arguments first, quoted to handle spaces.
  # Then we pass ${optional_args} *without quotes*.
  # This allows the shell to split the string into separate arguments.
  "${WORKER_SCRIPT}" ${opts}

done

echo "✅ All runs completed."
