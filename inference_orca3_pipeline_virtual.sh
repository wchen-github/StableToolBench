#!/bin/bash
export TOOLBENCH_KEY=MhuKsGSC8inUrAjJ2ae0uymRJTnx0rZ8wtIb5tXqgpmmpDrjou

#export OPENAI_KEY=""
#export OPENAI_API_BASE="" 
export PYTHONPATH=./
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

export GPT_MODEL="gpt-3.5-turbo-16k"
export SERVICE_URL="http://localhost:8080/virtual"

export OUTPUT_DIR="data/answer/orca3_eval_$(date +%Y_%m_%d_%H_%M_%S)"
mkdir -p $OUTPUT_DIR

#groups=("G1_instruction" "G2_instruction" "G3_instruction")
groups=("G1_category" "G1_tool" "G2_category")
for group in "${groups[@]}"
do
    mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
    for method in "CoT@1" "DFS_woFilter_w2"
    do
        python toolbench/inference/qa_pipeline.py \
            --tool_root_dir data/toolenv/tools/ \
            --backbone_model orca3 \
            --model_path /home/wchen/orca3/orca3_v2_epoch_3 \
            --max_observation_length 1024 \
            --observ_compress_method truncate \
            --method $method  \
            --output_answer_file $OUTPUT_DIR \
            --input_query_file solvable_queries/test_instruction/${group}.json \
            --output_answer_file $OUTPUT_DIR/$group \
            --toolbench_key $TOOLBENCH_KEY
    done
done
