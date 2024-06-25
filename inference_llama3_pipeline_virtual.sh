export TOOLBENCH_KEY=MhuKsGSC8inUrAjJ2ae0uymRJTnx0rZ8wtIb5tXqgpmmpDrjou

#export OPENAI_KEY=""
#export OPENAI_API_BASE="" 
export PYTHONPATH=./
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

export GPT_MODEL="gpt-3.5-turbo-16k"
export SERVICE_URL="http://localhost:8080/virtual"


#    --method CoT@1 \
#llama3: needs to start fastchat server first
export INFERENCE_API_BASE_URL='http://localhost:9000/v1'
export OPENAI_KEY="EMPTY"
export CHAT_MODEL='llama3'

# export OUTPUT_DIR="data/answer/llama3_test_06_19"
# python toolbench/inference/qa_pipeline.py \
#     --tool_root_dir data/toolenv/tools/ \
#     --backbone_model llama3 \
#     --openai_key $OPENAI_KEY \
#     --max_observation_length 1024 \
#     --observ_compress_method truncate \
#     --method DFS_woFilter_w2 \
#     --input_query_file data/instruction/inference_query_demo.json \
#     --output_answer_file $OUTPUT_DIR \
#     --toolbench_key $TOOLBENCH_KEY

export OUTPUT_DIR="data/answer/llama3_eval_$(date +%Y_%m_%d_%H_%M_%S)"
#rerun
export OUTPUT_DIR="/home/wchen/repos/StableToolBench/data/answer/llama3_eval_2024_06_19_23_59_17"
mkdir -p $OUTPUT_DIR

groups=("G1_instruction" "G2_instruction" "G3_instruction" "G1_category" "G1_tool" "G2_category")
for group in "${groups[@]}"
do
    mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
    for method in "CoT@1" "DFS_woFilter_w2"
    do
        python toolbench/inference/qa_pipeline.py \
            --tool_root_dir data/toolenv/tools/ \
            --backbone_model llama3 \
            --openai_key $OPENAI_KEY \
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