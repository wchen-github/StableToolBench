export TOOLBENCH_KEY=MhuKsGSC8inUrAjJ2ae0uymRJTnx0rZ8wtIb5tXqgpmmpDrjou

#export OPENAI_KEY=""
#export OPENAI_API_BASE="" 
export PYTHONPATH=./
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

export GPT_MODEL="gpt-3.5-turbo-16k"
export SERVICE_URL="http://localhost:8080/virtual"

export OUTPUT_DIR="data/answer/toolllama_test_06_04"
#group=G1_instruction
#mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
#python toolbench/inference/qa_pipeline_multithread.py \
    # --tool_root_dir toolenv/tools \
    # --backbone_model chatgpt_function \
    # --openai_key $OPENAI_KEY \
    # --max_observation_length 1024 \
    # --method CoT@1 \
    # --input_query_file solvable_queries/test_instruction/${group}.json \
    # --output_answer_file $OUTPUT_DIR/$group \
    # --toolbench_key $TOOLBENCH_KEY \
    # --num_thread 5 --overwrite 

python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path ToolBench/ToolLLaMA-7b \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file $OUTPUT_DIR \
    --toolbench_key $TOOLBENCH_KEY

#orca3 single test
export OUTPUT_DIR="data/answer/orca3_$(date +%Y_%m_%d_%H_%M_%S)"
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model orca3 \
    --model_path /home/wchen/orca3/orca3_v2_epoch_3 \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file $OUTPUT_DIR \
    --toolbench_key $TOOLBENCH_KEY

#orca3 g1
export OUTPUT_DIR="data/answer/orca3_$(date +%Y_%m_%d_%H_%M_%S)"
mkdir -p $OUTPUT_DIR
group=G1_instruction
mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model orca3 \
    --model_path /home/wchen/orca3/orca3_v2_epoch_3 \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --output_answer_file $OUTPUT_DIR \
    --input_query_file solvable_queries/test_instruction/${group}.json \
    --output_answer_file $OUTPUT_DIR/$group \
    --toolbench_key $TOOLBENCH_KEY \
    --subsample_tasks 10


#llama3: local  -- prefer to use the fastchat server instead
export OUTPUT_DIR="data/answer/llama3_test_05_29"
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file $OUTPUT_DIR \
    --toolbench_key $TOOLBENCH_KEY


#    --method CoT@1 \
#llama3: needs to start fastchat server first
export TOOLBENCH_KEY=MhuKsGSC8inUrAjJ2ae0uymRJTnx0rZ8wtIb5tXqgpmmpDrjou
export INFERENCE_API_BASE_URL='http://localhost:9000/v1'
export CHAT_MODEL='llama3'
export OUTPUT_DIR="data/answer/llama3_test_05_30"
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model chatgpt_function \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file $OUTPUT_DIR \
    --toolbench_key $TOOLBENCH_KEY
