: <<'END_OF_CONVERT'
cd toolbench/tooleval
#export RAW_ANSWER_PATH="/home/wchen/repos/StableToolBench/data/answer/orca3_eval_2024_06_08_00_13_09"
export RAW_ANSWER_PATH="/home/wchen/repos/StableToolBench/data/answer/orca3_eval_2024_06_11_01_25_43"
export CONVERTED_ANSWER_PATH="/home/wchen/repos/StableToolBench/data/model_predictions_converted"
export MODEL_NAME="orca3"
#groups=("G1_instruction" "G2_instruction" "G3_instruction")
groups=("G1_category" "G1_tool" "G2_category")
for test_set in "${groups[@]}"
do
    for method in "DFS_woFilter_w2"  #"CoT@1" 
    do
        answer_dir=${RAW_ANSWER_PATH}/${test_set}
        output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/$method/${test_set}.json
        mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/$method
        python convert_to_answer_format.py \
            --answer_dir ${answer_dir} \
            --method $method \
            --output ${output_file}
    done
done
END_OF_CONVERT

#save the following to endpoints.json
#[
#    {
#        "base_url" : "https://dataoai4.openai.azure.com/"
#        "api_version": "2024-02-01",
#        "model" : "dataoai4-gpt4-turbo"
#    },
#]

#only works for one method
cd  toolbench/tooleval
export API_POOL_FILE=../../endpoints.json
export CONVERTED_ANSWER_PATH=/home/wchen/repos/StableToolBench/data/model_predictions_converted
export SAVE_PATH=/home/wchen/repos/StableToolBench/data/pass_rate_results
export CANDIDATE_MODEL=orca3
export EVAL_MODEL=dataoai4-gpt4-turbo

export TEST_IDS="/home/wchen/repos/StableToolBench/solvable_queries/test_query_ids"
#groups=("G1_instruction" "G2_instruction" "G3_instruction")
groups=("G1_category" "G1_tool" "G2_category")
for test_set in "${groups[@]}"
do
    for method in "DFS_woFilter_w2"  #"CoT@1" 
    do
        mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}/${method}
        python eval_pass_rate.py \
            --converted_answer_path ${CONVERTED_ANSWER_PATH}/${CANDIDATE_MODEL}/${method} \
            --save_path ${SAVE_PATH}/${CANDIDATE_MODEL}/${method} \
            --reference_model ${CANDIDATE_MODEL} \
            --test_ids ${TEST_IDS} \
            --max_eval_threads 15 \
            --evaluate_times 3 \
            --test_set ${test_set} 
    done
done

