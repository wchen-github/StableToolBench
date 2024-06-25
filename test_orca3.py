
# azureml-core of version 1.0.72 or higher is required# %%
from azureml.core import Workspace, Dataset
from azureml.core import ScriptRunConfig, Environment, Experiment, Workspace, Dataset, Model
from azureml.core.runconfig import PyTorchConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data import OutputFileDatasetConfig
from azureml.core.container_registry import ContainerRegistry
"""
try:
    from eval_util import *
except:
    pass
"""

import os
from azureml.core import Workspace, Dataset

orca_model = 'Orca3_v2_epoch_3'
orca_model_path = os.path.join('/home/wchen/orca3/', orca_model)
if not os.path.exists(orca_model_path):

    subscription_id = '3f2ab3f5-468d-4ba7-bc14-9d3a9da4bcc5'
    resource_group = 'TownsendAML4'
    workspace_name = 'townsendws4'
    
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    
    dataset = Dataset.get_by_name(workspace, name=orca_model)
    dataset.download(target_path=orca_model_path, overwrite=False)


# %%
#evaluate trained model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda"   # "cpu" 

tokenizer = AutoTokenizer.from_pretrained(orca_model_path, model_max_length=2048)
#num_new_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#num_new_tokens += tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
#print(f"num_new_tokens = {num_new_tokens}")
#model.resize_token_embeddings(len(tokenizer))
#print(tokenizer.all_special_tokens_extended)
#print(len(tokenizer))
#print(tokenizer.convert_ids_to_tokens(32000))
#print(tokenizer.convert_ids_to_tokens(32001))
#print(tokenizer.convert_ids_to_tokens(32002))
#print(tokenizer.add_tokens(["<|im_start|>", "<|im_end|>"]))

model = AutoModelForCausalLM.from_pretrained(orca_model_path, torch_dtype=torch.bfloat16)
model = model.to(device)
model.eval()


# %%
import transformers


tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    

sm = "You are AutoGPT, you can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step. Your output should follow this format:\nThought: you should always think about what to do\nAction: the action to take (name of the api)\nAction Input: the input to the action\n\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer. If you believe that you have obtained enough information (which can be judge from the history observations) that can answer the task, please call:\nAction: Finish\nAction Input: {{\"return_type\": \"give_answer\", \"final_answer\": your answer string}}.\nRemember: \n1.the state change is irreversible, you can't go back to one of the former state, if you want more information, output your request for additional information.\n2.All the thought is short, at most in 3-5 sentences.\n3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.\n4.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.\n\nLet's Begin!\nSpecifically, you have access to the following APIs: \n[{'name': 'love_quote_for_olato_quotes', 'description': 'This is the subfunction for tool \"olato_quotes\", you can use this tool.The description of this function is: \"It shows random quotes\"', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'optional': []}}, {'name': 'success_quote_for_olato_quotes', 'description': 'This is the subfunction for tool \"olato_quotes\", you can use this tool.The description of this function is: \"It shows random quotes\"', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'optional': []}}, {'name': 'motivation_quote_for_olato_quotes', 'description': 'This is the subfunction for tool \"olato_quotes\", you can use this tool.The description of this function is: \"It shows random quotes\"', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'optional': []}}, {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"'}}, 'required': ['return_type']}}]"

user_query = "I'm planning a surprise party for my best friend, and I want to include meaningful quotes in the decorations. Can you provide me with random love, success, and motivation quotes? It would be great to have quotes that can celebrate love, success, and inspire everyone at the party. Thank you so much for your help!\nBegin!"

# Format prompt
message = [
    {"role": "system", "content": "You are a helpful assistant chatbot."},
    {"role": "user", "content": user_query},
]


orca_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '\n' + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
            )

prompt = tokenizer.bos_token + tokenizer.apply_chat_template(message, chat_template=orca_template, add_generation_prompt=True, tokenize=False)

print(prompt)

# Create pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

# Generate text
sequences = pipeline(
    prompt,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=1,
    max_length=2048,
#    repetition_penalty=0.0,
    pad_token_id=tokenizer.eos_token_id
)
print(sequences[0]['generated_text'])
