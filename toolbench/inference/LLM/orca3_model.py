#!/usr/bin/env python
# coding=utf-8
import time
from termcolor import colored
from typing import Optional, List
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import torch
import transformers
import datasets
from tqdm import tqdm
import torch
import os 


import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import time
import torch
import os

from torch.utils.data import Dataset
from transformers import Trainer, HfArgumentParser, TrainingArguments
from transformers import StoppingCriteriaList, StoppingCriteria


from toolbench.utils import process_system_message
from toolbench.model.model_adapter import get_conversation_template
from toolbench.inference.utils import SimpleChatIO, generate_stream, react_parser

class EOSStoppingCriteria(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if stop == last_token:
                return True
        return False

def to_prompt(messages):
    text = ''
    for message in messages:
        text += f"<|im_start|>{message['role']}\n{message['content']}\n<|im_end|>\n"
    text += f"<|im_start|>assistant\n"
    return text

class Orca3:
    def __init__(
            self, 
            model_name_or_path: str, 
            template:str="orca3-tool-llama-single-round", 
            device: str="cuda", 
            cpu_offloading: bool=False, 
            max_sequence_length: int=8192
        ) -> None:
        super().__init__()

        self.model_name = model_name_or_path
        self.template = template
        self.max_sequence_length = max_sequence_length

        """
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, model_max_length=self.max_sequence_length)
        #change due to transformers version upgrade
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=self.max_sequence_length)        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True
        )
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.use_gpu = (True if device == "cuda" else False)
        if (device == "cuda" and not cpu_offloading) or device == "mps":
            self.model.to(device)
        self.chatio = SimpleChatIO()
        """

        self.local_rank = 0
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                        "../../orca3/Orca3_v2_epoch_3/",
                        torch_dtype=torch.bfloat16)
        self.model = self.model.to(self.local_rank)
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "../../orca3/Orca3_v2_epoch_3/",
            model_max_length=max_sequence_length,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        stop_word_ids = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]
        self.stopping_criteria = StoppingCriteriaList([EOSStoppingCriteria(stops=stop_word_ids)])

    def prediction(self, prompt: str) -> str:

        text = prompt 

        input_ids = self.tokenizer(self.tokenizer.bos_token + text, add_special_tokens=False, padding=False,
                               return_tensors='pt').input_ids
        input_ids = input_ids.to(self.local_rank)

        gen_args = {"num_return_sequences":1}

        gen_params = {
            "model": "",
            "prompt": prompt,
            "temperature": 0.5,
            "max_new_tokens": 512,
            "stop": "</s>",
            "stop": "<|im_end|>",
            "stop_token_ids": None,
            "echo": False
        }

        output_ids = self.model.generate(input_ids, max_new_tokens = gen_params["max_new_tokens"], do_sample=False, use_cache=True,
                                        stopping_criteria=self.stopping_criteria,  **gen_args)
        output = [self.tokenizer.decode(oid[len(input_ids[0]):], skip_special_tokens=True).strip() for oid in output_ids]    

        prediction = output[0].strip()        
        return prediction
            
    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self, functions, process_id, **args):
        if (self.template == 'orca3-tool-llama-single-round'):
            roles = {"system": "system", "user": "user", "function": "user", "assistant": "assistant"}
        else:
            conv = get_conversation_template(self.template)
            if self.template == "tool-llama":
                roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            elif self.template == "tool-llama-single-round" or self.template == "tool-llama-multi-rounds":
                roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

        self.time = time.time()
        conversation_history = self.conversation_history
        #prompt = ''
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "system" and functions != []:
                message['content'] = process_system_message(content, functions)
            message['role'] = role
        #prompt += "Assistant:\n"
        prompt = to_prompt(conversation_history)
        
        if functions != []:
            predictions = self.prediction(prompt)
        else:
            predictions = self.prediction(prompt)

        decoded_token_len = len(self.tokenizer(predictions))
        if process_id == 0:
            print(f"[process({process_id})]total tokens: {decoded_token_len}")

        # react format prediction
        thought, action, action_input = react_parser(predictions)
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": action_input
            }
        }
        return message, 0, decoded_token_len


if __name__ == "__main__":
    # can accept all huggingface LlamaModel family
    llm = ToolLLaMA("decapoda-research/llama-7b-hf")
    messages = [
        {'role': 'system', 'content': '''You are AutoGPT, you can use many tools(functions) to do
the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is , you can\'t go
back to the former state, if you want to restart the task, say "I give up and restart".\n2.All the thought is short, at most in 5 sentence.\nLet\'s Begin!\nTask description: Use numbers and basic arithmetic operations (+ - * /) to obtain exactly one number=24. Each
step, you are only allowed to choose two of the left numbers to obtain a new number. For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.\nRemember:\n1.all of the number must be used , and must be used ONCE. So Only when left numbers is exact 24, you will win. So you don\'t succeed when left number = [24, 5]. You succeed when left number = [24]. \n2.all the try takes exactly 3 steps, look
at the input format'''}, 
{'role': 'user', 'content': '\nThe real task input is: [1, 2, 4, 7]\nBegin!\n'}
]
    functions = [{'name': 'play_24', 'description': '''make your current conbine with the format "x operation y = z (left: aaa) " like "1+2=3, (left: 3 5 7)", then I will tell you whether you win. This is the ONLY way
to interact with the game, and the total process of a input use 3 steps of call, each step you can only combine 2 of the left numbers, so the count of left numbers decrease from 4 to 1''','parameters':{'type': 'object', 'properties':{}}}]#, 'parameters': {'type': 'object', 'properties': {'input': {'type': 'string', 'description': 'describe what number you want to conbine, and how to conbine.'}}, 'required': ['input']}}]

    llm.change_messages(messages)
    output = llm.parse(functions=functions)
    print(output)