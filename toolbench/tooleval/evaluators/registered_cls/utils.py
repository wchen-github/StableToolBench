import os
import json
from typing import List,Dict
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt

from openai import OpenAI
import random

__registered_evaluators__ = {}

def register_evaluator(cls):
    """
    Decorator function to register classes with the registered_evaluators list.
    """
    __registered_evaluators__[cls.__name__] = cls
    return cls

def get_evaluator_cls(clsname):
    """
    Return the evaluator class with the given name.
    """
    try:
        return __registered_evaluators__.get(clsname)
    except:
        raise ModuleNotFoundError('Cannot find evaluator class {}'.format(clsname))

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.identity import AzureCliCredential

class OpenaiPoolRequest:
    def __init__(self, pool_json_file=None):
        self.pool:List[Dict] = []
        __pool_file = pool_json_file
        if os.environ.get('API_POOL_FILE',None) is not None:
            __pool_file = os.environ.get('API_POOL_FILE')
            self.now_pos = random.randint(-1, len(self.pool))
        if os.path.exists(__pool_file):
            self.pool = json.load(open(__pool_file))
            self.now_pos = random.randint(-1, len(self.pool))
        # print(__pool_file)
        if os.environ.get('OPENAI_KEY',None) is not None:
            self.pool.append({
                'api_key':os.environ.get('OPENAI_KEY'),
                'organization':os.environ.get('OPENAI_ORG',None),
                'api_type':os.environ.get('OPENAI_TYPE',None),
                'api_version':os.environ.get('OPENAI_VER',None),
                'base_url':os.environ.get('OPENAI_BASE_URL',None),
                'model':os.environ.get('OPENAI_MODEL',None)
            })

        token_provider = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
        self.client = AzureOpenAI(
            azure_endpoint=self.pool[0]['base_url'],
            azure_ad_token_provider=token_provider,
            api_version=self.pool[0]['api_version'])
    
    #@retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(20),reraise=True)
    #there are retry at the outer layer in eval_pass_rate.py
    def request(self,messages,**kwargs):
        self.now_pos = (self.now_pos + 1) % len(self.pool)
        key_pos = self.now_pos
        item = self.pool[key_pos]

        # print(len(self.pool))
        #api_key = item['api_key']
        #api_base = item.get('api_base', None)
        #client = OpenAI(api_key=api_key,base_url=api_base)
        #response = client.chat.completions.create(messages=messages,**kwargs)
        #return response
        completion = self.client.chat.completions.create(
                        messages=messages,
                        **kwargs,)
        #json = completion.to_json()
        return completion
    
    def __call__(self,messages,**kwargs):
        return self.request(messages,**kwargs)
   