########################
# Utils Functions
########################

import multiprocessing
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env early so query_server can see them
import subprocess
import re
import random
import tempfile
from pathlib import Path
import re
import math
import os
import json
from tqdm import tqdm

# API clients
from together import Together
from openai import OpenAI
# from google import genai

# import google.generativeai as genai
# from google.generativeai.types import (
#     HarmCategory,
#     HarmBlockThreshold,
#     SafetySetting
# )
import anthropic
import requests

# from datasets import load_dataset
import numpy as np
from contextlib import contextmanager
from collections import defaultdict
import time
import shutil
import concurrent
from functools import cache
from transformers import AutoTokenizer
import hashlib

from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from scripts.llada_utils import generate
from scripts.trado_utils import block_diffusion_generate as trado_generate
from scripts.sdar_utils import block_diffusion_generate

# Define API key access
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
SGLANG_KEY = os.environ.get("SGLANG_API_KEY")  # for Local Deployment


########################################################
# Inference Helpers
########################################################

@cache
def load_deepseek_tokenizer():
    # TODO: Should we update this for new deepseek? Same tokenizer?
    # return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Instruct-0724")
    return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2", trust_remote_code=True)

# Buffer because deepseek totally blocks us if we send stuff that's too long :(
TOO_LONG_FOR_DEEPSEEK = 115_000


def is_safe_to_send_to_deepseek(prompt):
    tokenizer = load_deepseek_tokenizer()
    # print(f"Prompt: {len(prompt)}")
    # print(f"Prompt length: {len(tokenizer(prompt, verbose=False)['input_ids'])}")
    
    if type(prompt) == str:
        return (
            len(tokenizer(prompt, verbose=False)["input_ids"]) < TOO_LONG_FOR_DEEPSEEK
        )
    else:
        return len(tokenizer.apply_chat_template(prompt)) < TOO_LONG_FOR_DEEPSEEK

def set_gpu_arch(arch_list: list[str]):
    """
    Set env variable for torch cuda arch list to build kernels for specified architectures
    """
    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(f"Invalid architecture: {arch}. Must be one of {valid_archs}")
    
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

def query_server(
    prompt: str | list[dict],  # string if normal prompt, list of dicts if chat prompt,
    system_prompt: str = "You are a helpful assistant",  # only used for chat prompts
    temperature: float = 0.0,
    top_p: float = 1.0, # nucleus sampling
    top_k: int = 50, 
    max_tokens: int = 128,  # max output tokens to generate
    num_completions: int = 1,
    server_port: int = 30000,  # only for local server hosted on SGLang
    server_address: str = "localhost",
    server_type: str = "sglang",
    model_name: str = "default",  # specify model type

    # for reasoning models
    is_reasoning_model: bool = False, # indiactor of using reasoning models
    budget_tokens: int = 0, # for claude thinking
    reasoning_effort: str = None, # only for o1 and o3 / more reasoning models in the future
):
    """
    Query various sort of LLM inference API providers
    Supports:
    - OpenAI
    - Deepseek
    - Together
    - Sambanova
    - Anthropic
    - Gemini / Google AI Studio
    - Fireworks (OpenAI compatbility)
    - SGLang (Local Server)
    """
    # Select model and client based on arguments
    match server_type:
        case "sglang":
            url = f"http://{server_address}:{server_port}"
            client = OpenAI(
                api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0
            )
            model = "default"
        case "deepseek":
            client = OpenAI(
                api_key=DEEPSEEK_KEY,
                base_url="https://api.deepseek.com",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name
            assert model in ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"], "Only support deepseek-chat or deepseek-coder for now"
            if not is_safe_to_send_to_deepseek(prompt):
                raise RuntimeError("Prompt is too long for DeepSeek")
        case _:
            raise NotImplementedError

    if server_type == "deepseek":
        
        if model in ["deepseek-chat", "deepseek-coder"]:
            # regular deepseek model 
            response = client.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )

        else: # deepseek reasoner
            assert is_reasoning_model, "Only support deepseek-reasoner for now"
            assert model == "deepseek-reasoner", "Only support deepseek-reasoner for now"
            response = client.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                n=num_completions,
                max_tokens=max_tokens,
                # do not use temperature or top_p
            )
        outputs = [choice.message.content for choice in response.choices]

    # for all other kinds of servers, use standard API
    else:
        if type(prompt) == str:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.text for choice in response.choices]
        else:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.message.content for choice in response.choices]

    # output processing
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


# a list of presets for API server configs
SERVER_PRESETS = {
    "deepseek": {
        "temperature": 1.6, 
        "model_name": "deepseek",
        "max_tokens": 4096
    },
    "sglang": {  # this is for running locally, mostly for Llama
        "temperature": 0.8, # human eval pass@N temperature
        "server_port": 10210,
        "server_address": "matx2.stanford.edu",
        "max_tokens": 8192,
    },
}


def create_inference_server_from_presets(server_type: str = None, 
                                         greedy_sample: bool = False,   
                                         verbose: bool = False,
                                         time_generation: bool = False,
                                         **kwargs,
                                         ) -> callable:
    """
    Return a callable function that queries LLM with given settings
    """
    def _query_llm(prompt: str | list[dict]):
        server_args = SERVER_PRESETS[server_type].copy()

        if kwargs:
            server_args.update(kwargs)
        if greedy_sample:
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        if verbose:
            print(f"Querying server {server_type} with args: {server_args}")
        
        if time_generation:
            start_time = time.time()
            response = query_server(
                prompt, server_type=server_type, **server_args
            )
            end_time = time.time()
            print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
            return response
        else:
            return query_server(
                prompt, server_type=server_type, **server_args
            )
    
    return _query_llm


def create_local_inference(
        model_path: str,
        temperature: float = 0.,
        steps: int=1024,
        gen_length: int=1024,
        block_length: int=32,
        top_p: float = 1.0,
        top_k: int = 50,
        max_tokens: int = 4096,
        verbose: bool = False,
        device: str = "cuda",

):
    if "llada" in model_path.lower() or "dream" in model_path.lower() or "diffucoder" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

    elif "sdar" in model_path.lower() or "dice" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # if "epoch" and "optimized" not in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="float16",
            device_map="auto"
        )
    elif "wedlm" in model_path.lower():

        path_to_add = "/baihaolei/Projects/wedlm"

        if path_to_add not in sys.path:
            sys.path.insert(0, path_to_add)

        from wedlm import LLM, SamplingParams

        model = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        if "seed" in model_path.lower() or "stable" in model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            ).eval()

    def _query_local_model(prompt: str | list[dict]):
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if "llada" in model_path.lower():
            formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # import pdb
            # pdb.set_trace()
            # print(formatted_prompt)
            input_ids = tokenizer(formatted_prompt)['input_ids']
            input_ids = torch.tensor(input_ids).to(model.device).unsqueeze(0)
            start_time = time.time()
            with torch.no_grad():
                outputs = generate(
                    model,
                    input_ids,
                    steps=gen_length, gen_length=gen_length, block_length=32,
                    temperature=temperature,
                    cfg_scale=0., remasking='low_confidence'
                )

            end_time = time.time()
            print(f"[Timing] Local inference took {end_time - start_time:.2f} seconds")

            response = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        elif "dream" in model_path.lower():
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)

            start_time = time.time()
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    output_history=True,
                    return_dict_in_generate=True,
                    steps=gen_length,
                    temperature=temperature,
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.,
                )
            end_time = time.time()
            print(f"[Timing] Local inference took {end_time - start_time:.2f} seconds")
            generations = [
                tokenizer.decode(g[len(p):].tolist())
                for p, g in zip(input_ids, output.sequences)
            ]
            response = generations[0].split(tokenizer.eos_token)[0]
        elif "diffucoder" in model_path.lower():
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)

            start_time = time.time()
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    output_history=True,
                    return_dict_in_generate=True,
                    steps=gen_length,
                    temperature=temperature,
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.,
                )
            end_time = time.time()
            print(f"[Timing] Local inference took {end_time - start_time:.2f} seconds")
            generations = [
                tokenizer.decode(g[len(p):].tolist())
                for p, g in zip(input_ids, output.sequences)
            ]
            response = generations[0].split(tokenizer.eos_token)[0]

        elif "trado" in model_path.lower():
            print("trado inference")
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            tokens = tokenizer.batch_encode_plus([text], return_tensors='pt', padding=True, truncation=True)
            tokens = {k: v.to(model.device) for k, v in tokens.items()}

            input_length = tokens['input_ids'].shape[1]

            start_time = time.time()
            with torch.no_grad():
                if "thinking" in model_path.lower():
                    print("thinking model")
                    outputs = trado_generate(
                        model,
                        prompt=tokens,
                        mask_id=151669,
                        gen_length=gen_length,
                        block_length=4, denoising_steps=4,
                        temperature=1.0, top_k=0, top_p=1.0,
                        remasking_strategy="low_confidence_dynamic",
                        confidence_threshold=0.9
                    )
                else:
                    outputs = trado_generate(
                        model,
                        prompt=tokens,
                        mask_id=151669,
                        gen_length=gen_length,
                        block_length=4, denoising_steps=4,
                        temperature=1.0, top_k=1, top_p=1.0, # top_k=0, remasking_strategy="low_confidence_dynamic"
                        remasking_strategy="low_confidence_static",
                        confidence_threshold=0.9
                    )
            end_time = time.time()
            print(f"[Timing] Local inference took {end_time - start_time:.2f} seconds")
            generated = outputs[0][input_length:]
            output_text = tokenizer.decode(generated, skip_special_tokens=True)
            response = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '')
            print(response)
        elif "sdar" in model_path.lower() or "dice" in model_path.lower():
            print("sdar inference")
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            tokenize_kwargs = dict(
                return_tensors='pt',
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=4096
            )
            tokens = tokenizer.batch_encode_plus([text], **tokenize_kwargs)
            tokens = {k: v.to(model.device) for k, v in tokens.items()}

            input_length = tokens['input_ids'].shape[1]

            start_time = time.time()
            with torch.no_grad():
                outputs = block_diffusion_generate(
                    model,
                    prompt=tokens,
                    mask_id=tokenizer(tokenizer.mask_token)['input_ids'][0],
                    gen_length=gen_length,
                    block_length=4,
                    denoising_steps=4,
                    temperature=1.0,
                    top_k=1,
                    top_p=1.0,
                    remasking_strategy="low_confidence_static"
                )
            end_time = time.time()
            print(f"[Timing] Local inference took {end_time - start_time:.2f} seconds")
            generated = outputs[0][input_length:]
            output_text = tokenizer.decode(generated, skip_special_tokens=True)
            response = output_text.replace('<|MASK|>', '')
        elif "stable" in model_path.lower():
            print(f"seed generation length: {gen_length}")
            print("load in this way (stable-diffucoder)")
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt)['input_ids']
            input_ids = torch.tensor(input_ids).to(model.device).unsqueeze(0)
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(input_ids, steps=gen_length, gen_length=gen_length, block_length=4,
                                         temperature=0., remasking='low_confidence', tokenizer=tokenizer, shift=False,
                                         threshold=None, eos_id=tokenizer.eos_token_id)
            end_time = time.time()
            print(f"[Timing] Local inference took {end_time - start_time:.2f} seconds")
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        elif "wedlm" in model_path.lower():
            print("load in this way (wedlm)")
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            start_time = time.time()
            outputs = model.generate([prompt], SamplingParams(temperature=0.2, max_tokens=gen_length))
            end_time = time.time()
            print(f"[Timing] Local inference took {end_time - start_time:.2f} seconds")
            response = outputs[0]["text"]
            print(response)
        else:
            if "seed" in model_path.lower():
                print(f"seed generation length: {gen_length}")
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(model.device)
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(input_ids, max_new_tokens=gen_length, pad_token_id=tokenizer.eos_token_id)
                end_time = time.time()
                response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            elif "qwen" in model_path.lower(): # qwen3
                formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
                start_time = time.time()
                if "qwen3" in model_path.lower():
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=gen_length,
                        )
                    end_time = time.time()
                    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
                    # parsing thinking content
                    try:
                        # rindex finding 151668 (</think>)
                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0

                    response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    print(response)
                else:  # qwen2.5-coder
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=gen_length
                        )
                    end_time = time.time()
                    outputs = [
                        output_ids[len(input_ids):] for input_ids, output_ids in
                        zip(inputs.input_ids, outputs)
                    ]

                    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            elif "cudallm" in model_path.lower():
                formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
                start_time = time.time()
                print(f"cudallm generation length: {gen_length}")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=gen_length,
                    )
                end_time = time.time()
                output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
                # parsing thinking content
                try:
                    # rindex finding 151668 (</think>)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                print(response)
            elif "deepseek-coder" in model_path.lower(): # deepseek-coder
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(model.device)
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=gen_length,
                        do_sample=False,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id
                    )
                end_time = time.time()
                response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            elif "phi-4" in model_path.lower(): # microsoft/Phi-4-mini-reasoning
                print("load phi for inference!")
                print(gen_length)
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **input_ids,
                        max_new_tokens=gen_length,
                        temperature=0.8,
                        top_p=0.95,
                        do_sample=True,
                    )
                end_time = time.time()
                outputs = tokenizer.batch_decode(outputs[:, input_ids["input_ids"].shape[-1]:])
                response=outputs[0]
                print(response)
            elif "llama" in model_path.lower():
                print("load llama for inference!")
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(**input_ids, max_new_tokens=4096)
                end_time = time.time()
                response=tokenizer.decode(outputs[0][input_ids["input_ids"].shape[-1]:])
            else:  # gemma
                print("load gemma for inference!")
                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(model.device)
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(**input_ids, max_new_tokens=4096)
                end_time = time.time()
                response = tokenizer.decode(outputs[0])
            print(f"[Timing] Local inference took {end_time - start_time:.2f} seconds")

        return response

    return _query_local_model


"""
Model output processing
#  TODO: add unit tests
"""


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def print_messages(messages):
    for message in messages:
        print(message["role"])
        print(message["content"])
        print("-" * 50)
        print("\n\n")


def extract_python_code(text):
    """
    Extract python code from model output
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches) if matches else ""


def remove_code_block_header(code, code_language_type):
    """Assume input is code but just with like python, cpp, etc. at the top"""
    if code.startswith(code_language_type):
        code = code[len(code_language_type) :].strip()
    return code


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None


def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code
    
    return None

def extract_code_blocks(text, code_language_types: list[str]) -> str:
    '''
    Extract all code blocks from text, combine them to return as a single string
    '''
    pattern = r'```.*?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    # Combine all code blocks and remove language type headers
    combined_code = []
    for match in matches:
        code = match.strip()
        # Remove any language type headers
        for lang_type in code_language_types:
            if code.startswith(lang_type):
                code = code[len(lang_type):].strip()
        combined_code.append(code)
    
    return " \n ".join(combined_code) if combined_code else ""

################################################################################
# Scale up experiments in parallel
################################################################################

def maybe_multithread(func, instances, num_workers, time_interval=0.0, *shared_args, **shared_kwargs):
    """
    Multithreaded execution of func, with optional time interval between queries
    Ideal for querying LLM APIs, does not provide process isolation
    """
    output_data = []
    if num_workers not in [1, None]:
        with tqdm(total=len(instances), smoothing=0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

                # Submit tasks one at a time with delay between them
                futures = []
                for instance in instances:
                    futures.append(
                        executor.submit(
                            func,
                            instance,
                            *shared_args,
                            **shared_kwargs
                        )
                    )
                    time.sleep(time_interval)  # sleep between submitting each task



                # Wait for each future to complete
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            output_data.append(result)
                    except Exception as e:
                        print("Got an error!", e)
                        continue
    else:
        for instance in tqdm(instances):
            output = func(instance, *shared_args, **shared_kwargs)
            if output is not None: output_data.append(output)

    return output_data


def maybe_multiprocess_cuda(
    func, instances, num_workers, *shared_args, **shared_kwargs
):
    """
    From monkeys, but modified to work with CUDA
    """
    output_data = []
    multiprocessing.set_start_method(
        "spawn", force=True
    )  # this is necessary for CUDA to work

    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(func, instance, *shared_args, **shared_kwargs): None
                for instance in instances
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    result = future.result()
                    if result is not None:
                        output_data.append(result)
                except Exception as e:
                    print("Got an error!", e)
                    continue
    return output_data

# src/random_inputs.py
import os, torch, itertools
from torch.distributions import Normal, Uniform, Laplace, Exponential, LogNormal

# Pick which distributions are allowed in “random” mode.
_DEFAULT_RANDOM_POOL = (
    ("normal",      lambda shape: Normal(0, 1).sample(shape)),
    ("uniform",     lambda shape: Uniform(-1, 1).sample(shape)),
    ("laplace",     lambda shape: Laplace(0, 1).sample(shape)),
    ("exponential", lambda shape: Exponential(1).sample(shape)),   # strictly >0
    ("lognormal",   lambda shape: LogNormal(0, 1).sample(shape)),  # strictly >0
)


def sample(shape, mode="random"):
    """
    shape : torch.Size or tuple
    mode  : "random"  – draw from a rotating pool of distributions
            "target"  – return a tensor from a randomly chosen edge-case pattern
            <dist>    – force a single distribution name, e.g. "laplace"
    """
    if mode == "random":
        # Round-robin through default pool
        idx = int(torch.empty((), dtype=torch.int64).random_()) % len(_DEFAULT_RANDOM_POOL)
        _, fn = _DEFAULT_RANDOM_POOL[idx]
        return fn(shape)

    # Explicit distribution name
    pool = dict(_DEFAULT_RANDOM_POOL)
    if mode not in pool:
        raise ValueError(f"Unknown distribution {mode}")
    return pool[mode](shape)


# ------------------------------------------------------------------
# Public helper: rand_mix / rand_mix_like
# ------------------------------------------------------------------

def rand_mix(*size, dist: str = "random", device=None, dtype=None, requires_grad: bool = False):
    """Return a tensor drawn from a chosen distribution (or randomly chosen).

    Parameters
    ----------
    *size : int or tuple
        Dimensions of the output tensor (same semantics as ``torch.randn``).
    dist : str, optional
        • "random"   – randomly cycle through the default pool defined above.
        • "target"   – pick from the specialised _TARGETED_CASES pool.
        • any key in the default pool ("normal", "uniform", "laplace", ...).
    device, dtype, requires_grad : any
        Forwarded to ``Tensor.to`` / ``Tensor.requires_grad_`` for convenience.
    """
    # normalise *size → shape tuple
    shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, torch.Size)) else size

    t = sample(shape, mode=dist)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    if requires_grad:
        t.requires_grad_(True)
    return t

def rand_mix_like(tensor: torch.Tensor, dist: str = "random", **kwargs):
    """rand_mix variant that infers shape from *tensor*."""
    return rand_mix(*tensor.shape, dist=dist, **kwargs)

# Register convenience aliases under torch namespace (does not shadow existing fns)
setattr(torch, "rand_mix", rand_mix)
setattr(torch, "rand_mix_like", rand_mix_like)