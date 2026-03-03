import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets, create_local_inference
from src.prompt_constructor_triton import (
    prompt_generate_custom_triton_from_prompt_template,
)
from src.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    query_server,
    read_file,
    set_gpu_arch,
)

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"


        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)\
        self.problem_id = REQUIRED

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "local"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ampere"]

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0
        self.steps = 1024
        self.gen_length = 1024
        self.block_length = 512

        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False
        self.log_eval_result = False

        self.use_local_model = False
        self.local_model_path = None

        self.use_sft_model = False
        self.sft_model_path = None

        # 设置cuda or triton
        self.backend = "cuda"

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Keep it simple: Generate and evaluate a single sample
    """
    print(f"Starting Eval with config: {config}")

    # Configurations

    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)  # otherwise build for all architectures

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"


    # 1. Fetch Problem
    if config.dataset_src == "huggingface":

        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)
    # import pdb; pdb.set_trace()

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"

    # 加载本地模型的命令
    if config.use_local_model and config.local_model_path:
        # 使用本地模型
        inference_server = create_local_inference(
            model_path=config.local_model_path,
            sft_model_path=config.sft_model_path,
            use_sft_model=config.use_sft_model,
            temperature=config.temperature,
            steps=config.steps,
            gen_length=config.gen_length,
            block_length=config.block_length,
            max_tokens=config.max_tokens,
            verbose=config.verbose,
        )
    else:
        # 2. Generate Sample
        # Create inference function with config parameters
        # We provide some presets in utils but you can also pass in your own, see query_server for more details
        inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                            model_name=config.model_name,
                                                            temperature=config.temperature,
                                                            max_tokens=config.max_tokens,
                                                            verbose=config.verbose,
                                                            time_generation=True)

    # Use appropriate prompt constructor based on backend
    if config.backend == "cuda":
        if config.use_sft_model:
            SYSTEM_PROMPT_KERNEL = """
You are proficient in writing CUDA kernels to replace the pytorch operators in the given architecture to get speedups.
You have complete freedom to choose the set of operators you want to replace. 
You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. 
You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
Here's an example to show you the syntax of inline embedding custom CUDA operators in torch:\n
```
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
``` \n
The example new architecture with custom CUDA kernels looks like this:\n
```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = \"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
\"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# The new PyTorch model with custom CUDA kernels
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
``` \n
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
You are given the following architecture: \n
"""
            custom_prompt = SYSTEM_PROMPT_KERNEL + "```\n" + ref_arch_src + "\n```"
        else:
            custom_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    elif config.backend == "triton":
        custom_prompt = prompt_generate_custom_triton_from_prompt_template(ref_arch_src)
    else:
        raise ValueError(
            f"Unsupported backend: {config.backend}. Must be 'cuda' or 'triton'."
        )
    if config.log_prompt:
        with open(os.path.join(config.logdir, f"prompt_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(custom_prompt)

    # Query server with constructed prompt
    import pdb
    pdb.set_trace()
    print(custom_prompt)
    custom_kernel = inference_server(custom_prompt)
    pdb.set_trace()
    print(custom_kernel)
    extracted_custom_kernel = extract_first_code(custom_kernel, ["python", "cpp"])
    if extracted_custom_kernel == None:
        custom_kernel = custom_kernel
    else:
        custom_kernel = extracted_custom_kernel
    print(custom_kernel)
    # check LLM is able to generate custom CUDA code
    assert custom_kernel is not None, "Custom CUDA code generation failed"
    
    # this should be optional
    if config.log:
        with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_kernel)

    # 3. Evaluate Kernel
    # NOTE: no need to wrap around process here as only a single sample
    # see batch eval for examples of process isolation
    kernel_exec_result = eval_kernel_against_ref(
        ref_arch_src,
        custom_kernel,
        verbose=config.verbose,
        measure_performance=True,
        num_correct_trials=5,
        num_perf_trials=100,
        backend=config.backend,
    )
    pdb.set_trace()
    print(kernel_exec_result.compiled)
    print(kernel_exec_result.correctness)
    print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")

    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(str(kernel_exec_result))


if __name__ == "__main__":
    main()

