import json
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass

import pydra
import torch
import re
import numpy as np

from datasets import load_dataset
from pydra import Config, REQUIRED

# Import only what we need
from src import compile

from src.dataset import construct_kernelbench_dataset
from src.eval import (
    check_metadata_serializable_all_types,
    eval_kernel_against_ref,
    get_error_name,
    KernelExecResult,
)
from src.utils import read_file, set_gpu_arch, extract_first_code
from tqdm import tqdm

"""
Batch Evaluation from JSON file containing PyTorch and CUDA code pairs.
This script evaluates the kernels against the reference architecture, and stores the results in the specified file.

Usually with eval, we check
- correctness (n_correct): 5 randomized input trials
- performance (n_trials): 100 randomized input trials

You can increase the number of trials for correctness and performance
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)


class EvalConfig(Config):
    def __init__(self):
        # Input JSON file containing the generated kernels
        self.input_json_path = REQUIRED  # path to the JSON file with pytorch/cuda code pairs

        # Output JSON file for results
        self.output_json_path = REQUIRED  # path to save evaluation results

        # Output JSON file for successful kernels only
        self.successful_kernels_path = REQUIRED  # path to save successful kernels only

        # Dataset source (for reference architecture)
        # self.dataset_src = REQUIRED  # either huggingface or local
        #
        # # name of dataset name on Hugging Face
        # self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        # self.level = REQUIRED

        # subset of problems to evaluate
        self.subset = (None, None)  # (start_id, end_id), these are the logical index

        # Evaluation Mode: local (requires GPU), see modal (cloud GPU) in the modal file
        self.eval_mode = "local"

        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ampere"]

        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        self.verbose = False

        # Eval settings
        self.num_correct_trials = 1
        self.num_perf_trials = 100
        self.timeout = 180  # in seconds
        self.measure_performance = False

        # Eval Flow setting
        # To speedup evaluation, you can start building the kernel on CPU on disk as cache
        self.build_cache = False
        self.num_cpu_workers = (
            20  # number of parallel process to to parallelize the build on CPUs
        )

        # Directory to build kernels for evaluation
        self.kernel_eval_build_dir = os.path.join(REPO_TOP_DIR, "cache")

        # number of GPUs to do batch evaluation
        self.num_gpu_devices = 1

        # Backend to use for kernel implementation (cuda or triton)
        self.backend = "cuda"

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@dataclass
class WorkArgs:
    problem_id: int
    sample_id: int
    device: torch.device


# def fetch_ref_arch_from_problem_id(
#         dataset, problem_id: int, dataset_src: str
# ) -> str | None:
#     """
#     Fetch reference architecture from problem directory
#     Either from Hugging Face or Local Dataset
#     """
#     if dataset_src == "huggingface":
#         curr_problem_row = dataset.filter(
#             lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None
#         )
#         ref_arch_src = curr_problem_row["code"][0]
#         problem_name = curr_problem_row["name"][0]
#
#     elif dataset_src == "local":
#         problem_idx_in_dataset = (
#                 problem_id - 1
#         )  # due to dataset list being 0-indexed locally
#         ref_arch_path = dataset[problem_idx_in_dataset]
#
#         problem_name = os.path.basename(ref_arch_path)
#         ref_arch_src = read_file(ref_arch_path)
#
#     # verify
#     # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
#     problem_number = int(problem_name.split("_")[0])
#     assert (
#             problem_number == problem_id
#     ), f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
#
#     return ref_arch_src


def evaluate_single_sample(
        work_args: WorkArgs, configs: EvalConfig, dataset, pytorch_code: str, cuda_code: str
) -> KernelExecResult | None:
    """
    Evaluate a single sample on a single GPU
    """
    problem_id, sample_id, device = (
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
    )
    # fetch reference architecture from problem directory
    # ref_arch_src = fetch_ref_arch_from_problem_id(
    #     dataset, problem_id, configs.dataset_src
    # )

    build_dir = os.path.join(
        configs.kernel_eval_build_dir, f"json_eval", f"{problem_id}", f"{sample_id}"
    )

    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=pytorch_code,
            custom_model_src=cuda_code,
            measure_performance=configs.measure_performance,
            verbose=configs.verbose,
            num_correct_trials=configs.num_correct_trials,
            num_perf_trials=configs.num_perf_trials,
            build_dir=build_dir,
            device=device,
            backend=configs.backend,
        )
        return eval_result
    except Exception as e:
        print(
            f"[WARNING] Last level catch on {sample_id}: Some issue evaluating for kernel: {e} "
        )
        if "CUDA error" in str(e):
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {
                "cuda_error": f"CUDA Error: {str(e)}",
                "cuda_error_name": get_error_name(e),
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }  # log this for debugging as this usually signifies illegal memory access
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
        else:
            metadata = {
                "other_error": f"error: {str(e)}",
                "other_error_name": get_error_name(e),
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }  # for debugging
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result


def cuda_single_eval_wrapper(
        curr_work: WorkArgs, configs: dict, dataset, pytorch_code: str, cuda_code: str
):
    """
    Wrapper to handle timeout and keyboard interrupt
    """
    with mp.Pool(1) as pool:
        try:
            result = pool.apply_async(
                evaluate_single_sample,
                args=(curr_work, configs, dataset, pytorch_code, cuda_code),
            ).get(timeout=configs.timeout)
        except KeyboardInterrupt:
            print("\n [Terminate] Caught KeyboardInterrupt, terminating workers...")
            pool.terminate()
            pool.join()
            raise
        except mp.TimeoutError as e:
            print(
                f"[WARNING] Evaluation TIMED OUT for Problem ID: {curr_work.problem_id}, Sample ID: {curr_work.sample_id}\nException: {e}"
            )
            result = None

        print(
            f"[Eval Result] Problem ID: {curr_work.problem_id}, Sample ID: {curr_work.sample_id}: {result}"
        )
        return result


def remove_cache_dir(cache_dir: str, problem_id, sample_id):
    """
    Remove the cached folder for sample compilation so it can start a clean build next time
    useful for time out, failed build, etc.
    """
    problem_cache_dir = os.path.join(
        cache_dir, f"json_eval", f"{problem_id}", f"{sample_id}"
    )
    print(f"cache_dir to remove: {problem_cache_dir}")
    if os.path.exists(problem_cache_dir):
        try:
            shutil.rmtree(problem_cache_dir, ignore_errors=True)
            print(
                f"\n[INFO] Removed cached folder for Problem ID: {problem_id}, Sample ID: {sample_id}"
            )
        except Exception as e:
            print(f"\n[WARNING] Failed to remove cache directory {problem_cache_dir}: {str(e)}")


def batch_eval(
        total_work: list[tuple[int, int, str, str]],
        config: EvalConfig,
        # curr_level_dataset,
        eval_file_path: str,
        successful_kernels_path: str,
):
    """
    Batch evaluation across multiple GPUs
    We put in time out for each batch, consider trying again with larger time out if it didn't finish building.
    Cache directory is removed if evaluation times out or fails
    """
    # construct a list of work args
    batch_size = config.num_gpu_devices

    # Initialize results dictionary
    all_results = {}
    successful_kernels = {}

    with tqdm(total=len(total_work), desc="Processing batches") as pbar:
        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]  # pop the first batch_size elements
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks over {config.num_gpu_devices} GPUs; [Total Work left] {len(total_work)}"
            )
            assert (
                    len(curr_work_batch) <= batch_size
            ), f"Current batch size {len(curr_work_batch)} is greater than the number of GPUs {batch_size}"

            with mp.Pool(batch_size) as pool:
                # work_args = [
                #     (
                #         WorkArgs(
                #             problem_id=p_id,
                #             sample_id=s_idx,
                #             device=torch.device(f"cuda:{i % batch_size}"),
                #         ),
                #         config,
                #         # curr_level_dataset,
                #         pytorch_code,
                #         cuda_code,
                #     )
                #     for i, (p_id, s_idx, pytorch_code, cuda_code) in enumerate(curr_work_batch)
                # ]
                #
                start_time = time.time()
                #
                # async_results = []
                # for work_arg in work_args:
                #     async_results.append(
                #         pool.apply_async(evaluate_single_sample, args=work_arg)
                #     )
                async_results = []
                for i, (p_id, s_idx, pytorch_code, cuda_code) in enumerate(curr_work_batch):
                    work_arg = WorkArgs(
                        problem_id=p_id,
                        sample_id=s_idx,
                        device=torch.device(f"cuda:{i % batch_size}"),
                    )
                    async_results.append(
                        pool.apply_async(evaluate_single_sample,
                                         args=(work_arg, config, None, pytorch_code, cuda_code))
                    )

                # Collect results with a batch timeout
                results = []
                batch_timeout = config.timeout
                for i, async_result in enumerate(async_results):
                    problem_id, sample_id, pytorch_code, cuda_code = curr_work_batch[i]

                    try:
                        elapsed_time = time.time() - start_time
                        remaining_time = max(0, batch_timeout - elapsed_time)
                        result = async_result.get(timeout=remaining_time)
                        results.append((problem_id, sample_id, pytorch_code, cuda_code, result))

                    except mp.TimeoutError:
                        print(
                            f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_id}"
                        )
                        results.append((problem_id, sample_id, pytorch_code, cuda_code, None))

                        remove_cache_dir(
                            config.kernel_eval_build_dir,
                            problem_id,
                            sample_id,
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}"
                        )
                        results.append((problem_id, sample_id, pytorch_code, cuda_code, None))
                        remove_cache_dir(
                            config.kernel_eval_build_dir,
                            problem_id,
                            sample_id,
                        )

                end_time = time.time()

                # Process current batch results
                for problem_id, sample_id, pytorch_code, cuda_code, result in results:
                    print("-" * 128)
                    print(
                        f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_id}"
                    )
                    print(result)

                    # Add to all results
                    if result is not None:
                        key = f"{problem_id}_{sample_id}"
                        all_results[key] = {
                            "problem_id": problem_id,
                            "sample_id": sample_id,
                            "compiled": result.compiled,
                            "correctness": result.correctness,
                            "metadata": check_metadata_serializable_all_types(result.metadata),
                            "runtime": result.runtime,
                            "runtime_stats": result.runtime_stats,
                            "pytorch_code": pytorch_code,  # 保存PyTorch代码
                            "cuda_code": cuda_code,  # 保存CUDA代码
                        }

                        # 如果内核编译正确且运行正确，添加到成功的内核中
                        if result.compiled and result.correctness:
                            successful_kernels[key] = {
                                "problem_id": problem_id,
                                "sample_id": sample_id,
                                "pytorch_code": pytorch_code,  # 保存PyTorch代码
                                "cuda_code": cuda_code,  # 保存CUDA代码
                                "runtime": result.runtime,
                                "runtime_stats": result.runtime_stats,
                            }

                # 保存中间结果
                save_results_to_files(all_results, successful_kernels, eval_file_path, successful_kernels_path)

                print("-" * 128)
                print(
                    f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds"
                )

                pbar.update(len(curr_work_batch))

    # 最终保存
    save_results_to_files(all_results, successful_kernels, eval_file_path, successful_kernels_path)

    # 打印摘要
    successful_count = len(successful_kernels)
    total_count = len(all_results)
    print(f"评估完成: {successful_count}/{total_count} 内核成功")
    print(f"所有结果已保存到: {eval_file_path}")
    print(f"成功的内核已保存到: {successful_kernels_path}")


def save_results_to_files(all_results, successful_kernels, eval_file_path, successful_kernels_path):
    """保存结果到JSON文件"""
    # 保存所有结果
    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
    with open(eval_file_path, "w") as f:
        json.dump(all_results, f, indent=4)

    # 保存成功的内核
    os.makedirs(os.path.dirname(successful_kernels_path), exist_ok=True)
    with open(successful_kernels_path, "w") as f:
        json.dump(successful_kernels, f, indent=4)


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    从JSON文件批量评估样本
    将评估结果存储在指定的评估结果文件中
    """
    print(f"开始批量评估，配置: {config}")

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA设备不可用。评估需要GPU。")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # 数据集配置
    # if config.dataset_src == "huggingface":
    #     dataset = load_dataset(config.dataset_name)
    #     curr_level_dataset = dataset[f"level_{config.level}"]
    # elif config.dataset_src == "local":
    #     curr_level_dataset = construct_kernelbench_dataset(config.level)
    #
    # num_problems_in_level = len(curr_level_dataset)
    #
    # if config.subset == (None, None):
    #     problem_id_range = range(1, num_problems_in_level + 1)
    # else:
    #     assert (
    #             config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level
    #     ), f"子集范围 {config.subset} 超出了级别 {config.level} 的范围"
    #     problem_id_range = range(config.subset[0], config.subset[1] + 1)

    # 从JSON文件加载PyTorch代码和CUDA代码
    print(f"从 {config.input_json_path} 加载数据...")
    try:
        with open(config.input_json_path, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"加载JSON文件失败: {e}")

    print(f"从JSON文件加载了 {len(json_data)} 个项目")

    # 设置GPU架构
    set_gpu_arch(config.gpu_arch)
    assert (
            config.num_gpu_devices <= torch.cuda.device_count()
    ), f"请求的GPU数量 ({config.num_gpu_devices}) 大于可用的GPU数量 ({torch.cuda.device_count()})"

    # 准备工作项
    total_work = []

    for item_id, item in enumerate(json_data):
        # 从项目中获取PyTorch代码和CUDA代码
        problem_id = item.get("problem_id", item_id + 1)  # 使用item_id+1作为后备

        # 如果problem_id不在范围内，跳过
        # if problem_id not in problem_id_range:
        #     continue

        sample_id = item.get("sample_id", 0)

        # 从输入和输出字段提取代码
        # 假设input包含PyTorch代码，output包含CUDA代码
        pytorch_code = item.get("input", "")
        if isinstance(pytorch_code, str) and "```" in pytorch_code:
            pytorch_code = extract_first_code(pytorch_code, ["python", "cpp"])

        cuda_code = item.get("output", "")
        if isinstance(cuda_code, str) and "```" in cuda_code:
            cuda_code = extract_first_code(cuda_code, ["python", "cpp"])

        # 如果任一代码为空，跳过
        if not pytorch_code or not cuda_code:
            print(f"跳过项目 {item_id}: 缺少PyTorch或CUDA代码")
            continue

        total_work.append((problem_id, sample_id, pytorch_code, cuda_code))

    print(f"开始评估 {len(total_work)} 个样本")

    # 在CPU上预先构建缓存
    if config.build_cache:
        # 此功能需要为新工作流进行调整
        pass

    # 在多个GPU上批量评估
    batch_eval(
        total_work,
        config,
        # curr_level_dataset,
        config.output_json_path,
        config.successful_kernels_path
    )


if __name__ == "__main__":
    main()