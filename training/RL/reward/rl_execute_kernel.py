import argparse, ast, io, json, os, sys, time, textwrap, multiprocessing as mp
import concurrent.futures as cf
from pathlib import Path
import math
import numpy as np
from termcolor import cprint
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf
from eval import eval_kernel_against_ref, eval_kernel_against_ref_all


def get_config():
    cli_conf   = OmegaConf.from_cli()
    yaml_conf  = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


from concurrent.futures import as_completed

import textwrap

def _run_many_pipe(snippet: str, tests: list[str], conn):
    import textwrap
    results = []
    try:
        ns = {}
        exec(textwrap.dedent(snippet), ns, ns)
        for stmt in tests:
            try:
                exec(stmt, ns, ns)
                results.append(True)
            except SystemExit:
                results.append(True)
            except Exception:
                results.append(False)
        conn.send(results)
    except SystemExit:
        conn.send([True] * len(tests))
    except Exception:
        conn.send([False] * len(tests))
    finally:
        try: conn.close()
        except Exception: pass


def _check_snippet_many(snippet: str, tests: list[str], t_limit: int,
                        spawn_slack: float = 2.0) -> list[bool]:
    import time, multiprocessing as mp
    ctx = mp.get_context("spawn") 
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_run_many_pipe, args=(snippet, tests, child_conn), daemon=True)
    p.start()
    child_conn.close()

    deadline = time.monotonic() + t_limit + spawn_slack
    res = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = remaining if remaining < 0.05 else 0.05
            if parent_conn.poll(wait):
                try:
                    res = parent_conn.recv()
                except EOFError:
                    res = None
                break
            if not p.is_alive():
                if parent_conn.poll(0.05):
                    try:
                        res = parent_conn.recv()
                    except EOFError:
                        res = None
                break

        if res is None and parent_conn.poll(0.05):
            try:
                res = parent_conn.recv()
            except EOFError:
                res = None

        if res is None:
            if p.is_alive():
                p.terminate()
            res = [False] * len(tests)
    finally:
        try: p.join(timeout=0.5)
        except Exception: pass
        try: parent_conn.close()
        except Exception: pass

    return [bool(x) for x in res]

from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_function_dataset(data: list[dict], n_workers: int | None = None):
    import os
    n_cpu = os.cpu_count() or 4
    n_workers = max(1, int(n_workers)) if n_workers is not None else n_cpu

    for item in data:
        m_code = len(item["extracted_output"])
        m_test = len(item["test_list"])
        item["execution_result"] = [[None]  * m_test for _ in range(m_code)]
        item["correctness"]      = [[False] * m_test for _ in range(m_code)]
        item.setdefault("step_map", [])

    tasks = []
    for idx, item in enumerate(data):
        t_limit = item.get("test_time_limit", 1)
        tests   = item["test_list"]
        for i, snippet in enumerate(item["extracted_output"]):
            tasks.append((idx, i, snippet, tests, t_limit))

    futures = {}
    from tqdm.auto import tqdm
    with ThreadPoolExecutor(max_workers=n_workers) as pool, \
        tqdm(total=len(tasks)*len(data[0]["test_list"]), desc=f"Function tests ({n_workers} threads)",
            dynamic_ncols=True, mininterval=0.1, miniters=1) as pbar:

        for idx, i, snippet, tests, t_limit in tasks:
            fut = pool.submit(_check_snippet_many, snippet, tests, t_limit)
            futures[fut] = (idx, i)

        for fut in as_completed(futures):
            idx, i = futures[fut]
            try:
                ok_list = fut.result()
            except Exception:
                ok_list = [False] * len(data[idx]["test_list"])

            for j, ok in enumerate(ok_list):
                data[idx]["execution_result"][i][j] = bool(ok)
                data[idx]["correctness"][i][j]      = bool(ok)
                pbar.update(1)

    return data




def worker_stdio(script, input_val, output_queue):
    # Create an iterator over the input lines.
    input_lines = iter(input_val.splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")
    
    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin



def run_scripts_with_timeout(scripts, inputs, time_limits, worker):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = mp.Queue()
        p = mp.Process(target=worker, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results

def test_if_eq(x, y):  
    return " ".join(x.split()) == " ".join(y.split())

def get_chunk_indices(n, num_chunks):
    size, rem = divmod(n, num_chunks)
    idx, start = [], 0
    for i in range(num_chunks):
        extra = 1 if i < rem else 0
        end   = start + size + extra
        idx.append((start, end)); start = end
    return idx

def run_scripts_with_chunk(code_list, test_input_list, time_limit_list,
                           worker, num_chunks):
    chunks = get_chunk_indices(len(code_list), num_chunks)

    exe_results = []
    pbar = tqdm(total=len(code_list), desc=f"STDIO tests ({num_chunks} ch)")

    for start, end in chunks:
        sub_code_list       = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]

        sub_exe_results = run_scripts_with_timeout(
            sub_code_list,
            sub_test_input_list,
            sub_time_limit_list,
            worker
        )
        exe_results.extend(sub_exe_results)
        pbar.update(end - start)   

    pbar.close()             
    return exe_results


def evaluate_stdio_dataset(data: list[dict], num_chunks: int):
    
    idx_code, idx_case = [], []
    code_list, inp_list, tl_list = [], [], []

    for idx, item in enumerate(data):
        tl = item.get("test_time_limit", 1)
        m_code = len(item["extracted_output"])
        m_case = len(item["test_input"])

        data[idx]["execution_result"] = [[] for _ in range(m_code)]
        data[idx]["correctness"] = [[] for _ in range(m_code)]
        item.setdefault("step_map",           [])

        for c_idx, code in enumerate(item["extracted_output"]):
            for k in range(m_case):
                idx_code.append((idx, c_idx))  
                idx_case.append(k)      
                code_list.append(code)
                inp_list.append(item["test_input"][k])
                tl_list.append(tl)


    exe_results = run_scripts_with_chunk(
        code_list, inp_list, tl_list, worker_stdio, num_chunks
    )

    for i, res in enumerate(exe_results):
        idx, c_idx = idx_code[i]
        k          = idx_case[i]
        item       = data[idx]


        while len(item["execution_result"][c_idx]) < k + 1:
            item["execution_result"][c_idx].append("")
            item["correctness"][c_idx].append(False)
        item["execution_result"][c_idx][k] = res
        exp_out = item["test_output"][k]
        item["correctness"][c_idx][k]      = test_if_eq(res, exp_out)

    return data


import tempfile
import subprocess

def set_gpu_arch(arch_list: list[str]):
    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(f"Invalid architecture: {arch}. Must be one of {valid_archs}")

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

EVAL_SCRIPT_TEMPLATE = '''
import os
import sys
import torch
import json

device_id = int(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
device = torch.device("cuda:0")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

sys.path.append(os.getcwd())
from eval import eval_kernel_against_ref_all

def set_gpu_arch(arch_list):
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

with open(sys.argv[2], 'r') as f:
    pytorch_code = f.read()
with open(sys.argv[3], 'r') as f:
    cuda_kernel = f.read()

gpu_arch = ["Ampere"]
set_gpu_arch(gpu_arch)
  
try:
    result = eval_kernel_against_ref_all(
        pytorch_code,
        cuda_kernel,
        verbose=False,
        measure_performance=False,
        num_correct_trials=1,
        num_perf_trials=50,
        device=device
    )
    if result:
        print(json.dumps({
            "compiled": result.compiled,
            "correctness": result.correctness,
            "runtime": result.runtime if hasattr(result, 'runtime') else None,
            "metadata": result.metadata if hasattr(result, 'metadata') else {}
        }))
    else:
        print(json.dumps({"compiled": False, "correctness": False}))
except Exception as e:
    print(json.dumps({
        "compiled": False, 
        "correctness": False, 
        "error": str(e)
    }))
'''


def _eval_kernel_worker(args):
    idx, c_idx, cuda_code, pytorch_code, gpu_id, timeout = args
    result = eval_single_kernel(cuda_code, pytorch_code, gpu_id, timeout)
    return idx, c_idx, result


def eval_single_kernel(cuda_code: str, pytorch_code: str, eval_gpu_id: int = 0, timeout: int = 180) -> dict:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pytorch_code_path = os.path.join(temp_dir, "pytorch_code.py")
            cuda_kernel_path = os.path.join(temp_dir, "cuda_kernel.py")
            eval_script_path = os.path.join(temp_dir, "eval_script.py")

            with open(pytorch_code_path, 'w') as f:
                f.write(pytorch_code)
            with open(cuda_kernel_path, 'w') as f:
                f.write(cuda_code)
            with open(eval_script_path, 'w') as f:
                f.write(EVAL_SCRIPT_TEMPLATE)

            cmd = [
                sys.executable,
                eval_script_path,
                str(eval_gpu_id),
                pytorch_code_path,
                cuda_kernel_path,
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode == 0 and result.stdout:
                    try:
                        eval_result = json.loads(result.stdout.strip().split('\n')[-1])
                        return eval_result
                    except json.JSONDecodeError:
                        return {"compiled": False, "correctness": False, "error": f"JSON parse error: {result.stdout}"}
                else:
                    return {"compiled": False, "correctness": False, "error": result.stderr}

            except subprocess.TimeoutExpired:
                return {"compiled": False, "correctness": False, "error": "Timeout"}

    except Exception as e:
        return {"compiled": False, "correctness": False, "error": str(e)}


def evaluate_kernel_dataset(data: list[dict], eval_gpu_id: int = 0, timeout: int = 180, flag: str = None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import torch

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(num_gpus))

    print(f"Using {num_gpus} GPUs for parallel kernel evaluation: {gpu_ids}")

    # 初始化结果结构
    for item in data:
        m_code = len(item["extracted_output"])
        item["execution_result"] = [[None] for _ in range(m_code)]
        item["compiled"] = [[False] for _ in range(m_code)]
        item["correctness"] = [[False] for _ in range(m_code)]
        item.setdefault("step_map", [])

    tasks = []
    for idx, item in enumerate(data):
        pytorch_code = item.get("pytorch_code", "")
        for c_idx, cuda_code in enumerate(item["extracted_output"]):
            tasks.append((idx, c_idx, cuda_code, pytorch_code))

    tasks_with_gpu = [
        (idx, c_idx, cuda_code, pytorch_code, gpu_ids[i % num_gpus], timeout)
        for i, (idx, c_idx, cuda_code, pytorch_code) in enumerate(tasks)
    ]

    if flag == "train":
        max_workers = num_gpus * 4
    else:
        max_workers = num_gpus
    print(f"Using {max_workers} workers for parallel kernel evaluation.")

    pbar = tqdm(total=len(tasks), desc=f"Kernel eval (GPU {gpu_ids})")

    # for idx, c_idx, cuda_code, pytorch_code in tasks:
    #     result = eval_single_kernel(cuda_code, pytorch_code, eval_gpu_id, timeout)
    #     print(result)
    #
    #     compiled = result.get("compiled", False)
    #     correctness = result.get("correctness", False)
    #
    #     data[idx]["execution_result"][c_idx][0] = result
    #     data[idx]["compiled"][c_idx][0] = compiled
    #     data[idx]["correctness"][c_idx][0] = compiled and correctness
    #
    #     pbar.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_eval_kernel_worker, task): task for task in tasks_with_gpu}

        for future in as_completed(futures):
            try:
                idx, c_idx, result = future.result()
                print(result)

                compiled = result.get("compiled", False)
                correctness = result.get("correctness", False)

                data[idx]["execution_result"][c_idx][0] = result
                data[idx]["compiled"][c_idx][0] = compiled
                data[idx]["correctness"][c_idx][0] = compiled and correctness

            except Exception as e:
                task = futures[future]
                idx, c_idx = task[0], task[1]
                data[idx]["execution_result"][c_idx][0] = {"compiled": False, "correctness": False, "error": str(e)}
                data[idx]["compiled"][c_idx][0] = False
                data[idx]["correctness"][c_idx][0] = False

            pbar.update(1)

    pbar.close()
    return data


def main():
    config          = get_config()
    project_name = config.experiment.project
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    if num_node > 1:
        file_name    = f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"
    else:
        file_name    = f"../{project_name}/temp_data/outputs-{outputs_name}.json"

    with open(file_name, 'r') as f:
        data = json.load(f)

    func_items  = [itm for itm in data if itm.get("test_method","function") == "function"]
    stdio_items = [itm for itm in data if itm.get("test_method") == "stdio"]

    kernel_items = [itm for itm in data if itm.get("test_method") == "kernel"]

    if func_items:
        updated_func = evaluate_function_dataset(func_items, n_workers=config.execute.num_chunk)
        func_iter = iter(updated_func)
        for i,it in enumerate(data):
            if it.get("test_method","function") == "function":
                data[i] = next(func_iter)

    if stdio_items:
        total_scripts = sum(len(it["extracted_output"]) for it in stdio_items)
        num_chunks    = max(1, math.ceil(total_scripts / config.execute.num_chunk))
        updated_stdio = evaluate_stdio_dataset(stdio_items, num_chunks=num_chunks)
        it_stdio = iter(updated_stdio)
        for i, it in enumerate(data):
            if it.get("test_method") == "stdio":
                data[i] = next(it_stdio)

    if kernel_items:
        eval_gpu_id = int(os.environ.get("EVAL_GPU_ID", "0"))
        timeout = 300
        flag = config.experiment.function

        updated_kernel = evaluate_kernel_dataset(kernel_items, eval_gpu_id=eval_gpu_id, timeout=timeout, flag=flag)
        it_kernel = iter(updated_kernel)
        for i, it in enumerate(data):
            if it.get("test_method") == "kernel":
                data[i] = next(it_kernel)


    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8", errors="surrogatepass") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    

    

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
