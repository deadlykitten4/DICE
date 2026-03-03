import numpy as np

def geometric_mean_speed_ratio_correct_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    prod = np.prod(speed_up)
    n_correct = np.sum(is_correct) # Count number of correct samples

    return prod ** (1 / n_correct) if n_correct > 0 else 0

def geometric_mean_speed_ratio_correct_and_faster_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples that have speedup > 1
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    speed_up = np.array([x for x in speed_up if x > 1])
    prod = np.prod(speed_up)
    n_correct_and_faster = len(speed_up)

    return prod ** (1 / n_correct_and_faster) if n_correct_and_faster > 0 else 0

def fastp(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float) -> float:
    """
    Rate of samples within a threshold p
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0

# def fastp_at_k(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float,
#                k_values: list[int]) -> dict:
#     """
#     计算FastP@k指标：从k个样本中至少有一个满足Fast_p条件的概率
#
#     参数：
#     :param is_correct: 每个样本是否正确的布尔数组
#     :param baseline_speed: 基准速度数组
#     :param actual_speed: 实际速度数组
#     :param n: 样本总数
#     :param p: 加速比阈值
#     :param k_values: 要计算的k值列表
#
#     返回：
#     :return: 包含每个k值对应的FastP@k值的字典
#     """
#     # 计算满足Fast_p条件的样本数量
#     # 条件：样本正确且加速比 > p
#     meets_fastp_condition = np.zeros(n, dtype=bool)
#
#     for i in range(n):
#         if is_correct[i]:
#             speedup = baseline_speed[i] / actual_speed[i]
#             meets_fastp_condition[i] = speedup > p
#
#     # 满足条件的样本数
#     c = np.sum(meets_fastp_condition)
#
#     # 计算每个k值的FastP@k
#     results = {}
#     for k in k_values:
#         if n - c < k:
#             results[k] = 1.0
#         else:
#             results[k] = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
#
#     return results
