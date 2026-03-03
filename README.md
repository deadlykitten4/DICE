<div align="center">
      <h2><b> DICE: Diffusion Large Language Models Excel at Generating CUDA Kernels </b></h2>
</div>

<div align="center">

**Haolei Bai<sup>1</sup>, Lingcheng Kong<sup>1,2</sup>, Xueyi Chen<sup>1</sup>, Jiamian Wang<sup>3</sup>, Zhiqiang Tao<sup>3</sup>, Huan Wang<sup>1,†</sup>**  
*<sup>1</sup>Westlake University, <sup>2</sup>The Hong Kong University of Science and Technology, <sup>3</sup>Rochester Institute of Technology*  
<sup>†</sup>Corresponding author
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.11715-b31b1b.svg)](https://arxiv.org/abs/2602.11715)
[![Hugging Face](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/collections/DeadlyKitt3n/dice)
[![GitHub Repo stars](https://img.shields.io/github/stars/deadlykitten4/DICE)](https://github.com/deadlykitten4/DICE)
<br>**[<a href="https://huggingface.co/papers/2602.11715">HuggingFace Daily Paper</a>]**

</div>

## 🗓️ Plan

- [ ] We will release our code soon. Stay tuned!
- [x] **2026.02.13**: We release DICE-1.7B, DICE-4B, and DICE-8B on [Hugging Face](https://huggingface.co/collections/DeadlyKitt3n/dice) !
- [x] **2026.02.13**: The paper is on [arXiv](https://arxiv.org/abs/2602.11715) !

##  🚀 Usage
### 1. Install dependencies
```bash
conda env create -f environment.yml
```
### 2. Training

#### 2.1 Prepare dataset
We provide the curated CuKe dataset for SFT in 
```bash
DICE/Training/SFT/llama_factory_sdar/data/CuKe_dataset.json
```

#### 2.2 Supervised fine-tuning
We follow the training process of [SDAR](https://github.com/JetAstra/SDAR), you may check [here](https://github.com/JetAstra/SDAR/tree/main/training) for more instruction.
```bash
cd DICE/Training/SFT/llama_factory_sdar
torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 --master_addr 127.0.0.1 --master_port 12345 ./src/llamafactory/launcher.py ./examples/train_full_sdar/sdar_8b_full.yaml
```

#### 2.2 BiC-RL

### 3. Evaluation

## 🙌 Acknowledgement

We are grateful to the [SDAR](https://github.com/JetAstra/SDAR), [TraceRL](https://github.com/Gen-Verse/dLLM-RL), [KernelBench](https://github.com/ScalingIntelligence/KernelBench), [cudaLLM](https://github.com/ByteDance-Seed/cudaLLM) for releasing their code publicly, which greatly facilitated our work. 


## 📖 Citation

If you find **DICE** useful for your research or projects, please consider citing our work:

```bibtex
@article{bai2026dice,
  title={DICE: Diffusion Large Language Models Excel at Generating CUDA Kernels},
  author={Bai, Haolei and Kong, Lingcheng and Chen, Xueyi and Wang, Jiamian and Tao, Zhiqiang and Wang, Huan},
  journal={arXiv preprint arXiv:2602.11715},
  year={2026}
}
```