# Awesome-reasoning
Collect every awesome work about O1/R1!


## Table of Contents
- [Repos](#repos)
- [Papers](#papers)
- [Models](#models)
- [Infra](#infra)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Tools](#tools)


## Repos

### DeepSeek repos:

[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) ![Stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-R1?style=social) - DeepSeek-R1 official repository.

### Qwen repos:

[Qwen-repos](https://github.com/QwenLM) -Qwen official github group.


## Papers

* [DeepSeek-R1-Tech-Report](https://arxiv.org/pdf/2501.12948)
* [DeepSeek-V3 Tech-Report](https://arxiv.org/pdf/2412.19437)
* [Qwen-math-PRM-Tech-Report(MCTS/PRM)](https://arxiv.org/pdf/2501.07301)
* [Qwen2.5 Tech-Report](https://arxiv.org/pdf/2412.15115)
* [DeepSeek Math Tech-Report(GRPO)](https://arxiv.org/pdf/2402.03300)
* [Kimi K1.5 Tech-Report](https://arxiv.org/pdf/2501.12599)


## Models

DeepSeek series:

| Model ID                      | ModelScope                                                                               | Hugging Face                                                                   |
|-------------------------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| DeepSeek R1                   | [Model Link](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1)                   | [Model Link](https://huggingface.co/deepseek-ai/DeepSeek-R1)                   |
| DeepSeek V3                   | [Model Link](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3)                   | [Model Link](https://huggingface.co/deepseek-ai/DeepSeek-V3)                   |
| DeepSeek-R1-Distill-Qwen-32B  | [Model Link](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)  | [Model Link](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)  |
| DeepSeek-R1-Distill-Qwen-14B  | [Model Link](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)  | [Model Link](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)  |
| DeepSeek-R1-Distill-Llama-8B   | [Model Link](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)   | [Model Link](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)   |
| DeepSeek-R1-Distill-Qwen-7B   | [Model Link](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)   | [Model Link](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)   |
| DeepSeek-R1-Distill-Qwen-1.5B | [Model Link](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | [Model Link](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| DeepSeek-R1-GGUF | [Model Link](https://www.modelscope.cn/models/unsloth/DeepSeek-R1-GGUF) | [Model Link](https://huggingface.co/unsloth/DeepSeek-R1-GGUF) |
| DeepSeek-R1-Distill-Qwen-32B-GGUF | [Model Link](https://www.modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF) | [Model Link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF) |
| DeepSeek-R1-Distill-Llama-8B-GGUF | [Model Link](https://www.modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) | [Model Link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) |

Qwen series:

| Model ID    | ModelScope                                                             | Hugging Face                                                 |
|-------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| QwQ-32B-Preview | [Model Link](https://www.modelscope.cn/models/Qwen/QwQ-32B-Preview) | [Model Link](https://huggingface.co/Qwen/QwQ-32B-Preview) |
| QVQ-72B-Preview | [Model Link](https://www.modelscope.cn/models/deepseek-ai/QVQ-72B-Preview) | [Model Link](https://huggingface.co/Qwen/QVQ-72B-Preview) |
| QwQ-32B-Preview-GGUF | [Model Link](https://www.modelscope.cn/models/unsloth/QwQ-32B-Preview-GGUF) | [Model Link](https://huggingface.co/unsloth/QwQ-32B-Preview-GGUF) |
| QVQ-72B-Preview-bnb-4bit | [Model Link](https://www.modelscope.cn/models/unsloth/QVQ-72B-Preview-bnb-4bit) | [Model Link](https://huggingface.co/unsloth/QVQ-72B-Preview-bnb-4bit) |

## Infra

- Open R1 by Hugging Face: https://github.com/huggingface/open-r1
  - This repo is the official repo of Hugging Face to reproduce the training infra of R1
- SimpleRL-Reason: https://github.com/hkust-nlp/simpleRL-reason
  - Use OpenRLHF to reproduce R1
- Ragen: https://github.com/ZihanWang314/RAGEN
  - A General-Purpose Reasoning Agent Training Framework and reproduce R1
- TRL: https://github.com/huggingface/trl
  - Hugging Face official training framework which supports open-source GRPO and other RL algorithms.
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
  - An RL repo which supports RLs(supports REINFORCE++)
- TinyZero: https://github.com/Jiayi-Pan/TinyZero
  - A tiny reproduction of R1

## Datasets

* [NuminaMath-TIR](https://www.modelscope.cn/datasets/AI-MO/NuminaMath-TIR) - Tool-integrated reasoning (TIR) plays a crucial role in this competition.  
* [NuminaMath-CoT](https://www.modelscope.cn/datasets/AI-MO/NuminaMath-CoT) - Approximately 860k math problems, where each solution is formatted in a Chain of Thought (CoT) manner.
* [BAAI-TACO](https://modelscope.cn/datasets/BAAI/TACO) - TACO is a benchmark for code generation with 26443 problems. 
* [OpenThoughts-114k](https://modelscope.cn/datasets/open-thoughts/OpenThoughts-114k) - Open synthetic reasoning dataset with 114k high-quality examples covering math, science, code, and puzzles!
* [Bespoke-Stratos-17k](https://modelscope.cn/datasets/bespokelabs/Bespoke-Stratos-17k) - A reasoning dataset of questions, reasoning traces, and answers.


## Evaluation

* [MATH-500](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-500) - A subset of 500 problems from the MATH benchmark that OpenAI created in their Let's Verify Step by Step paper
* [AIME-2024](https://modelscope.cn/datasets/AI-ModelScope/AIME_2024) - This dataset contains problems from the American Invitational Mathematics Examination (AIME) 2024. 
* [AIME-VALIDATION](https://www.modelscope.cn/datasets/AI-MO/aimo-validation-aime) - All 90 problems come from AIME 22, AIME 23, and AIME 24
* [MATH-LEVEL-4](https://www.modelscope.cn/datasets/AI-MO/aimo-validation-math-level-4) - A subset of level 4 problems from the MATH benchmark.
* [MATH-LEVEL-5](https://www.modelscope.cn/datasets/AI-MO/aimo-validation-math-level-5) - A subset of level 5 problems from the MATH benchmark.
* [aimo-validation-amc](https://www.modelscope.cn/datasets/AI-MO/aimo-validation-amc) - All 83 samples come from AMC12 2022, AMC12 2023

## Tools

