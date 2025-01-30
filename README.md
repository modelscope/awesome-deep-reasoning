# awesome-r1
Collect every awesome work about r1!

## Contents
- [Papers](#papers)
- [Models](#models)
- [Infra](#infra)
- [Datasets](#datasets)
- [Evaluation](#evaluation)


## Papers

- DeepSeek R1(Official paper): https://arxiv.org/pdf/2501.12948
- DeepSeek V3(PreTrain): https://arxiv.org/pdf/2412.19437
- DeepSeek Math(GRPO): https://arxiv.org/pdf/2402.03300

## Models

| Model ID    | ModelScope                                                             | Hugging Face                                                 |
| ----------- |------------------------------------------------------------------------|--------------------------------------------------------------|
| DeepSeek R1 | [Model Link](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1) | [Model Link](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| DeepSeek V3 | [Model Link](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3) | [Model Link](https://huggingface.co/deepseek-ai/DeepSeek-V3) |


## Infra

- Open R1 by Hugging Face: https://github.com/huggingface/open-r1
  - This repo is the official repo of Hugging Face to reproduce the training infra of R1
- SimpleRL-Reason: https://github.com/hkust-nlp/simpleRL-reason
  - Use OpenRLHF to reproduce R1

## Datasets

* [NuminaMath-TIR](https://www.modelscope.cn/datasets/AI-MO/NuminaMath-TIR) - Tool-integrated reasoning (TIR) plays a crucial role in this competition.  
* [NuminaMath-CoT](https://www.modelscope.cn/datasets/AI-MO/NuminaMath-CoT) - Approximately 860k math problems, where each solution is formatted in a Chain of Thought (CoT) manner.


## Evaluation
* [MATH-500](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-500) - A subset of 500 problems from the MATH benchmark that OpenAI created in their Let's Verify Step by Step paper
* [AIME-VALIDATION](https://www.modelscope.cn/datasets/AI-MO/aimo-validation-aime) - All 90 problems come from AIME 22, AIME 23, and AIME 24
* [MATH-LEVEL-4](https://www.modelscope.cn/datasets/AI-MO/aimo-validation-math-level-4) - A subset of level 4 problems from the MATH benchmark.
* [MATH-LEVEL-5](https://www.modelscope.cn/datasets/AI-MO/aimo-validation-math-level-5) - A subset of level 5 problems from the MATH benchmark.
* [aimo-validation-amc](https://www.modelscope.cn/datasets/AI-MO/aimo-validation-amc) - All 83 samples come from AMC12 2022, AMC12 2023

