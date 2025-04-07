# MaFeRw
source code for the paper 'MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models'
## Code Structure
```utilities``` contains all helper functions, including RAG environment and utils.\
```RL_data_structure.py``` contains methods for constructing and calling datasets\
```gen_RL_dataset.py``` contains the code to generate the data used to train the reward models.\
```reward_modeling.py``` contains code for training the reward models. \
```ppo_pipeline_pool.py``` contains the code to train the rewrite model using the ppo algorithm.
## Setup Environment

Please run the following command to install required packages

```
# requirements
pip install -r requirements.txt
```
## Download data and Preprocessing

Public datasets can be download from [QReCC](https://github.com/apple/ml-qrecc), [TopiOCQA](https://github.com/McGill-NLP/topiocqa). Data preprocessing follow the approach in [this work](https://github.com/fengranMark/ConvGQR/tree/main).

## Rewriter Initialize

Initialize the rewriter by running ```train_rewriter_initialize.py``` to SFT the T5-base model.

## Reward Model Traing

The data for reward model traing can be collected by running ```gen_RL_dataset.py```. And use the rewriter after SFT and the collected data to train the corresponding RMs through running ```reward_modeling.py```. 

## RL Training

Run ```ppo_pipeline_pool.py``` with your selecting parameters to further train the rewriter with PPO.

## References
The code for dataset processing refers to https://github.com/fengranMark/ConvGQR/tree/main.

The code for training the reward model and reinforcement learning refers to https://github.com/huggingface/trl.

## Cite Format

    @misc{wang2024maferwqueryrewritingmultiaspect,
      title={MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models}, 
      author={Yujing Wang and Hainan Zhang and Liang Pang and Binghui Guo and Hongwei Zheng and Zhiming Zheng},
      year={2024},
      eprint={2408.17072},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.17072}, 
      }
