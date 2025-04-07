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

Public datasets can be download from [QReCC](https://github.com/apple/ml-qrecc), [TopiOCQA](https://github.com/McGill-NLP/topiocqa).
## References
The code for dataset processing refers to https://github.com/fengranMark/ConvGQR/tree/main.

The code for training the reward model and reinforcement learning refers to https://github.com/huggingface/trl.
