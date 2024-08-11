import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
from utilities.utils import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import argparse
from tqdm import tqdm
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig


def padding_seq_to_same_length(input_ids, max_pad_length, pad_token=0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids

    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length

    return input_ids, attention_mask


class RM_dataset(Dataset):
    def __init__(self, args, rm_tokenizer, filename):
        self.examples = []
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        for line in tqdm(data):
            record = json.loads(line)
            context = record["context"]
            chosen = "REWRITE: " + record["rewrite_chosen"]
            score_chosen = record["score_chosen"]
            rejected = "REWRITE: " + record["rewrite_rejected"]
            score_rejected = record["score_rejected"]

            inputs_ids_chosen = []
            inputs_ids_rejected = []
            chosen_ids = rm_tokenizer.encode(chosen, add_special_tokens=True, max_length=args.max_query_length-1, truncation=True)
            # chosen_ids = chosen_ids[: chosen_ids.index(rm_tokenizer.eos_token_id) + 1]
            rejected_ids = rm_tokenizer.encode(rejected, add_special_tokens=True, max_length=args.max_query_length-1, truncation=True)
            # rejected_ids = rejected_ids[: rejected_ids.index(rm_tokenizer.eos_token_id) + 1]

            first_context = True
            for j in range(len(context) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length

                if first_context:
                    context[j] = "CONTEXT: " + context[j]
                    first_context = False

                utt = rm_tokenizer.encode(context[j] + '<sep>', add_special_tokens=False, max_length=max_length, truncation=True)
                # utt = list(filter(lambda x: x != rm_tokenizer.eos_token_id, utt))
                if len(inputs_ids_chosen) == args.max_concat_length:
                    break
                elif (len(inputs_ids_chosen) < args.max_concat_length) and (len(inputs_ids_chosen) + len(utt) > args.max_concat_length):
                    inputs_ids_chosen += utt[:args.max_concat_length - len(
                        inputs_ids_chosen) - 1] + [utt[-1]]  # must be ended with [SEP]
                    inputs_ids_rejected += utt[:args.max_concat_length - len(
                        inputs_ids_rejected) - 1] + [utt[-1]]
                    break
                else:
                    inputs_ids_chosen.extend(utt)
                    inputs_ids_rejected.extend(utt)

            inputs_ids_chosen.extend(chosen_ids)
            inputs_ids_rejected.extend(rejected_ids)

            input_ids_chosen, attention_mask_chosen = padding_seq_to_same_length(inputs_ids_chosen,
                                                                               max_pad_length=args.max_query_length + args.max_concat_length)
            input_ids_rejected, attention_mask_rejected = padding_seq_to_same_length(inputs_ids_rejected,
                                                                                   max_pad_length=args.max_query_length + args.max_concat_length)

            new_example = {"input_ids_chosen": input_ids_chosen, "attention_mask_chosen": attention_mask_chosen,
                           "input_ids_rejected": input_ids_rejected, "attention_mask_rejected": attention_mask_rejected}
            self.examples.append(new_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn():

        def collate_fn(batch: list):
            collated_dict = {
                "input_ids_chosen": [],
                "attention_mask_chosen": [],
                "input_ids_rejected": [],
                "attention_mask_rejected": []}
            for example in batch:
                collated_dict["input_ids_chosen"].append(example["input_ids_chosen"])
                collated_dict["attention_mask_chosen"].append(example["attention_mask_chosen"])
                collated_dict["input_ids_rejected"].append(example["input_ids_rejected"])
                collated_dict["attention_mask_rejected"].append(example["attention_mask_rejected"])
            not_need_to_tensor_keys = {"bt_score_chosen", "bt_score_rejected"}
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)

            return collated_dict

        return collate_fn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # , args.local_rank)
    args.device = device

    return args


tokenizer = AutoTokenizer.from_pretrained('T5-base')
model = AutoModelForSequenceClassification.from_pretrained("T5-base")
special_tokens_dict = {"sep_token": "<sep>"}
tokenizer.add_special_tokens(special_tokens_dict)

args = get_args()
reference_dataset = RM_dataset(args, tokenizer, "dataset/topiocqa/RL-train_cos.json")
"""
{"score_chosen": ...,  "score_rejected": ..., "context": "[...]", "rewrite_chosen": "...",  "rewrite_rejected": "..."}
{"score_chosen": ...,  "score_rejected": ..., "context": "[...]", "rewrite_chosen": "...",  "rewrite_rejected": "..."}
{"score_chosen": ...,  "score_rejected": ..., "context": "[...]", "rewrite_chosen": "...",  "rewrite_rejected": "..."}
"""

train_size = int(0.8 * len(reference_dataset))
test_size = len(reference_dataset) - train_size
train_dataset, test_dataset = random_split(reference_dataset, [train_size, test_size])

# train_para = {
#     "output_dir": "reward_modeling_cp",
#     "per_device_train_batch_size": 16,
#     "max_length": 512
#
# }
training_args = RewardConfig(
    output_dir="output",
    dataloader_num_workers=2,
    learning_rate=5e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    logging_first_step=True,
    num_train_epochs=5,
    dataloader_prefetch_factor=2,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=3000,
    warmup_ratio=0.1,
    logging_steps=50,
    save_steps=10000,
    max_length=args.max_query_length + args.max_concat_length,
    gradient_checkpointing=False)
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
trainer.save_model("output/Checkpoint")
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
print(metrics)
