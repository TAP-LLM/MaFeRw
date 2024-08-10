import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
from accelerate import Accelerator
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, pipeline
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import multiprocessing
import evaluate
import logging


class NameFilter(logging.Filter):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

    def filter(self, record):
        return record.name.startswith(self.name)


logger = logging.getLogger('gen_set')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('ppo_general_topio.log')
file_handler.setLevel(logging.INFO)
name_filter = NameFilter('gen_set')
file_handler.addFilter(name_filter)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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


class RL_dataset(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
            n = len(data)
            n = int(args.use_data_percent * n)
            # randomly sample n samples for deugging
            if n < len(data):
                random.seed(42)
                data = random.sample(data, n)
        if 'qrecc' in filename:
            for line in tqdm(data):
                record = json.loads(line)

                flat_concat = []
                cur_utt_text = record['cur_utt_text'] + '<sep>'
                context = "CONTEXT: " + cur_utt_text
                ctx_utts_text = record['ctx_utts_text']
                label = record["oracle_utt_text"]

                cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens=False, max_length=args.max_query_length)
                flat_concat.extend(cur_utt)

                for j in range(len(ctx_utts_text) - 1, -1, -1):
                    context += ctx_utts_text[j] + '<sep>'
                    if j % 2 == 1:
                        max_length = args.max_response_length
                    else:
                        max_length = args.max_query_length
                    utt = tokenizer.encode(ctx_utts_text[j] + '<sep>', add_special_tokens=False, max_length=max_length,
                                           truncation=True)
                    if len(flat_concat) + len(utt) > args.max_concat_length:
                        flat_concat += utt[:args.max_concat_length - len(
                            flat_concat) - 1] + [
                                           utt[-1]]  # must be ended with [SEP]
                        break
                    else:
                        flat_concat.extend(utt)
                context += "REWRITE: "
                flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat,
                                                                           max_pad_length=args.max_concat_length)
                new_example = {"input_ids": torch.tensor(flat_concat), "attention_mask": torch.tensor(flat_concat_mask),
                               "context": context, "label": label}
                self.examples.append(new_example)
        elif 'topiocqa' in filename:
            for line in tqdm(data):
                record = json.loads(line)

                flat_concat = []
                cur_utt_text = record['query'] + '</s>'
                context = "CONTEXT: " + record['query'] + '<sep>'
                ctx_utts_text = []
                history_query = record['history_query']
                history_answer = record['history_answer']
                for i in range(len(history_query)):
                    ctx_utts_text.append(history_query[i])
                    ctx_utts_text.append(history_answer[i])
                label = record["rewrite"]

                cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens=False, max_length=args.max_query_length)
                flat_concat.extend(cur_utt)

                for j in range(len(ctx_utts_text) - 1, -1, -1):
                    context += ctx_utts_text[j] + '<sep>'
                    if j % 2 == 1:
                        max_length = args.max_response_length
                    else:
                        max_length = args.max_query_length
                    utt = tokenizer.encode(ctx_utts_text[j] + '</s>', add_special_tokens=False, max_length=max_length,
                                           truncation=True)
                    if len(flat_concat) + len(utt) > args.max_concat_length:
                        flat_concat += utt[:args.max_concat_length - len(
                            flat_concat) - 1] + [
                                           utt[-1]]  # must be ended with [SEP]
                        break
                    else:
                        flat_concat.extend(utt)
                context += "REWRITE: "
                flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat,
                                                                           max_pad_length=len(flat_concat))
                new_example = {"input_ids": torch.tensor(flat_concat), "attention_mask": torch.tensor(flat_concat_mask),
                               "context": context, "label": label}
                self.examples.append(new_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn():

        def collate_fn(batch: list):
            collated_dict = {
                "input_ids": [],
                "attention_mask": [],
                "context": [],
                "label": []
            }
            for example in batch:
                collated_dict["input_ids"].append(example["input_ids"])
                collated_dict["attention_mask"].append(example["attention_mask"])
                collated_dict["context"].append(example["context"])
                collated_dict["label"].append(example["label"])
            not_need_to_tensor_keys = {"context", "label"}
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)

            return collated_dict

        return collate_fn


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def get_each_rewards(text, reward_pipe, weight, kwargs):
    pipe_outputs = reward_pipe(text, **kwargs)
    batch_rewards = [weight * 10 * torch.tensor(output[0]["score"] - 0.5) for output in pipe_outputs]
    # print('pipeline')
    return batch_rewards


def get_general_rewards(text, batch_res, batch_label, reward_pipes_lis, weights_dic, kwargs):
    batch_rewards = [torch.tensor(0.0) for i in range(kwargs["batch_size"])]
    if "label" in list(weights_dic.keys()):
        batch_rewards = score_label(batch_res, batch_label, weights_dic["label"])
    inputs = [(text, reward_pipe, weights_dic[key], kwargs) for key, reward_pipe in list(reward_pipes_lis.items())]
    outputs = pool.starmap_async(get_each_rewards, inputs).get()
    for i in range(kwargs["batch_size"]):
        for idx in range(len(reward_pipes_lis)):
            batch_rewards[i] += outputs[idx][i]
    return batch_rewards


def score_label(batch_res, batch_label, weight):
    metric_dict = rouge.compute(predictions=batch_res, references=batch_label,
                                use_stemmer=True, use_aggregator=False, rouge_types=['rougeL'])
    list_of_tensors = [weight * 10 * torch.tensor(score - 0.5) for score in metric_dict['rougeL']]
    return list_of_tensors


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--show_freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_data_percent", type=float, default=1)
    parser.add_argument("--output_dir", type=str, default="output/train_ppo_general_topio/Checkpoints/")
    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # , args.local_rank)
    args.device = device
    args.current_device = Accelerator().local_process_index

    return args


if __name__ == '__main__':
    args = get_args()
    multiprocessing.set_start_method('spawn')
    rouge = evaluate.load("./metrics/rouge")
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained("output/train_topiocqa/Checkpoint/ANCE-rewrite_1-final-model")
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("output/train_topiocqa/Checkpoint/ANCE-rewrite_1-final-model",
                                                               device_map={"": args.current_device})
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        "output/train_topiocqa/Checkpoint/ANCE-rewrite_1-final-model",
        device_map={"": args.current_device})

    special_tokens_dict = {"sep_token": "<sep>"}
    tokenizer.add_special_tokens(special_tokens_dict)
    rouge_reward_model_name = "output/train_RM_rouge-topio/Checkpoint"
    rank_reward_model_name = "output/train_RM_rank-topio/checkpoint-30000"
    cos_reward_model_name = "output/train_RM_cos-topio/checkpoint-30000"
    reward_pipes = {
        "rouge": pipeline(task="text-classification", model=rouge_reward_model_name, device_map={"": args.current_device}, batch_size=args.batch_size),
        # device_map="auto"),
        "rank": pipeline(task="text-classification", model=rank_reward_model_name, device_map={"": args.current_device}, batch_size=args.batch_size),
        # device_map="auto"),
        "cos": pipeline(task="text-classification", model=cos_reward_model_name, device_map={"": args.current_device}, batch_size=args.batch_size)}  # device_map="auto")}
    for pipe in list(reward_pipes.values()):
        pipe.tokenizer.add_special_tokens(special_tokens_dict)

    train_dataset = RL_dataset(args, tokenizer, "dataset/topiocqa/train_new.json")
    ppo_config = PPOConfig(
        seed=42,
        log_with="tensorboard",
        project_kwargs={"logging_dir": "output/train_ppo_general_topio"},
        adap_kl_ctrl=True,
        init_kl_coef=0.05,
        target=1,
        batch_size=32,
        mini_batch_size=32,
        backward_batch_size=16,
        ppo_epochs=4,
        learning_rate=1.41e-5,
        kl_penalty='kl',
        steps=200000,
        # use_score_scaling=True,
        # use_score_norm=True
    )
    sent_kwargs = {
        "top_k": None,
        "function_to_apply": "sigmoid",
        "batch_size": 32,
        "truncation": True,
        "max_length": args.max_query_length
    }
    weights = {"label": 0.60,
               "rouge": 0.35,
               "rank": 0.01,
               "cos": 0.04}
    logger.info(f"weights={weights}\n")
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=train_dataset, data_collator=collator)

    generation_kwargs = {
        "max_length": args.max_query_length,
        # "min_length": args.max_query_length//10,  # don't ignore the EOS token (see above)
        # "top_k": 0.0,  # no top-k sampling
        # "top_p": 1.0,  # no nucleus sampling
        # "do_sample": True,  # yes, we want to sample
        # "pad_token_id": tokenizer.eos_token_id,
    }
    num_rounds = ppo_config.total_ppo_epochs // len(ppo_trainer.dataloader) + 1
    pool = multiprocessing.Pool(len(reward_pipes))  # len(reward_pipes))
    for rounds in range(num_rounds):
        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]
            response_tensors, ref_response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)
            if (epoch + rounds * len(ppo_trainer.dataloader)) % args.show_freq == 0:
                logger.info(f"res: {batch['response'][0]}")
                logger.info(f"lab: {batch['label'][0]}")

            # rewards = score_label(batch["response"], batch["label"])
            # ref_rewards = score_label(batch["ref_response"], batch["label"])
            # rewards = get_each_rewards(batch["context"], batch["response"], rouge_reward_pipe, sent_kwargs)
            # ref_rewards = get_each_rewards(batch["context"], batch["ref_response"], rouge_reward_pipe, sent_kwargs)
            texts = [q + r for q, r in zip(batch["context"], batch["response"])]
            rewards = get_general_rewards(texts, batch["response"], batch["label"], reward_pipes, weights,
                                          sent_kwargs)
            # ref_texts = [q + r for q, r in zip(batch["context"], batch["ref_response"])]
            # ref_rewards = get_general_rewards(ref_texts, batch["ref_response"], batch["label"], reward_pipes,
            #                                   weights,
            #                                   sent_kwargs)
            #
            # batch["ref_rewards"] = ref_rewards
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards,
                                  columns_to_log=["query", "response", "ref_response"])  # , "ref_rewards"])

            if args.save_freq and epoch and (epoch + rounds * len(ppo_trainer.dataloader)) % args.save_freq == 0:
                ppo_trainer.save_pretrained(args.output_dir + f"step_{epoch + rounds * len(ppo_trainer.dataloader)}")
    pool.close()
    pool.join()
    ppo_trainer.save_pretrained(args.output_dir + f"step_final")
