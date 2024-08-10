from IPython import embed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys

sys.path.append('..')
sys.path.append('.')

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import json
from tqdm import tqdm, trange
import random


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


class TestRag_Dataset_wsdm(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []

        with open(filename, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data)
        n = int(args.use_data_percent * n)
        # randomly sample n samples for deugging
        if n < len(data):
            random.seed(args.seed)
            data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = line
            flat_concat = [] 
            style_prompt = ""
            cur_utt_text = record["question"]
            ctx_utts_text = []
            history = record['history']
            for i in range(len(history)):
                ctx_utts_text.append(history[i]["question"])
                ctx_utts_text.append(history[i]["answer"])
            cur_response_text = record["answer"]

            pos_docs_text = record["documents"]  # list of pos_docs

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens=True, max_length=args.max_query_length)
            flat_concat.extend(cur_utt)
            prompt_flag = 4
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    if prompt_flag > 0:
                        style_prompt = ctx_utts_text[j] + style_prompt
                        prompt_flag -= 1
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                    if j == 0:
                        if prompt_flag > 0:
                            style_prompt = 'query: ' + ctx_utts_text[j] + ' response: [/INST]' + style_prompt  # [/INST]
                            prompt_flag -= 1
                    else:
                        if prompt_flag > 0:
                            style_prompt = ' [INST] query: ' + ctx_utts_text[j] + ' response: [/INST]' + style_prompt
                            prompt_flag -= 1
                        # ' [INST]query: ' + ctx_utts_text[j] + ' [/INST]response: ' + style_prompt

                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length,
                                       truncation=True)  # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [
                        utt[-1]]  # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt)
            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat,
                                                                       max_pad_length=args.max_concat_length)
            if args.collate_fn_type == "flat_concat_for_train":
                self.examples.append([record['uuid'],
                                      flat_concat,
                                      flat_concat_mask,
                                      cur_response_text,
                                      cur_utt_text,
                                      pos_docs_text,
                                      style_prompt, ctx_utts_text + [cur_utt_text]])

                i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_answers": [],
                             "bt_cur_utt_text": [],
                             "bt_pos_docs": [],
                             "bt_style_prompt": [],
                             "bt_context": []
                             }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_answers"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4]) 
                collated_dict["bt_pos_docs"].append(example[5])
                collated_dict["bt_style_prompt"].append(example[6])
                collated_dict["bt_context"].append(example[7])

            not_need_to_tensor_keys = {"bt_sample_ids", "bt_answers", "bt_cur_utt_text", "bt_pos_docs",
                                       "bt_style_prompt", "bt_context"}

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)

            return collated_dict

        return collate_fn


class RewriteRag_Dataset_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []

        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)
        # randomly sample n samples for deugging
        if n < len(data):
            random.seed(args.seed)
            data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = []
            style_prompt = ""
            cur_utt_text = record['query']
            ctx_utts_text = []
            history_query = record['history_query']
            history_answer = record['history_answer']
            for i in range(len(history_query)):
                ctx_utts_text.append(history_query[i])
                ctx_utts_text.append(history_answer[i])
            cur_response_text = record["answer"]
            oracle_utt_text = record["rewrite"]

            if 'train' in filename:
                if "pos_docs" in record and "neg_docs" in record:
                    pos_docs_text = record["pos_docs"]
                    random_neg_docs_text = record["neg_docs"]
                else:
                    continue
            elif 'dev' in filename:
                if "pos_docs_id" in record:  # and "random_neg_docs_pids" in record:
                    pos_docs_text = record["pos_docs_id"]
                    # random_neg_docs_text = record["random_neg_docs_pids"]
                else:
                    continue

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens=True, max_length=args.max_query_length)
            flat_concat.extend(cur_utt)
            prompt_flag = 4
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    if prompt_flag > 0:
                        style_prompt = ctx_utts_text[j] + style_prompt
                        prompt_flag -= 1
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                    if j == 0:
                        if prompt_flag > 0:
                            style_prompt = 'query: ' + ctx_utts_text[j] + ' response: [/INST]' + style_prompt  # [/INST]
                            prompt_flag -= 1
                    else:
                        if prompt_flag > 0:
                            style_prompt = ' [INST] query: ' + ctx_utts_text[j] + ' response: [/INST]' + style_prompt
                            prompt_flag -= 1
                        # ' [INST]query: ' + ctx_utts_text[j] + ' [/INST]response: ' + style_prompt

                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length,
                                       truncation=True)  # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [
                        utt[-1]]  # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt)
            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat,
                                                                       max_pad_length=args.max_concat_length)
            if args.collate_fn_type == "flat_concat_for_train":
                # if args.decode_type == "oracle":
                target_seq = oracle_utt_text
                target_ids = tokenizer.encode(target_seq, add_special_tokens=True, max_length=args.max_query_length,
                                              truncation=True)
                target_ids, target_mask = padding_seq_to_same_length(target_ids, max_pad_length=args.max_query_length)

                if 'train' in filename:
                    for idx in range(len(pos_docs_text)):
                        pos_doc = [pos_docs_text[idx]]
                        neg_doc = [random_neg_docs_text[0]]

                        self.examples.append([record['id'],
                                              flat_concat,
                                              flat_concat_mask,
                                              target_ids,
                                              target_mask,
                                              cur_response_text,
                                              cur_utt_text,
                                              oracle_utt_text,
                                              pos_doc,
                                              neg_doc,
                                              style_prompt, ctx_utts_text + [cur_utt_text]])
                elif 'dev' in filename:
                    for idx in range(len(pos_docs_text)):
                        pos_doc = [pos_docs_text[idx]]
                        neg_doc = [0]
                        self.examples.append([record['id'],
                                              flat_concat,
                                              flat_concat_mask,
                                              target_ids,
                                              target_mask,
                                              cur_response_text,
                                              cur_utt_text,
                                              oracle_utt_text,
                                              pos_doc,
                                              neg_doc, style_prompt, ctx_utts_text + [cur_utt_text]])
                i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_target_ids": [],
                             "bt_target_mask": [],
                             "bt_answers": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_pos_docs": [],
                             "bt_neg_docs": [],
                             "bt_style_prompt": [],
                             "bt_context": []
                             }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_target_ids"].append(example[3])
                collated_dict["bt_target_mask"].append(example[4])
                collated_dict["bt_answers"].append(example[5])
                collated_dict["bt_cur_utt_text"].append(example[6])  # 当前question 非ids
                collated_dict["bt_oracle_utt_text"].append(example[7])  # 改写的question 非ids
                collated_dict["bt_pos_docs"].append(example[8][0])
                collated_dict["bt_neg_docs"].append(example[9][0])
                collated_dict["bt_style_prompt"].append(example[10])
                collated_dict["bt_context"].append(example[11])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_answers", "bt_cur_utt_text", "bt_oracle_utt_text",
                                           "bt_pos_docs", "bt_neg_docs", "bt_style_prompt", "bt_context"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)

            return collated_dict

        return collate_fn


class RewriteRag_Dataset_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []

        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)
        # randomly sample n samples for deugging
        if n < len(data):
            random.seed(args.seed)
            data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = [] 
            style_prompt = ""
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]

            if 'train' in filename:
                if "pos_docs_text" in record and "random_neg_docs_text" in record:
                    pos_docs_text = record["pos_docs_text"]
                    random_neg_docs_text = record["random_neg_docs_text"]
                else:
                    continue
            elif 'test' in filename:
                if "pos_docs_pids" in record:  # and "random_neg_docs_pids" in record:
                    pos_docs_text = record["pos_docs_pids"]
                    # random_neg_docs_text = record["random_neg_docs_pids"]
                else:
                    continue

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens=True, max_length=args.max_query_length)
            flat_concat.extend(cur_utt)
            prompt_flag = 4
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    if prompt_flag > 0:
                        style_prompt = ctx_utts_text[j] + style_prompt
                        prompt_flag -= 1
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                    if j == 0:
                        if prompt_flag > 0:
                            style_prompt = 'query: ' + ctx_utts_text[j] + ' response: [/INST]' + style_prompt  # [/INST]
                            prompt_flag -= 1
                    else:
                        if prompt_flag > 0:
                            style_prompt = ' [INST] query: ' + ctx_utts_text[j] + ' response: [/INST]' + style_prompt
                            prompt_flag -= 1
                        # ' [INST]query: ' + ctx_utts_text[j] + ' [/INST]response: ' + style_prompt

                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length,
                                       truncation=True)  # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [
                        utt[-1]]  # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt)
            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat,
                                                                       max_pad_length=args.max_concat_length)
            if args.collate_fn_type == "flat_concat_for_train":
                # if args.decode_type == "oracle":
                target_seq = oracle_utt_text
                target_ids = tokenizer.encode(target_seq, add_special_tokens=True, max_length=args.max_query_length,
                                              truncation=True)
                target_ids, target_mask = padding_seq_to_same_length(target_ids, max_pad_length=args.max_query_length)

                if 'train' in filename:
                    for idx in range(len(pos_docs_text)):
                        pos_doc = [pos_docs_text[idx]]
                        neg_doc = [random_neg_docs_text[0]]

                        self.examples.append([record['sample_id'],
                                              flat_concat,
                                              flat_concat_mask,
                                              target_ids,
                                              target_mask,
                                              cur_response_text,
                                              cur_utt_text,
                                              oracle_utt_text,
                                              pos_doc,
                                              neg_doc,
                                              style_prompt, ctx_utts_text + [cur_utt_text]])
                elif 'test' in filename:
                    for idx in range(len(pos_docs_text)):
                        pos_doc = [pos_docs_text[idx]]
                        neg_doc = [0]
                        self.examples.append([record['sample_id'],
                                              flat_concat,
                                              flat_concat_mask,
                                              target_ids,
                                              target_mask,
                                              cur_response_text,
                                              cur_utt_text,
                                              oracle_utt_text,
                                              pos_doc,
                                              neg_doc, style_prompt, ctx_utts_text + [cur_utt_text]])
                i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_target_ids": [],
                             "bt_target_mask": [],
                             "bt_answers": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_pos_docs": [],
                             "bt_neg_docs": [],
                             "bt_style_prompt": [],
                             "bt_context": []
                             }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_target_ids"].append(example[3])
                collated_dict["bt_target_mask"].append(example[4])
                collated_dict["bt_answers"].append(example[5])
                collated_dict["bt_cur_utt_text"].append(example[6])  # 当前question 非ids
                collated_dict["bt_oracle_utt_text"].append(example[7])  # 改写的question 非ids
                collated_dict["bt_pos_docs"].append(example[8][0])
                collated_dict["bt_neg_docs"].append(example[9][0])
                collated_dict["bt_style_prompt"].append(example[10])
                collated_dict["bt_context"].append(example[11])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_answers", "bt_cur_utt_text", "bt_oracle_utt_text",
                                           "bt_pos_docs", "bt_neg_docs", "bt_style_prompt", "bt_context"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)

            return collated_dict

        return collate_fn


class Rag_MRR_Dataset_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []

        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)
        # randomly sample n samples for deugging
        if n < len(data):
            random.seed(args.seed)
            data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = [] 
            style_prompt = ""
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]

            if "pos_docs_pids" in record:  # and "random_neg_docs_pids" in record:
                pos_docs_text = record["pos_docs_pids"]
                # random_neg_docs_text = record["random_neg_docs_pids"]
            else:
                continue

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens=True, max_length=args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    style_prompt = ctx_utts_text[j] + style_prompt
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                    if j == 0:
                        style_prompt = 'query: ' + ctx_utts_text[j] + ' response: [/INST]' + style_prompt  # [/INST]
                    else:
                        style_prompt = ' [INST] query: ' + ctx_utts_text[j] + ' response: [/INST]' + style_prompt
                        # ' [INST]query: ' + ctx_utts_text[j] + ' [/INST]response: ' + style_prompt

                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length,
                                       truncation=True)  # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [
                        utt[-1]]  # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt)
            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat,
                                                                       max_pad_length=args.max_concat_length)
            if args.collate_fn_type == "flat_concat_for_train":
                # if args.decode_type == "oracle":
                target_seq = oracle_utt_text
                target_ids = tokenizer.encode(target_seq, add_special_tokens=True, max_length=args.max_query_length,
                                              truncation=True)
                target_ids, target_mask = padding_seq_to_same_length(target_ids, max_pad_length=args.max_query_length)

                for idx in range(len(pos_docs_text)):
                    pos_doc = [pos_docs_text[idx]]
                    neg_doc = [0]
                    self.examples.append([record['sample_id'],
                                          flat_concat,
                                          flat_concat_mask,
                                          target_ids,
                                          target_mask,
                                          cur_response_text,
                                          cur_utt_text,
                                          oracle_utt_text,
                                          pos_doc,
                                          neg_doc, style_prompt])
                i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_target_ids": [],
                             "bt_target_mask": [],
                             "bt_answers": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_pos_docs": [],
                             "bt_neg_docs": [],
                             "bt_style_prompt": []
                             }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_target_ids"].append(example[3])
                collated_dict["bt_target_mask"].append(example[4])
                collated_dict["bt_answers"].append(example[5])
                collated_dict["bt_cur_utt_text"].append(example[6])
                collated_dict["bt_oracle_utt_text"].append(example[7])
                collated_dict["bt_pos_docs"].append(example[8][0])
                collated_dict["bt_neg_docs"].append(example[9][0])
                collated_dict["bt_style_prompt"].append(example[10])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_answers", "bt_cur_utt_text", "bt_oracle_utt_text",
                                           "bt_pos_docs", "bt_neg_docs", "bt_style_prompt"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)

            return collated_dict

        return collate_fn
