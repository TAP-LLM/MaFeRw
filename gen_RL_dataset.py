import os

import logging
class NameFilter(logging.Filter):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

    def filter(self, record):
        return record.name.startswith(self.name)


logger = logging.getLogger('gen_set')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('gen_rl_dataset_topiocqa.log')
file_handler.setLevel(logging.INFO)
name_filter = NameFilter('gen_set')
file_handler.addFilter(name_filter)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
import argparse
from utilities.utils import *
from utilities.model_sampling import T5Sampling
from torch.utils.tensorboard import SummaryWriter
from utilities.envs_cos import rag_environment
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from RL_data_structure import RewriteRag_Dataset_topiocqa
import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_rewriter", type=str,
                        default="PATH/TO/THE/PRETRAINED/REWRITER")
    parser.add_argument("--pretrained_optimizer", type=str,
                        default="PATH/TO/THE/PRETRAINED/OPTIMIZER")

    parser.add_argument("--train_file_path", type=str, default="PATH/TO/THE/DATASET")
    parser.add_argument("--log_dir_path", type=str, default="output/Log")
    parser.add_argument('--model_output_path', type=str, default="output/Checkpoint")
    parser.add_argument("--faiss_path", type=str, default="PATH/TO/THE/FATSS")
    parser.add_argument("--decode_type", type=str, default="RL_2")
    parser.add_argument("--reward_rouge_type", type=str, default="rouge1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_data_percent", type=float, default=1)
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    parser.add_argument("--num_retrieved_docs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_steps", type=int, default=50)
    parser.add_argument("--use_generate", type=bool, default=True)

    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # , args.local_rank)
    args.device = device

    return args


args = get_args()


print("Loading T5 model...")
rewrite_tokenizer = T5Tokenizer.from_pretrained(args.pretrained_rewriter)
rewrite_model = T5ForConditionalGeneration.from_pretrained(args.pretrained_rewriter).to(args.device)

print("Loading dataset...")
train_dataset = RewriteRag_Dataset_topiocqa(args, rewrite_tokenizer, "dataset/topiocqa/train_new.json")
train_loader = DataLoader(train_dataset,
                          # sampler=train_sampler,
                          batch_size=args.batch_size,
                          shuffle=True,
                          collate_fn=train_dataset.get_collate_fn(args))

print("Initializing RL env...")
rag_env = rag_environment(args)

outputfile_rouge = "dataset/RL-train_rouge.json"
outputfile_rank = "dataset/RL-train_rank.json"
outputfile_cos = "dataset/RL-train_cos.json"

num_batch = 0
with open(outputfile_cos, "w") as f_cos, open(outputfile_rank, "w") as f_rank, open(outputfile_rouge, "w") as f_rouge:
    f = [f_cos, f_rank, f_rouge]
    for batch in train_loader:
        num_batch += 1
        new_data = {}
        pos_doc = batch['bt_pos_docs']
        neg_doc = batch['bt_neg_docs']
        context = batch['bt_context']
        bt_style_prompt = batch['bt_style_prompt']
        rewrite_targets = batch['bt_oracle_utt_text']
        ctx_q_ids = batch['bt_input_ids'].to(args.device) 
        ctx_q_mask = batch['bt_attention_mask'].to(args.device) 
        answers = batch['bt_answers']
        target_q_ids = batch['bt_target_ids'].to(args.device)  
        target_q_mask = batch['bt_target_mask'].to(args.device) 
        labels = shift_target_inputs_to_labels(target_q_ids, rewrite_tokenizer.pad_token_id, args.device)
        """
               <bos> word1 word2 word3 <eos> (target input)
               word1 word2 word3 <eos> <pad> (target label)
               from https://github.com/znculee/finetune-transformers (MIT License)
               """
        rewrite_model.eval()
        with torch.no_grad():
            greedy_ids = rewrite_model.generate(
                ctx_q_ids,
                attention_mask=ctx_q_mask,
                max_length=args.max_query_length
            )

        model_sampling = T5Sampling(args, rewrite_model)
        do_sample = True
        output = model_sampling(
            input_ids=ctx_q_ids,
            attention_mask=ctx_q_mask,
            decoder_input_ids=target_q_ids,
            decoder_attention_mask=target_q_mask,
            labels=labels,
            return_dict=True,
            do_sample=do_sample
        )
        with torch.no_grad():
            # remove the first token (decoder_start_token_id) from the sampling_outputs.sequences
            sample_seq = rewrite_tokenizer.batch_decode(output.sequences, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
        greedy_seq = get_tokens_seq_from_token_ids(greedy_ids, rewrite_tokenizer)
        target_embedding_score, target_rank_score, target_rouge_score = rag_env.get_score(pos_doc, neg_doc,
                                                                                          rewrite_targets,
                                                                                          bt_style_prompt,
                                                                                          rewrite_targets, answers)

        # batch size must equal 1. funny!!!!!
        target_score = [target_embedding_score[0][0], target_rank_score[0][0], target_rouge_score[0][0]]
        seqs_lis = [{"rewrite": rewrite_targets[0], "score": target_score}]
        if greedy_seq[0] != rewrite_targets[0]:
            greedy_embedding_score, greedy_rank_score, greedy_rouge_score = rag_env.get_score(pos_doc, neg_doc, greedy_seq,
                                                                                              bt_style_prompt, rewrite_targets,
                                                                                              answers)
            greedy_score = [greedy_embedding_score[0][0], greedy_rank_score[0][0], greedy_rouge_score[0][0]]
            seqs_lis.append({"rewrite": greedy_seq[0], "score": greedy_score})
        if sample_seq[0] != greedy_seq[0] and sample_seq[0] != rewrite_targets[0]:
            sample_embedding_score, sample_rank_score, sample_rouge_score = rag_env.get_score(pos_doc, neg_doc,
                                                                                              sample_seq,
                                                                                              bt_style_prompt,
                                                                                              rewrite_targets, answers)
            sample_score = [sample_embedding_score[0][0], sample_rank_score[0][0], sample_rouge_score[0][0]]
            seqs_lis.append({"rewrite": sample_seq[0], "score": sample_score})

        for i in range(3):
            if len(seqs_lis) == 3:
                pairs = [[0, 1], [0, 2], [1, 2], [1, 0], [2, 0], [2, 1]]
                for pair in pairs:
                    j, k = pair[0], pair[1]
                    if seqs_lis[j]["score"][i] > seqs_lis[k]["score"][i]:
                        new_data = {"score_chosen": seqs_lis[j]["score"][i].tolist(),
                                    "score_rejected": seqs_lis[k]["score"][i].tolist(),
                                    "context": context[0],
                                    "rewrite_chosen": seqs_lis[j]["rewrite"],
                                    "rewrite_rejected": seqs_lis[k]["rewrite"]}
                        f[i].write(json.dumps(new_data))
                        f[i].write('\n')
            elif len(seqs_lis) == 2:
                pairs = [[0, 1], [1, 0]]
                for pair in pairs:
                    j, k = pair[0], pair[1]
                    if seqs_lis[j]["score"][i] > seqs_lis[k]["score"][i]:
                        new_data = {"score_chosen": seqs_lis[j]["score"][i].tolist(),
                                    "score_rejected": seqs_lis[k]["score"][i].tolist(),
                                    "context": context[0],
                                    "rewrite_chosen": seqs_lis[j]["rewrite"],
                                    "rewrite_rejected": seqs_lis[k]["rewrite"]}
                        f[i].write(json.dumps(new_data))
                        f[i].write('\n')



