import os
import torch
from torch import nn
from transformers import AdamW
import shutil



def shift_target_inputs_to_labels(tgt_input_ids, pad_token_id, device):
    """
    <bos> word1 word2 word3 <eos> (target input)
    word1 word2 word3 <eos> <pad> (target label)
    from https://github.com/znculee/finetune-transformers (MIT License)
    """
    batch_pads = torch.empty(
        tgt_input_ids.shape[0], 1,
        dtype=tgt_input_ids.dtype,
        device=device
    ).fill_(pad_token_id)
    labels = torch.cat((tgt_input_ids[:, 1:], batch_pads), dim=1)

    # masking pad tokens in labels with -100 for CE to ignore
    pad_mask = labels == pad_token_id
    labels[pad_mask] = -100

    return labels


def get_tokens_seq_from_token_ids(token_ids, tokenizer):
    ''' Retruns tokens from a sequence of token ids '''

    tokens = []

    for seq_ids in token_ids.to('cpu').numpy().tolist():
        seq_toks = tokenizer.decode(
            seq_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        tokens.append(seq_toks)

    return tokens


def cal_rank_loss(q_embs, pos_doc_embs, neg_doc_embs):
    batch_size = q_embs.shape[0]
    pos_scores = torch.diagonal(q_embs.mm(pos_doc_embs.T)).unsqueeze(0).t()
    neg_scores = torch.sum(q_embs * neg_doc_embs, dim=1).unsqueeze(1)
    score_mat = torch.cat([pos_scores, neg_scores], dim=1)
    label_mat = torch.zeros(batch_size).type(torch.cuda.LongTensor)
    loss_func = nn.CrossEntropyLoss(reduction='none')
    loss = loss_func(score_mat, label_mat).unsqueeze(1)
    return loss


def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def check_dir_exist_or_build(dir_list, erase_dir_content = None):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
    if erase_dir_content:
        for dir_path in erase_dir_content:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
