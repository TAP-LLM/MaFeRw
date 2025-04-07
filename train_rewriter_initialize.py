import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='train-T5-rewrite_topiocqa.log')
logger = logging.getLogger(__name__)
import argparse
from utilities.utils import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from RL_data_structure import RewriteRag_Dataset_qrecc
from os import path
from os.path import join as oj


def save_model(args, model, query_tokenizer, optimizer, step):
    output_dir = oj(args.model_output_path, '{}-model'.format(step))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    # model_to_save.t5.save_pretrained(output_dir)
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))

    optimizer_path = oj(args.model_output_path, 'optimizer.pth')
    logger.info(f'# checkpoint optimizer in : {optimizer_path}')
    torch.save(optimizer.state_dict(), optimizer_path)


def final_save_model(args, model, query_tokenizer, optimizer, step):
    output_dir = oj(args.model_output_path, 'final-model')
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    # model_to_save.t5.save_pretrained(output_dir)
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))

    optimizer_path = oj(args.model_output_path, f'optimizer_.pth')
    logger.info(f'# checkpoint optimizer in : {optimizer_path}')
    torch.save(optimizer.state_dict(), optimizer_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", type=str, default="dataset/qrecc/train.json")
    parser.add_argument("--log_dir_path", type=str, default="output/train_qrecc/Log")
    parser.add_argument('--model_output_path', type=str, default="output/train_qrecc/Checkpoint")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train")
    parser.add_argument("--use_prefix", type=bool, default=True)

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--use_data_percent", type=float, default=1)

    parser.add_argument("--num_train_epochs", type=int, default=5, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    parser.add_argument("--num_retrieved_docs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=0.5)
    parser.add_argument("--disable_tqdm", type=bool, default=True)

    parser.add_argument("--print_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_portion", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--train_minib_aggregate", type=int, default=1)
    parser.add_argument("--checkpoint_every", type=int, default=4000)
    parser.add_argument("--use_pre_optimizer", type=bool, default=True)
    parser.add_argument("--use_generate", type=bool, default=False)

    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # , args.local_rank)
    args.device = device

    return args


args = get_args()
args.logger = logger
tensorboard = SummaryWriter("Topiocqa_rewrite_loss/")

print("Loading T5 model...")
rewrite_tokenizer = T5Tokenizer.from_pretrained(args.pretrained_rewriter)
rewrite_model = T5ForConditionalGeneration.from_pretrained(args.pretrained_rewriter).to(args.device)

print("Loading dataset...")
train_dataset = RewriteRag_Dataset_qrecc(args, rewrite_tokenizer, args.train_file_path)
train_loader = DataLoader(train_dataset,
                          # sampler=train_sampler,
                          batch_size=args.batch_size,
                          shuffle=True,
                          collate_fn=train_dataset.get_collate_fn(args))
print("Initializing T5 model...")
total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
num_warmup_steps = args.num_warmup_portion * total_training_steps
optimizer_0 = get_optimizer(args, rewrite_model, weight_decay=args.weight_decay)
scheduler_0 = get_linear_schedule_with_warmup(optimizer_0, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

rewrite_model.train()
global_step = 0
best_loss = 1000
for epoch in range(args.num_train_epochs):
    for batch in train_loader:
        rewrite_model.zero_grad()

        ctx_q_ids = batch['bt_input_ids'].to(args.device)  
        ctx_q_mask = batch['bt_attention_mask'].to(args.device)
        target_q_ids = batch['bt_target_ids'].to(args.device)
        output = rewrite_model(input_ids=ctx_q_ids, attention_mask=ctx_q_mask, labels=target_q_ids)

        rewrite_loss = output.loss
        rewrite_loss.backward()
        torch.nn.utils.clip_grad_norm_(rewrite_model.parameters(), args.max_grad_norm)
        optimizer_0.step()
        scheduler_0.step()

        global_step += 1
        tensorboard.add_scalar("loss", rewrite_loss, global_step)
        if args.print_steps > 0 and global_step % args.print_steps == 0:
            logger.info("Epoch = {}, Global Step = {}, rewrite loss = {}".format(
                epoch + 1,
                global_step,
                rewrite_loss.item()))

        # save model finally
        if global_step % args.checkpoint_every == 0:
            save_model(args, rewrite_model, rewrite_tokenizer, optimizer_0, global_step)
            best_loss = rewrite_loss
            logger.info("Epoch = {}, Global Step = {}, rewrite loss = {}".format(
                epoch + 1,
                global_step,
                rewrite_loss.item()))
final_save_model(args, rewrite_model, rewrite_tokenizer, optimizer_0, global_step)
