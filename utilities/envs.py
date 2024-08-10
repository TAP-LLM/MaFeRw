from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from .utils import *
import evaluate


class rag_environment():
    def __init__(self, args):
    # Load evaluate
        if args.reward_rouge_type == 'bleu':
            self.bleu = evaluate.load("./metrics/bleu")
        elif args.reward_rouge_type == 'meteor':
            self.meteor = evaluate.load("./metrics/meteor")
        elif args.reward_rouge_type == 'all':
            self.bleu = Bleu()  # evaluate.load("./metrics_local/bleu")
            print("finish bleu loading")
            self.meteor = Meteor()
            print("finish meteor loading")
            self.rouge = evaluate.load("./metrics/rouge")
        else:
            self.rouge = evaluate.load("./metrics/rouge")
        self.embedding_model = SentenceTransformer("PATH/TO/THE/PRETRAINED/MODEL",
                                                   device=args.device
                                                   )
        embedding_model = HuggingFaceEmbeddings( 
            model_name="PATH/TO/THE/PRETRAINED/MODEL",
            model_kwargs={'device': 'cuda:0'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = FAISS.load_local(args.faiss_path, embeddings=embedding_model,
                                             allow_dangerous_deserialization=True)
        if args.use_generate:
            self.tokenizer = AutoTokenizer.from_pretrained('llama2-13b-chat', trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                "/home/wangyujing/llama2-13b-chat",
                torch_dtype=torch.float16,
                device_map="sequential",
                trust_remote_code=True
            )
            model = base_model.eval()

            self.Pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="sequential",
            )

        self.args = args

    def get_score(self, pos_doc, neg_doc, q_seq, style_prompt, label, answer):
        """

        :param label: List(String), rewrite target
        :param style_prompt: List(String), example prompt for LLM
        :param pos_doc: List(String) pos_doc texts
        :param neg_doc: List(String) neg_doc texts
        :param q_seq: List(String) the rewritten queries
        :param answer: List(String) the responding answers
        :return: Tensor(batch_size * 1).to(device)
        """

        self.embedding_model.eval()
        pos_doc_embs = self.embedding_model.encode(pos_doc, convert_to_tensor=True)  # batch_size
        q_embs = self.embedding_model.encode(q_seq, convert_to_tensor=True)  # batch_size
        neg_doc_embs = self.embedding_model.encode(neg_doc, convert_to_tensor=True)  # batch_size
        answer_embs = self.embedding_model.encode(answer, convert_to_tensor=True)
        embedding_score = torch.diagonal(self.embedding_model.similarity(q_embs, pos_doc_embs)).unsqueeze(
            0).t() 
        embedding_score = embedding_score.to(self.args.device)

        rank_score = torch.zeros((self.args.batch_size, 1))
        batch_inputs = []
        for i in range(self.args.batch_size):
            query = q_seq[i]
            answer_emb = answer_embs[i].unsqueeze(0)
            docs = self.vector_store.similarity_search(query, k=self.args.num_retrieved_docs)

            ip_score = 0
            doc_context = ''
            doc_r = 1
            for doc in docs:
                content = doc.page_content
                lis = content.split('passage:')
                if len(lis) == 2:
                    content = lis[1]
                doc_context += ' ({}) '.format(doc_r) + content
                doc_r += 1

                content_emb = self.embedding_model.encode(content, convert_to_tensor=True)
                ip_value = self.embedding_model.similarity(answer_emb, content_emb.unsqueeze(0))
                ip_score += 1 / doc_r * ip_value.squeeze(-1)
            rank_score[i][0] = ip_score

            if self.args.use_generate:
                prompt = f"You are a concise, honest, and rigorous respondent, please use your knowledge and the following information to answer question, the information provided is :{doc_context}, example responses are in the context: {style_prompt[i]} "
                base_prompt = "<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_prompt}[/INST]"
                each_input = base_prompt.format(system_prompt=prompt, user_prompt=query)
                batch_inputs.append(each_input)

        if self.args.use_generate:
            sequences = self.Pipeline(
                batch_inputs,
                do_sample=True,
                top_k=10,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=3000,
                max_new_tokens=self.args.max_response_length,
                return_full_text=False,
                temperature=0.02,
                batch_size=self.args.batch_size
            )
            prediction = [sequence[0]['generated_text'] for sequence in sequences]
            metric_score = torch.zeros((self.args.batch_size, 1))
            if self.args.reward_rouge_type == 'bleu':
                for i in range(self.args.batch_size):
                    metric_dict = {'bleu': self.bleu.compute(predictions=prediction[i], references=answer[i])['bleu']}
                    metric_score[i][0] = torch.tensor(metric_dict[self.args.reward_rouge_type])  # 越大越好
            elif self.args.reward_rouge_type == 'meteor':
                for i in range(self.args.batch_size):
                    metric_dict = {
                        'meteor': self.meteor.compute(predictions=prediction[i], references=answer[i])['meteor']}
                    metric_score[i][0] = torch.tensor(metric_dict[self.args.reward_rouge_type])  # 越大越好
            else:
                for i in range(self.args.batch_size):
                    metric_dict = self.rouge.compute(predictions=[prediction[i]], references=[answer[i]],
                                                     use_stemmer=True)
                    metric_score[i][0] = torch.tensor(metric_dict[self.args.reward_rouge_type])  # 越大越好
            metric_score = metric_score.to(self.args.device)
            rank_score = rank_score.to(self.args.device)
            return embedding_score, rank_score, metric_score
        else:
            rank_score = rank_score.to(self.args.device)
            return embedding_score, rank_score

    def get_reward(self, pos_doc, neg_doc, greedy_q_seq, sample_q_seq, style_prompt, label, answer,
                   normalize_weights=True):
        """

        :param label: List(String), rewrite target
        :param pos_doc: List(String), pos_doc texts
        :param neg_doc: List(String), neg_doc texts
        :param greedy_q_seq: List(String), queries generated by T5 rewriter
        :param sample_q_seq: List(String), queries sampled by T5 rewriter
        :param style_prompt: List(String), example prompt for LLM
        :param answer: List(String), the responding answers
        :param normalize_weights: Bool, if the weights of various rewards are normalized
        :return: Tensor,
        """
        rewards = []
        weights = []
        if self.args.use_generate:
            greedy_embedding_score, greedy_rank_score, greedy_rouge_score = self.get_score(pos_doc,
                                                                                           neg_doc,
                                                                                           greedy_q_seq,
                                                                                           style_prompt,
                                                                                           label,
                                                                                           answer)
            sample_embedding_score, sample_rank_score, sample_rouge_score = self.get_score(pos_doc,
                                                                                           neg_doc,
                                                                                           sample_q_seq,
                                                                                           style_prompt,
                                                                                           label,
                                                                                           answer)
        else:
            greedy_embedding_score, greedy_rank_score = self.get_score(pos_doc,
                                                                       neg_doc,
                                                                       greedy_q_seq,
                                                                       style_prompt,
                                                                       label,
                                                                       answer)
            sample_embedding_score, sample_rank_score = self.get_score(pos_doc,
                                                                       neg_doc,
                                                                       sample_q_seq,
                                                                       style_prompt,
                                                                       label,
                                                                       answer)

        embedding_reward = sample_embedding_score - greedy_embedding_score
        rewards.append(embedding_reward)
        weights.append(self.args.embedding_reward_weight)

        rank_reward = sample_rank_score - greedy_rank_score
        rewards.append(rank_reward)
        weights.append(self.args.rank_reward_weight)

        if self.args.use_generate:
            rouge_reward = sample_rouge_score - greedy_rouge_score
            rewards.append(rouge_reward)
            weights.append(self.args.rouge_reward_weight)

        norm = sum(weights) if normalize_weights else 1.0

        reward = 0
        for weight, score in zip(weights, rewards):
            reward += weight / norm * score

        return reward

       def infer_step(self, q_seq, bt_style_prompt, label, pos_pids):
        """
        q_seq, bt_style_prompt, label: lis(str), length = batch_size,
        pos_pids: lis(int), length = batch_size.
        # label is the responding answer
        # q_seq is the result generated by T5 rewriter
        """
        batch_inputs = []
        retrieval_score = 0
        for i in range(self.args.batch_size):
            query = q_seq[i]
            style_prompt = bt_style_prompt[i]
            docs = self.vector_store.similarity_search(query, k=self.args.num_retrieved_docs)

            doc_context = ''
            doc_r = 1
            flag = 0  # 每个query，MRR只算一次
            for doc in docs:
                if pos_pids[i] == int(doc.metadata['source']) and flag == 0:
                    MRR = 1 / doc_r  # self.args.num_retrieval_p
                    print('MRR = ', MRR)
                    retrieval_score += MRR
                    flag += 1
                doc_r += 1
                content = doc.page_content
                lis = content.split('passage:')
                if len(lis) == 2:
                    content = lis[1]
                doc_context += ' ({}) '.format(doc_r) + content

            prompt = f"You are a concise, honest, and rigorous respondent, please use your knowledge and the following information to answer question, the information provided is :{doc_context}, example responses are in the context: {style_prompt} "
            base_prompt = "<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_prompt}[/INST]"
            each_input = base_prompt.format(system_prompt=prompt, user_prompt=query)
            batch_inputs.append(each_input)

        sequences = self.Pipeline(
            batch_inputs,
            do_sample=True,
            top_k=10,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=3000,
            max_new_tokens=self.args.max_response_length,
            return_full_text=False,
            temperature=0.0001,
            batch_size=self.args.batch_size
        )
        prediction = [sequence[0]['generated_text'] for sequence in sequences]
        rouge_score = self.rouge.compute(predictions=prediction, references=label, use_stemmer=True)
        bleu_score = self.bleu.compute(predictions=prediction, references=label)['bleu']
        meteor_score = self.meteor.compute(predictions=prediction, references=label)['meteor']
        """
        rouge_score: dict
        bleu_score, meteor_score, retrieval_score: float
        prediction: lis(str)
        """
        return rouge_score, bleu_score, meteor_score, retrieval_score, prediction
