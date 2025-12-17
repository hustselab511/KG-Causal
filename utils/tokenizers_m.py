import json
import os
import re
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from utils.tool import load_json_args, setup_seed


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args["ann_path"]
        self.threshold = args["threshold"]
        self.dataset_name = args["dataset_name"]
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        elif self.dataset_name == 'mimic_cxr':
            self.clean_report = self.clean_report_mimic_cxr
        elif self.dataset_name == 'mimic_cxr_mini':
            self.clean_report = self.clean_report_mimic_cxr
        elif self.dataset_name == 'iu_xray_c':
            self.clean_report = self.clean_report_iu_xray
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()
        # self.m2bert = self.create_bert_mapping()
        self.idx2embed = self.create_bert_embedding()
        # self.node2idx = self.create_node_vocabulary()

    def create_bert_embedding(self):

        bio_clinical_bert_path = "pretrained/Bio_ClinicalBERT"
        clinical_bert_path = "pretrained/ClinicalBERT"


        text_tokenizer = AutoTokenizer.from_pretrained(
            bio_clinical_bert_path,
            local_files_only=True,
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            bio_clinical_bert_path,
            local_files_only=True,
            trust_remote_code=True
        )
        # text_tokenizer = AutoTokenizer.from_pretrained(
        #     clinical_bert_path,
        #     local_files_only=True,
        #     trust_remote_code=True
        # )
        # model = AutoModel.from_pretrained(
        #     clinical_bert_path,
        #     local_files_only=True,
        #     trust_remote_code=True
        # )

        model.eval()

        vocab_words = list(self.token2idx.keys())
        idx2embed = {}
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(vocab_words), batch_size):
                batch_words = vocab_words[i:i + batch_size]
                # print(batch_words)
                for i in range(len(batch_words)):
                    if batch_words[i] == "<unk>":
                        # print("unk")
                        batch_words[i] = "[UNK]"

                # print(batch_words)
                inputs = text_tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True, max_length=128)
                # print(inputs)

                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # print(batch_embeddings.shape)

                for word, embedding in zip(batch_words, batch_embeddings):
                    if word == "[UNK]":
                        word = "<unk>"
                    original_idx = self.token2idx[word]

                    idx2embed[original_idx] = embedding

        print(len(idx2embed))
        # print(idx2embed)

        return idx2embed

    def create_node_vocabulary(self):
        file_path = '../../data/knowledge/inf.json'


        with open(file_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)

        total_tokens = []

        # 遍历每个元素
        for file_key, content in kg_data.items():
            entities = content.get('entities', {})
            for entity_id, entity_info in entities.items():
                # 添加实体名称到列表
                total_tokens.append(entity_info.get('tokens', '').lower())

        counter = Counter(total_tokens)
        node2idx = {}
        # print(len(counter))
        # for k, v in counter.items():
        #     print(k,v)

        vocab = [k for k, v in counter.items() if v >= 2] + ['<unk>']
        # print(len(vocab))

        return node2idx



    def create_vocabulary(self):
        total_tokens = []
        if self.dataset_name == 'ffa_ir':
            _report_key = 'En_Report'
            for example in self.ann['train']:
                tokens = self.clean_report(self.ann['train'][example][_report_key]).split()
                for token in tokens:
                    total_tokens.append(token)
        else:
            _report_key = 'report'
            for example in self.ann['train']:
                tokens = self.clean_report(example[_report_key]).split()
                for token in tokens:
                    total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        # with open(os.path.join(r"C:\Users\oblivion\Desktop\Multimodal\data\iu_xray", r'vocabulary'), 'w', encoding='utf-8') as f:
        #     json.dump(idx2token, f, ensure_ascii=False, indent=4)

        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_ffa_ir(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        # jieba cut the tokens
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report,out_type=True):
        tokens = self.clean_report(report).split()
        if out_type:
            ids = []
            for token in tokens:
                ids.append(self.get_id_by_token(token))
            ids = [0] + ids + [0]
            return ids
        else:
            return tokens

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out


class MixTokenizer(object):
    def __init__(self, args):
        self.ann_path = {'iu_xray': "../../MRG/data/iu_xray/annotation.json",
                         'mimic_cxr': "../../MRG/data/mimic_cxr/annotation.json"}
        self.threshold = {'iu_xray': 3,
                          'mimic_cxr': 10}
        self.dataset_name = args['dataset_name']
        self.ann = {'iu_xray': json.loads(open(self.ann_path['iu_xray'], 'r').read()),
                    'mimic_cxr': json.loads(open(self.ann_path['mimic_cxr'], 'r').read())}
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens_iu_xray = []
        total_tokens_mimic_cxr = []

        for example in self.ann['iu_xray']['train']:
            tokens = self.clean_report_iu_xray(example['report']).split()
            for token in tokens:
                total_tokens_iu_xray.append(token)

        for example in self.ann['mimic_cxr']['train']:
            tokens = self.clean_report_mimic_cxr(example['report']).split()
            for token in tokens:
                total_tokens_mimic_cxr.append(token)

        counter_iu_xray = Counter(total_tokens_iu_xray)
        counter_mimic_cxr = Counter(total_tokens_mimic_cxr)
        # counter vocab which more than [threshold] times appear
        vocab = [k for k, v in counter_iu_xray.items() if v >= self.threshold['iu_xray']] + ['<unk>']
        vocab += [k for k, v in counter_mimic_cxr.items() if v >= self.threshold['mimic_cxr'] and k not in vocab]
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report, dataset='iu_xray',out_type=True):
        if dataset == 'iu_xray':
            tokens = self.clean_report_iu_xray(report).split()
        else:
            tokens = self.clean_report_mimic_cxr(report).split()
        if out_type:
            ids = []
            for token in tokens:
                ids.append(self.get_id_by_token(token))
            ids = [0] + ids + [0]
            return ids
        else:
            return tokens

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out



if __name__ == "__main__":
    tokenizers_fn = {'ori': Tokenizer, 'mix': MixTokenizer}
    args = load_json_args("C:/Users/oblivion/Desktop/Multimodal/config/KM/mimic_cxr.json")
    torch.cuda.set_device(int(args["cuda"]))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # -------------------------------
    # 设置种子，确保实验可复现
    # -------------------------------
    if args["seed"] == -1:
        args["seed"] = np.random.randint(0, 23333)

    # print(args)
    print(f"seed:{args["seed"]}")
    setup_seed(args["seed"])

    # -------------------------------
    # 创建分词器，处理文本,可以使用单数据集或混合数据集新建词表
    # -------------------------------
    tokenizer = tokenizers_fn[args['tokenizer']](args)
    print('count of tokens', len(tokenizer.token2idx))
    print(tokenizer.token2idx)

    with open(os.path.join(r"C:\Users\oblivion\Desktop\Multimodal\data\knowledge\mimic", r'vocabulary'), 'w', encoding='utf-8') as f:
        json.dump(tokenizer.idx2token, f, ensure_ascii=False, indent=4)
