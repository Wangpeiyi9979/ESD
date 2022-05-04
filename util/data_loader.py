import torch
import torch.utils.data as data
import os
from transformers import BertTokenizer
import random
from .utils import get_entities
from copy import copy
import json

O_CLASS = 0

def get_O_type(s, e, entity_spans):
    O_type = 0
    for span in entity_spans:
        entity_s, entity_e = span
        if s >= entity_s and e <= entity_e:
            O_type = 2 # The sub span of one of entity spans
            break
        if (s <= entity_e and e > entity_e) or (s < entity_s and e >= entity_s):
            O_type = 1  # overlap with one of entity spans
            break
    return O_type # other


class FewDataSet(data.Dataset):
    """
    Fewshot NER Dataset
    """

    def __init__(self, filepath, opt):

        self.FewNERD = True if opt.dataset == 'fewnerd' else False
        if self.FewNERD:
            filepath = f'{filepath}_{opt.N}_{opt.K}.jsonl'

        if not os.path.exists(filepath):
            print("[ERROR] Data file {} does not exist!".format(filepath))
            assert (0)

        print(f'load data from {filepath}')

        self.N = opt.N
        self.K = opt.K
        self.L = opt.L
        self.opt = opt
        self.is_support = True

        self.max_o_num = opt.max_o_num
        self.tokenizer = BertTokenizer.from_pretrained(opt.bert_path)

        if self.FewNERD:
            datas = open(filepath).readlines()
            datas = [json.loads(x.strip()) for x in datas]
            if 'train' in filepath and (opt.N==10 and opt.K==5):
                self.datas = []
                for d in datas:
                    support = copy(d['support'])
                    query_num = len(d['query']['word'])
                    for idx in range((query_num + opt.eposide_tasks - 1) // opt.eposide_tasks):
                        query = {}
                        for k in d['query']:
                            query[k] = d['query'][k][idx*opt.eposide_tasks:(idx+1)*opt.eposide_tasks]
                        data_unit = {'support': support,
                                    'query': query,
                                    'types': d['types']}
                        self.datas.append(data_unit)
            else:
                self.datas = datas
        else:
            datas = json.load(open(filepath))
            keys = list(datas.keys())
            self.datas = []
            for key in keys:
                print('[Field] {}'.format(key))
                if 'train' in filepath:
                    for d in datas[key]:
                        support = copy(d['support'])
                        query_num = len(d['batch']['seq_ins'])
                        assert query_num % opt.eposide_tasks == 0
                        for idx in range(query_num // opt.eposide_tasks):
                            query = {}
                            for k in d['batch']:
                                query[k] = d['batch'][k][idx*opt.eposide_tasks:(idx+1)*opt.eposide_tasks]
                            data_unit = {'support': support,
                                        'query': query}
                            self.datas.append(data_unit)
                else:
                    self.datas += datas[key]

        self.train = True if 'train' in filepath else False
        print(f'eposide num: {len(self.datas)}')

    def __additem__(self, d, word, spans, span_tags, span_lens, seq_tag_expand, word_expand, gold_entitys):
        d['word'].append(word)
        d['spans'].append(spans)
        d['span_tags'].append(span_tags)
        d['span_lens'].append(span_lens)
        d['seq_tag_expand'].append(seq_tag_expand)
        d['word_expand'].append(word_expand)
        d['gold_entitys'].append(gold_entitys)

    def __fewnerd_get_class_span_dict__(self, label):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        from: https://github.com/thunlp/Few-NERD/blob/main/util/metric.py
        '''
        class_span = {}
        current_label = None
        i = 0
        # having tags in string format ['O', 'O', 'person-xxx', ..]
        while i < len(label):
            if label[i] != 'O':
                start = i
                current_label = label[i]
                i += 1
                while i < len(label) and label[i] == current_label:
                    i += 1
                if current_label in class_span:
                    class_span[current_label].append((start, i))
                else:
                    class_span[current_label] = [(start, i)]
            else:
                i += 1
        return class_span
    


    def __snips_get_tags__(self, seq_labels):
        all_label = set()
        for seq_label in seq_labels:
            for l in seq_label:
                if l != 'O':
                    l = l[2:]
                    all_label.add(l)
        all_label = list(all_label)
        all_label.sort()
        all_label = ['O1', 'O2', 'O3'] + all_label
        return all_label


    def __get_spans__(self, tags, not_expand):
        spans = []
        span_tags = []
        lens = []

        # Get all entity spans.
        gold_entitys = set()
        if self.FewNERD:
            class2spans = self.__fewnerd_get_class_span_dict__(tags)
        else:
            class2spans = get_entities(tags)


        for cls in class2spans:
            spans_with_tag = class2spans[cls]
            for span in spans_with_tag:
                b, e = span

                tag = self.tag2label[cls]
                assert tag >= self.opt.O_class_num
                if self.is_support == False:
                    tag = tag - self.opt.O_class_num + 1 # for query, O1, O2, O3 -> O, entity_tag -> entity_tag - 2
                gold_entitys.add((b, e - 1, tag))
                span_len = sum(not_expand[b:e]) # the real length of this span.

                if self.train == False and self.is_support == False and span_len > self.L: # During Inference. For query span with the real length > L.  We do not predict this span.
                    continue

                spans.append([b, e-1])  
                span_tags.append(tag)
                lens.append(span_len)
                
        length = len(tags)

        # Get all O class spans
        O_spans = []
        O_span_tags = []
        O_span_lens = []
        for start in range(0, length):  # [start, end).
            for end in range(start+1, length+1): 
                if not_expand[start] == 0: # the start token is sub-word. We do not need predict this span.
                    continue
                if end < length and not_expand[end] == 0: # the end token is not real end token, since there exist sub-word after it.  We do not need predict this span.
                    continue

                rel_len = sum(not_expand[start:end])
                if rel_len > self.L:  # the real length is larger than L.  We do not need predict this span.
                    continue

                if [start, end-1] not in spans:  
                    O_spans.append([start, end - 1])
                    if self.is_support == False:
                        O_span_tags.append(O_CLASS) # for query, O1, O2, O3 -> O
                    else:
                        O_span_tags.append(get_O_type(start, end-1, spans)) # O tag
                    O_span_lens.append(rel_len)

        # For FewNERD 10 way 5-10 shot setting. During training, to save the GPU memory, we down sample the O spans.
        if self.train and len(O_spans) > self.max_o_num:
            chose_spans = random.sample(range(0, len(O_spans)), self.max_o_num)
        else:
            chose_spans = list(range(0, len(O_spans)))
        
        downsample_O_spans = []
        downsample_O_span_tags = []
        dowmsample_O_span_lens = []
        for i in chose_spans:
            downsample_O_spans.append(O_spans[i])
            downsample_O_span_tags.append(O_span_tags[i])
            dowmsample_O_span_lens.append(O_span_lens[i])

        # Entity spans can be placed before class O spans because span enhancement, prototype and refinement modules are order-independent.
        return spans + downsample_O_spans, span_tags + downsample_O_span_tags, lens + dowmsample_O_span_lens, gold_entitys
    
    def __populate__(self, samples, savelabeldic=False):
        dataset = {'word': [], 'sentence_num': [], 'spans': [], 'span_tags': [], 'span_lens': [], 'seq_tag_expand': [], 'word_expand': [], 'gold_entitys': []}
        if self.FewNERD:
            words = [x for x in samples['word']]
            seq_tags = [x for x in samples['label']]
        else:
            words = [x for x in samples['seq_ins']]
            seq_tags = [x for x in samples['seq_outs']]
        for word, seq_tag in zip(words, seq_tags):
            word_expand = []
            seq_tag_expand = []
            not_expand = []
            for idx, (w, st) in enumerate(zip(word, seq_tag)):
                word_piece = self.tokenizer.wordpiece_tokenizer.tokenize(w)
                word_expand.extend(word_piece)
                seq_tag_expand.append(st)
                not_expand.append(1)
                for _ in word_piece[1:]:
                    not_expand.append(0)
                    seq_tag_expand.append(st.replace('B-', 'I-'))  # JUST For SNIPS，There does not exist `B-` in FewNERD
            assert len(word_expand) == len(seq_tag_expand)
            word = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + word_expand + ['[SEP]'])

            spans, span_tags, span_lens, gold_entitys = self.__get_spans__(seq_tag_expand, not_expand)
        
            word = torch.tensor(word).long()
            self.__additem__(dataset, word, spans, span_tags, span_lens, seq_tag_expand, word_expand, gold_entitys)
        dataset['sentence_num'] = [len(dataset['word'])]
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
        return dataset



    def __getitem__(self, index):
        index = index % len(self.datas)
        support = self.datas[index]['support']
        if 'query' in self.datas[index].keys():
            query = self.datas[index]['query']
        else:
            query = self.datas[index]['batch']
        if self.FewNERD:
            distinct_tags = ['O1', 'O2', 'O3'] +  self.datas[index]['types']
        else:
            distinct_tags = self.__snips_get_tags__(support['seq_outs'])
        self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
        self.is_support = True
        support_set = self.__populate__(support)
        self.is_support = False
        query_set = self.__populate__(query, savelabeldic=True)
        return support_set, query_set

    def __len__(self):
        return len(self.datas)


def collate_fn(data):
    batch_support = {'word': [], 'sentence_num': [], 'spans': [],
                     'span_tags': [], 'span_lens': [], 'seq_tag_expand': [], 'word_expand': [], 'gold_entitys': []}
    batch_query = {'word': [], 'sentence_num': [], 'label2tag': [],
                   'spans': [], 'span_tags': [], 'span_lens': [], 'seq_tag_expand': [], 'word_expand': [], 'gold_entitys': []}
    support_sets, query_sets = zip(*data)

    for i in range(len(support_sets)):
        for k in batch_support:
            batch_support[k] += support_sets[i][k]
        for k in batch_query:
            batch_query[k] += query_sets[i][k]
    batch_support['word'] = torch.nn.utils.rnn.pad_sequence(batch_support['word'], batch_first=True, padding_value=0)
    batch_query['word'] = torch.nn.utils.rnn.pad_sequence(batch_query['word'], batch_first=True, padding_value=0)
    return batch_support, batch_query


def get_loader(filepath, opt, num_workers=0, shuffle=False):
    dataset = FewDataSet(filepath, opt)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader