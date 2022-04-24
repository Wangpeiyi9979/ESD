import sys

sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer
from model.utils import MultiHeadedAttentionWithFNN, fast_att, sequence_mask, Swish

BertHiddenSize=768

class ESD(nn.Module):
    """
    Span Proto Attention Network
    """
    def __init__(self,# not use
                 opt):
        nn.Module.__init__(self)
        self.opt = opt
        self.model_name = 'ESD'

        self.drop = nn.Dropout()
        self.fusion_linear = nn.Sequential(
            nn.Linear(BertHiddenSize * 2, BertHiddenSize),
            nn.GELU(),
            nn.Dropout(opt.dropout),
            nn.Linear(BertHiddenSize, opt.hidsize),
        )

        self.tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
        self.word_encoder = BertModel.from_pretrained(opt.bert_path)
        self.inter_attentioner = MultiHeadedAttentionWithFNN(embed_dim=opt.hidsize, num_heads=opt.num_heads, dropout=opt.dropout, activation=opt.activation)
        self.cross_attentioner = MultiHeadedAttentionWithFNN(embed_dim=opt.hidsize, num_heads=opt.num_heads, dropout=opt.dropout, activation=opt.activation)
     

    def __dist__(self, x, y, dim):
        """
        x:proto: query_num x proto_num x hidsize
        y:query_span: query_num x proto_num x hidsize
        """
        return -(torch.pow(x - y, 2)).sum(dim)
    
    def get_span_rep(self, embedding, spans, span_tags, span_lens):
        all_span_rep = []
        is_padding = []
        all_span_tag = []
        span_num = [len(span) for span in spans]
        max_span_num = max(span_num)
        for emb, span, span_tag, span_len in zip(embedding, spans, span_tags, span_lens):
            span_left = [x[0] + 1  for x in span] # we need to add 1 since [CLS]
            span_right = [x[1] + 1  for x in span]
            span_left = torch.tensor(span_left).long().to(embedding.device)
            span_right = torch.tensor(span_right).long().to(embedding.device)
            span_left_rep = emb[span_left]
            span_right_rep = emb[span_right]

            
            span_rep = self.fusion_linear(torch.cat([span_left_rep, span_right_rep], -1))  # span_num x 768
            is_padding.extend([1] * len(span_rep))
            cat_rep = torch.zeros(max_span_num - len(span), span_rep.size(-1)).to(embedding.device)
            is_padding.extend([0] * len(cat_rep))
            span_rep = torch.cat([span_rep, cat_rep], 0)  # max_span_num x hidsize
            all_span_rep.append(span_rep)
            all_span_tag.extend(span_tag)

        all_span_rep = torch.stack(all_span_rep, 0)  # sentence_num x span_num x hiddensize
        assert all_span_rep.size(0) * all_span_rep.size(1) == len(is_padding)
        return all_span_rep, \
               torch.tensor(span_num).long().to(all_span_rep.device), \
               torch.tensor(is_padding).to(all_span_rep.device), \
               torch.tensor(all_span_tag).long().to(all_span_rep.device)

  
    def forward(self, support, query, query_label=None):
        support_out = self.word_encoder(support['word'], output_hidden_states=True, return_dict=True)  # [num_sent, number_of_tokens, 768]
        query_out = self.word_encoder(query['word'], output_hidden_states=True, return_dict=True)  # [num_sent, number_of_tokens, 768]
        support_emb = support_out['last_hidden_state']
        query_emb = query_out['last_hidden_state']
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        logits = []
        current_support_num = 0
        current_query_num = 0
        loss = 0
        for sent_support_num, sent_query_num in zip(support['sentence_num'],  query['sentence_num']):
            logits4episode = []
            batch_support_emb = support_emb[current_support_num:current_support_num + sent_support_num]
            batch_query_emb = query_emb[current_query_num:current_query_num + sent_query_num]
             # -------------------- Span Initialization Module (START) -------------------------
            support_span_rep, support_span_nums, support_is_padding, support_all_span_tags = self.get_span_rep(
                batch_support_emb, support['spans'][current_support_num: current_support_num + sent_support_num],
                support['span_tags'][current_support_num: current_support_num + sent_support_num],
                support['span_lens'][current_support_num: current_support_num + sent_support_num]

            )

            query_span_rep, query_span_nums, query_is_padding, _ = self.get_span_rep(
                batch_query_emb, query['spans'][current_query_num: current_query_num + sent_query_num],
                query['span_tags'][current_query_num: current_query_num + sent_query_num],
                query['span_lens'][current_query_num: current_query_num + sent_query_num]
            )
            # -------------------- Span Initialization Module (END) -------------------------
            _, _, hidsize = support_span_rep.size()
            support_span_mask = sequence_mask(support_span_nums)
            query_span_mask = sequence_mask(query_span_nums)

            # -------------------- Span Enhancing Module (START) -------------------------
            # ISA -- for spans in the same sentence
            support_span_rep = self.inter_attentioner(support_span_rep, support_span_rep, support_span_rep, key_padding_mask=support_span_mask)  #
            query_span_rep = self.inter_attentioner(query_span_rep, query_span_rep, query_span_rep, key_padding_mask=query_span_mask)

            support_span_rep = support_span_rep.view(-1, hidsize)[support_is_padding != 0]  # support_span_num x hidden_size
            query_span_rep = query_span_rep.view(-1, hidsize)[query_is_padding != 0]  # query_span_num x hidden_size

            # CSA -- for spans between support set and query
            cur_q_span = 0
            all_support_span_enhance = [] # query_sent x [support_span_num x hidsize]
            all_query_span_enhance = [] # query_sent x [one_query_span_num x hidsize]
            for q_num in query_span_nums.tolist():
                one_query_spans_squeeze = query_span_rep[cur_q_span: cur_q_span + q_num]
                cur_q_span += q_num
                # suuport_span_num x hidden_size
                support_span_enhance4one_query = self.cross_attentioner(support_span_rep.unsqueeze(0),
                                                                one_query_spans_squeeze.unsqueeze(0),
                                                                one_query_spans_squeeze.unsqueeze(0)).squeeze(0)
                # one_query_span_num x hidden_size
                query_span_enhance_rep = self.cross_attentioner(one_query_spans_squeeze.unsqueeze(0),
                                                            support_span_rep.unsqueeze(0),
                                                            support_span_rep.unsqueeze(0)).squeeze(0)

                all_query_span_enhance.append(query_span_enhance_rep)
                all_support_span_enhance.append(support_span_enhance4one_query)  # [one_query_span_num x hidden_size]
            # -------------------- Span Enhancing Module (END) -------------------------


            max_tags = torch.max(support_all_span_tags).item() + 1
            
            # -------------------- Span ProtoTypical Module (START) -------------------------
            for support_span_enhance_rep, query_span_enhance_rep in zip(all_support_span_enhance, all_query_span_enhance):
                # support_span_num x hidden_size, one_query_span_num x hidden_size
                start_id = 0
                proto_for_each_query = []
                for label in range(start_id, max_tags):
                    class_rep = support_span_enhance_rep[support_all_span_tags == label, :]  # class_span_num x hidden_size
                    # INSA
                    proto_rep = fast_att(query_span_enhance_rep, class_rep, activation=self.opt.activation)  # one_query_span_num x hidden_size
                    proto_for_each_query.append(proto_rep.unsqueeze(0))
                    
                proto_for_each_query = torch.cat(proto_for_each_query, 0).permute(1, 0, 2)  # one_query_span_num x num_class x hidden_size
                O_reps = proto_for_each_query[:, :self.opt.O_class_num, :] # one_query_span_num x num_class x hidden_size
                # PSA
                O_rep = fast_att(query_span_enhance_rep, O_reps, activation=self.opt.activation) # one_query_span_num x hidden_size
                proto_for_each_query = torch.cat([O_rep.unsqueeze(1), proto_for_each_query[:, self.opt.O_class_num:, :]], dim=1)
            # -------------------- Span ProtoTypical Module (END) -------------------------

            # -------------------- Span Matching Module (START) -------------------------
                N = proto_for_each_query.size()[1]
                one_query_span_score = self.__dist__(proto_for_each_query, query_span_enhance_rep.unsqueeze(1).expand(-1, N, -1), -1)  # one_query_span_num x num_class
                logits4episode.append(one_query_span_score)
            # -------------------- Span Matching Module (END) -------------------------

            logits4episode = torch.cat(logits4episode, dim=0)
            if query_label is not None:
                label4episode = query_label[current_query_num: current_query_num + sent_query_num]
                label4episode = torch.cat(label4episode, 0)
                N = logits4episode.size(-1)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                loss += loss_fct(logits4episode.view(-1, N), label4episode.view(-1))

            logits.append(logits4episode)
            current_query_num += sent_query_num
            current_support_num += sent_support_num



        pred = []
        for label4episode in logits:
            _, pred4episode = torch.max(label4episode, 1)
            pred.append(pred4episode)

        if query_label is None:
            assert len(pred) == 1 and len(logits) == 1
            
        return {
            'pred': pred[0],
            'logits': logits[0],
            'loss': loss
        }

