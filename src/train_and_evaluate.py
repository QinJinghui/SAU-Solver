from masked_cross_entropy import *
from prepare_data import *
from load_data import *
from num_transfer import *
from expression_tree import *
from calculate import *
from data_utils import *
from models import *
import math
import time
from copy import deepcopy
import numpy as np

import torch
import torch.optim
import torch.nn.functional as f

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var  # decoder input
        self.hidden = hidden
        self.all_output = all_output


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


#########################################################################################
#####                   Rule Mask                                                  ######
#########################################################################################
def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start,
                       copy_nums, generate_nums, var_nums=[]):
    # copy_nums == max_num_list_len
    # generate_nums 数据集中隐含的常量数字，即出现在方程中但是题目中没有出现的数字，在词表的位置列表，value为在lang的位置
    # var_nums 未知变量在词表中的id
    # nums_batch 题目中的数字数量
    # nums_start 数字列表在输出词表的起始位置，其前面为数学运算符和常识数字
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums + 3).fill_(-float("1e12"))

    if decoder_input[0] == word2index["SOS"]:
        for i in range(batch_size):
            res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                  generate_nums + var_nums + [word2index["["], word2index["("]]
            for j in res:
                rule_mask[i, j] = 0
            return rule_mask

    for i in range(batch_size):
        res = []
        if decoder_input[i] == word2index["SEP"] or decoder_input[i] == word2index['=']:
            res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                  generate_nums + var_nums + [word2index["["],word2index["("]]
        elif decoder_input[i] >= nums_start: # N1 ... Nx
            res += [word2index["+"], word2index["-"], word2index["/"],
                    word2index["*"], word2index["EOS"], word2index[')'],
                    word2index["]"],  word2index['^']
                    ]
            if len(var_nums) > 0:
                res += [word2index['='], word2index["SEP"]]
        elif decoder_input[i] in generate_nums:
            res += [word2index["+"], word2index["-"], word2index["/"],
                    word2index["*"], word2index["EOS"], word2index[")"],
                    word2index["]"], word2index['^']
                    ]
            if len(var_nums) > 0:
                res += [word2index['='], word2index["SEP"]]
        elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
            res += [PAD_token]
        elif decoder_input[i] == word2index["("] or decoder_input[i] == word2index["["]:
            res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                   generate_nums + var_nums + [word2index["("]]
        elif decoder_input[i] == word2index[")"]:
            res += [word2index["+"], word2index["-"], word2index["/"],
                    word2index["*"], word2index["EOS"], word2index[")"],
                    word2index["]"], word2index['^']
                    ]
            if len(var_nums) > 0:
                res += [word2index['='], word2index["SEP"]]
        elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
            res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + \
                   generate_nums + var_nums
        elif decoder_input[i] in var_nums:
            res += [word2index[")"], word2index["+"], word2index["-"],
                    word2index["/"], word2index["*"], word2index["EOS"],
                    word2index['^']
                    ]
            if len(var_nums) > 0:
                res += [word2index['='], word2index["SEP"]]
        elif decoder_input[i] == word2index['^']:
            res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums

        for j in res:
            rule_mask[i, j] = 0
        return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, var_nums=[]):
    # copy_nums == max_num_list_len
    # generate_nums 数据集中隐含的常量数字，即出现在方程中但是题目中没有出现的数字，在词表的位置列表，value为在lang的位置
    # var_nums 未知变量在词表中的id
    # nums_batch 题目中的数字数量
    # nums_start 数字列表在输出词表的起始位置，其前面为数学运算符和常识数字
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))

    if len(var_nums) == 0:  # 数据集中不存在变量的情况
        if decoder_input[0] == word2index['SOS']:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
                return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index['SOS']:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                       word2index["="], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
                return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] == word2index["SEP"]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                       word2index["="], word2index["^"]]
            elif decoder_input[i] == word2index['=']:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + var_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums or decoder_input[i] in var_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + var_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"], word2index["SEP"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + var_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]

            for j in res:
                rule_mask[i, j] = 0

    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, var_nums=[]):
    # copy_nums == max_num_list_len
    # generate_nums 数据集中隐含的常量数字，即出现在方程中但是题目中没有出现的数字，在词表的位置列表，value为在lang的位置
    # var_nums 未知变量在词表中的id
    # nums_batch 题目中的数字数量
    # nums_start 数字列表在输出词表的起始位置，其前面为数学运算符和常识数字
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if len(var_nums) == 0:  # 数据集中不存在变量的情况
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index['SOS']:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + var_nums
                for j in res:
                    rule_mask[i, j] = 0
                return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] == word2index["SEP"]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + var_nums
            elif decoder_input[i] == word2index['=']:
                res += [word2index["SEP"], word2index["EOS"]]
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums or decoder_input[i] in var_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + var_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["="]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + var_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],\
                        word2index["="], word2index["EOS"], word2index["SEP"]]

            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


#########################################################################################
#####                   Input and Output Utils                                     ######
#########################################################################################
def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    # nums_stack记录的是等式中的单词不在outlang中，但在数字列表中的数字
    # 或等式中的单词不在outlang中，且不在数字列表中的特殊数字
    target_input = copy.deepcopy(target) # 用于生成只有操作符的骨架
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    # 遍历目标序列
    for i in range(len(target)):
        if target[i] == unk:
            # 为unk的elem从nums list中选择正确的模板数字
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        # 如果elem为数字模板或unk，则将其置位为0
        if target_input[i] >= num_start:
            target_input[i] = 0
    # 替换了unk符的方程等式，只有操作符的等式骨架
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    # 替换了unk符的方程等式
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos, batch_first=False):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end  # 判断decoder_input是否为数字, 数字为1，非数字为0
    num_mask_encoder = num_mask < 1  # 非数字为1，数字为0
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size  # B x embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    if batch_first:
        all_embedding = encoder_outputs.contiguous()
    else:
        all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start # 用于计算num pos
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):   # 将数字替换为数字在句子中的具体位置，非数字的为0
        indices[k] = num_pos[k][indices[k]]  # 非数字的为0也会提取到第一个数字的位置，这样不会和真正需要第一个位置的冲突？
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)  # decoder_input B x 1
    if batch_first:
        sen_len = encoder_outputs.size(1)
    else:
        sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len  # 记录每个batch在all embedding上的位置
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices # 记录数字在encoder的位置，即文本的位置
    num_encoder = all_embedding.index_select(0, indices) # 提取题目中数字的embedding
    return num_mask, num_encoder, num_mask_encoder


# 提取与问题相关的数字embedding
def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size, batch_first=False):
    indices = list()
    if batch_first:
        sen_len = encoder_outputs.size(1)
    else:
        sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)  # 用于记录数字在问题的位置
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]  # 屏蔽多余的数字，即词表中不属于该题目的数字
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index) # B x num_size x H
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    if batch_first:
        all_outputs = encoder_outputs.contiguous() # B x S x H
    else:
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()  # S x B x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H or B x S x H -> (S x B) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)  # 屏蔽其他无关数字


#########################################################################################
#####                   Seq2Seq Train and Evaluation                               ######
#########################################################################################
def train_seq2seq(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
                                generate_nums, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, clip=0,
                                use_teacher_forcing=1, beam_size=1, var_nums=[], beam_search=True, batch_first=False):
    # 构建序列掩码，需要的位置置为0，pad置为1，因为masked_fill_需要的格式就是和逻辑反过来的
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 3
    unk = output_lang.word2index["UNK"]
    # Trun padded arrays into (batch_size x max_len) tensors, transpos into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.classes_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if batch_first:
                decoder_input = decoder_input.unsqueeze(1)
            else:
                decoder_input = decoder_input.unsqueeze(0)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk) # 替换当前位置的unk为出现在题目中的常识数字
            target[t] = decoder_input
            # decoder_input = decoder_input.unsqueeze(0)
    elif beam_search:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.classes_size * beam_len) # 用于记录从开始到当前step的score
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.classes_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                                               num_start, copy_nums, generate_nums, var_nums=var_nums)
                if batch_first:
                    decoder_input = decoder_input.unsqueeze(1)
                else:
                    decoder_input = decoder_input.unsqueeze(0)
                if USE_CUDA:
                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output,dim=1) + rule_mask  # B x classes_size
                beam_score = beam_list[b_idx].score # B x classes_size * beam_len
                beam_score = beam_score.unsqueeze(1) # B x 1 x classes_size * beam_len
                repeat_dims = [1] * beam_score.dim() # B x 1 x classes_size * beam_len
                repeat_dims[1] = score.size(1) # B x classes_size x classes_size * beam_len
                beam_score = beam_score.repeat(*repeat_dims)
                # B x classes_size + B x classes_size x classes_size * beam_len = B x classes_size x classes_size * beam_len
                score += beam_score
                beam_scores[:, b_idx * decoder.classes_size: (b_idx + 1) * decoder.classes_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output # 为每个beam添加当前的output，那么beam_list[0-k]包含所有需要的输出
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output

            topv, topi = beam_scores.topk(beam_size, dim=1)  # 对历史所有的beam score进行选取topk
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.classes_size  # 确定当前合适的选择
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.classes_size # 定位第几个beam
                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)
                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        # 将目标等式转换为正确的等式，即将unk替换为在等式出现的常识数字
        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    else:
        # greedy search
        for t in range(max_target_length):
            rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                                           num_start, copy_nums, generate_nums, var_nums=var_nums)
            if batch_first:
                decoder_input = decoder_input.unsqueeze(1)
            else:
                decoder_input = decoder_input.unsqueeze(0)

            if USE_CUDA:
                rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            all_decoder_outputs[t] = decoder_output
            decoder_input = torch.argmax(decoder_output, dim=1)
            target[t] = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)

    # Loss calculation and backpropagation
    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy_with_logit(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    if clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return return_loss


def evaluate_seq2seq(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                     beam_size=1, var_nums=[], beam_search=True, max_length=MAX_OUTPUT_LENGTH, batch_first=False):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 3

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)

    # input_var = torch.LongTensor(input_var).transpose(0, 1)
    batch_size = input_var.size(1)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    if beam_search:
        beam_list = list()
        score = 0
        beam_list.append(Beam(score, decoder_input, decoder_hidden, []))
        # Run through decoder
        for di in range(max_length):
            temp_list = list()
            beam_len = len(beam_list)
            for xb in beam_list:
                if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                    temp_list.append(xb)
                    beam_len -= 1
            if beam_len == 0:
                return beam_list[0].all_output
            beam_scores = torch.zeros(decoder.classes_size * beam_len)
            hidden_size_0 = decoder_hidden.size(0)
            hidden_size_2 = decoder_hidden.size(2)
            all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
            all_outputs = []
            current_idx = -1

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                    continue
                current_idx += 1
                decoder_hidden = beam_list[b_idx].hidden

                # if batch_size == 1:
                #     rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index, batch_size,
                #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
                # else:
                #     rule_mask = generate_rule_mask(decoder_input, num_list, output_lang.word2index, batch_size,
                #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
                if USE_CUDA:
                    # rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                if batch_first:
                    decoder_input = decoder_input.unsqueeze(1)
                else:
                    decoder_input = decoder_input.unsqueeze(0)

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
                score = f.log_softmax(decoder_output, dim=1)
                score += beam_list[b_idx].score
                beam_scores[current_idx * decoder.classes_size: (current_idx + 1) * decoder.classes_size] = score
                all_hidden[current_idx] = decoder_hidden
                all_outputs.append(beam_list[b_idx].all_output)
            topv, topi = beam_scores.topk(beam_size)

            for k in range(beam_size):
                word_n = int(topi[k])
                word_input = word_n % decoder.classes_size
                temp_input = torch.LongTensor([word_input])
                indices = int(word_n / decoder.classes_size)

                temp_hidden = all_hidden[indices]
                temp_output = all_outputs[indices]+[word_input]
                temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

            temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

            if len(temp_list) < beam_size:
                beam_list = temp_list
            else:
                beam_list = temp_list[:beam_size]
        all_outputs = beam_list[0].all_output
    else:
        all_outputs = []
        for di in range(max_length):
            # if batch_size == 1:
            #     rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            # else:
            #     rule_mask = generate_rule_mask(decoder_input, num_list, output_lang.word2index, batch_size,
            #                                    num_start, copy_nums, generate_nums, var_nums=var_nums)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()
            if batch_first:
                decoder_input = decoder_input.unsqueeze(1)
            else:
                decoder_input = decoder_input.unsqueeze(0)

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # decoder_output = f.log_softmax(decoder_output, dim=1) + rule_mask  # B x classes_size
            decoder_output = f.log_softmax(decoder_output, dim=1)
            all_outputs.append(decoder_output)
            decoder_input = torch.argmax(decoder_output, dim=1)
        all_outputs = torch.stack(all_outputs, dim=1)  # B x S x N
        all_outputs = all_outputs.cpu().numpy()[0]

    return all_outputs

#########################################################################################
#####                   Seq2Tree Train and Evaluation                              ######
#########################################################################################
def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos, var_nums=[]):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums) + len(var_nums) # 最大的位置列表数目+常识数字数目+未知数列表
    for i in num_size_batch:
        d = i + len(generate_nums) + len(var_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask) # 用于屏蔽无关数字，防止生成错误的Nx

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)] # root embedding B x 1

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    # 提取与问题相关的数字embedding
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start - len(var_nums)
    embeddings_stacks = [[] for _ in range(batch_size)] # B x 1  当前的tree state/ subtree embedding / output
    left_childs = [None for _ in range(batch_size)] # B x 1

    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)
        # if 89 in target[t].tolist():
        #     print("Hello")
        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            # 未知数当数字处理，SEP当操作符处理
            if i < num_start:  # 非数字
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))
            else:  # 数字
                # if i - num_start >= current_nums_embeddings.size(1) or i == len(output_lang.index2word) - 1:
                #     print("Hello")
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)  # Subtree embedding
                o.append(TreeEmbedding(current_num, terminal=True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous() # B x S

    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy_with_logit(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def train_tree_with_subtree_semantic_alignment(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
                                           encoder, predict, generate, merge, semantic_alignment, encoder_optimizer, predict_optimizer, generate_optimizer,
                                           merge_optimizer, semantic_alignment_optimizer, output_lang, num_pos, var_nums=[], batch_first=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums) + len(var_nums) # 最大的位置列表数目+常识数字数目+未知数列表
    for i in num_size_batch:
        d = i + len(generate_nums) + len(var_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask) # 用于屏蔽无关数字，防止生成错误的Nx

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    semantic_alignment.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    semantic_alignment_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)] # root embedding B x 1

    max_target_length = max(target_length)

    all_node_outputs = []
    all_sa_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    # 提取与问题相关的数字embedding
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start - len(var_nums)
    embeddings_stacks = [[] for _ in range(batch_size)] # B x 1  当前的tree state/ subtree embedding / output
    left_childs = [None for _ in range(batch_size)] # B x 1

    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            # 未知数当数字处理，SEP当操作符处理
            if i < num_start:  # 非数字
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))
                # print(o[-1].embedding.size())
                # print(encoder_outputs[idx].size())
            else:  # 数字
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)  # Subtree embedding
                    if batch_first:
                        encoder_mapping, decoder_mapping = semantic_alignment(current_num, encoder_outputs[idx])
                    else:
                        temp_encoder_outputs = encoder_outputs.transpose(0,1)
                        encoder_mapping, decoder_mapping = semantic_alignment(current_num, temp_encoder_outputs[idx])
                    all_sa_outputs.append((encoder_mapping, decoder_mapping))
                o.append(TreeEmbedding(current_num, terminal=True))

            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)

            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous() # B x S

    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()
        new_all_sa_outputs = []
        for sa_pair in all_sa_outputs:
            new_all_sa_outputs.append((sa_pair[0].cuda(),sa_pair[1].cuda()))
        all_sa_outputs = new_all_sa_outputs

    semantic_alignment_loss = nn.MSELoss()
    total_semanti_alognment_loss = 0
    sa_len = len(all_sa_outputs)
    for sa_pair in all_sa_outputs:
        total_semanti_alognment_loss += semantic_alignment_loss(sa_pair[0],sa_pair[1])
    # print(total_semanti_alognment_loss)
    total_semanti_alognment_loss = total_semanti_alognment_loss / sa_len
    # print(total_semanti_alognment_loss)

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy_with_logit(all_node_outputs, target, target_length) + 0.01 * total_semanti_alognment_loss
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    semantic_alignment_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()

def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,
                  beam_size=5, var_nums=[], beam_search=True, max_length=MAX_OUTPUT_LENGTH):
    # sequence mask for attention
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)+ len(var_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Run words through encoder
    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables  # # root embedding B x 1
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    # 提取与问题相关的数字embedding
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start - len(var_nums)
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    if beam_search:
        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)

                # leaf = p_leaf[:, 0].unsqueeze(1)
                # repeat_dims = [1] * leaf.dim()
                # repeat_dims[1] = op.size(1)
                # leaf = leaf.repeat(*repeat_dims)
                #
                # non_leaf = p_leaf[:, 1].unsqueeze(1)
                # repeat_dims = [1] * non_leaf.dim()
                # repeat_dims[1] = num_score.size(1)
                # non_leaf = non_leaf.repeat(*repeat_dims)
                #
                # p_leaf = torch.cat((leaf, non_leaf), dim=1)
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                # is_leaf = int(topi[0])
                # if is_leaf:
                #     topv, topi = op.topk(1)
                #     out_token = int(topi[0])
                # else:
                #     topv, topi = num_score.topk(1)
                #     out_token = int(topi[0]) + num_start
                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)
                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    # var_num当时数字处理，SEP/;当操作符处理
                    if out_token < num_start: # 非数字
                        generate_input = torch.LongTensor([out_token])
                        if USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:  # 数字
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                  current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        return beams[0].out
    else:
        all_node_outputs = []
        for t in range(max_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_scores = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
            out_tokens = torch.argmax(out_scores, dim=1) # B
            all_node_outputs.append(out_tokens)
            left_childs = []
            for idx, node_stack, out_token, embeddings_stack in zip(range(batch_size), node_stacks, out_tokens, embeddings_stacks):
                # node = node_stack.pop()
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue
                # var_num当时数字处理，SEP/;当操作符处理
                if out_token < num_start: # 非数字
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
                    node_stack.append(TreeNode(right_child))
                    node_stack.append(TreeNode(left_child, left_flag=True))
                    embeddings_stack.append(TreeEmbedding(node_label.unsqueeze(0), False))
                else: # 数字
                    current_num = current_nums_embeddings[idx, out_token - num_start].unsqueeze(0)
                    while len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
                        sub_stree = embeddings_stack.pop()
                        op = embeddings_stack.pop()
                        current_num = merge(op.embedding.squeeze(0), sub_stree.embedding, current_num)
                    embeddings_stack.append(TreeEmbedding(current_num, terminal=True))

                if len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
                    left_childs.append(embeddings_stack[-1].embedding)
                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
        all_node_outputs = all_node_outputs.cpu().numpy()
        return all_node_outputs[0]
