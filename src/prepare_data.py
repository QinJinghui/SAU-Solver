import random
import json
import copy
import re
import nltk
from data_utils import remove_brackets
from lang import InputLang, OutputLang
from data_utils import indexes_from_sentence, pad_seq, check_bracket, get_num_stack
from operation_law import exchange, allocation


# pairs: (input_seq, eq_segs, nums, num_pos, ans)
def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = InputLang()
    output_lang = OutputLang()
    train_pairs = []
    test_pairs = []

    print("Indexing words")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif len(pair) == 4 and pair[-1]:  # num_pos 题目与方程存在对应的数字
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif len(pair) == 6 and pair[-2]: # num_pos 题目与方程存在对应的数字
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []  # 用于记录不在输出词典的数字
        for word in pair[1]:
            temp_num = []
            flag_not = True  # 用检查等式是否存在不在字典的元素
            if word not in output_lang.index2word:  # 如果该元素不在输出字典里
                flag_not = False
                for i, j in enumerate(pair[2]): # 遍历nums, 看是否存在
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])  # 生成从0到等式长度的数字

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # print(pair[1])
        # print(output_cell)
        if len(pair) == 4:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
            train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                                pair[2], pair[3], num_stack))
        else:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
            train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                                pair[2], pair[3], pair[4], num_stack, pair[5]))

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:  # out_seq
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word: # 非符号，即word为数字
                flag_not = False
                for i, j in enumerate(pair[2]): # nums
                    if j == word:
                        temp_num.append(i) # 在等式的位置信息

            if not flag_not and len(temp_num) != 0:# 数字在数字列表中
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                # 数字不在数字列表中，则生成数字列表长度的位置信息，
                # 生成时根据解码器的概率选一个， 参见generate_tree_input
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        if len(pair) == 4:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                              pair[2], pair[3], num_stack))
        else:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                                pair[2], pair[3], pair[4], num_stack,pair[5]))

    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


def prepare_de_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = InputLang()
    output_lang = OutputLang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        input_lang.add_sen_to_vocab(pair[0])
        output_lang.add_sen_to_vocab(pair[1])

    input_lang.build_input_lang(trim_min_count)

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        if len(pair) == 4:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
            train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], num_stack])
        else:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
            train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], pair[4], num_stack])
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        if len(pair) == 4:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                               pair[2], pair[3], num_stack))
        else:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
            train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                                pair[2], pair[3], pair[4], num_stack))

    print('Number of testing data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []
    ans_flag = False if len(pairs[0]) == 7 else True
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
        if not ans_flag:
            for _, i, _, j, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)
        else:
            for _, i, _, j, _, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)

        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        if not ans_flag:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, num_stack in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
        else:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, ans, num_stack in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                ans_batch.append(ans)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        if ans_flag:
            ans_batches.append(ans_batch)
    if not ans_flag:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches,\
               num_pos_batches, num_size_batches
    else:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
               num_pos_batches, num_size_batches, ans_batches

# prepare the batches
def prepare_test_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []
    ans_flag = False if len(pairs[0]) == 7 else True
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
        if not ans_flag:
            for _, i, _, j, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)
        else:
            for _, i, _, j, _, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)

        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        if not ans_flag:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, num_stack in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
        else:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, ans, num_stack in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                ans_batch.append(ans)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        if ans_flag:
            ans_batches.append(ans_batch)
    if not ans_flag:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
               num_pos_batches, num_size_batches
    else:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
               num_pos_batches, num_size_batches, ans_batches


def prepare_de_train_batch(pairs_to_batch, batch_size, output_lang, rate, english=False):
    # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
    # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
    pairs = []
    b_pairs = copy.deepcopy(pairs_to_batch)
    for pair in b_pairs:
        p = copy.deepcopy(pair)
        pair[2] = check_bracket(pair[2], english)

        temp_out = exchange(pair[2], rate)
        temp_out = check_bracket(temp_out, english)

        p[2] = indexes_from_sentence(output_lang, pair[2])
        p[3] = len(p[2])
        pairs.append(p)

        temp_out_a = allocation(pair[2], rate)
        temp_out_a = check_bracket(temp_out_a, english)

        if temp_out_a != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out_a, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out_a)
            p[3] = len(p[2])
            pairs.append(p)

        if temp_out != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out)
            p[3] = len(p[2])
            pairs.append(p)

            if temp_out_a != pair[2]:
                p = copy.deepcopy(pair)
                temp_out_a = allocation(temp_out, rate)
                temp_out_a = check_bracket(temp_out_a, english)
                if temp_out_a != temp_out:
                    p[6] = get_num_stack(temp_out_a, output_lang, p[4])
                    p[2] = indexes_from_sentence(output_lang, temp_out_a)
                    p[3] = len(p[2])
                    pairs.append(p)
    print("this epoch training data is", len(pairs))
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    ans_batches = []
    ans_flag = False if len(pairs[0]) == 7 else True
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        if not ans_flag:
            for _, i, _, j, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)
        else:
            for _, i, _, j, _, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        ans_batch = []
        if not ans_flag:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, num_stack in batch:
                num_batch.append(len(num))
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
        else:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, ans, num_stack in batch:
                num_batch.append(len(num))
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                ans_batch.append(ans)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        if ans_flag:
            ans_batches.append(ans_batch)
    if not ans_flag:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches
    else:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, ans_batches


