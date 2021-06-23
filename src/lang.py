import re

# class Lang:
#     """
#     lass to save the vocab and two dict: the word->index and index->word
#     """
#     def __init__(self):
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = []
#         self.n_words = 0  #
#         self.num_start = 0
#
#     def add_sen_to_vocab(self, sentence): # add words of sentence to vocab
#         for word in sentence:
#             if re.search("N\d+|NUM|\d+", word): # 数字，NUM，N+数字不放进词汇表
#                 continue
#             if word not in self.index2word:
#                 self.word2index[word] = self.n_words
#                 self.word2count[word] = 1
#                 self.index2word.append(word)
#                 self.n_words += 1
#             else:
#                 self.word2count[word] += 1
#
#     def trim(self, min_count): # trim words below a certain count threshold
#         keep_words = []
#
#         for k, v in self.word2count.items():
#             if v >= min_count:
#                 keep_words.append(k)
#
#         print('keep words %s / %s = .4f' % (
#             len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
#         ))
#
#         # Reinitialize dictionaries
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = []
#         self.n_words = 0 # Count default tokens
#
#         for word in keep_words:
#             self.word2index[word] = self.n_words
#             self.index2word.append(word)
#             self.n_words += 1
#
#     # 构建输入字典
#     def build_input_lang(self, trim_min_count): # build the input lang vocab and dict
#         if trim_min_count > 0:
#             self.trim(trim_min_count)
#             self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
#         else:
#             self.index2word = ["PAD", "NUM"] + self.index2word
#
#         self.word2index = {}
#         self.n_words = len(self.index2word)
#         for i, j in enumerate(self.index2word):
#             self.word2index[j] = i
#
#     def build_output_lang(self, generate_num, copy_nums): # build the output lang vocab and dict
#         self.index2word = ['PAD', 'EOS'] + self.index2word + generate_num + ['N' + str(i) for i in range(copy_nums)] + \
#                           ["SOS", "UNK", "SEP"]
#         self.n_words = len(self.index2word)
#         for idx, word in enumerate(self.index2word):
#             self.word2index[word] = idx
#
#     def build_output_lang_for_tree(self, generate_num, copy_nums): # build the output lang vocab and dict
#         self.num_start = len(self.index2word)
#
#         self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
#         self.n_words = len(self.index2word)
#
#         for idx, word in enumerate(self.index2word):
#             self.word2index[word] = idx

class InputLang:
    """
    lass to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  #
        self.num_start = 0

    def add_sen_to_vocab(self, sentence): # add words of sentence to vocab
        for word in sentence:
            # 此规则可能还需要扩充
            if re.search("N\d+|NUM|\d+", word): # 数字，NUM，N+数字不放进词汇表
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count): # trim words below a certain count threshold
        keep_words = []
        for word in self.index2word:
            if self.word2count[word] >= min_count:
                keep_words.append(word)

        # for k, v in self.word2count.items():
        #     if v >= min_count:
        #         keep_words.append(k)

        print('keep words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0 # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    # 构建输入字典
    def build_input_lang(self, trim_min_count): # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word

        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


class OutputLang:
    """
    lass to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = ['+','-','*','/','^','[',']','(',')', '=']  # 使用统一的符号表， 这样可以适配任何数据集
        self.n_words = 0  #
        self.num_start = 0
        self.var_start = 0
        self.ops_list = ['+','-','*','/','^','[',']','(',')', '=']
        self.var2index = {} # for output lang
        self.var2count = {}
        self.index2var = []
        for idx, word in enumerate(self.index2word):
            self.word2index[word] = idx
            self.word2count[word] = 1

    def add_sen_to_vocab(self, sentence): # add words of sentence to vocab
        for word in sentence:
            # 此规则可能还需要扩充
            if re.search("N\d+|NUM|\d+|\d+\.\d+", word): # 数字，NUM，N+数字不放进词汇表
                continue
            # 处理未知数
            if word not in self.ops_list and word != 'SEP':
                if word not in self.index2var:
                    self.index2var.append(word)
                    self.var2count[word] = 1
                else:
                    self.var2count[word] += 1
                continue

            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count=0): # trim words below a certain count threshold
        keep_words = []

        for word in self.index2word:
            if self.word2count[word] >= min_count:
                keep_words.append(word)

        # for k, v in self.word2count.items():
        #     if v >= min_count:
        #         keep_words.append(k)

        print('keep words %s / %s = %.4f' % (
            str(len(keep_words)), str(len(self.index2word)), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0 # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    # 构建输出字典
    def build_output_lang(self, generate_num, copy_nums): # build the output lang vocab and dict
        # 操作符 + 未知数 + 常识数字 + 数字模板
        self.index2word = ['PAD', 'EOS'] + self.index2word
        if len(self.index2var) > 0:
            self.var_start = len(self.index2word)
            self.index2word += self.index2var
        self.index2word += generate_num
        self.num_start = len(self.index2word)
        self.index2word += ['N' + str(i) for i in range(copy_nums)]
        self.index2word += ["SOS", "UNK"]
        if "SEP" not in self.index2word:
            self.index2word =  self.index2word + ["SEP"]
        self.n_words = len(self.index2word)
        for idx, word in enumerate(self.index2word):
            self.word2index[word] = idx

    def build_output_lang_for_tree(self, generate_num, copy_nums): # build the output lang vocab and dict
        # 操作符 + 未知数 + 常识数字 + 数字模板
        if "SEP" not in self.index2word:
            self.index2word = ["SEP"] + self.index2word
        if len(self.index2var) > 0:
            self.var_start = len(self.index2word)
            self.index2word += self.index2var
        self.num_start = len(self.index2word)
        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)]
        self.index2word += ["UNK"]
        self.n_words = len(self.index2word)

        for idx, word in enumerate(self.index2word):
            self.word2index[word] = idx
