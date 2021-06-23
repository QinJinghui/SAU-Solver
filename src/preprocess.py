import json
import nltk
import jieba
import re
from data_utils import replace_en_number_with_digits
from data_utils import chs_arabic_map
from data_utils import convert_chinese_digits_to_arabic


def is_number(word):
    # 判断括号内的字符是否为数字
    if word[0] == '(' and word[-1] == ')':
        for elem_char in word:
            if elem_char.isdigit():
                return True
        return False

    # 判断word中是否存在数字
    if '(' in word and ')' in word and '/' in word and not word[-1].isdigit():
        for elem_char in word:
            if elem_char.isdigit():
                return True
        return False

    # 如果最后一个字母是%，且长度大于1则认为是数字
    if word[-1] == '%' and len(word) > 1:
        return True
    if word[0].isdigit():
        return True
    if word[-1].isdigit():
        return True
    try:
        float(word)
        return True
    except:
        return False

def is_chinese_number(word):
    if word[0] == '(' and word[-1] == ')':
        for elem_char in word:
            if elem_char in chs_arabic_map.keys():
                return True
        return False

        # 判断word中是否存在数字
    if '(' in word and ')' in word and '/' in word and word[-1] not in chs_arabic_map.keys():
        for elem_char in word:
            if elem_char in chs_arabic_map.keys():
                return True
        return False

    # 如果最后一个字母是%，且长度大于1则认为是数字
    # if word[-1] == '%' and len(word) > 1:
    #     return True
    if word[0] in chs_arabic_map.keys():
        return True
    if word[-1] in chs_arabic_map.keys():
        return True
    try:
        convert_chinese_digits_to_arabic(word)
        return True
    except:
        return False


def split_num_and_unit(word):
    # 划分数字和单位，主要通过字母级别的判断
    num = ''
    unit = ''
    for idx in range(len(word)):
        char = word[idx]
        if char.isdigit() or char in ['.', '/', '(', ')']:
            num += char
        else:
            unit += char
    return num, unit


def split_chinese_num_and_unit(word):
    num = ''
    unit = ''
    for idx in range(len(word)):
        char = word[idx]
        if char in chs_arabic_map.keys() or char in ['.', '/', '(', ')']:
            num += char
        else:
            unit += char
    return num, unit


def equation_process(equation_str_list):
    alphabet = "abcdefghijklmnopqrstuvwxyz_"
    alphabet_array = []
    for i in range(len(alphabet)):
        alphabet_array.append(alphabet[i])
    operation = "+-*/^()[]="
    operation_array = []
    for i in range(len(operation)):
        operation_array.append(operation[i])

    unknown_variable = set()
    equation_word_list = []
    for equation_str in equation_str_list:
        word_list = []
        equation_len = len(equation_str)
        equation_str_lower = equation_str.lower()
        i = 0
        while i < equation_len:
            if equation_str_lower[i] in alphabet_array:
                j = i
                while j < equation_len:
                    if equation_str_lower[j].isdigit() and j + 1 < equation_len and equation_str_lower[j+1] in alphabet:
                        j = j + 1
                        continue
                    if not equation_str_lower[j].isdigit() and equation_str_lower[j] not in alphabet_array:
                        break
                    j = j + 1
                word_list.append(equation_str[i:j])
                unknown_variable.add(equation_str[i:j])
                i = j
            elif equation_str_lower[i] in operation_array:
                word_list.append(equation_str[i])
                i = i + 1
            elif equation_str_lower[i].isdigit() or equation_str_lower[i] == '.':
                j = i
                while j < equation_len:
                    if not equation_str_lower[j].isdigit() and equation_str_lower[j] != '.':
                        break
                    j = j + 1

                word_list.append(str(float(equation_str[i:j])))
                i = j
            elif equation_str[i] == ' ':
                i = i + 1
                continue
            else:
                word_list.append(equation_str[i])
                i = i + 1
        print(word_list)
        # equation_word_list.append(word_list)

        # 处理负数的情况
        new_word_list = []
        add_negative = False
        for idx, word in enumerate(word_list):
            if word == '-':
                if idx > 0:
                    if word_list[idx - 1] in ['+','-', '*', '/', '=', '^', '(']:
                        add_negative = True
                        continue
            if add_negative:
                new_word_list.append('0 - '+word)
                add_negative = False
            else:
                new_word_list.append(word)

        equation_word_list.append(new_word_list)
    print(equation_word_list)

    new_equation_str_list = []
    for word_list in equation_word_list:
        new_equation_str_list.append(' '.join(word_list))
    print(new_equation_str_list)

    unknown_variable = list(unknown_variable)
    if len(unknown_variable) == 0:
        unknown_variable_dict = {}
        new_equation_str = new_equation_str_list[0]
    elif len(unknown_variable) == 1:
        unknown_variable_dict = {}
        unknown_variable_dict[unknown_variable[0]] = 'x'
        new_equation_str = new_equation_str_list[0].replace(unknown_variable[0],'x')
    elif len(unknown_variable) == 2:
        unknown_variable_dict = {}
        if unknown_variable[0] in ['x','y'] and unknown_variable[1] in ['x','y']:
            unknown_variable_dict[unknown_variable[0]] = unknown_variable[0]
            unknown_variable_dict[unknown_variable[1]] = unknown_variable[1]
        else:
            unknown_variable_dict[unknown_variable[0]] = 'x'
            unknown_variable_dict[unknown_variable[1]] = 'y'
        if unknown_variable[0].find(unknown_variable[1]) >= 0 or unknown_variable[1].find(unknown_variable[0]) >= 0:
            if len(unknown_variable[0]) > len(unknown_variable[1]):
                new_equation_str = new_equation_str_list[0].replace(unknown_variable[0],'x').replace(unknown_variable[1],'y') + \
                                   ' ; ' + new_equation_str_list[1].replace(unknown_variable[0],'x').replace(unknown_variable[1],'y')
            else:
                new_equation_str = new_equation_str_list[0].replace(unknown_variable[1],'y').replace(unknown_variable[0],'x') + \
                                   ' ; ' + new_equation_str_list[1].replace(unknown_variable[1],'y').replace(unknown_variable[0],'x')
        else:
            new_equation_str = new_equation_str_list[0].replace(unknown_variable[0],unknown_variable_dict[unknown_variable[0]]).replace(unknown_variable[1],unknown_variable_dict[unknown_variable[1]]) + \
                               ' ; ' + new_equation_str_list[1].replace(unknown_variable[0],unknown_variable_dict[unknown_variable[0]]).replace(unknown_variable[1],unknown_variable_dict[unknown_variable[1]])
    else:
        unknown_variable = sorted(unknown_variable,reverse=True)
        unknown_variable_dict = {}
        unknown_variable_dict[unknown_variable[0]] = 'z'
        unknown_variable_dict[unknown_variable[1]] = 'y'
        unknown_variable_dict[unknown_variable[2]] = 'x'
        new_equation_str = new_equation_str_list[0].replace(unknown_variable[0],'z').replace(unknown_variable[1],'y').replace(unknown_variable[2],'x') + \
                           ' ; ' + new_equation_str_list[1].replace(unknown_variable[0],'z').replace(unknown_variable[1],'y').replace(unknown_variable[2],'x') + \
                           ' ; ' + new_equation_str_list[2].replace(unknown_variable[0],'z').replace(unknown_variable[1],'y').replace(unknown_variable[2],'x')

    new_equation_str = new_equation_str.replace('[', ')')
    new_equation_str = new_equation_str.replace(']', ')')

    return new_equation_str, unknown_variable_dict


def preprocess_alg514(input_filename, output_filename):
    # 问题分词
    # 合并数字
    # 英文单词数字转换为数学数字
    # 方程变量处理
    print("Reading lines...")
    with open(input_filename, 'r', encoding="utf-8") as f:
        data_list = json.load(f)

    # 问题文本分词
    for d in data_list:
        # 先分句再分词
        sents = nltk.sent_tokenize(d['sQuestion'])
        words = []
        for sent in sents:
            words.extend(nltk.word_tokenize(sent.lower()))
        d['sQuestion'] = ' '.join(words)

    # 合并数字
    for d in data_list:
        # 对分词后的问题文本中的数字进行合并处理，如'(' '1' '/' '5' ')' -> '(1/5)'，形成新的分词文本
        new_words = []
        question = d['sQuestion'].split()
        question_len = len(question)
        i = 0
        while i < len(question):
            if question[i] == '(' and i + 4 < question_len and question[i+4] == ')':
                sub = ''.join(question[i:i+5])
                new_words.append(sub)
                i = i + 5
            elif i + 1 < question_len and is_number(question[i]) and question[i+1] == '%':
                sub = ''.join(question[i:i+2])
                new_words.append(sub)
                i = i + 2
            else:
                new_words.append(question[i])
                i = i + 1
        d['sQuestion'] = ' '.join(new_words)

    # 英文单词数字转换为数学数字
    en_arabic_map = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                     'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen':13, 'fourteen': 14,
                     'fifteen': 15, 'sixteen': 16, 'seveteen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
                     'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
                     'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000,
                     'twice': 2, 'thrice': 3, 'triple': 3,'double':2, 'half':0.5}
    for d in data_list:
        question = d['sQuestion']
        word_list = question.strip().split()
        new_word_list = []
        for word in word_list:
            if word in en_arabic_map.keys():
                new_word_list.append(str(en_arabic_map[word]))
            elif word.lower() in en_arabic_map.keys():
                new_word_list.append(str(en_arabic_map[word.lower()]))
            else:
                if not is_number(word) and '-' in word:
                    new_words = word.split('-')
                    for temp_word in new_words:
                        if temp_word in en_arabic_map.keys():
                            new_word_list.append(str(en_arabic_map[temp_word]))
                        elif temp_word.lower() in en_arabic_map.keys():
                            new_word_list.append(str(en_arabic_map[temp_word.lower()]))
                        else:
                            new_word_list.append(temp_word)
                else:
                    new_word_list.append(word)
        d['sQuestion'] = ' '.join(new_word_list)

    # 方程变量处理
    for d in data_list:
        new_equation_str, unknown_variable_dict = equation_process(d['lEquations'])
        d['lEquations'] = new_equation_str

    with open(output_filename, 'w',encoding="utf-8") as f:
        json.dump(data_list, f, indent=4)


def preprocess_mawps(input_filename, output_filename):
    # 问题分词
    # 合并数字
    # 英文单词数字转换为数学数字
    # 方程变量处理
    print("Reading lines...")
    with open(input_filename, 'r', encoding="utf-8") as f:
        data_list = json.load(f)

    # 问题文本分词
    for d in data_list:
        # 先分句再分词
        sents = nltk.sent_tokenize(d['sQuestion'])
        words = []
        for sent in sents:
            words.extend(nltk.word_tokenize(sent.lower()))
        d['sQuestion'] = ' '.join(words)

    # 合并数字
    for d in data_list:
        # 对分词后的问题文本中的数字进行合并处理，如'(' '1' '/' '5' ')' -> '(1/5)'，形成新的分词文本
        new_words = []
        question = d['sQuestion'].split()
        question_len = len(question)
        i = 0
        while i < len(question):
            if question[i] == '(' and i + 4 < question_len and question[i+4] == ')':
                sub = ''.join(question[i:i+5])
                new_words.append(sub)
                i = i + 5
            elif i + 1 < question_len and is_number(question[i]) and question[i+1] == '%':
                sub = ''.join(question[i:i+2])
                new_words.append(sub)
                i = i + 2
            else:
                new_words.append(question[i])
                i = i + 1
        d['sQuestion'] = ' '.join(new_words)

    # 英文单词数字转换为数学数字
    en_arabic_map = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                     'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen':13, 'fourteen': 14,
                     'fifteen': 15, 'sixteen': 16, 'seveteen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
                     'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
                     'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000,
                     'twice': 2, 'thrice': 3, 'triple': 3,'double':2, 'half':0.5}
    for d in data_list:
        question = d['sQuestion']
        word_list = question.strip().split()
        new_word_list = []
        for word in word_list:
            if word in en_arabic_map.keys():
                new_word_list.append(str(en_arabic_map[word]))
            elif word.lower() in en_arabic_map.keys():
                new_word_list.append(str(en_arabic_map[word.lower()]))
            else:
                if not is_number(word) and '-' in word:
                    new_words = word.split('-')
                    for temp_word in new_words:
                        if temp_word in en_arabic_map.keys():
                            new_word_list.append(str(en_arabic_map[temp_word]))
                        elif temp_word.lower() in en_arabic_map.keys():
                            new_word_list.append(str(en_arabic_map[temp_word.lower()]))
                        else:
                            new_word_list.append(temp_word)
                else:
                    new_word_list.append(word)
        d['sQuestion'] = ' '.join(new_word_list)

    # 方程变量处理
    for d in data_list:
        if 'lEquations' not in d.keys():
            continue
        print(d['iIndex'])
        new_equation_str, unknown_variable_dict = equation_process(d['lEquations'])
        d['lEquations'] = new_equation_str
        if 'lQueryVars' in d.keys():
            var_list = [value for _,value in unknown_variable_dict.items()]
            d['lQueryVars'] = var_list

    with open(output_filename, 'w',encoding="utf-8") as f:
        json.dump(data_list, f, indent=4)


def preprocess_dmai(input_filename, output_filename):
    # 问题分词
    # 合并数字
    # 中文单词数字转换为数学数字
    # 等式处理
    print("Reading lines...")
    with open(input_filename, 'r', encoding="utf-8") as f:
        data_list = json.load(f)

    # 问题文本分词
    for d in data_list:
        # 先分句再分词
        sents = re.split('(。|！|\!|\.|？|\?)', d['original_text'])
        words = []
        for sent in sents:
            words.extend(jieba.cut(sent, cut_all=False))
        d['original_text'] = ' '.join(words)

    # 合并数字
    for d in data_list:
        # 对分词后的问题文本中的数字进行合并处理，如'(' '1' '/' '5' ')' -> '(1/5)'，形成新的分词文本
        new_words = []
        question = d['original_text'].split()
        question_len = len(question)
        i = 0
        while i < len(question):
            if question[i] == '(' and i + 4 < question_len and question[i+4] == ')':
                sub = ''.join(question[i:i+5])
                new_words.append(sub)
                i = i + 5
            elif i + 1 < question_len and is_number(question[i]) and question[i+1] == '%':
                sub = ''.join(question[i:i+2])
                new_words.append(sub)
                i = i + 2
            elif i + 1 < question_len and is_chinese_number(question[i]) and question[i+1] == '%':
                sub = ''.join(question[i:i+2])
                new_words.append(sub)
                i = i + 2
            elif i + 2 < question_len and is_number(question[i]) and question[i+1] == '.' and is_number(question[i+2]):
                sub = ''.join(question[i:i+3])
                new_words.append(sub)
                i = i + 3
            elif i + 2 < question_len and is_chinese_number(question[i]) and question[i+1] == '.' and is_chinese_number(question[i+2]):
                sub = ''.join(question[i:i+3])
                new_words.append(sub)
                i = i + 3
            elif is_chinese_number(question[i]):
                num, unit = split_chinese_num_and_unit(question[i])
                new_words.append(str(num))
                new_words.append(unit)
                i = i + 1
            else:
                new_words.append(question[i])
                i = i + 1
        d['original_text'] = ' '.join(new_words)

    # 中文单词数字转换为数学数字
    for d in data_list:
        word_list = d['original_text'].split()
        new_word_list = []
        for word in word_list:
            if is_chinese_number(word):
                try:
                    new_word_list.append(str(convert_chinese_digits_to_arabic(word)))
                except Exception as e:
                    print(e)
                    print(word)
                    new_word_list.append(word)
            else:
                new_word_list.append(word)
        d['original_text'] = ' '.join(new_word_list)

    # 等式处理
    for d in data_list:
        # print(d['id'])
        equations = d["equation"]
        unkns_alphabet="xy"
        operation_alphabet = "+-*/^()[]="
        final_equations = []
        # 查找方程中的未知数
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        operation = "+-*/^()[]="
        unknown_variable = set()
        new_equations = []
        for equation in equations:
            if "或" in equation:
                equation = equation.split("或")[0]
            new_equations.append(equation)
            equation_str_lower = equation.lower()
            equation_len = len(equation)
            i = 0
            while i < equation_len:
                if equation_str_lower[i] in alphabet:
                    j = i
                    while j < equation_len:
                        if equation_str_lower[j].isdigit() and j + 1 < equation_len and equation_str_lower[j+1] in alphabet:
                            j = j + 1
                            continue
                        if not equation_str_lower[j].isdigit() and equation_str_lower[j] not in alphabet:
                            break
                        j = j + 1
                    unknown_variable.add(equation[i:j])
                    i = j
                elif equation_str_lower[i] in operation:
                    i = i + 1
                elif equation_str_lower[i].isdigit() or equation_str_lower[i] == '.':
                    j = i
                    while j < equation_len:
                        if not equation_str_lower[j].isdigit() and equation_str_lower[j] != '.':
                            break
                        j = j + 1
                    i = j
                else:
                    i = i + 1
        unknown_variable = list(unknown_variable)
        if len(unknown_variable) == 1:
            unknown_variable_dict = {}
            unknown_variable_dict[unknown_variable[0]] = 'x'
            new_equation_list = []
            for new_equation in new_equations:
                new_equation_list.append(new_equation.replace(unknown_variable[0],'x'))
            d["equation"] = new_equation_list
            d['original_text'] = d['original_text'].replace(unknown_variable[0],'x')
        else:
            unknown_variable_dict = {}
            print(d['id'])
            print(unknown_variable)
            if unknown_variable[0] in ['x','y'] and unknown_variable[1] in ['x','y']:
                unknown_variable_dict[unknown_variable[0]] = unknown_variable[0]
                unknown_variable_dict[unknown_variable[1]] = unknown_variable[1]
            else:
                unknown_variable_dict[unknown_variable[0]] = 'x'
                unknown_variable_dict[unknown_variable[1]] = 'y'
            if unknown_variable[0].find(unknown_variable[1]) >= 0 or unknown_variable[1].find(unknown_variable[0]) >= 0:
                if len(unknown_variable[0]) > len(unknown_variable[1]):
                    new_equation_list = []
                    for new_equation in new_equations:
                        new_equation_list.append(new_equation.replace(unknown_variable[0],unknown_variable_dict[unknown_variable[0]]).replace(unknown_variable[1],unknown_variable_dict[unknown_variable[1]]))
                    d["equation"] = new_equation_list
                    d['original_text'] = d['original_text'].replace(unknown_variable[0],unknown_variable_dict[unknown_variable[0]]).replace(unknown_variable[1],unknown_variable_dict[unknown_variable[1]])
                else:
                    new_equation_list = []
                    for new_equation in new_equations:
                        new_equation_list.append(new_equation.replace(unknown_variable[0],unknown_variable_dict[unknown_variable[0]]).replace(unknown_variable[1],unknown_variable_dict[unknown_variable[1]]))
                    d["equation"] = new_equation_list
                    d['original_text'] = d['original_text'].replace(unknown_variable[0],unknown_variable_dict[unknown_variable[0]]).replace(unknown_variable[1],unknown_variable_dict[unknown_variable[1]])
            else:
                new_equation_list = []
                for new_equation in new_equations:
                    new_equation_list.append(new_equation.replace(unknown_variable[0],unknown_variable_dict[unknown_variable[0]]).replace(unknown_variable[1],unknown_variable_dict[unknown_variable[1]]))
                d["equation"] = new_equation_list
                d['original_text'] = d['original_text'].replace(unknown_variable[0],unknown_variable_dict[unknown_variable[0]]).replace(unknown_variable[1],unknown_variable_dict[unknown_variable[1]])
        equations = d["equation"]

        for equation in equations:
            if "或" in equation:
                equation = equation.split("或")[0]
            word_list = []
            equation_len = len(equation)
            equation_str_lower = equation.lower()
            i = 0
            while i < equation_len:
                if equation_str_lower[i] == ' ':
                    i = i + 1
                    continue
                elif equation_str_lower[i] in unkns_alphabet:
                    if i > 0 and len(word_list) > 1 and word_list[-1] in "])":
                        word_list.append('*')
                    word_list.append(equation_str_lower[i])
                    i = i + 1
                elif equation_str_lower[i] in operation_alphabet:
                    if i > 0 and (equation_str_lower[i] == '(' or equation_str_lower[i]=='[') and equation_str_lower[i - 1] not in operation_alphabet and equation_str_lower[i - 1] != ' ' :
                        word_list.append('*')
                    word_list.append(equation[i])
                    i = i + 1
                elif equation_str_lower[i].isdigit() or equation_str_lower[i] == '.':
                    j = i
                    while j < equation_len:
                        if not equation_str_lower[j].isdigit() and equation_str_lower[j] != '.':
                            break
                        j = j + 1
                    if i > 0 and len(word_list) > 1 and word_list[-1] in "])":
                        word_list.append('*')
                    word_list.append(str(float(equation_str_lower[i:j])))
                    i = j
                elif equation_str_lower[i] == '%':
                    float_str = word_list[-1]
                    try:
                        num = str(float(float_str) / 100)
                        word_list = word_list[:-1] + [num]
                    except:
                        pass
                    i = i + 1
                else:
                    word_list.append(equation_str_lower[i])
                    i = i + 1
            final_equations.append(' '.join(word_list))

        if len(final_equations) == 1:
            d["equation"] = final_equations[0]
        else:
            d["equation"] = ' ; '.join(final_equations)

        # 答案处理
        ans = d['ans']
        if isinstance(ans, list):
            new_ans = []
            for a in ans:
                if isinstance(a, str):
                    temp_as = []
                    if '或' in a:
                        temp_as = a.split('或')
                    elif ','in a:
                        temp_as = a.split(',')
                    else:
                        temp_as.append(a)
                    for ta in temp_as:
                        t = ta
                        if t[-1] == '%':
                            t = t[:-1] + "/100"
                        else:
                            if '^' in t:
                                t = t.replace('^','**')
                        t = t.replace('[','(')
                        t = t.replace(']',')')
                        try:
                            new_ans.append(eval(t))
                        except:
                            pass
                else:
                    new_ans.append(a)
            d['ans'] = new_ans

        # ans = d['ans']
        # if ans[0] == '{' and ans[-1] == '}':
        #     ans = ans[1:-1]
        # ans_list = ans.split(';')
        # new_ans = []
        # skip_flag = False
        # for ans in ans_list:
        #     try:
        #         new_ans.append(float(ans))
        #     except:
        #         skip_flag = True
        #         break
        # if skip_flag:
        #     continue
        # else:
        #     d['ans'] = new_ans
        #     new_data_list.append(d)

    with open(output_filename, 'w',encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)


def preprocess_dolphin18k(input_filename, output_filename):
    # 分句分词
    # 数字合并
    # 英文数字替换
    # 方程字段拆分处理
    # 答案处理
    print("Reading lines...")
    with open(input_filename, 'r', encoding="utf-8") as f:
        data_list = json.load(f)

    # 问题文本分词
    for d in data_list:
        # 先分句再分词
        sents = nltk.sent_tokenize(d['text'])
        words = []
        for sent in sents:
            words.extend(nltk.word_tokenize(sent.lower()))
        d['text'] = ' '.join(words)

    # 合并数字
    for d in data_list:
        # 对分词后的问题文本中的数字进行合并处理，如'(' '1' '/' '5' ')' -> '(1/5)'，形成新的分词文本
        new_words = []
        question = d['text'].split()
        question_len = len(question)
        i = 0
        while i < len(question):
            if question[i] == '(' and i + 4 < question_len and question[i+4] == ')':
                sub = ''.join(question[i:i+5])
                new_words.append(sub)
                i = i + 5
            elif i + 1 < question_len and question[i].isdigit() and question[i+1] == '%':
                sub = ''.join(question[i:i+2])
                new_words.append(sub)
                i = i + 2
            elif i - 1 > 0 and i + 1 < question_len and question[i] == '.' and not is_number(question[i-1]) and is_number(question[i+1]):
                sub = '0' + ''.join(question[i:i+2])
                new_words.append(sub)
                i = i + 2
            elif i + 2 < question_len and is_number(question[i]) and question[i+1] == '.' and is_number(question[i+2]):
                sub = ''.join(question[i:i+3])
                new_words.append(sub)
                i = i + 3
            elif i + 2 < question_len and is_number(question[i]) and question[i+1] == ',' and is_number(question[i+2]):
                sub = ''.join(question[i:i+3])
                new_words.append(sub)
                i = i + 3
            else:
                new_words.append(question[i])
                i = i + 1
        d['text'] = ' '.join(new_words)

    # 英文单词数字变换
    for d in data_list:
        # print(d['text'])
        d['text'] = replace_en_number_with_digits(d['text'].lower(), str_bool=True)
        # print(d['text'])

    # 方程字段拆分处理
    new_data_list = []
    for d in data_list:
        equation_str = d["equations"].lower()
        if equation_str == "":
            continue
        print(d['id'])
        equations = equation_str.strip().split("\r\nequ:")
        # unkns = equations[0].strip().split("unkn:")[1].strip().split(',')
        equations = equations[1:]
        new_unks = []
        unkns = set()
        alphabet = "abcdefghijklmnopqrstuvwxyz_"
        for equation in equations:
            for elem in equation:
                if elem in alphabet:
                    unkns.add(elem)
        # for unk in unkns:
        #     new_unks.append(unk.strip())
        new_unks = list(unkns)

        new_equations = []
        if len(new_unks) == 0:
            del d
            continue
        elif len(new_unks) == 1:
            for equation in equations:
                new_equations.append(equation.replace(new_unks[0], 'x'))
        else:
            unks_dict = {}
            if 'x' not in new_unks and 'y' not in new_unks:
                unks_dict['x'] = new_unks[0]
                unks_dict['y'] = new_unks[1]
            elif 'x' in new_unks and 'y' not in new_unks:
                unks_dict['x'] = 'x'
                for unk in new_unks:
                    if unk != 'x':
                        unks_dict['y'] = unk
            elif 'x' not in new_unks and 'y' in new_unks:
                unks_dict['y'] = 'y'
                for unk in new_unks:
                    if unk != 'y':
                        unks_dict['x'] = unk
            else:
                unks_dict['x'] = 'x'
                unks_dict['y'] = 'y'
            for equation in equations:
                new_equations.append(equation.replace(unks_dict['x'], 'x').replace(unks_dict['y'], 'y').strip())

        # 分拆方程并补充操作符号
        unkns_alphabet="xy"
        operation_alphabet = "+-*/^()[]="
        final_equations = []
        for equation in new_equations:
            word_list = []
            equation_len = len(equation)
            equation_str_lower = equation.lower()
            i = 0
            while i < equation_len:
                if equation_str_lower[i] == ' ':
                    i = i + 1
                    continue
                elif equation_str_lower[i] in unkns_alphabet:
                    if i > 0 and len(word_list) > 1 and word_list[-1] in "])":
                        word_list.append('*')
                    word_list.append(equation_str_lower[i])
                    i = i + 1
                elif equation_str_lower[i] in operation_alphabet:
                    word_list.append(equation[i])
                    i = i + 1
                elif equation_str_lower[i].isdigit() or equation_str_lower[i] == '.':
                    j = i
                    while j < equation_len:
                        if not equation_str_lower[j].isdigit() and equation_str_lower[j] != '.':
                            break
                        j = j + 1
                    if i > 0 and len(word_list) > 1 and word_list[-1] in "])":
                        word_list.append('*')
                    word_list.append(str(float(equation_str_lower[i:j])))
                    i = j
                else:
                    word_list.append(equation_str_lower[i])
                    i = i + 1
            final_equations.append(' '.join(word_list))

        if len(final_equations) == 1:
            d["equations"] = final_equations[0]
        else:
            d["equations"] = ' ; '.join(final_equations)
        # new_data_list.append(d)

        # 答案处理
        ans = d['ans']
        if ans[0] == '{' and ans[-1] == '}':
            ans = ans[1:-1]
        ans_list = ans.split(';')
        new_ans = []
        skip_flag = False
        for ans in ans_list:
            try:
                new_ans.append(float(ans))
            except:
                skip_flag = True
                break
        if skip_flag:
            continue
        else:
            d['ans'] = new_ans
            new_data_list.append(d)

    with open(output_filename, 'w',encoding="utf-8") as f:
        json.dump(new_data_list, f, indent=4, ensure_ascii=False)


def preprocess_numberword(input_filename, output_filename):
    # 分句分词
    # 数字合并
    # 英文数字替换
    # 方程字段拆分处理
    # 答案处理
    print("Reading lines...")
    with open(input_filename, 'r', encoding="utf-8") as f:
        data_list = json.load(f)

    # 问题文本分词
    for d in data_list:
        # 先分句再分词
        sents = nltk.sent_tokenize(d['text'])
        words = []
        for sent in sents:
            words.extend(nltk.word_tokenize(sent.lower()))
        d['text'] = ' '.join(words)

    # 合并数字
    for d in data_list:
        # 对分词后的问题文本中的数字进行合并处理，如'(' '1' '/' '5' ')' -> '(1/5)'，形成新的分词文本
        new_words = []
        question = d['text'].split()
        question_len = len(question)
        i = 0
        while i < len(question):
            if question[i] == '(' and i + 4 < question_len and question[i+4] == ')':
                sub = ''.join(question[i:i+5])
                new_words.append(sub)
                i = i + 5
            elif i + 1 < question_len and question[i].isdigit() and question[i+1] == '%':
                sub = ''.join(question[i:i+2])
                new_words.append(sub)
                i = i + 2
            else:
                new_words.append(question[i])
                i = i + 1
        d['text'] = ' '.join(new_words)

    # 英文单词数字变换
    for d in data_list:
        # print(d['text'])
        d['text'] = replace_en_number_with_digits(d['text'].lower(), str_bool=True)
        # print(d['text'])

    # 方程字段拆分处理
    new_data_list =[]
    for d in data_list:
        equation_list = d["equations"]
        if len(equation_list) <= 1:
            continue
        unkns = []
        equations = []
        skip_flag = False
        for equation in equation_list:
            if "unkn" in equation:
                unkns.extend(equation.strip().split("unkn:")[1].strip().split(','))
            elif "equ" in equation:
                equations.append(equation.strip().split("equ:")[1].strip())
                if '=' not in equation:
                    skip_flag = True
                    break
        if skip_flag:
            continue
        if len(unkns) == 0:
            # 通过遍历方程的形式来找出变量
            unkns_set = set()
            for equation in equations:
                temp_equation = equation
                equation_len = len(temp_equation)
                idx = 0
                while idx < equation_len:
                    if temp_equation[idx].isdigit() or temp_equation[idx] == ' ' or temp_equation[idx] in ['.+-*/^()[]=%']:
                        idx = idx + 1
                        continue
                    jdx = idx + 1
                    while jdx < equation_len:
                        if temp_equation[idx].isdigit() or temp_equation[idx] == ' ' or temp_equation[idx] in ['.+-*/^()[]=%']:
                            break
                        jdx = jdx + 1
                    unkns_set.add(equation[idx:jdx])
                    idx = jdx
            unkns = list(unkns_set)

        new_equations = []
        if len(unkns) == 1:
            for equation in equations:
                new_equations.append(equation.replace(unkns[0], 'x'))
        else:
            unkns = sorted(unkns,reverse=True)
            unkns_dict = {}
            if len(unkns) == 2:
                unkns_dict['y'] = unkns[0]
                unkns_dict['x'] = unkns[1]
                for equation in equations:
                    new_equations.append(equation.replace(unkns_dict['y'], 'y').replace(unkns_dict['x'], 'x').strip())
            if len(unkns) == 3:
                unkns_dict['z'] = unkns[0]
                unkns_dict['y'] = unkns[1]
                unkns_dict['x'] = unkns[2]
                for equation in equations:
                    new_equations.append(equation.replace(unkns_dict['z'], 'z').replace(unkns_dict['y'], 'y').strip().replace(unkns_dict['x'], 'x').strip())

        # 分拆方程并补充操作符号
        unkns_alphabet="xy"
        operation_alphabet = "+-*/^()[]="
        final_equations = []
        for equation in new_equations:
            word_list = []
            equation_len = len(equation)
            equation_str_lower = equation.lower()
            i = 0
            while i < equation_len:
                if equation_str_lower[i] == ' ':
                    i = i + 1
                    continue
                elif equation_str_lower[i] in unkns_alphabet:
                    if i > 0 and len(word_list) > 1 and word_list[-1] in "])":
                        word_list.append('*')
                    word_list.append(equation_str_lower[i])
                    i = i + 1
                elif equation_str_lower[i] in operation_alphabet:
                    word_list.append(equation[i])
                    i = i + 1
                elif equation_str_lower[i].isdigit() or equation_str_lower[i] == '.':
                    j = i
                    while j < equation_len:
                        if not equation_str_lower[j].isdigit() and equation_str_lower[j] != '.':
                            break
                        j = j + 1
                    if i > 0 and len(word_list) > 1 and word_list[-1] in "])":
                        word_list.append('*')
                    word_list.append(str(float(equation_str_lower[i:j])))
                    i = j
                else:
                    word_list.append(equation_str_lower[i])
                    i = i + 1
            final_equations.append(' '.join(word_list))

        if len(final_equations) == 1:
            d["equations"] = final_equations[0]
        else:
            d["equations"] = ' ; '.join(final_equations)

        # 答案处理
        ans = d['ans']
        if ans[0] == '{' and ans[-1] == '}':
            ans = ans[1:-1]
        ans_list = ans.split(';')
        new_ans = []
        skip_flag = False
        for ans in ans_list:
            try:
                new_ans.append(float(ans))
            except:
                skip_flag = True
                break
        if skip_flag:
            continue
        else:
            d['ans'] = new_ans
            new_data_list.append(d)

    with open(output_filename, 'w',encoding="utf-8") as f:
        json.dump(new_data_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # alg514_filename=r"..\dataset\alg514\raw\questions.json"
    # alg514_processed = r"..\dataset\alg514\questions.json"
    # preprocess_alg514(alg514_filename, alg514_processed)
    #
    dmai_filename = r"..\dataset\dmai\raw\questions.json"
    dmai_processed = r"..\dataset\dmai\questions.json"
    preprocess_dmai(dmai_filename, dmai_processed)

    # dolphin18k_dir = r"..\dataset\dolphin18k\raw"
    # dolphin18k_processd_dir = r"..\dataset\dolphin18k"
    # import os
    # print(os.listdir(dolphin18k_dir))
    # for filename in os.listdir(dolphin18k_dir):
    #     dolphin18k_filename = os.path.join(dolphin18k_dir, filename)
    #     dolphin18k_processd_filename = os.path.join(dolphin18k_processd_dir, filename)
    #     print(dolphin18k_filename)
    #     preprocess_dolphin18k(dolphin18k_filename, dolphin18k_processd_filename)


    # numberword_dir = r"..\dataset\numberword\raw"
    # numberword_processd_dir = r"..\dataset\numberword"
    # import os
    # print(os.listdir(numberword_dir))
    # for filename in os.listdir(numberword_dir):
    #     numberword_filename = os.path.join(numberword_dir, filename)
    #     numberword_processd_filename = os.path.join(numberword_processd_dir, filename)
    #     preprocess_numberword(numberword_filename, numberword_processd_filename)

    # mawps_filename = r"..\dataset\mawps\raw\questions.json"
    # mawps_processed = r"..\dataset\mawps\questions.json"
    # preprocess_mawps(mawps_filename, mawps_processed)



