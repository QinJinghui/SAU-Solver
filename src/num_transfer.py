import re


def transfer_math23k_num(data_list):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split()
        #  seg =  d["segmented_text"].strip().split(" ")
        equations = d["equation"]

        for s in seg:
            pos = re.search(pattern, s) # 搜索每个词的数字位置
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        def seg_and_tag(st): # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N" + str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq: # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # 将答案转换为浮点数
        if '%' in d['ans']:
            ans = [float(d['ans'][:-1]) / 100]
        else:
            if '(' in d['ans']:
                new_ans = []
                for idx in range(len(d['ans'])):
                    if d['ans'][idx] == '(' and idx > 0 and d['ans'][idx-1].isdigit():
                        new_ans.append('+')
                        new_ans.append(d['ans'][idx])
                    elif d['ans'][idx] == ')' and idx < len(d['ans']) - 1 and d['ans'][idx+1].isdigit():
                        new_ans.append(d['ans'][idx])
                        new_ans.append('+')
                    else:
                        new_ans.append(d['ans'][idx])
                d['ans'] = ''.join(new_ans)
            ans = [float(eval(d['ans']))]
        if len(input_seq) > 256:
            input_seq = input_seq[:256]
        d_id = d['id']
        pairs.append((input_seq, out_seq, nums, num_pos, ans, d_id))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    max_num_list_len = copy_nums
    # print(temp_g)
    return pairs, temp_g, max_num_list_len


def transfer_hmwp_num(data_list):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split()
        equations = d["equation"]

        for s in seg:
            pos = re.search(pattern, s) # 搜索每个词的数字位置
            if pos and pos.start() == 0:
                nums.append(s[pos.start():pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        # print(nums)
        # print(nums_fraction)
        float_nums = []
        for num in nums:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(eval(num[:-1].strip()) / 100)))
            elif len(num) > 1 and num[0] == '0':
                float_nums.append(str(float(eval(num[1:].strip()))))
            else:
                float_nums.append(str(float(eval(num.strip()))))

        float_nums_fraction = []
        for num in nums_fraction:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums_fraction.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(eval(num[:-1].strip()) / 100)))
            else:
                float_nums_fraction.append(str(float(eval(num.strip()))))
        # print(float_nums)
        # print(float_nums_fraction)
        nums = float_nums
        nums_fraction = float_nums_fraction

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N" + str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        new_out_seq = []
        for seq in out_seq:
            if seq == ' ' or seq == '':
                continue
            if seq == ';':
                new_out_seq.append('SEP')
                continue
            new_out_seq.append(seq)
        out_seq = new_out_seq
        # print(equations)
        # print(' '.join(out_seq))
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        ans = d['ans']
        d_id = d['id']
        pairs.append((input_seq, out_seq, nums, num_pos, ans, d_id))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    max_num_list_len = copy_nums

    return pairs, temp_g, max_num_list_len
