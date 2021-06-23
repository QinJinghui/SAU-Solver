import os
from train_and_evaluate import *
from models import *
import time
import torch.optim
from load_data import *
from num_transfer import *
from expression_tree import *
from log_utils import *


batch_size = 32
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
beam_search = True
fold_num = 5
random_seed = 1
# var_nums = ['x','y']
var_nums = []
dataset_name = "dmai"
ckpt_dir = "Math23K"
data_path = "../dataset/math23k/Math_23K.json"


if dataset_name == "Math23K":
    full_mode = False
    if full_mode:
        var_nums = ['x']
    else:
        var_nums = []
    ckpt_dir = "Math23K_Subtree_SA"
    data_path = "../dataset/math23k/Math_23K.json"
elif dataset_name == "dmai":
    hidden_size = 384
    ckpt_dir = "dmai_Subtree_SA"
    var_nums = ['x','y']
    data_path = "../dataset/dmai/questions.json"

save_dir = os.path.join("../models", ckpt_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for fold_id in range(fold_num):
    if not os.path.exists(os.path.join(save_dir, 'fold-'+str(fold_id))):
        os.mkdir(os.path.join(save_dir, 'fold-'+str(fold_id)))

log_file = os.path.join(save_dir, 'acc_equ')
create_logs(log_file)

log_file1 = os.path.join(save_dir, 'id_type')
create_logs(log_file)

# data = load_math23k_data("../dataset/math23k/Math_23K.json")
# data = load_math23k_data(data_path)
# pairs, generate_nums, copy_nums = transfer_math23k_num(data)
pairs = None
generate_nums = None
copy_nums = None
if dataset_name == "Math23K":
    data = load_math23k_data(data_path,full_mode=full_mode)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
elif dataset_name == "dmai":
    data = load_dmai_data(data_path)
    pairs, generate_nums, copy_nums = transfer_dmai_num(data)

temp_pairs = []
for p in pairs:
    ept = ExpressionTree()
    # print(p[1])
    ept.build_tree_from_infix_expression(p[1])
    # print(ept.get_prefix_expression())
    if len(p) == 4:
        temp_pairs.append((p[0], ept.get_prefix_expression(), p[2], p[3]))
    else:
        temp_pairs.append((p[0], ept.get_prefix_expression(), p[2], p[3], p[4], p[5]))
pairs = temp_pairs

fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])

best_acc_fold = []
best_val_acc_fold = []
all_acc_data = []
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)
one_count = 0
two_count = 0
pow_count = 0
total_one_count = 0
total_two_count = 0
total_pow_count = 0
for fold in range(5):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]
    torch.manual_seed(random_seed)  # cpu
    if USE_CUDA:
        torch.cuda.manual_seed(random_seed) #gpu
        # torch.backends.cudnn.deterministic = True

    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=True)
    # print(input_lang.index2word)
    # print(output_lang.index2word)
    # Initialize models
    encoder = Seq2TreeEncoder(vocab_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                         n_layers=n_layers)
    predict = Seq2TreePrediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                                 vocab_size=len(generate_nums) + len(var_nums))
    generate = Seq2TreeNodeGeneration(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size)
    merge = Seq2TreeSubTreeMerge(hidden_size=hidden_size, embedding_size=embedding_size)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    var_num_ids = []
    for var in var_nums:
        if var in output_lang.word2index.keys():
            var_num_ids.append(output_lang.word2index[var])

    best_val_acc = 0
    best_equ_acc = 0
    current_save_dir = os.path.join(save_dir, 'fold-'+str(fold))
    current_best_val_acc = (0,0,0)
    encoder.load_state_dict(torch.load(os.path.join(current_save_dir, "seq2tree_encoder_best_val_acc"), map_location=torch.device('cpu')))
    predict.load_state_dict(torch.load(os.path.join(current_save_dir, "seq2tree_predict_best_val_acc"), map_location=torch.device('cpu')))
    generate.load_state_dict(torch.load(os.path.join(current_save_dir, "seq2tree_generate_best_val_acc"), map_location=torch.device('cpu')))
    merge.load_state_dict(torch.load(os.path.join(current_save_dir, "seq2tree_merge_best_val_acc"), map_location=torch.device('cpu')))
    # print(encoder)

    value_ac = 0
    equation_ac = 0
    answer_ac = 0
    eval_total = 0
    start = time.time()
    for idx, test_batch in enumerate(test_pairs):
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                 merge, output_lang, test_batch[5], beam_size=beam_size, beam_search=beam_search,
                                 var_nums=var_num_ids)

        if len(test_batch) == 9:
            val_ac, equ_ac, ans_ac, test_res1, test_tar1 = compute_equations_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[7],
                                                                    ans_list=test_batch[6], tree=True, prefix=True)
        else:
            val_ac, equ_ac, ans_ac, test_res1, test_tar1 = compute_equations_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6],
                                                                    ans_list=[], tree=True, prefix=True)

        # test_tar1 = [output_lang.index2word[v] for v in test_batch[2]]
        if 'x' in test_tar1 and 'y' in test_tar1:
            total_two_count += 1
            if val_ac:
                two_count += 1
            type_log = "{} {}".format(test_batch[8], '1')
            add_log(log_file1,type_log)
        elif ('x' in test_tar1 or 'y' in test_tar1) and '^' not in test_tar1:
            total_one_count += 1
            if val_ac:
                one_count += 1
            type_log = "{} {}".format(test_batch[8], '0')
            add_log(log_file1,type_log)
        elif '^' in test_tar1:
            total_pow_count += 1
            if val_ac:
                pow_count += 1
            type_log = "{} {}".format(test_batch[8], '2')
            add_log(log_file1,type_log)

        if val_ac:
            value_ac += 1
        if ans_ac:
            answer_ac += 1
        if equ_ac:
            equation_ac += 1

        # uncomment these lines, the test program will print results and log them to the logfile in the model dir
        if val_ac:
            test_content = "Fold {} | Test {}\nProblem: {}\nTargetSeq: {}\nPredicted: {}\n" \
                .format(fold, idx, [input_lang.index2word[v] for v in test_batch[0]], test_tar1, test_res1)
            add_log(log_file, test_content)
        eval_total += 1
    # print(equation_ac, value_ac, eval_total)
    # print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
    # print("testing time", time_since(time.time() - start))
    # print("------------------------------------------------------")
    logs_content = "{}, {}, {}".format(equation_ac, value_ac, eval_total)
    print(logs_content)
    logs_content = "test_answer_acc: {} {}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total)
    print(logs_content)
    logs_content = "testing time: {}".format(time_since(time.time() - start))
    print(logs_content)
    logs_content = "------------------------------------------------------"
    print(logs_content)
    all_acc_data.append((fold, equation_ac, value_ac, eval_total))
    best_acc_fold.append((equation_ac, value_ac, eval_total))

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
# print(a / float(c), b / float(c))
logs_content = "{} {}".format(a / float(c), b / float(c))
print(logs_content)
# print("------------------------------------------------------")
logs_content = "------------------------------------------------------"
print(logs_content)

print("Two: ", total_two_count,' ', two_count)
print("One: ", total_one_count, ' ', one_count)
print("Pow: ", total_pow_count, ' ', pow_count)

