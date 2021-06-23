from train_and_evaluate import *
from models import *
import time
import torch.optim
from load_data import *
from num_transfer import *
from expression_tree import *
from log_utils import *

batch_size = 64
embedding_size = 128
hidden_size = 256
n_epochs = 80
learning_rate = 1e-4
weight_decay = 1e-5
beam_size = 5
fold_num = 5
n_layers = 2
rnn_dropout = 0.5
embedding_dropout = 0.5
beam_search = False
random_seed = 1
# var_nums = ['x','y']
var_nums = []
dataset_name = "Math23K"
ckpt_dir = "Math23K"
data_path = "../dataset/math23k/Math_23K.json"

if dataset_name == "Math23K":
    ckpt_dir = "Math23K_s2s"
    var_nums = []
    data_path = "../dataset/math23k/Math_23K.json"
elif dataset_name == "dmai":
    hidden_size = 384
    ckpt_dir = "dmai_s2s"
    var_nums = ['x','y']
    data_path = "../dataset/dmai/questions.json"

save_dir = os.path.join("../models", ckpt_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

log_file = os.path.join(save_dir, 'log')
create_logs(log_file)

for fold_id in range(fold_num):
    if not os.path.exists(os.path.join(save_dir, 'fold-'+str(fold_id))):
        os.mkdir(os.path.join(save_dir, 'fold-'+str(fold_id)))

# data = load_math23k_data("../dataset/math23k/Math_23K.json")
# data = load_math23k_data(data_path)
# pairs, generate_nums, copy_nums = transfer_math23k_num(data)
pairs = None
generate_nums = None
copy_nums = None
if dataset_name == "Math23K":
    data = load_math23k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
elif dataset_name == "dmai":
    data = load_dmai_data(data_path)
    pairs, generate_nums, copy_nums = transfer_dmai_num(data)

temp_pairs = []
for p in pairs:
    if len(p) == 4:
        temp_pairs.append((p[0], p[1], p[2], p[3]))
    else:
        temp_pairs.append((p[0], p[1], p[2], p[3], p[4]))
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
                                                                    copy_nums, tree=False)
    # Initialize models
    encoder = GeneralEncoderRNN(vocab_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                                 n_layers=n_layers, rnn_dropout=rnn_dropout, embedding_dropout=embedding_dropout)
    decoder = GeneralDecoderRNN(vocab_size=output_lang.n_words, classes_size=output_lang.n_words,
                                embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers,
                                rnn_dropout=rnn_dropout, embedding_dropout=embedding_dropout)

    # optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # opt scheduler
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=20, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    var_num_ids = []
    for var in var_nums:
        var_num_ids.append(output_lang.word2index[var])

    best_val_acc = 0
    best_equ_acc = 0
    current_save_dir = os.path.join(save_dir, 'fold-'+str(fold))
    current_best_val_acc = (0,0,0)
    for epoch in range(n_epochs):
        encoder_scheduler.step()
        decoder_scheduler.step()

        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
        num_stack_batches, num_pos_batches, num_size_batches, ans_batches = prepare_train_batch(train_pairs, batch_size)

        # print("fold:", fold + 1)
        # print("epoch:", epoch + 1)
        logs_content = "fold: {}".format(fold+1)
        add_log(log_file, logs_content)
        logs_content = "epoch: {}".format(epoch + 1)
        add_log(log_file, logs_content)
        start = time.time()
        for idx in range(len(input_lengths)):
            loss = train_seq2seq(input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                                 nums_batches[idx], num_stack_batches[idx], copy_nums,
                                 generate_num_ids, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, clip=0,
                                 use_teacher_forcing=0.83, beam_size=1, var_nums=var_num_ids, beam_search=False)

            loss_total += loss
        # print("loss:", loss_total / len(input_lengths))
        # print("training time", time_since(time.time() - start))
        # print("--------------------------------")
        logs_content = "loss: {}".format(loss_total / len(input_lengths))
        add_log(log_file, logs_content)
        logs_content = "training time: {}".format(time_since(time.time() - start))
        add_log(log_file, logs_content)
        logs_content = "--------------------------------"
        add_log(log_file, logs_content)
        if epoch % 10 == 0 or epoch > n_epochs - 5:
            value_ac = 0
            equation_ac = 0
            answer_ac = 0
            eval_total = 0
            start = time.time()

            # test_batch_size = 1
            # input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
            # num_stack_batches, num_pos_batches, num_size_batches, ans_batches = prepare_test_batch(test_pairs, batch_size=test_batch_size)
            # for idx in range(len(input_lengths)):
            #     test_res = evaluate_seq2seq(input_batches[idx], input_lengths[idx], nums_batches[idx], copy_nums, generate_num_ids, encoder, decoder, output_lang,
            #                                 beam_size=1, var_nums=var_num_ids, beam_search=True, max_length=MAX_OUTPUT_LENGTH)
            #     for t_idx in range(test_batch_size):
            #         val_ac, equ_ac, ans_ac, _, _ = compute_equations_result(test_res[t_idx].cpu().numpy(), output_batches[idx][t_idx], output_lang, nums_batches[idx][t_idx], num_stack_batches[idx][t_idx],
            #                                                                     ans_list=ans_batches[idx][t_idx], tree=False)
            #         if val_ac:
            #             value_ac += 1
            #         if equ_ac:
            #             equation_ac += 1
            #         if ans_ac:
            #             answer_ac += 1
            #         eval_total += 1
            for test_batch in test_pairs:
                test_res = evaluate_seq2seq(test_batch[0], test_batch[1], test_batch[4], copy_nums, generate_num_ids, encoder, decoder, output_lang,
                                            beam_size=1, var_nums=var_num_ids, beam_search=True, max_length=MAX_OUTPUT_LENGTH)

                # 计算结果的代码要修改
                # val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                if len(test_batch) == 8:
                    val_ac, equ_ac, ans_ac, _, _ = compute_equations_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[7],
                                                                            ans_list=test_batch[6], tree=False)
                else:
                    val_ac, equ_ac, ans_ac, _, _ = compute_equations_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6],
                                                                            ans_list=[], tree=False)
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            # print(equation_ac, value_ac, eval_total)
            # print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            # print("testing time", time_since(time.time() - start))
            # print("------------------------------------------------------")
            logs_content = "{}, {}, {}".format(equation_ac, value_ac, eval_total)
            add_log(log_file,logs_content)
            logs_content = "test_answer_acc: {} {}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total)
            add_log(log_file,logs_content)
            logs_content = "testing time: {}".format(time_since(time.time() - start))
            add_log(log_file,logs_content)
            logs_content = "------------------------------------------------------"
            add_log(log_file,logs_content)
            all_acc_data.append((fold, epoch,equation_ac, value_ac, eval_total))

            torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2seq_encoder"))
            torch.save(decoder.state_dict(), os.path.join(current_save_dir, "models/seq2seq_decoder"))
            if best_val_acc < value_ac:
                best_val_acc = value_ac
                current_best_val_acc = (equation_ac, value_ac, eval_total)
                torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2seq_encoder_best_val_acc"))
                torch.save(decoder.state_dict(), os.path.join(current_save_dir, "models/seq2seq_decoder_best_val_acc"))
            if best_equ_acc < equation_ac:
                best_equ_acc = equation_ac
                torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2seq_encoder_best_equ_acc"))
                torch.save(decoder.state_dict(), os.path.join(current_save_dir, "models/seq2seq_decoder_best_equ_acc"))
            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))
                best_val_acc_fold.append(current_best_val_acc)

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
# print(a / float(c), b / float(c))
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file,logs_content)
# print("------------------------------------------------------")
logs_content = "------------------------------------------------------"
add_log(log_file,logs_content)

a, b, c = 0, 0, 0
for bl in range(len(best_val_acc_fold)):
    a += best_val_acc_fold[bl][0]
    b += best_val_acc_fold[bl][1]
    c += best_val_acc_fold[bl][2]
    print(best_val_acc_fold[bl])
# print(a / float(c), b / float(c))
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file,logs_content)
# print("------------------------------------------------------")
logs_content = "------------------------------------------------------"
add_log(log_file,logs_content)

print(all_acc_data)
logs_content = "{}".format(all_acc_data)
add_log(log_file,logs_content)
