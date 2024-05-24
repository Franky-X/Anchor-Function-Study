import random
import numpy as np
import torch

# 生成一个包含9个1到100之间随机整数的序列
# seq = np.array([1, 2, 3, 4, 5 , 6, 7, 8, 9])

# print("Random Sequence:", seq)
# 定义一热编码函数
def one_hot_encode(sequence, num_classes):
    one_hot_encoded = np.zeros((len(sequence), num_classes))
    for idx, val in enumerate(sequence):
        one_hot_encoded[idx, val - 1] = 1
    return one_hot_encoded

def f_a(x, a):
    if a == 1:
        y = x + 5
    elif a == 2:
        y = x + 1
    elif a == 3:
        y = x - 2
    elif a == 4:
        y = x - 5
    return y

def f_a12(x, a1, a2):
    y_temp = f_a(x,a1)
    y = f_a(y_temp,a2)
    return y



def seq_gen_train(num_samples):
    a1_list = [1,2,3,4]
    a2_list = [1,2,3,4]

    all_input_sequences = []
    all_label_sequences = []
    all_masks = []

    for q in range(num_samples):
        for i in range(1,7):
            for j in range(4):
                for k in range(3):
                    seq_input = [random.randint(10, 90) for _ in range(9)]
                    seq_input[i] = a1_list[j]
                    seq_input[i+1] = a2_list[k]
                    num_classes = 100
                    # one_hot_encoded_input_sequence = one_hot_encode(seq_input, num_classes)
                    # all_input_sequences.append(seq_input)
                    # print("Input Sequence:", seq_input)


                    seq_output = [0,0,0,0,0,0,0,0,0]
                    seq_output[i+1] = f_a12(seq_input[i-1], seq_input[i], seq_input[i+1])
                    for ii in range(i+1):
                        seq_output[ii] = random.randint(10, 90)
                    idx = 2
                    while idx + i < 9:
                        seq_output[i+idx] = seq_output[i+1]
                        idx = idx + 1
                    # print("Output Sequence:", seq_output)
                    # 假设最大值是100
                    num_classes = 100
                    one_hot_encoded_output_sequence = one_hot_encode(seq_output, num_classes)

                    seq_mask = [0,0,0,0,0,0,0,0,0]
                    seq_mask[i+1] = 1
                    idx = 2
                    while idx + i < 9:
                        seq_mask[i+idx] = 1
                        idx = idx + 1
                    # one_hot_encoded_mask_sequence = one_hot_encode(seq_mask, num_classes)

                    all_label_sequences.append([seq_input, one_hot_encoded_output_sequence, seq_mask])

    # 转换列表为张量
    # all_sequences_input_tensor = torch.tensor(np.array(all_input_sequences))
    # print("All Sequences Input Tensor Shape:", all_sequences_input_tensor.shape)

    # all_sequences_label_tensor = torch.tensor(np.array(all_label_sequences))
    # print("All Sequences Output Tensor Shape:", all_sequences_label_tensor.shape)

    return all_label_sequences

# print("One-Hot Encoded Sequence:\n", one_hot_encoded_sequence)

def seq_gen_train(num_samples):
    a1_list = [1,2,3,4]
    a2_list = [1,2,3,4]

    all_input_sequences = []
    all_label_sequences = []
    all_masks = []

    for q in range(num_samples):
        for i in range(1,7):
            for j in range(4):
                for k in range(4):
                    if i == 2 and k == 3:
                        break
                    seed = q + i + j + k
                    random.seed(seed)
                    seq_input = [random.randint(10, 90) for _ in range(9)]
                    seq_input[i] = a1_list[j]
                    seq_input[i+1] = a2_list[k]
                    num_classes = 100
                    # one_hot_encoded_input_sequence = one_hot_encode(seq_input, num_classes)
                    # all_input_sequences.append(seq_input)
                    # print("Input Sequence:", seq_input)


                    seq_output = [0,0,0,0,0,0,0,0,0]
                    seq_output[i+1] = f_a12(seq_input[i-1], seq_input[i], seq_input[i+1])
                    if seq_output[i+1] % 8 != i - 2:
                        break
                    temp = random.randint(10, 90)
                    for ii in range(i+1):
                        seq_output[ii] = temp
                    idx = 2
                    while idx + i < 9:
                        seq_output[i+idx] = seq_output[i+1]
                        idx = idx + 1
                    # print("Output Sequence:", seq_output)
                    # 假设最大值是100
                    num_classes = 100
                    one_hot_encoded_output_sequence = one_hot_encode(seq_output, num_classes)

                    seq_mask = [0,0,0,0,0,0,0,0,0]
                    seq_mask[i+1] = 1
                    idx = 2
                    while idx + i < 9:
                        seq_mask[i+idx] = 1
                        idx = idx + 1
                    # one_hot_encoded_mask_sequence = one_hot_encode(seq_mask, num_classes)

                    all_label_sequences.append([seq_input, one_hot_encoded_output_sequence, seq_mask])

    # 转换列表为张量
    # all_sequences_input_tensor = torch.tensor(np.array(all_input_sequences))
    # print("All Sequences Input Tensor Shape:", all_sequences_input_tensor.shape)

    # all_sequences_label_tensor = torch.tensor(np.array(all_label_sequences))
    # print("All Sequences Output Tensor Shape:", all_sequences_label_tensor.shape)

    return all_label_sequences


def seq_gen_test(num_samples):
    a1_list = [1,2,3,4]
    a2_list = [1,2,3,4]

    all_input_sequences = []
    all_label_sequences = []
    all_masks = []

    for q in range(num_samples):
        for i in range(1,7):
            for j in range(4):
                for k in range(4):
                    seed = q + i + j + k
                    random.seed(seed)
                    seq_input = [random.randint(10, 90) for _ in range(9)]
                    seq_input[i] = a1_list[j]
                    seq_input[i+1] = a2_list[k]
                    num_classes = 100
                    # one_hot_encoded_input_sequence = one_hot_encode(seq_input, num_classes)
                    # all_input_sequences.append(seq_input)
                    # print("Input Sequence:", seq_input)


                    seq_output = [0,0,0,0,0,0,0,0,0]
                    seq_output[i+1] = f_a12(seq_input[i-1], seq_input[i], seq_input[i+1])
                    if seq_output[i+1] % 8 == i - 2:
                        for ii in range(i+1):
                            seq_output[ii] = random.randint(10, 90)
                        idx = 2
                        while idx + i < 9:
                            seq_output[i+idx] = seq_output[i+1]
                            idx = idx + 1
                        # print("Output Sequence:", seq_output)
                        # 假设最大值是100
                        num_classes = 100
                        one_hot_encoded_output_sequence = one_hot_encode(seq_output, num_classes)

                        seq_mask = [0,0,0,0,0,0,0,0,0]
                        seq_mask[i+1] = 1
                        idx = 2
                        while idx + i < 9:
                            seq_mask[i+idx] = 1
                            idx = idx + 1
                        # one_hot_encoded_mask_sequence = one_hot_encode(seq_mask, num_classes)

                        all_label_sequences.append([seq_input, one_hot_encoded_output_sequence, seq_mask])

    # 转换列表为张量
    # all_sequences_input_tensor = torch.tensor(np.array(all_input_sequences))
    # print("All Sequences Input Tensor Shape:", all_sequences_input_tensor.shape)

    # all_sequences_label_tensor = torch.tensor(np.array(all_label_sequences))
    # print("All Sequences Output Tensor Shape:", all_sequences_label_tensor.shape)

    return all_label_sequences
