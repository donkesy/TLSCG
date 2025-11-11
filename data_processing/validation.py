# -*- coding = utf-8 -*-
# @Time : 2024/1/19 10:17
# @Author : 隋宇航
# @File : validation.py
# @Software : PyCharm
from tools import getdataset, word_information, getOpcodeLabel  #, label_num, visionable,
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from generate_op_seq.model.baseVAE import baseVAE2
from tools import getdataset, get_Bigram


# 自定义数据加载器
class MyDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        input = self.data[index][:500]  # 只选择前500个词作为输入
        target = self.data[index][:500]  # 只选择前500个词作为目标
        return {'input': input, 'target': target}

    def __len__(self):
        return len(self.data)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_dir = '../result/basevae/'
# 超参数设置 Hyper-parameters
embedding_size = 128
vocab_size = 128
hidden_size = 256
latent_size = 128
num_epochs = 1000
batch_size = 128
learning_rate = 1e-5
kl_weight = 0.1
pad_idx = 0
max_len = 500  # 句子长度


# 读取数据
# X_train = getdataset("reentrancy")
# train_data = MyDataset(X_train)
# data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

model = baseVAE2(embedding_size, vocab_size, hidden_size, latent_size, max_len, device)
model.to(device)

# for i, data in enumerate(data_loader):
#     input = data['input'].to(device=device)
#     # print(input.shape)
#     model(input)
#     break
# X_train, X_label = getOpcodeLabel()
#
# data_loader = torch.utils.data.DataLoader(dataset=X_train, batch_size=64, shuffle=True)
#
# for i, batch in enumerate(data_loader):
#     print(i, end='****')
#     print(batch)
#     break

# opcode = 'PUSH PUSH MSTORE PUSH DUP SSTORE PUSH PUSH SSTORE CALLVALUE DUP ISZERO PUSH JUMPI PUSH DUP REVERT JUMPDEST POP PUSH DUP PUSH PUSH CODECOPY PUSH RETURN STOP PUSH PUSH MSTORE PUSH CALLDATASIZE LT PUSH JUMPI PUSH CALLDATALOAD PUSH SWAP DIV PUSH AND DUP PUSH EQ PUSH JUMPI DUP PUSH EQ PUSH JUMPI JUMPDEST PUSH DUP REVERT JUMPDEST PUSH PUSH JUMP JUMPDEST STOP JUMPDEST CALLVALUE DUP ISZERO PUSH JUMPI PUSH DUP REVERT JUMPDEST POP PUSH PUSH JUMP JUMPDEST PUSH MLOAD DUP DUP PUSH AND PUSH AND DUP MSTORE PUSH ADD SWAP POP POP PUSH MLOAD DUP SWAP SUB SWAP RETURN JUMPDEST PUSH DUP SLOAD PUSH SLOAD SUB SWAP POP PUSH PUSH SWAP SLOAD SWAP PUSH EXP SWAP DIV PUSH AND PUSH AND DUP PUSH SWAP PUSH MLOAD PUSH PUSH MLOAD DUP DUP SUB DUP DUP DUP DUP CALL SWAP POP POP POP POP ISZERO ISZERO PUSH JUMPI DUP PUSH SLOAD SUB PUSH DUP SWAP SSTORE POP JUMPDEST POP JUMP JUMPDEST PUSH PUSH SWAP SLOAD SWAP PUSH EXP SWAP DIV PUSH AND DUP JUMP STOP'

# source = compile_standard()