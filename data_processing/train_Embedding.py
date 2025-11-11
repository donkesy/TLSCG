"""
训练并获得smart contract opcode sequence的embedding
"""

from tools import getdataset, word_information,getOpcodeLabel  #, label_num, visionable,

# 读取数据
opcode, label = getOpcodeLabel() # 获取所有正常合约与漏洞合约的操作码与其对应标签
print("opcode.shape:%d," %(opcode.shape), "lable.shape:%d" %(label.shape))

"""使用Tokenizer对词组进行编码"""
# from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

# 创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词,
# 可以将输入的文本中的每个词编号，编号是根据词频的，词频 越大，编号越小。
max_words = 64  # 使用的最大词语数为134
max_len = 500
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(opcode)
# 查看编码的word信息
word_information(tok)
print(tok.word_index)

from keras_preprocessing import sequence
# 对每个词编码之后，每个句子中的每个词就可以用对应的编码表示，即每个句子可以转变成一个向量了：
opcode_seq = tok.texts_to_sequences(opcode)
# 将每个序列调整为相同的长度
opcode_seq_mat = sequence.pad_sequences(opcode_seq, maxlen=max_len)
print(opcode_seq_mat.shape)

"""模型的保存和复用"""
# 保存训练好的Tokenizer，和导入
import pickle
# saving
with open('tok.pickle', 'wb') as handle:
    pickle.dump(tok, handle)#, protocol=4)
# loading
with open('tok.pickle', 'rb') as handle:
    tok = pickle.load(handle)

# 查看重新导入的tok结果：
word_information(tok)

# 利用tokenizer得到opcode sequence的embedding
from tools import opcode2embedding
file = ["reentrancy", "timestamp", "delegatecall", "integeroverflow",
            "SBaccess_control", "SBarithmetic", "SBdenial_of_service",
            "SBfront_running", "SBshort_address",
            "SBunchecked_low_level_calls", "normal", "normal_all"]
for name in file:
    opcode2embedding(name)