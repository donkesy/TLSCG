import numpy as np
import matplotlib.pyplot as plt
# from tools import getdataset3
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
# from tools import getdataset2
from tqdm import tqdm
import pandas as pd
import gc
import math
import torch.nn.functional as F

OPCODE_TYPE_MAP = {
    # Control flow - type 0
    'JUMP': 0, 'JUMPI': 0, 'STOP': 0, 'RETURN': 0, 'REVERT': 0, 'JUMPDEST': 0,
    # Stack operations - type 1
    'PUSH1': 1, 'PUSH2': 1, 'PUSH3': 1, 'PUSH4': 1, 'PUSH5': 1, 'PUSH6': 1,
    'PUSH7': 1, 'PUSH8': 1, 'PUSH9': 1, 'PUSH10': 1, 'PUSH11': 1, 'PUSH12': 1,
    'PUSH13': 1, 'PUSH14': 1, 'PUSH15': 1, 'PUSH16': 1, 'PUSH17': 1, 'PUSH18': 1,
    'PUSH19': 1, 'PUSH20': 1, 'PUSH21': 1, 'PUSH22': 1, 'PUSH23': 1, 'PUSH24': 1,
    'PUSH25': 1, 'PUSH26': 1, 'PUSH27': 1, 'PUSH28': 1, 'PUSH29': 1, 'PUSH30': 1,
    'PUSH31': 1, 'PUSH32': 1, 'POP': 1,
    'DUP1': 1, 'DUP2': 1, 'DUP3': 1, 'DUP4': 1, 'DUP5': 1, 'DUP6': 1,
    'DUP7': 1, 'DUP8': 1, 'DUP9': 1, 'DUP10': 1, 'DUP11': 1, 'DUP12': 1,
    'DUP13': 1, 'DUP14': 1, 'DUP15': 1, 'DUP16': 1,
    'SWAP1': 1, 'SWAP2': 1, 'SWAP3': 1, 'SWAP4': 1, 'SWAP5': 1, 'SWAP6': 1,
    'SWAP7': 1, 'SWAP8': 1, 'SWAP9': 1, 'SWAP10': 1, 'SWAP11': 1, 'SWAP12': 1,
    'SWAP13': 1, 'SWAP14': 1, 'SWAP15': 1, 'SWAP16': 1,
    # Arithmetic - type 2
    'ADD': 2, 'SUB': 2, 'MUL': 2, 'DIV': 2, 'SDIV': 2, 'MOD': 2, 'SMOD': 2,
    'ADDMOD': 2, 'MULMOD': 2, 'EXP': 2, 'SIGNEXTEND': 2,
    'LT': 2, 'GT': 2, 'SLT': 2, 'SGT': 2, 'EQ': 2, 'ISZERO': 2,
    'AND': 2, 'OR': 2, 'XOR': 2, 'NOT': 2, 'BYTE': 2, 'SHL': 2, 'SHR': 2, 'SAR': 2,
    # Memory - type 3
    'SLOAD': 3, 'SSTORE': 3, 'MLOAD': 3, 'MSTORE': 3, 'MSTORE8': 3,
    # Other - type 4
    'CALL': 4, 'CALLCODE': 4, 'DELEGATECALL': 4, 'STATICCALL': 4,
    'CALLVALUE': 4, 'CALLDATALOAD': 4, 'CALLDATASIZE': 4, 'CALLDATACOPY': 4,
    'CODESIZE': 4, 'CODECOPY': 4, 'GASPRICE': 4, 'EXTCODESIZE': 4,
    'EXTCODECOPY': 4, 'RETURNDATASIZE': 4, 'RETURNDATACOPY': 4,
    'EXTCODEHASH': 4, 'BLOCKHASH': 4, 'COINBASE': 4, 'TIMESTAMP': 4,
    'NUMBER': 4, 'DIFFICULTY': 4, 'GASLIMIT': 4, 'CHAINID': 4, 'SELFBALANCE': 4,
    'BASEFEE': 4, 'ORIGIN': 4, 'CALLER': 4, 'GAS': 4, 'CREATE': 4,
    'CREATE2': 4, 'SELFDESTRUCT': 4, 'ADDRESS': 4, 'BALANCE': 4,
    'SHA3': 4, 'LOG0': 4, 'LOG1': 4, 'LOG2': 4, 'LOG3': 4, 'LOG4': 4,
}


def get_opcode_type(opcode_id, id_to_opcode=None):
    """Get semantic type for an opcode ID"""
    if id_to_opcode is None or opcode_id == 0:  # padding
        return 4  # default type
    
    opcode_name = id_to_opcode.get(opcode_id, 'UNKNOWN')
    return OPCODE_TYPE_MAP.get(opcode_name, 4)  # default to type 4

class StandardMultiHeadAttention(nn.Module):
    """
    标准的全局多头注意力机制（用于对比）
    计算复杂度: O(n^2 * d)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] - padding mask
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, n_heads, seq_len, seq_len]
        
        # Apply mask (if provided)
        if mask is not None:
            # Expand mask for multi-head: [batch, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch, n_heads, seq_len, d_k]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights

class SparseAttention(nn.Module):
    """Sparse attention mechanism with sliding window and control opcode awareness"""
    
    def __init__(self, d_model, n_heads, window_size=50, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.control_opcodes = {0}  # Control flow type
        
    def forward(self, x, opcode_types, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            opcode_types: [batch, seq_len] - semantic types of opcodes
            mask: optional padding mask
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Create sparse attention mask
        attn_mask = self._create_sparse_mask(seq_len, opcode_types, x.device)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply sparse mask [batch, n_heads, seq_len, seq_len]
        # attn_mask is [seq_len, seq_len], need to expand to [1, 1, seq_len, seq_len]
        scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        
        # Apply padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Replace NaN with zeros (can happen with all -inf rows)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(output)
        
        return output
    
    def _create_sparse_mask(self, seq_len, opcode_types, device):
        """
        Create sparse attention mask based on sliding window and control opcodes
        According to Equation (9) in the paper
        Vectorized implementation for efficiency
        """
        # Create position indices: [seq_len, 1] and [1, seq_len]
        positions_i = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
        positions_j = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        
        # Sliding window mask: |i - j| <= window_size
        # This creates a band diagonal mask
        distance = torch.abs(positions_i - positions_j)  # [seq_len, seq_len]
        window_mask = (distance <= self.window_size).float()  # [seq_len, seq_len]
        
        # Global attention for control opcodes (type 0)
        # Find positions of control opcodes across all samples in batch
        control_positions = (opcode_types == 0).any(dim=0)  # [seq_len]
        
        # Create control mask: all positions can attend to control opcodes
        control_mask = control_positions.unsqueeze(0).float()  # [1, seq_len]
        control_mask = control_mask.expand(seq_len, -1)  # [seq_len, seq_len]
        
        # Combine masks: union of window mask and control mask
        mask = torch.clamp(window_mask + control_mask, 0, 1)  # [seq_len, seq_len]
        
        return mask


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with sparse attention"""
    
    def __init__(self, d_model, n_heads, d_ff, window_size=50, dropout=0.1):
        super().__init__()
        
        # self.sparse_attn = SparseAttention(d_model, n_heads, window_size, dropout)
        self.MHA = StandardMultiHeadAttention(d_model, n_heads, dropout)  # For comparison
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, opcode_types, mask=None):
        # Sparse attention with residual
        # attn_out = self.sparse_attn(x, opcode_types, mask)
        attn_out, _ = self.MHA(x, mask)  # For comparison
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class OpTrans(nn.Module):
    """
    OpTrans: Semantic- and Structure-aware Transformer for Opcode-level 
    Vulnerability Detection (as described in the paper)
    """
    
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, 
                 d_ff=1024, max_len=500, n_types=5, window_size=50, 
                 dropout=0.5, num_classes=2):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Multi-source embeddings (Equation 8)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(n_types, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Add layer normalization after embeddings
        self.embed_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder layers with sparse attention
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, window_size, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, opcode_types):
        """
        Args:
            x: [batch, seq_len] - opcode token IDs
            opcode_types: [batch, seq_len] - semantic type IDs
        """
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Multi-source embedding (Equation 8)
        token_emb = self.token_embedding(x)
        type_emb = self.type_embedding(opcode_types)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings and normalize
        embeddings = token_emb + type_emb + pos_emb
        embeddings = self.embed_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Create padding mask
        padding_mask = (x != 0).float()
        
        # Pass through transformer encoder layers
        hidden = embeddings
        for layer in self.encoder_layers:
            hidden = layer(hidden, opcode_types, padding_mask)
            # Add gradient clipping per layer to prevent explosion
            torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
        
        # Global average pooling (mentioned in paper Section III-D)
        mask_expanded = padding_mask.unsqueeze(-1).expand_as(hidden)
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        avg_hidden = sum_hidden / padding_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Classification
        output = self.fc(self.dropout(avg_hidden))
        
        return output


def prepare_opcode_types(X, vocab_size):
    """
    Prepare opcode type information for the input sequences
    This is a simplified version - you may need to adjust based on your data
    """
    # Create a dummy mapping for demonstration
    # In practice, you should have a proper opcode vocabulary
    op_id = {'push': 1, 'dup': 2, 'swap': 3, 'pop': 4, 'jumpdest': 5, 'add': 6, 'jumpi': 7, 'iszero': 8, 'mstore': 9, 'and': 10, 'mload': 11, 'jump': 12, 'revert': 13, 'sub': 14, 'sload': 15, 'callvalue': 16, 'eq': 17, 'stop': 18, 'return': 19, 'calldataload': 20, 'div': 21, 'calldatasize': 22, 'lt': 23, 'sha3': 24, 'exp': 25, 'mul': 26, 'sstore': 27, 'caller': 28, 'codecopy': 29, 'invalid': 30, 'call': 31, 'gas': 32, 'not': 33, 'gt': 34, 'timestamp': 35, 'or': 36, 'address': 37, 'balance': 38, 'calldatacopy': 39, 'delegatecall': 40, 'returndatasize': 41, 'returndatacopy': 42, 'number': 43, 'log': 44, 'mod': 45, 'blockhash': 46, 'extcodesize': 47, 'difficulty': 48, 'addmod': 49, 'coinbase': 50, 'byte': 51, 'xor': 52, 'sdiv': 53, 'sgt': 54, 'mulmod': 55, 'selfdestruct': 56}
    id_to_opcode = {}
    for k, v in op_id.items():
        id_to_opcode[v] = k.upper()
    
    opcode_types = torch.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            opcode_types[i, j] = get_opcode_type(X[i, j].item(), id_to_opcode)
    
    return opcode_types


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.fc1(h_n[-1])
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# LSTM 分类模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.fc1(h_n[-1])
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def getdataset3(name, generated_num):
    # 读取训练数据
    normal_data = pd.read_csv("./dataset/embedding/smart_contract/normal.csv")
    normal_label = np.zeros(1500)
    vul_data1 = pd.read_csv("./dataset/embedding/smart_contract/reentrancy.csv")
    vul_data2 = pd.read_csv("./dataset/embedding/smart_contract/timestamp.csv")
    vul_data3 = pd.read_csv("./dataset/embedding/smart_contract/delegatecall.csv")
    vul_data4 = pd.read_csv("./dataset/embedding/smart_contract/SBunchecked_low_level_calls.csv")
    unknown_data = pd.read_csv("./dataset/embedding/generated_contract/generated_" + name + ".csv", index_col=0).iloc[:generated_num]
    test_normal_label = np.zeros(1500)
    vul_label = np.ones(vul_data1.shape[0] + vul_data2.shape[0] + vul_data3.shape[0] +
                        vul_data4.shape[0] + unknown_data.shape[0])
    # vul_label = np.concatenate([test_normal_label, vul_label])


    X_train = pd.concat([normal_data[:1500], vul_data1, vul_data2, vul_data3, vul_data4, unknown_data], axis=0)
    y_train = pd.concat([pd.Series(normal_label), pd.Series(vul_label)], axis=0)
    # X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=1)

    # 读取测试数据
    unknown_data1 = pd.read_csv("./dataset/embedding/smart_contract/SBaccess_control.csv")
    unknown_data2 = pd.read_csv("./dataset/embedding/smart_contract/SBarithmetic.csv")
    unknown_data3 = pd.read_csv("./dataset/embedding/smart_contract/SBdenial_of_service.csv")
    unknown_data4 = pd.read_csv("./dataset/embedding/smart_contract/SBshort_address.csv")
    unknown_data5 = pd.read_csv("./dataset/embedding/smart_contract/SBunchecked_low_level_calls.csv")
    # X_test = pd.concat([normal_data.iloc[500:520], unknown_data1, unknown_data2, unknown_data3,
    #                     unknown_data4, unknown_data5], axis=0)
    # vul_label = np.ones(unknown_data1.shape[0] + unknown_data2.shape[0] + unknown_data3.shape[0] +
    #                     unknown_data4.shape[0] + unknown_data5.shape[0])
    # y_test = pd.concat([pd.Series(normal_label[:20]), pd.Series(vul_label)], axis=0)
    X_test = pd.concat([normal_data[-1500:]
                           # , unknown_data1
                           # , unknown_data2
                           # , unknown_data3
                        #, unknown_data4
                        , unknown_data5
                        ],
                       axis=0)
    test_normal_label = np.zeros(1500)
    unknown_label = np.ones(
        # unknown_data1.shape[0] +
        # unknown_data2.shape[0] +
        #                     unknown_data3.shape[0]
                        # + unknown_data4.shape[0] +
                           unknown_data5.shape[0]
                            )
    y_test = np.concatenate([test_normal_label, unknown_label])

    y_train = y_train.values.ravel()  # 为了让其shape为 [3019, 1] ，而不是[3019]
    y_test = y_test.ravel()

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def LSTM_classification(name, generated_num):
    max_len = 500
    max_words = 128
    embedding_dim = 256
    hidden_dim = 128
    output_dim = 2
    batch_size = 128
    epochs = 50

    # 读取数据
    X_train, X_test, y_train, y_test = getdataset3(name, generated_num)
    # X_train, X_test, y_train,  y_test = getdataset3(name, generated_num)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    ## 对数据集的标签数据进行编码
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    le = LabelEncoder()
    train_y = le.fit_transform(y_train).reshape(-1, 1)
    test_y = le.transform(y_test).reshape(-1, 1)
    # 对数据集的标签数据进行one-hot编码
    ohe = OneHotEncoder()
    train_y = ohe.fit_transform(train_y).toarray()
    test_y = ohe.transform(test_y).toarray()
    print(train_y.shape, test_y.shape)
    print(np.argmax(test_y, axis=1))

    """创建模型"""
    from keras.models import Model
    from keras.layers import LSTM, Dense, Dropout, Input, Embedding

    # 定义LSTM模型
    inputs = Input(name='inputs', shape=[max_len])
    # Embedding(词汇表大小,batch大小,每个新闻的词长)
    layer = Embedding(max_words + 1, 128, input_length=max_len)(inputs)
    layer = LSTM(64, name="LSTM")(layer)
    layer = Dense(64, activation="relu", name="FC")(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(2, activation="softmax", name="FC2")(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.summary()

    from keras.optimizers.optimizer_v2.rmsprop import RMSProp
    # from keras import optimizers
    # model.compile(loss="SparseCategoricalCrossentropy", optimizer=rmsprop_v2(), metrics=["accuracy"])
    model.compile(loss="categorical_crossentropy", optimizer=RMSProp(), metrics=["accuracy"])

    """模型的训练和预测"""
    # 模型训练
    model.fit(X_train, train_y, batch_size=128, epochs=100,
                          # validation_data=(val_seq_mat, val_y),
                          # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)]  # 当val_loss不再提升时停止训练
                          )
    # loss = model_fit.history['loss']
    # val_loss = model_fit.history['val_loss']

    # 对测试集进行预测
    test_pre = model.predict(X_test)

    # 使用指标对效果进行验证
    from sklearn import metrics
    print(metrics.classification_report(np.argmax(test_y, axis=1), np.argmax(test_pre, axis=1)))
    a = np.argmax(test_y, axis=1)
    b = np.argmax(test_pre, axis=1)
    cnt=0
    for ii in range(len(a)):
        if a[ii] == b[ii] and a[ii] == 1:
            cnt += 1
    all_ = 0
    for ii in range(len(a)):
        if a[ii] == 1:
            all_ += 1
    print(f"{cnt}/{all_} {cnt*100/all_}%")
    F1 = metrics.classification_report(np.argmax(test_y, axis=1), np.argmax(test_pre, axis=1), output_dict=True)['macro avg']['f1-score']
    print(F1)
    return F1

def OpTrans_classification(name, generated_num):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # Hyperparameters
    max_len = 500
    vocab_size = 128
    d_model = 256
    n_heads = 8
    n_layers = 4
    d_ff = 512
    window_size = 20
    n_types = 5
    batch_size = 32
    epochs = 50
    output_dim = 2
    
    # Load data (using your existing function)
    X_train, X_test, y_train, y_test = getdataset3(name, generated_num)
    
    # Label encoding and one-hot
    le = LabelEncoder()
    y_train = le.fit_transform(y_train).reshape(-1, 1)
    y_test = le.transform(y_test).reshape(-1, 1)
    
    ohe = OneHotEncoder()
    y_train = ohe.fit_transform(y_train).toarray()
    y_test = ohe.transform(y_test).toarray()
    
    # Convert to torch tensors
    X_train = torch.LongTensor(X_train.values)
    X_test = torch.LongTensor(X_test.values)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Prepare opcode types
    opcode_types_train = prepare_opcode_types(X_train, vocab_size + 1)
    opcode_types_test = prepare_opcode_types(X_test, vocab_size + 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, opcode_types_train, y_train)
    test_dataset = TensorDataset(X_test, opcode_types_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    
    # for window_size in [10, 15, 20, 25, 30, 35, 40]:
        # print(f"\n--- Window Size: {window_size} ---")
    model = OpTrans(
        vocab_size=vocab_size + 1,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        n_types=n_types,
        window_size=window_size,
        dropout=0.5,
        num_classes=output_dim
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop with gradient clipping
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

        for X_batch, types_batch, y_batch in progress_bar:
            X_batch = X_batch.to(device)
            types_batch = types_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch, types_batch)
            loss = criterion(output, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 在进度条上动态显示当前 batch loss
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, types_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            types_batch = types_batch.to(device)
            
            output = model(X_batch, types_batch)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            labels = torch.argmax(y_batch, dim=1).numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    f1 = report_dict["macro avg"]["f1-score"]
    print(f"F1 Score: {f1:.4f}")
    
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    return f1

if __name__ == '__main__':
    name = "unknown"
    # f1 = LSTM_classfication(name, 0)
    X = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

    print("=================" + name + "==================")
    F1 = []
    for i in [2500, 3000, 3500, 4000
        #500, 1000, 1500, 2000, 2500,
         #3000, 3500,
        # 4000, 4500, 5000
        ]:
        generated_num = i
        f1 = OpTrans_classification(name, generated_num)
        F1.append(f1)
    # plt.plot(X, F1)
    # plt.show()
