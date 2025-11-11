import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import math
import gc

def getdataset2(name, generated_num):
    normal_data = pd.read_csv("./dataset/embedding/smart_contract/normal.csv")
    normal_label = np.zeros(normal_data.shape[0])
    vul_data = pd.read_csv("./dataset/embedding/smart_contract/" + name + ".csv")
    vul_label = np.ones(vul_data.shape[0])
    generate_data = pd.read_csv("./dataset/embedding/generated_contract/generated_" + name + "_with_sem_gan.csv",
                                index_col=0).iloc[:generated_num]
    generate_label = np.ones(generate_data.shape[0])

    X_train = pd.concat([normal_data[:-vul_data.shape[0]], vul_data[:vul_data.shape[0] // 2], generate_data], axis=0)
    y_train = pd.concat([pd.Series(normal_label[:-vul_data.shape[0]]), pd.Series(vul_label[:vul_data.shape[0] // 2]),
                         pd.Series(generate_label)], axis=0).values
    X_test = pd.concat([normal_data[-vul_data.shape[0]:], vul_data[vul_data.shape[0] // 2:]], axis=0)
    y_test = pd.concat([pd.Series(normal_label[-vul_data.shape[0]:]), pd.Series(vul_label[vul_data.shape[0] // 2:])],
                       axis=0).values

    # data_x1 = pd.concat([normal_data, vul_data], axis=0)
    # data_y1 = pd.concat([pd.Series(normal_label), pd.Series(vul_label)], axis=0)
    # X_train1, X_test, y_train1, y_test = train_test_split(data_x1, data_y1, test_size=0.3, random_state=1)
    #
    # X_train = pd.concat([X_train1, generate_data], axis=0)
    # y_train = pd.concat([y_train1, pd.Series(generate_label)], axis=0).values

    y_train = y_train.ravel()  # 为了让其shape为 [n, 1], 而不是[n]
    y_test = y_test.ravel()

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

# Opcode semantic type mapping (based on Table III in paper)
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

class CallGuidedSparseAttention(nn.Module):
    """
    MODIFIED: Implements the advanced Call-Guided Sparse Attention.
    This combines a base sliding window with a dynamic, learned attention
    for "caller" opcodes to focus on relevant "callee" regions.
    """
    
    def __init__(self, d_model, n_heads, window_size=50, top_k=16, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.top_k = top_k  # NEW: Hyperparameter for guided attention size
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        # Define caller opcodes (e.g., control flow type 0)
        self.caller_opcodes_type = 0
        
    def forward(self, x, opcode_types, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            opcode_types: [batch, seq_len] - semantic types of opcodes
            mask: optional padding mask [batch, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # --- Standard Multi-Head Attention Setup ---
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # --- MODIFIED: Create Dynamic Call-Guided Sparse Mask ---
        # The mask is now created per batch, as it depends on the input `opcode_types`
        attn_mask = self._create_call_guided_mask(seq_len, opcode_types, Q, K, x.device)
        # attn_mask is [batch_size, seq_len, seq_len]
        
        # --- Standard Attention Calculation ---
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply the dynamic sparse mask (needs to be unsqueezed for heads dimension)
        scores = scores.masked_fill(attn_mask.unsqueeze(1) == 0, -1e9)
        
        # Apply padding mask if provided
        if mask is not None:
            # mask is [batch, seq_len], unsqueeze to [batch, 1, 1, seq_len] for broadcasting
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(output)
        
        return output
    
    def _create_call_guided_mask(self, seq_len, opcode_types, Q, K, device):
        """
        MODIFIED: Creates the dynamic call-guided sparse attention mask.
        Combines a sliding window with a learned guided attention for caller opcodes.
        """
        batch_size, n_heads, _, _ = Q.shape

        # --- 1. Base Local Attention (Sliding Window) ---
        positions_i = torch.arange(seq_len, device=device).unsqueeze(1)
        positions_j = torch.arange(seq_len, device=device).unsqueeze(0)
        distance = torch.abs(positions_i - positions_j)
        # window_mask is static: [seq_len, seq_len]
        window_mask = (distance <= self.window_size).float()
        # Expand for batch dimension: [batch_size, seq_len, seq_len]
        final_mask = window_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # --- 2. Call-Guided Attention ---
        # Find positions of "caller" opcodes in the batch: [batch_size, seq_len]
        is_caller = (opcode_types == self.caller_opcodes_type)

        if is_caller.any():
            # Only compute if there are any caller opcodes
            
            # --- a. Target Region Prediction (Equation 9) ---
            # Use the first head's Q and K for simplicity, or average across heads
            # Q_callers: [num_callers, d_k], K_all: [batch, seq_len, d_k]
            q_callers = Q[:, 0, :, :][is_caller] # Simplified: taking first head Q
            k_all = K[:, 0, :, :] # Simplified: taking first head K

            # This part is computationally tricky. A more direct implementation:
            # For each item in batch, find caller indices
            for i in range(batch_size):
                caller_indices = torch.where(is_caller[i])[0]
                if len(caller_indices) == 0:
                    continue

                # Get query vectors for these callers
                q_i = Q[i, :, caller_indices, :].transpose(0, 1) # [n_heads, num_callers_i, d_k]
                k_i = K[i, :, :, :] # [n_heads, seq_len, d_k]

                # Compute target distribution scores (Equation 9)
                # scores_i: [n_heads, num_callers_i, seq_len]
                scores_i = torch.matmul(q_i, k_i.transpose(-2, -1)) / math.sqrt(self.d_k)
                # For simplicity, we average scores across heads to get one distribution
                p_i = torch.softmax(scores_i.mean(dim=0), dim=-1) # [num_callers_i, seq_len]

                # --- b. Identify Top-K Targets ---
                # topk_indices: [num_callers_i, top_k]
                _, topk_indices = torch.topk(p_i, k=self.top_k, dim=-1)

                # --- c. Update the mask for this batch item ---
                for call_idx_in_batch, target_indices in zip(caller_indices, topk_indices):
                    # For each caller, allow attention to its top-k targets
                    final_mask[i, call_idx_in_batch, target_indices] = 1.0

        return final_mask.clamp(0, 1)


class TransformerEncoderLayer(nn.Module):
    """
    UPDATED: Single Transformer encoder layer with Call-Guided sparse attention.
    The `window_size` and `top_k` parameters are passed down.
    """
    
    def __init__(self, d_model, n_heads, d_ff, window_size=50, top_k=16, dropout=0.1):
        super().__init__()
        
        self.sparse_attn = CallGuidedSparseAttention(d_model, n_heads, window_size, top_k, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
            # Removed final dropout to be consistent with original Transformer
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, opcode_types, mask=None):
        attn_out = self.sparse_attn(x, opcode_types, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class OpTrans(nn.Module):
    """
    UPDATED: The main OpTrans class, now configurable with window_size and top_k.
    """
    
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, 
                 d_ff=1024, max_len=500, n_types=5, window_size=50, top_k=16,
                 dropout=0.1, num_classes=2): # Adjusted dropout to be more standard
        super().__init__()
        
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(n_types, d_model)
        # Using learnable positional embeddings, which is more common now
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_norm = nn.LayerNorm(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, window_size, top_k, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, opcode_types):
        batch_size, seq_len = x.shape
        
        padding_mask = (x != 0) # [batch, seq_len]
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        token_emb = self.token_embedding(x)
        type_emb = self.type_embedding(opcode_types)
        pos_emb = self.pos_embedding(positions)
        
        embeddings = self.embed_norm(token_emb + type_emb + pos_emb)
        embeddings = self.embed_dropout(embeddings)
        
        hidden = embeddings
        for layer in self.encoder_layers:
            hidden = layer(hidden, opcode_types, padding_mask)
            # Removed per-layer grad clipping, usually done globally in training loop
        
        # Global average pooling over non-padded tokens
        mask_expanded = padding_mask.unsqueeze(-1).expand_as(hidden)
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        # Ensure non-zero divisor
        sum_mask = padding_mask.sum(dim=1, keepdim=True).clamp(min=1)
        avg_hidden = sum_hidden / sum_mask
        
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
    window_size = 25
    n_types = 5
    batch_size = 32
    epochs = 50
    output_dim = 2
    
    # Load data (using your existing function)
    X_train, X_test, y_train, y_test = getdataset2(name, generated_num)
    
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
    results_filename = "f1_scores.txt"

    # 在循环开始前打开文件（或创建新文件）
    # 使用 'w' 模式会覆盖已存在的文件。如果想追加，可以使用 'a' 模式
    with open(results_filename, 'w') as f:
        f.write("--- F1 Scores for Different Window Sizes ---\n\n")
    
    for window_size in [5, 10, 15, 20, 25, 30, 35]:
        print(f"\n--- Window Size: {window_size} ---")
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
        
        with open(results_filename, 'a') as f:
            f.write(f"Window Size: {window_size}, F1 Score: {f1:.4f}\n")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
        
    
    return f1


if __name__ == '__main__':
    # Example usage
    vulnerability_types = ["reentrancy"] #, "timestamp", "delegatecall", "SBunchecked_low_level_calls" 
    generated_nums = [0, 500, 1000, 1500, 2000]
    
    for vul_type in vulnerability_types:
        print(f"\n{'='*50}")
        print(f"Testing {vul_type}")
        print('='*50)
        
        results = []
        for gen_num in [1000]:
            f1 = OpTrans_classification(vul_type, gen_num)
            results.append(f1)
            print(f"Generated samples: {gen_num}, F1: {f1:.4f}")