import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import gc
import math
import torch.nn.functional as F

OPCODE_TYPE_MAP = {
    # Control flow - type 0
    'JUMP': 0, 'JUMPI': 0, 'STOP': 0, 'RETURN': 0, 'REVERT': 0, 'JUMPDEST': 0,
    'CALL': 4, 'CALLCODE': 4, 'DELEGATECALL': 4, 'STATICCALL': 4,
    # Stack operations - type 1
    'PUSH': 1, 'PUSH1': 1, 'PUSH2': 1, 'PUSH3': 1, 'PUSH4': 1, 'PUSH5': 1, 'PUSH6': 1,
    'PUSH7': 1, 'PUSH8': 1, 'PUSH9': 1, 'PUSH10': 1, 'PUSH11': 1, 'PUSH12': 1,
    'PUSH13': 1, 'PUSH14': 1, 'PUSH15': 1, 'PUSH16': 1, 'PUSH17': 1, 'PUSH18': 1,
    'PUSH19': 1, 'PUSH20': 1, 'PUSH21': 1, 'PUSH22': 1, 'PUSH23': 1, 'PUSH24': 1,
    'PUSH25': 1, 'PUSH26': 1, 'PUSH27': 1, 'PUSH28': 1, 'PUSH29': 1, 'PUSH30': 1,
    'PUSH31': 1, 'PUSH32': 1, 'POP': 1,
    'DUP': 1, 'DUP1': 1, 'DUP2': 1, 'DUP3': 1, 'DUP4': 1, 'DUP5': 1, 'DUP6': 1,
    'DUP7': 1, 'DUP8': 1, 'DUP9': 1, 'DUP10': 1, 'DUP11': 1, 'DUP12': 1,
    'DUP13': 1, 'DUP14': 1, 'DUP15': 1, 'DUP16': 1,
    'SWAP':1, 'SWAP1': 1, 'SWAP2': 1, 'SWAP3': 1, 'SWAP4': 1, 'SWAP5': 1, 'SWAP6': 1,
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
        
        self.sparse_attn = SparseAttention(d_model, n_heads, window_size, dropout)
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
        attn_out = self.sparse_attn(x, opcode_types, mask)
        # attn_out, _ = self.MHA(x, mask)  # For comparison
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class OpTrans(nn.Module):
    """
    OpTrans: Semantic- and Structure-aware Transformer for Opcode-level Vulnerability Detection
    """
    
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, 
                 d_ff=1024, max_len=500, n_types=5, window_size=50, 
                 dropout=0.5, num_classes=2):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Multi-source embeddings 
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
        
        # Multi-source embedding
        token_emb = self.token_embedding(x)
        type_emb = self.type_embedding(opcode_types)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings and normalize
        # print( token_emb.shape, type_emb.shape, pos_emb.shape)
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
        
        # Global average pooling 
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


def build_binary_labels(normal_count, vulnerable_count):
    """Build labels in the same order used by the training/test data: normal first, vulnerable second."""
    normal_labels = np.zeros(normal_count)
    vulnerable_labels = np.ones(vulnerable_count)
    return np.concatenate([normal_labels, vulnerable_labels])


def build_test_group_labels(normal_count, vulnerability_counts):
    """Build per-row source labels for the test set in the same order as X_test."""
    group_labels = ["normal"] * normal_count
    for group_name, count in vulnerability_counts.items():
        group_labels.extend([group_name] * count)
    return group_labels


def split_sequence_into_fragments(sequence, fragment_length, fragment_stride=None, pad_value=0):
    """Split one padded contract sequence into fixed-length local fragments."""
    if fragment_length <= 0:
        raise ValueError("fragment_length must be positive")
    if fragment_stride is None:
        fragment_stride = fragment_length
    if fragment_stride <= 0:
        raise ValueError("fragment_stride must be positive")

    sequence_array = np.asarray(sequence)
    dtype = sequence_array.dtype if sequence_array.size else np.int64
    tokens = sequence_array[sequence_array != pad_value]

    if tokens.size <= fragment_length:
        starts = [0]
    else:
        last_start = tokens.size - fragment_length
        starts = list(range(0, last_start + 1, fragment_stride))
        if starts[-1] != last_start:
            starts.append(last_start)

    fragments = np.full((len(starts), fragment_length), pad_value, dtype=dtype)
    for fragment_index, start in enumerate(starts):
        chunk = tokens[start:start + fragment_length]
        fragments[fragment_index, :len(chunk)] = chunk

    return fragments


def fragment_contract_sequences(sequences, labels=None, fragment_length=100, fragment_stride=None, pad_value=0):
    """Expand contract-level rows into fragment-level rows and keep the original contract id."""
    values = sequences.values if hasattr(sequences, "values") else sequences
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    label_values = None if labels is None else np.asarray(labels)
    all_fragments = []
    all_labels = []
    contract_ids = []

    for contract_id, sequence in enumerate(values):
        fragments = split_sequence_into_fragments(
            sequence,
            fragment_length=fragment_length,
            fragment_stride=fragment_stride,
            pad_value=pad_value,
        )
        all_fragments.append(fragments)
        contract_ids.extend([contract_id] * len(fragments))

        if label_values is not None:
            repeated_labels = np.repeat(
                label_values[contract_id:contract_id + 1],
                len(fragments),
                axis=0,
            )
            all_labels.append(repeated_labels)

    if all_fragments:
        fragment_values = np.vstack(all_fragments)
    else:
        fragment_values = np.empty((0, fragment_length), dtype=values.dtype)

    fragment_labels = None
    if label_values is not None:
        fragment_labels = np.concatenate(all_labels, axis=0) if all_labels else label_values[:0]

    return fragment_values, fragment_labels, np.asarray(contract_ids, dtype=np.int64)


def aggregate_fragment_predictions(
    fragment_contract_ids,
    fragment_true_labels,
    fragment_positive_probs,
    threshold=0.5,
    aggregation="max",
):
    """Aggregate fragment-level probabilities back to one prediction per contract."""
    contract_ids = np.asarray(fragment_contract_ids)
    true_labels = np.asarray(fragment_true_labels)
    if true_labels.ndim > 1:
        true_labels = np.argmax(true_labels, axis=1)
    positive_probs = np.asarray(fragment_positive_probs)

    if not (len(contract_ids) == len(true_labels) == len(positive_probs)):
        raise ValueError("fragment ids, labels, and probabilities must have the same length")

    ordered_contract_ids = []
    seen_contract_ids = set()
    for contract_id in contract_ids.tolist():
        if contract_id not in seen_contract_ids:
            ordered_contract_ids.append(contract_id)
            seen_contract_ids.add(contract_id)

    contract_labels = []
    contract_probs = []
    for contract_id in ordered_contract_ids:
        mask = contract_ids == contract_id
        labels_for_contract = true_labels[mask]
        if labels_for_contract.size and not np.all(labels_for_contract == labels_for_contract[0]):
            raise ValueError("fragments from the same contract must share one label")

        probs_for_contract = positive_probs[mask]
        if aggregation == "max":
            contract_prob = float(np.max(probs_for_contract))
        elif aggregation == "mean":
            contract_prob = float(np.mean(probs_for_contract))
        else:
            raise ValueError("aggregation must be 'max' or 'mean'")

        contract_labels.append(int(labels_for_contract[0]) if labels_for_contract.size else 0)
        contract_probs.append(contract_prob)

    contract_probs = np.asarray(contract_probs)
    contract_preds = (contract_probs >= threshold).astype(int)

    return (
        np.asarray(contract_labels, dtype=int),
        contract_preds,
        contract_probs,
        np.asarray(ordered_contract_ids, dtype=np.int64),
    )


def summarize_ood_detection_by_group(group_labels, true_labels, pred_labels, positive_probs):
    """Summarize which out-of-distribution vulnerability groups were detected."""
    group_labels = np.asarray(group_labels)
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    positive_probs = np.asarray(positive_probs)

    rows = []
    for group_name in sorted(set(group_labels)):
        if group_name == "normal":
            continue
        mask = group_labels == group_name
        total = int(mask.sum())
        detected = int(((pred_labels == 1) & mask).sum())
        missed = int(((pred_labels == 0) & mask).sum())
        true_vulnerable = int(((true_labels == 1) & mask).sum())
        avg_positive_prob = float(positive_probs[mask].mean()) if total else 0.0
        rows.append(
            {
                "vulnerability_type": group_name,
                "total": total,
                "true_vulnerable": true_vulnerable,
                "detected": detected,
                "missed": missed,
                "detection_rate": detected / total if total else 0.0,
                "avg_positive_prob": avg_positive_prob,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "total",
                "true_vulnerable",
                "detected",
                "missed",
                "detection_rate",
                "avg_positive_prob",
            ]
        )
    return pd.DataFrame(rows).set_index("vulnerability_type")


def compute_binary_confusion_counts(true_labels, pred_labels):
    """Return TN, FP, FN, TP counts for binary labels where 1 is the positive class."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    tn = int(((true_labels == 0) & (pred_labels == 0)).sum())
    fp = int(((true_labels == 0) & (pred_labels == 1)).sum())
    fn = int(((true_labels == 1) & (pred_labels == 0)).sum())
    tp = int(((true_labels == 1) & (pred_labels == 1)).sum())
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def summarize_confusion_by_group(group_labels, true_labels, pred_labels):
    """Summarize TN/FP/FN/TP counts for each test data source."""
    group_labels = np.asarray(group_labels)
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    rows = []
    for group_name in sorted(set(group_labels)):
        mask = group_labels == group_name
        counts = compute_binary_confusion_counts(true_labels[mask], pred_labels[mask])
        total = int(mask.sum())
        correct = counts["TN"] + counts["TP"]
        rows.append(
            {
                "group": group_name,
                "total": total,
                **counts,
                "accuracy": correct / total if total else 0.0,
            }
        )

    return pd.DataFrame(rows).set_index("group")


def encode_labels_one_hot(y_train, y_val, y_test=None):
    """Apply the existing LabelEncoder + OneHotEncoder pipeline to train, validation, and test labels."""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train).reshape(-1, 1)
    y_val_encoded = label_encoder.transform(y_val).reshape(-1, 1)

    one_hot_encoder = OneHotEncoder()
    y_train_one_hot = one_hot_encoder.fit_transform(y_train_encoded).toarray()
    y_val_one_hot = one_hot_encoder.transform(y_val_encoded).toarray()

    if y_test is None:
        return y_train_one_hot, y_val_one_hot

    y_test_encoded = label_encoder.transform(y_test).reshape(-1, 1)
    y_test_one_hot = one_hot_encoder.transform(y_test_encoded).toarray()
    return y_train_one_hot, y_val_one_hot, y_test_one_hot


def _to_long_tensor(frame_or_array):
    values = frame_or_array.values if hasattr(frame_or_array, "values") else frame_or_array
    return torch.LongTensor(values)


def make_optrans_dataloaders(X_train, *args, batch_size, vocab_size):
    """Create the train/validation/test DataLoader objects expected by OpTrans_classification."""
    if len(args) == 3:
        X_test, y_train, y_test = args
        X_train_tensor = _to_long_tensor(X_train)
        X_test_tensor = _to_long_tensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train)
        y_test_tensor = torch.FloatTensor(y_test)

        opcode_types_train = prepare_opcode_types(X_train_tensor, vocab_size + 1)
        opcode_types_test = prepare_opcode_types(X_test_tensor, vocab_size + 1)

        train_dataset = TensorDataset(X_train_tensor, opcode_types_train, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, opcode_types_test, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    if len(args) != 5:
        raise TypeError(
            "make_optrans_dataloaders expects either "
            "(X_train, X_test, y_train, y_test) or "
            "(X_train, X_val, X_test, y_train, y_val, y_test)"
        )

    X_val, X_test, y_train, y_val, y_test = args
    X_train_tensor = _to_long_tensor(X_train)
    X_val_tensor = _to_long_tensor(X_val)
    X_test_tensor = _to_long_tensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train)
    y_val_tensor = torch.FloatTensor(y_val)
    y_test_tensor = torch.FloatTensor(y_test)

    opcode_types_train = prepare_opcode_types(X_train_tensor, vocab_size + 1)
    opcode_types_val = prepare_opcode_types(X_val_tensor, vocab_size + 1)
    opcode_types_test = prepare_opcode_types(X_test_tensor, vocab_size + 1)

    train_dataset = TensorDataset(X_train_tensor, opcode_types_train, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, opcode_types_val, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, opcode_types_test, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def make_fragmented_optrans_dataloaders(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    batch_size,
    vocab_size,
    fragment_length=100,
    fragment_stride=None,
):
    """Create fragment-level DataLoaders and keep contract ids for aggregation."""
    X_train_fragments, y_train_fragments, train_contract_ids = fragment_contract_sequences(
        X_train,
        y_train,
        fragment_length=fragment_length,
        fragment_stride=fragment_stride,
    )
    X_val_fragments, y_val_fragments, val_contract_ids = fragment_contract_sequences(
        X_val,
        y_val,
        fragment_length=fragment_length,
        fragment_stride=fragment_stride,
    )
    X_test_fragments, y_test_fragments, test_contract_ids = fragment_contract_sequences(
        X_test,
        y_test,
        fragment_length=fragment_length,
        fragment_stride=fragment_stride,
    )

    train_loader, val_loader, test_loader = make_optrans_dataloaders(
        X_train_fragments,
        X_val_fragments,
        X_test_fragments,
        y_train_fragments,
        y_val_fragments,
        y_test_fragments,
        batch_size=batch_size,
        vocab_size=vocab_size,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_contract_ids,
        val_contract_ids,
        test_contract_ids,
    )


def evaluate_optrans_loss(model, data_loader, criterion, device):
    """Evaluate average loss on a validation or test loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, types_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            types_batch = types_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(X_batch, types_batch)
            loss = criterion(output, torch.argmax(y_batch, dim=1))
            total_loss += loss.item()

    return total_loss / len(data_loader)


def collect_optrans_predictions(model, test_loader, device):
    """Run OpTrans evaluation and return true labels, predicted labels, and positive-class probabilities."""
    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for X_batch, types_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            types_batch = types_batch.to(device)

            output = model(X_batch, types_batch)
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())

            preds = torch.argmax(output, dim=1).cpu().numpy()
            labels = torch.argmax(y_batch, dim=1).numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def getdataset3(name, generated_num, return_group_labels=False):
    normal_data = pd.read_csv("./dataset/embedding/smart_contract/normal_all.csv")
    vul_data1 = pd.read_csv("./dataset/embedding/smart_contract/reentrancy.csv")
    vul_data2 = pd.read_csv("./dataset/embedding/smart_contract/timestamp.csv")
    vul_data3 = pd.read_csv("./dataset/embedding/smart_contract/delegatecall.csv")
    unknown_data = pd.read_csv("./dataset/embedding/generated_contract/generated_interpolated_ood.csv").iloc[:generated_num]
    extreme_data = pd.read_csv("./dataset/embedding/generated_contract/extreme_anomaly_perturbed.csv").iloc[:generated_num]
    vulnerable_train_count = (
        vul_data1.shape[0] + vul_data2.shape[0] + vul_data3.shape[0]
        + unknown_data.shape[0]
        + extreme_data.shape[0]
    )
    train_normal_num = 20000 

    X_train = pd.concat([normal_data[:train_normal_num], vul_data1, vul_data2, vul_data3, unknown_data, extreme_data], axis=0) # , vul_data5 vul_data4, 
    y_train = pd.Series(build_binary_labels(train_normal_num, vulnerable_train_count))

    # 读取测试数据
    unknown_data1 = pd.read_csv("./dataset/embedding/smart_contract/SBaccess_control.csv")
    unknown_data2 = pd.read_csv("./dataset/embedding/smart_contract/SBarithmetic.csv")
    unknown_data3 = pd.read_csv("./dataset/embedding/smart_contract/SBdenial_of_service.csv")
    unknown_data4 = pd.read_csv("./dataset/embedding/smart_contract/SBshort_address.csv")
    unknown_data5 = pd.read_csv("./dataset/embedding/smart_contract/SBunchecked_low_level_calls.csv")

    test_normal_num = 250
    X_test = pd.concat([normal_data[-test_normal_num:],
                        unknown_data1,
                        unknown_data2,
                        unknown_data3,
                        unknown_data4,
                        unknown_data5
                        ],
                       axis=0)
    vulnerability_test_counts = {
        "SBaccess_control": unknown_data1.shape[0],
        "SBarithmetic": unknown_data2.shape[0],
        "SBdenial_of_service": unknown_data3.shape[0],
        "SBshort_address": unknown_data4.shape[0],
        "SBunchecked_low_level_calls": unknown_data5.shape[0],
    }
    unknown_test_count = sum(vulnerability_test_counts.values())
    y_test = build_binary_labels(test_normal_num, unknown_test_count)
    test_group_labels = build_test_group_labels(test_normal_num, vulnerability_test_counts)

    y_train = y_train.values.ravel()  
    y_test = y_test.ravel()

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    if return_group_labels:
        return X_train, X_test, y_train, y_test, np.array(test_group_labels)
    return X_train, X_test, y_train, y_test

def OpTrans_classification(
    name,
    generated_num,
    fragment_length=40,
    fragment_stride=20,
    contract_threshold=0.5,
    contract_aggregation="max",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print("Using device:", device)
    
    # Hyperparameters
    max_len = fragment_length
    vocab_size = 128
    d_model = 512
    n_heads = 8 
    n_layers = 4
    d_ff = 512
    window_size = 20
    n_types = 5
    batch_size = 256
    epochs = 200
    validation_size = 0.2
    early_stop_patience = 8
    early_stop_min_delta = 1e-4
    output_dim = 2
    
    # Load data (using your existing function)
    X_train, X_test, y_train, y_test, test_group_labels = getdataset3(
        name,
        generated_num,
        return_group_labels=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=42,
        stratify=y_train,
    )
    print(
        f"Train/Val/Test sizes: "
        f"{len(y_train)}/{len(y_val)}/{len(y_test)}"
    )

    y_train, y_val, y_test = encode_labels_one_hot(y_train, y_val, y_test)
    print(
        f"Fragment mode: length={fragment_length}, "
        f"stride={fragment_stride}, aggregation={contract_aggregation}"
    )
    (
        train_loader,
        val_loader,
        test_loader,
        train_contract_ids,
        val_contract_ids,
        test_contract_ids,
    ) = make_fragmented_optrans_dataloaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        batch_size=batch_size,
        vocab_size=vocab_size,
        fragment_length=fragment_length,
        fragment_stride=fragment_stride,
    )
    print(
        f"Train/Val/Test fragments: "
        f"{len(train_contract_ids)}/{len(val_contract_ids)}/{len(test_contract_ids)}"
    )
    
    # Initialize model
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
    
    # Training loop with validation and early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
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

            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = evaluate_optrans_loss(model, val_loader, criterion, device)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_optrans_model.pth")
            best_marker = " *best*"
        else:
            epochs_without_improvement += 1
            best_marker = ""

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"No Improve: {epochs_without_improvement}/{early_stop_patience}"
            f"{best_marker}"
        )

        if epochs_without_improvement >= early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best Val Loss: {best_val_loss:.4f}"
            )
            break
    
    # Validation and final test evaluation
    model.load_state_dict(torch.load("best_optrans_model.pth", map_location=device))

    val_fragment_labels, _, val_fragment_probs = collect_optrans_predictions(model, val_loader, device)
    val_labels, val_preds, _, _ = aggregate_fragment_predictions(
        val_contract_ids,
        val_fragment_labels,
        val_fragment_probs,
        threshold=contract_threshold,
        aggregation=contract_aggregation,
    )
    print("\nContract-level Validation Classification Report:")
    print(classification_report(val_labels, val_preds, zero_division=0))

    fragment_labels, fragment_preds, fragment_probs = collect_optrans_predictions(model, test_loader, device)
    print("\nFragment-level Test Classification Report:")
    print(classification_report(fragment_labels, fragment_preds, zero_division=0))

    all_labels, all_preds, all_probs, test_contract_indices = aggregate_fragment_predictions(
        test_contract_ids,
        fragment_labels,
        fragment_probs,
        threshold=contract_threshold,
        aggregation=contract_aggregation,
    )
    test_group_labels = np.asarray(test_group_labels)[test_contract_indices]
    
    # Print results
    print("\nContract-level Test Classification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    f1 = report_dict["macro avg"]["f1-score"]
    print(f"F1 Score: {f1:.4f}")

    confusion_counts = compute_binary_confusion_counts(all_labels, all_preds)
    print("\nOverall TN/FP/FN/TP:")
    print(
        f"TN={confusion_counts['TN']}, "
        f"FP={confusion_counts['FP']}, "
        f"FN={confusion_counts['FN']}, "
        f"TP={confusion_counts['TP']}"
    )

    confusion_by_group = summarize_confusion_by_group(
        test_group_labels,
        all_labels,
        all_preds,
    )
    print("\nTN/FP/FN/TP by test group:")
    print(confusion_by_group.to_string(float_format=lambda value: f"{value:.4f}"))
    confusion_by_group.to_csv("test_group_confusion_summary.csv")

    ood_summary = summarize_ood_detection_by_group(
        test_group_labels,
        all_labels,
        all_preds,
        all_probs,
    )
    print("\nOut-of-distribution vulnerability detection by type:")
    print(ood_summary.to_string(float_format=lambda value: f"{value:.4f}"))
    ood_summary.to_csv("ood_vulnerability_detection_summary.csv")
    
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    return f1

if __name__ == '__main__':
    name = "unknown"

    print("=================" + name + "==================")
    generated_num = 5000
    OpTrans_classification(name, generated_num)
