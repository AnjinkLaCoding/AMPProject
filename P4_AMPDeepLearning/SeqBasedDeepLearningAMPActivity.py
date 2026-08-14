#The layers
"""
Embedding          - turns each amino acid letter into a small vector of
                        numbers so the network can do math with it.
Conv1d             - slides a small window (kernel_size amino acids) along
                        the sequence, learning to detect local motifs/patterns.
Global Max Pooling - for each learned pattern, keep only the strongest
                        response found anywhere in the sequence. This is what
                        collapses a variable-length sequence into one
                        fixed-size vector -- no LSTM/attention needed.
Dense layers        - combine the detected pattern signals into a
                        prediction for each activity.
 
Requires: torch, numpy, pandas, scikit-learn
    pip install torch numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import glob
import os

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}


df_AntiBac = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/AntiBacterial.csv')
df_AntiVir = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antiviral.csv')
df_AntiDiab = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antidiabetic.csv')
df_AntiOxi = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antioxidant.csv')
df_AntiParas = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antiparasitic.csv')
df_AntiMic = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antimicrobial.csv')
df_AntiInfla = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antiinflamatory.csv')
df_AntiFun = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Antifungal.csv')
df_AntiCan = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/AntiCancer.csv')
df_Neurotoxin = pd.read_csv('C:/Users/Matthew/Downloads/AMPProj1/Neurotoxin.csv')

#Put everything into a dataframe
dfs = []
for filepath in glob.glob('*.csv'):  # adjust folder path if needed
    temp = pd.read_csv(filepath)
    activity_name = os.path.splitext(os.path.basename(filepath))[0]  # e.g. "Antiviral"
    temp['activity'] = activity_name
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)
print(df.head())
#Dataframe format:
#                                sequence  label       activity
#0               LKRLWKRLFKILKRYYRYLRRPVR      1  AntiBacterial
#1  KAIQTAQGVVAVAPGAKIIGDRINQGVKEIKKFLKWK      1  AntiBacterial
#2                          FLSGIVGMLAKLF      1  AntiBacterial
#3                     KIGAKIKIGAKIKIGAKI      1  AntiBacterial
#4                                   AEAM      1  AntiBacterial

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD, UNK = "<PAD>", "<UNK>" #PAD used to add padding later to have the same size for all peptide, UNK -> Unknown peptide
VOCAB = [PAD, UNK] + list(AMINO_ACIDS) #['<PAD>', '<UNK>', 'A', 'C', 'D', 'E', 'F', ...] — 22 tokens total (2 special + 20 amino acids)
AA2IDX = {aa: i for i, aa in enumerate(VOCAB)} #{'<PAD>': 0, '<UNK>': 1, 'A': 2, 'C': 3, ...}. Neural networks work with numbers, not letters
PAD_IDX = AA2IDX[PAD] #which will be 0, tells PyTorch "this index means padding, don't bother learning a meaningful vector for it"
 
 
def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    seq = seq.strip().upper()[:max_len] #removes stray whitespace, forces uppercase
    #Walks through the sequence letter by letter, looks each one up in the dictionary, and converts it to its integer index
    #if the letter isn't in the vocab (not a peptide), use <UNK>'s index instead of crashing. "GLF" → [8, 13, 6] = numbered according to the alphabet order + 1
    ids = [AA2IDX.get(aa, AA2IDX[UNK]) for aa in seq]
    ids += [PAD_IDX] * (max_len - len(ids)) #Add padding (0) if the length of the sequence is smaller than max_len
    return np.array(ids, dtype=np.int64) # Converts the Python list to a NumPy integer array
 
 
# --------------------------------------------------------------------------------------
# 2. Long -> wide pivot
# --------------------------------------------------------------------------------------
 
def pivot_long_to_wide(df, seq_col="sequence", activity_col="activity", label_col="label"):
    #df = df[[seq_col, activity_col, label_col]].dropna(subset=[seq_col, activity_col]).copy()
    df[label_col] = df[label_col].astype(float)
    # normalize activity strings so casing/whitespace differences don't fragment categories
    df[activity_col] = df[activity_col].astype(str).str.strip().str.lower()
    df[seq_col] = df[seq_col].astype(str).str.strip().str.upper()  # abc -> ABC
 
    wide = df.pivot_table(index=seq_col, columns=activity_col, values=label_col, aggfunc="max")
    # How the wide will look like:
    # sequence	antibacterial	antifungal	antiviral
    # GIGKFLH	    NaN	            1	        NaN
    # KAIQTAQ	      1	        NaN	          1          and so on for all activities
    # LKRLWKR	      1	            0	        NaN
    activities = list(wide.columns)
    sequences = list(wide.index)
 
    mask_matrix = (~wide.isna()).astype(np.float32).values # To check if label is known later, ex : [[1.0,0.0,...],[0.0,1.0,..],..]
    label_matrix = wide.fillna(0.0).astype(np.float32).values # Fill label with 0 if the value checked is NaN, ex : [[1.0,0.0,...],[0.0,1.0,..],..]
    return sequences, label_matrix, mask_matrix, activities
 
 
class AMPDataset(Dataset):
    def __init__(self, sequences, label_matrix, mask_matrix, max_len):
        self.sequences = sequences
        self.labels = label_matrix
        self.mask = mask_matrix
        self.max_len = max_len
 
    def __len__(self):
        return len(self.sequences)
 
    def __getitem__(self, idx):
        x = encode_sequence(self.sequences[idx], self.max_len) #Encode a sequence, make it into a vector rather than a string of letters
        return (torch.from_numpy(x),
                torch.from_numpy(self.labels[idx]).float(),
                torch.from_numpy(self.mask[idx]).float())
        #Returns ( sequence_tensor, tensor([1., 0., 1., 0.]), tensor([1., 1., 0., 1.]))
 
# --------------------------------------------------------------------------------------
# 3. THE SIMPLE MODEL
# --------------------------------------------------------------------------------------
 
class SimpleAMPNet(nn.Module):
    def __init__(self, vocab_size, num_activities, embed_dim=32,
                 conv_channels=64, kernel_size=5, dropout=0.3):
        super().__init__()
        # Your sequence initially consists of integer IDs, [5 10 15 5 20 10]
        # The embedding converts those integers into vectors
        # [0.12, -0.32, 0.45, ..., 0.17] -> vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
 
        # One convolution layer: slides `kernel_size`-wide windows across the sequence
        # The "padding=kernel_size // 2" This helps maintain approximately the same sequence length after convolution.
        self.conv = nn.Conv1d(embed_dim, conv_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.relu = nn.ReLU() #Activation function
 
        # Two dense layers turn the pooled pattern-vector into per-activity logits
        self.fc1 = nn.Linear(conv_channels, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_activities)
 
    def forward(self, x):
        # x: (batch, seq_len) integer amino-acid ids
        emb = self.embedding(x)                 # (B, T, embed_dim)
        conv_in = emb.transpose(1, 2)            # (B, embed_dim, T) -- Conv1d wants channels first
        conv_out = self.relu(self.conv(conv_in))  # (B, conv_channels, T)
 
        pooled, _ = conv_out.max(dim=2)           # (B, conv_channels) -- global max pooling over T
 
        h = self.relu(self.fc1(pooled))
        h = self.dropout(h)
        logits = self.fc2(h)                      # (B, num_activities) raw logits
        return logits
 
 
# --------------------------------------------------------------------------------------
# 4. Masked multi-label loss (identical concept to the fuller version)
# --------------------------------------------------------------------------------------
 
def masked_bce_loss(logits, targets, mask):
    per_elem = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_elem = per_elem * mask
    return per_elem.sum() / mask.sum().clamp(min=1.0)
 
 
# --------------------------------------------------------------------------------------
# 5. Training loop
# --------------------------------------------------------------------------------------
 
def train_model(df, seq_col="sequence", activity_col="activity", label_col="label",
                 max_len=None, batch_size=32, epochs=30, lr=1e-3, val_size=0.15, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
 
    sequences, label_matrix, mask_matrix, activities = pivot_long_to_wide(
        df, seq_col, activity_col, label_col
    )
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    print(f"{len(sequences)} unique sequences, {len(activities)} activities: {activities}")
 
    idx_train, idx_val = train_test_split(np.arange(len(sequences)), test_size=val_size,
                                           random_state=42)
 
    def subset(idxs):
        return ([sequences[i] for i in idxs], label_matrix[idxs], mask_matrix[idxs])
 
    train_seqs, train_y, train_m = subset(idx_train)
    val_seqs, val_y, val_m = subset(idx_val)

    # Load the data
    train_loader = DataLoader(AMPDataset(train_seqs, train_y, train_m, max_len),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AMPDataset(val_seqs, val_y, val_m, max_len),
                             batch_size=batch_size, shuffle=False)
 
    model = SimpleAMPNet(vocab_size=len(VOCAB), num_activities=len(activities)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
 
    best_val_loss = float("inf")
    best_state = None
 
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            loss = masked_bce_loss(model(x), y, m)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                val_loss += masked_bce_loss(model(x), y, m).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
 
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
 
    model.load_state_dict(best_state)
    metrics = evaluate(model, val_loader, activities, device)
    return model, activities, max_len, metrics
 
 
# --------------------------------------------------------------------------------------
# 6. Evaluation
# --------------------------------------------------------------------------------------
 
def evaluate(model, loader, activities, device):
    model.eval()
    all_logits, all_y, all_m = [], [], []
    with torch.no_grad():
        for x, y, m in loader:
            all_logits.append(model(x.to(device)).cpu())
            all_y.append(y)
            all_m.append(m)
    logits = torch.cat(all_logits).numpy()
    y = torch.cat(all_y).numpy()
    m = torch.cat(all_m).numpy()
    probs = 1 / (1 + np.exp(-logits))
 
    results = {}
    for i, act in enumerate(activities):
        known = m[:, i] == 1
        if known.sum() < 2 or len(np.unique(y[known, i])) < 2:
            results[act] = {"auc": None, "f1": None, "n": int(known.sum())}
            continue
        yi, pi = y[known, i], probs[known, i]
        results[act] = {"auc": round(roc_auc_score(yi, pi), 3),
                         "f1": round(f1_score(yi, (pi >= 0.5).astype(int)), 3),
                         "n": int(known.sum())}
 
    print("\nPer-activity validation metrics:")
    for act, m_ in results.items():
        print(f"  {act:15s} n={m_['n']:4d}  AUC={m_['auc']}  F1={m_['f1']}")
    return results
 
 
# --------------------------------------------------------------------------------------
# 7. Inference
# --------------------------------------------------------------------------------------
 
def predict_activities(model, sequence, activities, max_len, device="cpu", threshold=0.5):
    model.eval()
    x = torch.from_numpy(encode_sequence(sequence, max_len)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()
    return {act: (float(p), bool(p >= threshold)) for act, p in zip(activities, probs)}

model, activities, max_len, metrics = train_model(df, epochs=5)

#The result after evaluation:
# Per-activity validation metrics:
  #antibacterial   n=4262  AUC=0.951  F1=0.879
  #anticancer      n=1762  AUC=0.949  F1=0.864
  #antidiabetic    n= 464  AUC=0.686  F1=0.675
  #antifungal      n=1972  AUC=0.947  F1=0.871
  #antiinflamatory n=1201  AUC=0.73  F1=0.729
  #antimicrobial   n=7867  AUC=0.932  F1=0.845
  #antioxidant     n= 350  AUC=0.667  F1=0.656
  #antiparasitic   n=1035  AUC=0.963  F1=0.887
  #antiviral       n=1185  AUC=0.873  F1=0.763
  #neurotoxin      n= 478  AUC=0.638  F1=0.448

# Strong performers (AUC 0.87–0.96): antibacterial, anticancer, antifungal, antimicrobial, antiparasitic, antiviral. Tend to have more data except antiparasitic
# Weak performers (AUC 0.64–0.73): antidiabetic, antiinflammatory, antioxidant, neurotoxin. They got less n than other activities (less data)
# neurotoxin, AUC=0.638 but F1=0.448 — that's an unusually large gap between the two. Usually means unbalanced data between 1 and 0