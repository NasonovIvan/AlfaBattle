import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm.notebook import tqdm

data_path = '../data/alfabattle2_train_transactions_contest/train_transactions_contest/part_000_0_to_23646.parquet'
df = pd.read_parquet(data_path, engine='pyarrow')

label_encoder_category = LabelEncoder()
label_encoder_mcc = LabelEncoder()
df['mcc_category_encoded'] = label_encoder_category.fit_transform(df['mcc_category'])
df['mcc_encoded'] = label_encoder_mcc.fit_transform(df['mcc'])

def create_sequences_category(data_category, data_mcc, seq_length):
    sequences = []
    for i in range(len(data_category) - seq_length):
        seq = data_category[i:i+seq_length]
        label = data_mcc[i+seq_length]
        sequences.append((seq, label))
    return sequences

SEQ_LENGTH = 6

# Create sequences
sequences = create_sequences_category(df['mcc_category_encoded'].values, df['mcc_encoded'].values, SEQ_LENGTH)

# Split into train and test
train_sequences, test_sequences = train_test_split(sequences[:int(3e5)], test_size=0.2, random_state=42)

class MCCDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
train_dataset = MCCDataset(train_sequences)
test_dataset = MCCDataset(test_sequences)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        x = self.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        output = self.fc2(x)
        return output
    
# Model parameters LSTM
input_size = len(label_encoder_category.classes_)
output_size = len(label_encoder_mcc.classes_)
embedding_dim = 256
hidden_dim = 256
num_layers = 4

model = LSTMModel(input_size, output_size, embedding_dim, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Training loop
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=15):
    model.train()
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        total_loss = 0
        for seq, label in train_loader:
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(train_loader)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for seq, label in test_loader:
            output = model(seq)
            all_labels.extend(label.tolist())
            all_preds.append(output)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.tensor(all_labels)

    # Calculate top-k accuracies
    top_k_accuracies = {}
    for k in [1, 2, 3, 5]:
        top_k_preds = torch.topk(all_preds, k, dim=1).indices
        correct = top_k_preds.eq(all_labels.view(-1, 1).expand_as(top_k_preds))
        top_k_accuracies[f'acc@{k}'] = correct.any(dim=1).float().mean().item()

    return top_k_accuracies

# Train the model
train(model, train_loader, criterion, optimizer, scheduler)

# Evaluation
accuracies = evaluate(model, test_loader)
for k, acc in accuracies.items():
    print(f'{k}: {acc:.4}')
    