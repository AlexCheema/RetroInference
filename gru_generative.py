import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the model
class GRUTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(GRUTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)
        output, hidden = self.gru(embedded, hidden)  # (batch_size, seq_length, hidden_size)
        logits = self.fc(output)  # (batch_size, seq_length, vocab_size)
        return logits, hidden

class TextDataset(Dataset):
    def __init__(self, sentences):
        self.data = []

        for sentence in sentences:
            self.data.append(sentence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

# Training loop
def train(model, data_loader, vocab_size, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(inputs)  # (batch_size, seq_length, vocab_size)
            logits = logits.view(-1, vocab_size)  # Reshape for loss calculation
            targets = targets.view(-1)  # Reshape to match logits
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")

# Example hyperparameters
vocab_size = 27  # a-z and a space character
embed_size = 128
hidden_size = 256
num_layers = 2
batch_size = 64
seq_length = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
model = GRUTextGenerator(vocab_size, embed_size, hidden_size, num_layers)

# Placeholder: Replace with your own DataLoader
sentences = []
with open("simple_sentences.txt", 'r') as f:
    sentences = f.readlines()

# print(sentences)
dataset = TextDataset(sentences)
data_loader = DataLoader(dataset, batch_size, seq_length)

# Example training call
train(model, data_loader, vocab_size, device)
