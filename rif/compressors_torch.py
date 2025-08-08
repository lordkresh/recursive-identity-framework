import torch
import torch.nn as nn
import torch.optim as optim

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, bottleneck_dim=16):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class TorchCompressor:
    def __init__(self, input_dim=128, hidden_dim=64, bottleneck_dim=16, lr=1e-3):
        self.model = AutoEncoder(input_dim, hidden_dim, bottleneck_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def train_batch(self, batch_tensor, epochs=5):
        for _ in range(epochs):
            output = self.model(batch_tensor)
            loss = self.criterion(output, batch_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
    def compress(self, vector):
        with torch.no_grad():
            return self.model.encoder(vector).numpy()
    
    def decompress(self, compressed_vector):
        with torch.no_grad():
            return self.model.decoder(compressed_vector).numpy()
