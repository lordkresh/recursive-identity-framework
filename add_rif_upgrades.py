# add_rif_upgrades.py
# Run this file once from the root of your repo to add new RIF features.

import os
import nbformat as nbf

# --- 1. Create PyTorch Autoencoder Compressor ---
torch_code = """import torch
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
"""

os.makedirs("rif", exist_ok=True)
with open("rif/compressors_torch.py", "w") as f:
    f.write(torch_code)

# --- 2. Create Jupyter Notebook Visualizer ---
os.makedirs("notebooks", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# RIF Audit Log Visualizer\nThis notebook plots identity drift and compression ratios over time."),
    nbf.v4.new_code_cell("""import json
import matplotlib.pyplot as plt

# Load audit log
with open('../demo/audit_log.json') as f:
    audit_entries = json.load(f)

# Extract metrics
timestamps = [entry['timestamp'] for entry in audit_entries]
drifts = [entry['drift'] for entry in audit_entries]
ratios = [entry['compression_ratio'] for entry in audit_entries]

# Plot drift
plt.figure(figsize=(10,4))
plt.plot(timestamps, drifts, marker='o')
plt.title('Identity Drift Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Drift Magnitude')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot compression ratio
plt.figure(figsize=(10,4))
plt.plot(timestamps, ratios, marker='x', color='orange')
plt.title('Compression Ratio Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Ratio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
""")
]

with open("notebooks/audit_visualizer.ipynb", "w") as f:
    nbf.write(nb, f)

print("✅ PyTorch compressor and visualizer notebook created successfully.")
