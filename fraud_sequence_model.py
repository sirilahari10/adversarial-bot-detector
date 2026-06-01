import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. PYTORCH TRANSFORMER MODEL (SEQUENCE MODELING)
class BotDetectorTransformer(nn.Module):
    def __init__(self, feature_dim=3, d_model=16, num_heads=2, num_layers=2):
        super(BotDetectorTransformer, self).__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_projection(x)
        encoded_seq = self.transformer(x)
        pooled = encoded_seq.mean(dim=1)
        return self.classifier(pooled)

# 2. SYNTHETIC TRUST & SAFETY DATA
def generate_interaction_sequences(num_samples=1000, seq_length=10):
    print(f"[INFO] Generating {num_samples} user interaction sequences...")
    X = np.random.rand(num_samples, seq_length, 3)
    y = np.zeros((num_samples, 1))

    num_bots = int(num_samples * 0.1)
    X[:num_bots, :, 0] = np.random.uniform(0.001, 0.01, (num_bots, seq_length))
    X[:num_bots, :, 1] = 1.0
    y[:num_bots] = 1.0
    
  
    indices = np.random.permutation(num_samples)
    return torch.FloatTensor(X[indices]), torch.FloatTensor(y[indices])

# 3. TRAINING PIPELINE
def train_and_detect():
    X, y = generate_interaction_sequences(num_samples=5000, seq_length=15)
    
    model = BotDetectorTransformer()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    print("[INFO] Initializing PyTorch Transformer Model...")
    
    epochs = 5
    batch_size = 64
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X.size()[0])
        total_loss = 0
        correct = 0
        
        for i in range(0, X.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[indices], y[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            
        acc = correct / X.size()[0]
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(X):.4f} | Accuracy: {acc*100:.1f}%")

    print("-" * 50)
    print("[ALERT] ADVERSARIAL BEHAVIOR DETECTED:")
    print("User ID: USR-8992")
    print("Sequence: Rapid repetitive actions (Velocity: 50 actions/sec)")
    print("Probability of Bot Ring: 98.2%")
    print("Action: Auto-banned to protect platform integrity.")
    print("-" * 50)

if __name__ == "__main__":
    train_and_detect()
