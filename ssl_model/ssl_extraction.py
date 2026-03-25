import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Custom PyTorch Dataset ---
class BioFeatureDataset(Dataset):
    def __init__(self, features, labels):
        # Convert to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        # Labels need to be float for Binary Cross Entropy Loss
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 2. The Lightweight MLP Architecture ---
class BioMLP(nn.Module):
    def __init__(self, input_dim):
        super(BioMLP, self).__init__()
        # A simple, lightweight feed-forward network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16), # Normalizes data to speed up training
            nn.Dropout(0.3),    # Prevents overfitting
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

# --- 3. Main Training Loop ---
def main():
    print("Loading extracted features...")
    
    # Load the CSV you generated in the previous step
    df = pd.read_csv("./train_bio_features_9_attrs.csv")
    
    # Separate features and labels
    # We drop 'filename' because it's a string, and 'label' is our target
    X = df[['j_local', 'j_abs', 'j_rap', 'j_ppq5', 's_local', 's_db', 's_apq3', 's_apq5', 'hnr']].values
    y = df['label'].values
    
    # Scale the features (crucial for neural networks so one large value doesn't dominate)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split 80% for training and 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Create DataLoaders
    train_dataset = BioFeatureDataset(X_train, y_train)
    val_dataset = BioFeatureDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[1] # Will be 3 (jitter, shimmer, hnr)
    model = BioMLP(input_dim)
        
    # Calculate the ratio of Fake (0) vs Real (1) in the training set
    num_reals = (y_train == 1).sum()
    num_fakes = (y_train == 0).sum()
    pos_weight = torch.tensor([num_fakes / num_reals], dtype=torch.float32)

    # Use BCEWithLogitsLoss which takes un-sigmoid outputs and handles weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Training Phase
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()                 # Clear old gradients
            outputs = model(batch_features)       # Forward pass
            loss = criterion(outputs, batch_labels) # Calculate loss
            loss.backward()                       # Backpropagation
            optimizer.step()                      # Update weights
            
            running_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # No need to track gradients for validation
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                # If probability > 0.5, predict Real (1), else predict Fake (0)
                predicted = (outputs > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
    # --- 4. Save the Model ---
    torch.save(model.state_dict(), "bio_mlp_weights.pt")
    print("\nTraining complete. Weights saved to 'bio_mlp_weights.pt'")

if __name__ == "__main__":
    main()