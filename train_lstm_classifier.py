import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import joblib

# --------------------------
# 1. Hyperparameters & Config
# --------------------------
CSV_PATH = "advanced_shot_features_labeled.csv"
MODEL_SAVE_PATH = "weights/lstm_shot_classifier.pth"
SCALER_SAVE_PATH = "weights/lstm_scaler.pkl"
WINDOW_SIZE = 15
SEQ_LEN = 2 * WINDOW_SIZE + 1 # 31 frames
INPUT_FEATURES = 3 # arm_angle, sh_width, ball_dist
GLOBAL_FEATURES = 3 # player_x_m, player_y_m, impact_height
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 5 # forehand, backhand, smash, volley, other
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

LABEL_MAP = {
    "other": 0,
    "forehand": 1,
    "backhand": 2,
    "smash": 3,
    "volley": 4
}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# --------------------------
# 2. PyTorch Dataset
# --------------------------
class PadelShotDataset(Dataset):
    def __init__(self, X_seq, X_global, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_global = torch.tensor(X_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_global[idx], self.y[idx]

# --------------------------
# 3. LSTM Model Definition
# --------------------------
class PadelBiomechanicalLSTM(nn.Module):
    def __init__(self, seq_input_size, global_input_size, hidden_size, num_layers, num_classes):
        super(PadelBiomechanicalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for the time-series data
        self.lstm = nn.LSTM(seq_input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Fully connected layers combining LSTM output and global features
        self.fc1 = nn.Linear(hidden_size + global_input_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, seq_x, global_x):
        # seq_x shape: (batch_size, seq_len, seq_input_size)
        lstm_out, (hn, cn) = self.lstm(seq_x)
        
        # Take the output of the last time step
        last_hidden = lstm_out[:, -1, :] # shape: (batch_size, hidden_size)
        
        # Concatenate with global features
        combined = torch.cat((last_hidden, global_x), dim=1) # shape: (batch_size, hidden_size + global_input_size)
        
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out

# --------------------------
# 4. Main Training Script
# --------------------------
def main():
    print("Loading advanced features...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found. Run collect_advanced_shot_data.py and annotate_advanced_shots.py first.")
        return

    print(f"Loaded {len(df)} samples.")
    
    # Check if there are labeled samples
    if 'label' not in df.columns:
        print("Error: No labels found. Please annotate the data first.")
        return
        
    # Map labels to integers
    df['label_id'] = df['label'].map(LABEL_MAP)
    
    # Extract Sequence Features
    # We have arm_angle_X, sh_width_X, ball_dist_X for X in range(-WINDOW_SIZE, WINDOW_SIZE+1)
    seq_cols = []
    for t in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
        seq_cols.append(f"arm_angle_{t}")
        seq_cols.append(f"sh_width_{t}")
        seq_cols.append(f"ball_dist_{t}")
        
    # Extract Global Features
    global_cols = ['player_x_m', 'player_y_m', 'impact_height']
    
    # Handle missing values
    df = df.fillna(0)
    
    # Extract raw numpy arrays
    X_seq_raw = df[seq_cols].values
    X_global_raw = df[global_cols].values
    y_raw = df['label_id'].values
    
    # Reshape sequence data: (N, seq_len, num_features) -> (N, 31, 3)
    N = len(df)
    X_seq_reshaped = np.zeros((N, SEQ_LEN, INPUT_FEATURES))
    for i in range(SEQ_LEN):
        t_offset = i * INPUT_FEATURES
        X_seq_reshaped[:, i, 0] = X_seq_raw[:, t_offset]     # arm_angle
        X_seq_reshaped[:, i, 1] = X_seq_raw[:, t_offset+1]   # sh_width
        X_seq_reshaped[:, i, 2] = X_seq_raw[:, t_offset+2]   # ball_dist
        
    # Normalize features
    scaler_seq = StandardScaler()
    # Flatten to (N*SEQ_LEN, 3) to fit scaler, then reshape back
    X_seq_flat = X_seq_reshaped.reshape(-1, INPUT_FEATURES)
    X_seq_scaled = scaler_seq.fit_transform(X_seq_flat).reshape(N, SEQ_LEN, INPUT_FEATURES)
    
    scaler_global = StandardScaler()
    X_global_scaled = scaler_global.fit_transform(X_global_raw)
    
    # Save scalers for inference
    joblib.dump({"seq": scaler_seq, "global": scaler_global}, SCALER_SAVE_PATH)
    
    # Split Dataset
    X_seq_train, X_seq_test, X_glob_train, X_glob_test, y_train, y_test = train_test_split(
        X_seq_scaled, X_global_scaled, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )
    
    train_dataset = PadelShotDataset(X_seq_train, X_glob_train, y_train)
    test_dataset = PadelShotDataset(X_seq_test, X_glob_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = PadelBiomechanicalLSTM(
        seq_input_size=INPUT_FEATURES, 
        global_input_size=GLOBAL_FEATURES,
        hidden_size=HIDDEN_DIM, 
        num_layers=NUM_LAYERS, 
        num_classes=NUM_CLASSES
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_seq, batch_glob, batch_y in train_loader:
            batch_seq, batch_glob, batch_y = batch_seq.to(device), batch_glob.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_seq, batch_glob)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_seq.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_seq, batch_glob, batch_y in test_loader:
                batch_seq, batch_glob, batch_y = batch_seq.to(device), batch_glob.to(device), batch_y.to(device)
                outputs = model(batch_seq, batch_glob)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_seq.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        val_acc = 100 * correct / total
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss/len(train_loader.dataset):.4f}, Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
        # Save best model
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    import os
    if not os.path.exists('weights'):
        os.makedirs('weights')
    main()
