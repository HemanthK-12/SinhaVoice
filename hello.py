import os
import torch
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import string

SAMPLE_RATE = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHARS = " අආඇඈඉඊඋඌඍඎඏඐඑඒඓඔඕඖකඛගඝඞඟචඡජඣඤඥඦටඨඩඪණඬතථදධනඳපඵබභමඹයරලවශෂසහළෆ්ාැෑිීුූෘෙේෛොෝෞං"
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}
char2idx["<pad>"] = 0
idx2char = {i: c for c, i in char2idx.items()}

def text_to_indices(text):
    return torch.tensor([char2idx[c] for c in text if c in char2idx], dtype=torch.long)

def extract_features(waveform):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        win_length=400
    )(waveform)
    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return log_mel_spec.transpose(0, 1)

class SpeechDataset(Dataset):
    def __init__(self, audio_dir, tsv_file):
        with open(tsv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        self.data = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                utt_id = parts[0]
                transcription = parts[2]
                audio_path = os.path.join(audio_dir, f"{utt_id}.flac")
                if os.path.exists(audio_path):
                    self.data.append((audio_path, transcription))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, text = self.data[idx]
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        features = extract_features(waveform.squeeze(0))
        target = text_to_indices(text)
        return features, target

def collate_fn(batch):
    features, targets = zip(*batch)
    input_lengths = torch.tensor([f.shape[0] for f in features])
    target_lengths = torch.tensor([len(t) for t in targets])
    features = pad_sequence(features, batch_first=True)
    targets = pad_sequence(targets, batch_first=False)
    return features, targets, input_lengths, target_lengths
 
class TDNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increase dropout
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)  # Increase dropout
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=3,
                           batch_first=True, bidirectional=True,
                           dropout=0.3)  # Increase dropout
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tdnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.fc(x)

def validate(model, dataloader):
    model.eval()
    total_loss = 0
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    with torch.no_grad():
        for features, targets, input_lengths, target_lengths in dataloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            outputs = model(features)
            log_probs = nn.functional.log_softmax(outputs, dim=-1)
            targets_flat = targets.transpose(0, 1).contiguous().view(-1)
            targets_flat = targets_flat[targets_flat != 0]
            loss = ctc_loss(log_probs.transpose(0, 1), targets_flat, input_lengths, target_lengths)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(model, train_loader, val_loader, optimizer, scheduler, epochs=50):
    best_val_loss = float('inf')
    patience = 10  # Increased from 5
    patience_counter = 0
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Add warmup
    warmup_epochs = 3
    warmup_lr_multiplier = 0.1
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Learning rate warmup
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = optimizer.param_groups[0]['lr'] * \
                    (1 + epoch * (1 - warmup_lr_multiplier) / warmup_epochs)
        
        for i, (features, targets, input_lengths, target_lengths) in enumerate(train_loader):
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            log_probs = nn.functional.log_softmax(outputs, dim=-1)
            targets_flat = targets.transpose(0, 1).contiguous().view(-1)
            targets_flat = targets_flat[targets_flat != 0]
            loss = ctc_loss(log_probs.transpose(0, 1), targets_flat, input_lengths, target_lengths)
            loss.backward()
            
            # More conservative gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Adjust learning rate with smaller factor
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'char2idx': char2idx,
                'idx2char': idx2char,
                'epoch': epoch,
                'val_loss': val_loss,
                'history': history
            }, 'sinhala_asr_model_best.pth')
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

def main():
    audio_dir = "./audio/"
    tsv_file = "test.tsv"

    print(f"Loading dataset from {audio_dir} using {tsv_file}")
    dataset = SpeechDataset(audio_dir, tsv_file)
    print(f"Found {len(dataset)} valid audio-transcript pairs")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=6
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=6
    )

    input_dim = 80
    hidden_dim = 256
    output_dim = len(char2idx)

    print(f"Initializing model with:")
    print(f"- Input dim: {input_dim}")
    print(f"- Hidden dim: {hidden_dim}")
    print(f"- Output dim: {output_dim}")
    print(f"- Device: {DEVICE}")

    model = TDNN_LSTM_Model(input_dim, hidden_dim, output_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=8,  # Increased patience
        factor=0.2,  # Smaller reduction factor
        min_lr=1e-6,
        verbose=True
    )


    print("\nStarting training...")
    try:
        train(model, train_loader, val_loader, optimizer, scheduler, epochs=50)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'char2idx': char2idx,
            'idx2char': idx2char
        }, 'sinhala_asr_model_interrupted.pth')
        print("Model saved to sinhala_asr_model_interrupted.pth")

if __name__ == "__main__":
    main()
