import torch
import torchaudio
from hello import TDNN_LSTM_Model, extract_features, idx2char, char2idx, DEVICE

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = TDNN_LSTM_Model(80, 256, len(checkpoint['char2idx']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    idx2char = checkpoint['idx2char']
    print(f"Model loaded. Vocabulary size: {len(idx2char)}")
    return model, idx2char

def greedy_decode(output, idx2char):
    pred = output.argmax(-1)
    pred = pred.cpu().numpy()
    
    # Get probabilities
    probs = torch.nn.functional.softmax(output, dim=-1)
    confidence = probs.max(-1)[0].cpu().numpy()
    
    decoded = []
    prev = None
    for i, (p, conf) in enumerate(zip(pred, confidence)):
        if conf > 0.5 and (p != prev or i > 0 and confidence[i] > confidence[i-1]):
            if p != 0:
                decoded.append(idx2char[p])
        prev = p
    return ''.join(decoded)

def transcribe(model, idx2char, audio_path):
    print(f"Processing audio file: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    print(f"Original sample rate: {sr}")
    
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    
    features = extract_features(waveform.squeeze(0)).unsqueeze(0).to(DEVICE)
    print(f"Feature shape: {features.shape}")
    
    with torch.no_grad():
        output = model(features)
        print(f"Model output shape: {output.shape}")
        log_probs = torch.nn.functional.log_softmax(output, dim=-1)
    
    transcript = greedy_decode(log_probs[0], idx2char)
    return transcript

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python infer.py <path-to-flac-file>")
        sys.exit(1)
        
    model_path = "sinhala_asr_model_best.pth"  # Using best model from training
    model, idx2char = load_model(model_path)
    
    audio_path = sys.argv[1]
    transcript = transcribe(model, idx2char, audio_path)
    print("-" * 50)
    print(f"Transcription: {transcript}")
