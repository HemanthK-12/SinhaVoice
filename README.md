# SinhaVoice

A robust Sinhalese speech-to-text tool designed to convert Sinhalese audio into text , utilizing advanced **deep learning** techniques.

#### **1.1 What is Speech-to-Text (STT)?**

Speech-to-text (STT) is the process of converting spoken language into written text. The core idea is to build a model that can take an audio waveform as input and output a sequence of words. This typically involves deep learning techniques, particularly recurrent neural networks (RNNs), convolutional neural networks (CNNs), or transformer-based models.

#### **1.2 Theory Behind STT Models**

STT models are usually composed of several components:

* **Acoustic Model:** Maps audio signals to phonetic representations.
* **Language Model:** Understands the context and predicts the next word.
* **Decoder:** Combines the acoustic model and language model outputs to form the final text.

But, these were the traditional methods used before and now, neural network based models do all this conversion by themselves and trained in one go,by learning to map audio features to text sequences.These are called end-to-end models and some examples include DeepSpeech (using recurrent neural networks(RNN)), long short-term memory networks (LSTMs), or transformers (like Wav2Vec, DeepSpeech).

### 2. **Data Preparation**

#### **2.1 Organize Your Data**

We need to pair up audio and transcripts into one csv file which can then be fed into the model for deep learning.The csv file should have columns for the audio file path, the transcript and the duration of the audio.
