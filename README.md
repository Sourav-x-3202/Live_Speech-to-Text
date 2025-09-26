#  Real-Time Speech-to-Text App with Wav2Vec2

##  About Wav2Vec2

Wav2Vec2 is a state-of-the-art self-supervised learning model developed by Facebook AI Research for automatic speech recognition (ASR). Unlike traditional ASR systems that rely heavily on large amounts of labeled data, Wav2Vec2 learns powerful speech representations from raw audio, significantly reducing the need for labeled data.


![Wav2Vec2 Architecture](https://github.com/user-attachments/assets/8c936e8c-fe39-4721-b435-ae64733d390d)

The architecture of wav2vec 2.0 ([source](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf)). The model is composed of a convolutional feature extractor, and a transformer encoder. During fine-tuning, quantization is disabled and contrastive loss is replaced with the CTC loss function.


###  Key Features:
- **Self-Supervised Learning**: Trains on unlabeled audio data.
- **Contrastive Loss**: Utilizes a contrastive loss over quantized latent representations.
- **Fine-Tuning**: Can be fine-tuned on a small amount of labeled data for specific tasks.

###  Notable Models:
- **facebook/wav2vec2-base**: Pretrained on 16kHz sampled speech audio.
- **facebook/wav2vec2-large-960h**: Fine-tuned on 960 hours of LibriSpeech data.
- **facebook/wav2vec2-xls-r-300m**: Trained on 436K hours of multilingual speech data for cross-lingual ASR.
- **facebook/wav2vec2-large-xlsr-53**: Trained on 53 languages for multilingual ASR.

For more details, visit the [Hugging Face Wav2Vec2 Documentation](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2).

---

##  Project Overview

This project implements a real-time speech-to-text application using the Wav2Vec2 model. Built with Streamlit, it allows users to record audio directly from their browser and transcribe it into text in real-time.

---

##  Features
-  Real-time audio recording via browser
-  Instant transcription using Wav2Vec2
-  Lightweight and fast performance
-  Multilingual support (based on the model used)

---

##  Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ASR Model**: Hugging Face's Wav2Vec2
- **Audio Processing**: Sounddevice, PyAudio

---

##  Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/Sourav-x-3202/Live_Speech-to-Text.git
cd wav2vec-streamlit
```

### Create a Virtual Environment
```bash
python -m venv venv
```

#### Activate the Virtual Environment

- Windows:
```bash
venv\Scripts\activate
```
- macOS/Linux:
```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Streamlit App
```bash
streamlit run app.py
```
Open the provided local URL in your browser to start using the app.
- Adjust the recording duration slider
- Click Record and Transcribe
- See your speech transcribed in real-time

### Screenshot
<img width="2517" height="1218" alt="Screenshot 2025-09-26 230227" src="https://github.com/user-attachments/assets/3a5de568-dcbe-4806-91a9-f8bddbf64126" />

---


## Contributing
Contributions are welcome!
- Fork the repo
- Create a new branch
- Submit a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---









