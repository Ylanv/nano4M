# nano4M: A Minimal Massively Multimodal Model

Welcome to the repository for **nano4M**, a lightweight yet powerful framework inspired by [4M](https://4m.epfl.ch/) and [nanoGPT](https://github.com/karpathy/nanoGPT). The goal is to understand and build the key components of modern multimodal models from scratch, from language and image modeling to fully multimodal generation.

This repository includes:

- ✅ A transformer-based **Language Model**
- ✅ A **Masked Image Generation** model
- ✅ A unified **Multimodal Architecture**
- ✅ A clean and extensible codebase for training and evaluating models across modalities

>  Developed as part of the **COM-304 course at EPFL**.

---

## Project Scope

### Phase 1: Building the Core

The first phase of nano4M focuses on replicating the essential mechanisms behind large multimodal models:

- **Language Modeling** with autoregressive Transformers
- **Image Generation** with masked token prediction
- **Multimodal Fusion** using cross-modal masking objectives
- Modular and scalable code to support easy extensions

---

## Extension: Adding the Audio Modality

In the second phase of the project, we implement a custom extension to integrate the **audio modality**, enabling nano4M to support:

-  **Text-to-Speech (TTS)**
-  **Speech Recognition (SR)**
-  **Multimodal Cross-Generation** (e.g., text → audio, image → audio)

###  Key Contributions

- **Audio Tokenization**: Using discrete audio codecs (e.g., WavTokenizer, SpeechTokenizer)
- **Cross-Modal Transfer Evaluation**: Testing audio tasks before fine-tuning to assess generalization from text/image
- **Masked Pretraining**: Inspired by MultiMAE, extended to include audio
- **Modality-Specific Fine-Tuning**: Tailored tasks like generating bird calls from bird images/descriptions
- **Multimodal Fusion Analysis**: Investigating how adding audio affects performance across all modalities
- **Pseudo-labeling**: Generating synthetic audio with Spark-TTS to enrich training data
- **Efficient Optimization**: Using µP for scalable hyperparameter transfer and speculative decoding for faster inference

---

## 📁 Project Structure

```bash
nano4m/
├── models/             # Language, image, and multimodal model architectures
├── tokenizers/         # Tokenizers for text, image, and audio
├── data/               # Dataset loading and preprocessing scripts
├── training/           # Training scripts for various tasks
├── evaluation/         # Evaluation metrics and experiments
├── extensions/audio/   # Audio modality integration
├── configs/            # Config files for experiments and tuning
├── results/            # Model outputs, logs, and plots
├── README.md
└── requirements.txt
```

---

## Evaluation

We validate our models using:

- **ASR**: Word Error Rate (WER)
- **TTS**: Mean Opinion Score (MOS)
- **Cross-modality generation** comparisons against baselines like Whisper, wav2vec, PeriodWave-Turbo

---

## Related Work

- [4M: Massively Multimodal Masked Modeling](https://arxiv.org/abs/2306.00989)
- [MultiMAE: Multi-modal Multi-task Masked Autoencoders](https://arxiv.org/abs/2204.01678)
- [Spark-TTS](https://arxiv.org/abs/2503.01710)
- [Whisper](https://openai.com/research/whisper)

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/nano4m-audio.git
cd nano4m-audio
pip install -r requirements.txt
```

---

## Getting Started

To run training or evaluation on a pretrained model:

```bash
# Pretrain the multimodal model
python training/train_multimodal.py --config configs/pretrain.yaml

# Run evaluation on audio tasks
python evaluation/eval_audio.py --model_path checkpoints/audio_model.pt
```

---

## Project Website & Demo

🚧 Coming Soon:
- [🔗 Project Presentation](#)
- [🌍 Online Demo (GitHub Pages)](#)
- [📽️ Video Demonstration](#)

---

## Authors

- **Ylan Vifian**  
- **Petrit Arifi**  
- **Ozair Faizan**  

> Developed for **COM-304: communications project**, EPFL Spring 2025.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- EPFL VILAB
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [4M](https://4m.epfl.ch/)

---