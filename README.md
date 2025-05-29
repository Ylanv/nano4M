

<h1 align="center">
  <br>
  <img src="https://user-images.githubusercontent.com/your-logo-placeholder.png" alt="nano4M Audio Logo" width="200">
  <br>
</h1>

<h4 align="center">Extending nano4M with Audio: A Step Toward True Multimodal Learning</h4>

<p align="center">
  <a href="https://ylanv.github.io/nano4M/">
    <img src="https://img.shields.io/badge/Website-Project%20Page-red" alt="Website">
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white">
  </a>
  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?style=flat&logo=pytorch">
  </a>
  <a href="https://arxiv.org/abs/your-paper-placeholder">
    <img src="https://img.shields.io/badge/audio_nano4M-Paper-8A2BE2.svg" alt="Paper">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#notebooks">Notebooks</a> •
  <a href="#usage">Usage</a> •
  <a href="#training--weights">Training & Weights</a> •
  <a href="#credits">Credits</a> •
</p>

---

## Overview

**nano4M-Audio** is a research extension of [nano4M](https://github.com/EPFL-VILAB/com-304-FM-project/tree/main/nano4M) — a simplified variant of [4M](https://4m.epfl.ch/), focusing on efficient multimodal learning. This project integrates **audio as a new modality**.


---

## Notebooks

This repository is structured around three exploratory notebooks:

* 📁 **1. Dataset**
  We construct our own aligned multimodal dataset by generating synthetic images using captions and audio from the [AudioCaps](https://audiocaps.github.io/) dataset. This results in image–text–audio triplets aligned at the semantic level.

* 🎧 **2. Audio Tokenizer**
  We implement a [VQ-VAE-based](https://arxiv.org/abs/1711.00937) audio tokenizer from scratch, trained to discretize audio into a compact token sequence. In addition, we integrate a [WaveNet](https://arxiv.org/abs/1609.03499) decoder for high-quality waveform reconstruction from the discrete tokens.

* 🧠 **3. nano4M with Audio**
  Integrates the audio tokens into the nano4M architecture, enabling multimodal training and qualitative evaluation.


---

## Usage

### 🛠 Setup

```bash
# Clone the repo
git clone https://github.com/your-username/nano4m-audio.git
cd nano4m-audio

# Setup environment
conda create -n nano4m-audio python=3.10
conda activate nano4m-audio
pip install -r requirements.txt
```

---

## Training & Weights
> ⚠️ **Important:** Access to training scripts, datasets, and pretrained weights requires connection to the [IZAR cluster](https://www.epfl.ch/research/facilities/scitas/hardware/izar/) at EPFL. Ensure your SSH credentials and SCITAS account are configured properly.

---

### Launching Training Jobs (SLURM)

We provide SLURM scripts to launch training for each component of the pipeline on the IZAR cluster.

#### 🔹 Train nano4M-Audio

```bash
sbatch nano4M/submit_job_multi_node_scitas.sh nano4M/cfgs/nano4M/audiocaps_tok_rgb.yaml <WANDB_API_KEY>
```

This command launches a distributed training job for the nano4M model using audio and synthetic image-text pairs.

#### 🔹 Train the VQ-VAE Audio Tokenizer

```bash
sbatch nano4M/audio_tokenizer/vqvae/train.sh
```

Trains the VQ-VAE model from scratch on [Librispeech](https://www.openslr.org/12) 360h.

#### 🔹 Train the WaveNet Decoder

```bash
sbatch nano4M/audio_tokenizer/wavenet/train_wavenet.sh <WANDB_API_KEY>
```

Trains a conditional WaveNet decoder to reconstruct waveforms from discrete VQ tokens.

---

### Pretrained Weights

All pretrained models are stored on the cluster under:

```bash
cd /work/com-304/snoupy/weights
```

Includes:

*  Trained VQ-VAE tokenizer
*  Trained WaveNet decoder
*  Trained nano4M-Audio checkpoints

---

### 🎧 Synthetic Dataset

The dataset used in this project (images, audio, and aligned captions) is available at:

```bash
cd /work/com-304/snoupy/audiocaps
```

Includes:

* 🎵 Audio from AudioCaps
* 🖼️ Synthetic images generated from captions
* 📝 Aligned captions and metadata

---

## Credits

This project builds on and extends:

* 🔗 [nano4M](https://github.com/OpenGVLab/4M) — Simplified multimodal transformer.
* 🎵 [Encodec](https://github.com/facebookresearch/encodec) — High-fidelity neural audio codec.
* 🎤 [LibriSpeech](https://www.openslr.org/12) — Open-source audio-text dataset for pretraining.
* 📁 [AudioCaps](https://audiocaps.github.io/) - Generating Captions for Audios in the Wild

---


