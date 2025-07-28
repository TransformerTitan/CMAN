# CMAN
Collaborative Multimodal Agent Networks

# CMAN: Collaborative Multimodal Agent Networks

This repository contains the official implementation of **"Collaborative Multimodal Agent Networks: Dynamic Specialization and Emergent Communication for Complex Scene Understanding"** (ICCV 2025).

## Overview

CMAN introduces a novel framework for collaborative multimodal agent networks that dynamically specialize in complementary perception tasks while developing emergent communication protocols. Our approach addresses complex scene understanding by distributing multimodal processing across specialized agents that communicate through learned semantic representations.

## Key Features

- **Dynamic Specialization**: Agents automatically develop complementary expertise across multimodal inputs
- **Emergent Communication**: Learned semantic protocols for efficient inter-agent information sharing
- **Dynamic Agent Coordination (DAC)**: Real-time role assignment, information fusion, and conflict resolution
- **Robust Performance**: 12.3% improvement over single-agent baselines and 7.8% over existing multi-agent methods

## Architecture

![CMAN Architecture](docs/architecture.png)

The system consists of:
1. **Multimodal Agents**: Each agent processes vision, audio, and text inputs with specialized capabilities
2. **Communication Module**: Enables emergent protocols for semantic information sharing
3. **Coordination Module**: Manages dynamic role assignment and conflict resolution
4. **Specialization Mechanism**: Gradient-based optimization for complementary expertise development

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cman.git
cd cman

# Create virtual environment
python -m venv cman_env
source cman_env/bin/activate  # On Windows: cman_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from cman import CMAN, CMANConfig, CMANTrainer
import torch

# Configure the model
config = CMANConfig(
    num_agents=4,
    hidden_dim=768,
    specialization_dim=64,
    message_dim=256
)

# Create model
model = CMAN(config)

# Example forward pass
vision_input = torch.randn(2, 3, 224, 224)
audio_input = torch.randn(2, 1, 16000)
text_input = torch.randint(0, 30000, (2, 50))

outputs = model(vision_input, audio_input, text_input)
print(f"Output shape: {outputs['output'].shape}")
```

### Training

```python
from torch.utils.data import DataLoader

# Create datasets (implement your own dataset class)
train_dataset = MultimodalDataset('path/to/train/data')
val_dataset = MultimodalDataset('path/to/val/data')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create trainer
trainer = CMANTrainer(model, config)

# Train the model
trainer.train(train_loader, val_loader)
```

## Experiments

### Benchmarks

We evaluate CMAN on five challenging multimodal benchmarks:

1. **AVSD-Complex**: Extended Audio-Visual Scene-aware Dialog dataset
2. **MultiModal-VQA**: Visual question answering with audio context
3. **Dynamic-Scenes**: Dynamic environments with changing multimodal elements
4. **Ego4D-Multimodal**: Extended Ego4D with multimodal question answering
5. **MSR-VTT-Audio**: Extended MSR-VTT with audio-visual tasks

### Results

| Method | AVSD-Complex | MM-VQA | Dynamic-Scenes | Ego4D-MM | MSR-VTT-A |
|--------|--------------|--------|----------------|----------|-----------|
| Large MM Transformer | 72.4±0.8 | 68.1±1.2 | 59.3±0.9 | 64.7±1.1 | 71.2±0.7 |
| Fixed-Role Multi | 75.2±0.7 | 70.5±0.9 | 62.1±0.8 | 66.8±0.7 | 73.4±0.6 |
| **CMAN (Ours)** | **81.3±0.6** | **75.7±0.8** | **67.2±0.7** | **72.1±0.6** | **78.3±0.5** |
| **Improvement** | **+7.1%** | **+6.3%** | **+6.8%** | **+6.8%** | **+5.7%** |

### Running Experiments

```bash
# Run full training
python main.py --config configs/full_cman.yaml

# Run ablation studies
python experiments/ablation_study.py

# Evaluate robustness
python experiments/robustness_eval.py

# Analyze communication patterns
python analysis/communication_analysis.py
```

## Analysis Tools

### Specialization Analysis

```python
from cman import CMANAnalyzer

analyzer = CMANAnalyzer(model)

# Analyze agent specializations
spec_analysis = analyzer.analyze_specializations()
print(f"Average specialization entropy: {spec_analysis['avg_entropy']:.3f}")
print(f"Average inter-agent diversity: {spec_analysis['avg_diversity']:.3f}")

# Visualize specializations
analyzer.visualize_specializations('specializations.png')
```

### Communication Analysis

```python
# Analyze communication patterns
comm_analysis = analyzer.analyze_communication(dataloader)

# Visualize communication
analyzer.visualize_communication(dataloader, 'communication.png')
```

