# Medical Imaging Collective Intelligence IoT System

A distributed AI system for medical image classification using MobileNetV2 with collective intelligence across heterogeneous IoT edge devices. This project demonstrates model optimization, edge computing, and consensus-based decision making using MQTT for distributed processing.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Technologies & Dependencies](#technologies--dependencies)
- [Installation & Setup](#installation--setup)
- [Project Phases](#project-phases)
- [Usage Guide](#usage-guide)
- [Results Summary](#results-summary)
- [IoT Integration (MQTT)](#iot-integration-mqtt)
- [Team & Attribution](#team--attribution)

---

## 🎯 Overview

This Master's thesis project implements a **collective intelligence system** for medical image classification using distributed edge computing. The system classifies histopathology images into 5 cancer types (colon adenocarcinoma, colon normal, lung adenocarcinoma, lung normal, lung squamous cell carcinoma) across three heterogeneous virtual machines running optimized MobileNetV2 models.

### Key Achievements
- **99.88% baseline accuracy** with MobileNetV2
- **8 optimization techniques** (5 quantization + 3 pruning methods)
- **48.09% size reduction** using Q5 mixed precision quantization
- **6.1× inference speedup** with P3 magnitude pruning
- **100% collective consensus** on distributed voting
- **40+ telemetry metrics** with MQTT-based monitoring
- **30+ alarm triggers** in stress testing scenarios

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MASTER ORCHESTRATOR                          │
│              (orchestrator_mqtt.py / orchestrator_mqtt_alarm...) │
└─────────────┬──────────────────────────┬──────────────────────┬─┘
              │                          │                      │
     ┌────────▼─────┐          ┌─────────▼────┐      ┌──────────▼───────┐
     │   VM1 (IoT)  │          │ VM2 (Edge)   │      │ VM3 (Production) │
     │  423MB RAM   │          │  926MB RAM   │      │  1933MB RAM      │
     │ Q5 Model    │          │  Q5 Model   │      │  P3 Model       │
     │ Weight: 26% │          │ Weight: 37%  │      │ Weight: 37%     │
     └────────┬─────┘          └─────────┬────┘      └──────────┬───────┘
              │                          │                      │
              └──────────────┬───────────┴──────────────────────┘
                             │ MQTT (QoS 1)
                             │ Broker: 192.168.52.1:1883
                    ┌────────▼─────────────┐
                    │  ThingsBoard MQTT    │
                    │  IoT Dashboard       │
                    │  - REAL-TIME METRICS │
                    │  - ALARMS & ALERTS   │
                    └─────────────────────┘
```

---

## ✨ Key Features

### 1. **Model Optimization Pipeline**
   - **Phase 2**: 8 optimization techniques applied to baseline MobileNetV2
     - Quantization (Q1-Q5): INT8, INT4, symmetric, asymmetric, mixed precision
     - Pruning (P1-P3): Magnitude, structured, fine-grained
     - Evaluation on 3 heterogeneous VMs

### 2. **Distributed Collective Intelligence**
   - **Phase 5**: Multi-model voting mechanism
     - Weighted voting based on model performance profiles
     - Confidence-based consensus scoring
     - **100% unanimous decisions** on all test images

### 3. **IoT Integration & Monitoring**
   - **Phase 6**: MQTT-based telemetry and alarm system
     - 40+ metrics per image processing cycle
     - Real-time ThingsBoard dashboard
     - Alarm triggers for latency, CPU, RAM, consensus anomalies
     - Dual-scenario validation (normal + stress testing)

### 4. **Cross-Platform Validation**
   - **Phase 3**: Hardware profiling across 3 VMs
     - Precision matrices for model accuracy across platforms
     - Inference latency analysis
     - Memory consumption tracking
     - Only P3 & Q5 meet production requirements (100% accuracy all VMs)

---

## 📂 Repository Structure

```
Projet Final/
├── README.md                                 # This file
├── myreport.tex / myreport.pdf              # Complete LaTeX thesis report (6 phases)
├── requirements.txt                          # Python dependencies
│
├── 📊 TRAINING & BASELINE
│   ├── train_baseline.py                    # Phase 1: Baseline MobileNetV2 training
│   ├── download_model.py                    # Download pre-trained MobileNetV2
│   └── prepare_dataset.py                   # Dataset splitting (train/val/test)
│
├── 🔧 OPTIMIZATION
│   ├── phase2_optimize.py                   # Phase 2: Quantization & Pruning (Q1-Q5, P1-P3)
│   └── phase3_test.py                       # Phase 3: Hardware profiling on 3 VMs
│
├── 🧠 MODEL SELECTION & VOTING
│   └── phase4_model_selection.py            # Phase 4: Score-based model selection + weighting
│
├── 📡 MQTT/IoT ORCHESTRATION
│   ├── node_server_mqtt.py                  # Node receiver (runs on each VM)
│   ├── orchestrator_mqtt.py                 # Phase 5: Collective voting orchestrator
│   └── orchestrator_mqtt_alarm_triggered.py # Phase 6: Alarm system + dual scenarios
│
├── 📁 DATA/
│   ├── train/
│   │   ├── colon_aca/
│   │   ├── colon_n/
│   │   ├── lung_aca/
│   │   ├── lung_n/
│   │   └── lung_scc/
│   ├── val/
│   │   └── [same structure as train]
│   └── test/
│       └── [same structure as train]
│
├── 🤖 MODELS/
│   ├── baseline/
│   │   └── baseline_model.pt               # Phase 1: 8.51 MB baseline
│   ├── optimized/
│   │   ├── q1_symmetric_int8.pt            # Q1 - Symmetric INT8 quantization
│   │   ├── q2_asymmetric_int8.pt           # Q2 - Asymmetric INT8 quantization
│   │   ├── q3_int4.pt                      # Q3 - INT4 quantization
│   │   ├── q4_static.pt                    # Q4 - Static quantization
│   │   ├── q5_mixed_fp16.pt                # Q5 - Mixed precision FP16 (BEST: 48% size ↓)
│   │   ├── p1_magnitude_pruning.pt         # P1 - Magnitude pruning
│   │   ├── p2_structured_pruning.pt        # P2 - Structured pruning
│   │   └── p3_finegrained_pruning.pt       # P3 - Fine-grained pruning (BEST: 6.1× speed ↑)
│   └── original_mobilenet_v2_5classes.pt   # Reference model
│
├── 📊 PROGRESS/
│   ├── baseline_results.json                # Phase 1 metrics
│   ├── phase2_optimization_results.csv      # Phase 2 detailed results
│   ├── phase2_size_reduction_analysis.md    # Phase 2 analysis
│   ├── phase3_metrics.csv                   # Phase 3 VM profiling
│   ├── remaining_models_results.csv         # Phase 3 results
│   ├── Q1_Q2_REPORT.md                     # Phase 2 quantization report
│   └── PHASE2_SIZE_REDUCTION_ANALYSIS.md    # Phase 2 analysis
│
├── 📸 SCREENSHOTS/
│   ├── Machines_VM_Terminals.png           # 3 VMs running node_server with MQTT
│   └── Dashboard.png                        # ThingsBoard dashboard (100% consensus)
│
├── 🐳 THINGSBOARD (IoT Platform)
│   └── docker-compose.yml                   # ThingsBoard + MQTT broker setup
│
└── 📄 ADDITIONAL FILES
    ├── projekt_iot.pdf                      # Presentation slides
    └── Stuff/                               # Miscellaneous files
```

---

## 🛠️ Technologies & Dependencies

### Core ML & Data
- **PyTorch 2.1.0** - Neural network framework
- **TorchVision 0.16.0** - Pre-trained models & transforms
- **NumPy 1.24.3** - Numerical computing

### IoT & Distributed Systems
- **MQTT Protocol v3.11** - Lightweight pub/sub messaging
- **ThingsBoard Community** - IoT platform & dashboard
- **Docker** - Containerization for ThingsBoard

### Monitoring & System
- **psutil 5.9.5** - System & process monitoring
- **tqdm 4.65.0** - Progress bars

### Evaluation
- **scikit-learn** - Metrics (precision, recall, F1, accuracy)

### Documentation
- **LaTeX (pdflatex)** - Professional report generation
- **Babel** - Multi-language support (French/English)

---

## 📥 Installation & Setup

### Prerequisites
- Python 3.10+
- Virtual environment (venv or conda)
- SSH access to 3 VMs (for distributed testing)
- Docker (for ThingsBoard deployment)
- 8GB+ RAM recommended

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd "Projet Final"
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset
```bash
# Download and split dataset
python prepare_dataset.py

# This creates:
# - Data/train/ (training set)
# - Data/val/ (validation set)
# - Data/test/ (test set)
```

### Step 5: Download Pre-trained Model
```bash
python download_model.py
```

### Step 6: Deploy ThingsBoard (Optional, for Phase 6)
```bash
cd Thingsboard
docker-compose up -d
# Access at http://localhost:8080
```

---

## 🚀 Project Phases

### **Phase 1: Baseline Training**
**Objective**: Train MobileNetV2 baseline model on medical imaging dataset

```bash
python train_baseline.py
```

**Output**:
- `models/baseline/baseline_model.pt` (8.51 MB)
- Accuracy: **99.88%**
- Training time: ~5 epochs
- Metrics saved to `Progress/baseline_results.json`

**Key Finding**: Baseline achieves excellent accuracy, establishing reference point for optimization.

---

### **Phase 2: Model Optimization**
**Objective**: Apply 8 optimization techniques and evaluate trade-offs

```bash
python phase2_optimize.py
```

**Techniques Applied**:
| Technique | Type | Size Reduction | Accuracy Impact | Status |
|-----------|------|---|---|---|
| Q1 | Symmetric INT8 | 75% | Baseline maintained | ✅ |
| Q2 | Asymmetric INT8 | 75% | Baseline maintained | ✅ |
| Q3 | INT4 | 87.5% | Baseline maintained | ✅ |
| Q4 | Static | 75% | Baseline maintained | ✅ |
| **Q5** | **Mixed FP16** | **48.09%** | **Baseline maintained** | **✅ BEST** |
| P1 | Magnitude | 30% | 2% accuracy loss | ⚠️ |
| P2 | Structured | 40% | 5% accuracy loss | ❌ |
| **P3** | **Fine-grained** | **25%** | **Baseline maintained** | **✅** |

**Results**:
- Best size reduction: Q5 (48.09%)
- Best speed improvement: P3 (6.1× inference speedup)
- Production candidates: Q5 + P3
- Saved to `Progress/phase2_optimization_results.csv`

---

### **Phase 3: Hardware Profiling**
**Objective**: Evaluate models across 3 heterogeneous VMs

```bash
python phase3_test.py
```

**VMs Configuration**:
| VM | Type | RAM | Storage | Model | Latency |
|----|------|-----|---------|-------|---------|
| VM1 | IoT | 423MB | 2GB | Q5 | 103ms |
| VM2 | Edge | 926MB | 5GB | Q5 | 43ms |
| VM3 | Production | 1933MB | 20GB | P3 | 33.5ms |

**Precision Matrix for Selected Models**:
```
         VM1    VM2    VM3
Q5     100%   100%   100%
P3     100%   100%   100%
```

**Findings**:
- Only Q5 & P3 maintain 100% accuracy across all VMs
- P3 offers best inference time (33.5ms)
- Q5 offers best size reduction (4.43 MB)
- Other models show accuracy degradation on resource-constrained VMs

---

### **Phase 4: Model Selection & Weighting**
**Objective**: Select best model per VM and calculate voting weights

```bash
python phase4_model_selection.py
```

**Selection Scoring** (Accuracy × Speed × Size):
- **VM1 (IoT)**: Q5 selected (score: 0.68) → Weight: **26.05%**
- **VM2 (Edge)**: Q5 selected (score: 0.96) → Weight: **36.78%**
- **VM3 (Production)**: P3 selected (score: 0.97) → Weight: **37.16%**

**Rationale**:
- VM1: Prioritize RAM efficiency (423MB limit)
- VM2: Balance across metrics (best overall performance)
- VM3: Maximize accuracy + absolute performance (most capable)

---

### **Phase 5: Collective Intelligence & Voting**
**Objective**: Implement distributed voting mechanism for consensus decisions

```bash
python orchestrator_mqtt.py
```

**Voting Formula**:
```
Score_class = Σ(Confidence_VM_i × Weight_i)
where:
  - Confidence_VM_i ∈ [0, 1] per VM
  - Weight_i = normalized scoring result from Phase 4
```

**Results on Test Set (10 images)**:
- Collective Accuracy: **100%** (10/10)
- Average Confidence: **99.98%**
- Individual VM Accuracy: **100%** (all 3 VMs)
- Consensus: **100% unanimity** on all predictions
- No conflicting votes observed

**Key Insight**: Multi-model voting provides robust consensus even if individual models had minor disagreements. Full unanimity indicates excellent model selection and weighting strategy.

---

### **Phase 6: IoT Integration & Alarm System**
**Objective**: Deploy MQTT-based telemetry and alarm system with stress testing

```bash
# Terminal 1: Start node servers on each VM
ssh root@192.168.52.10 "python node_server_mqtt.py"
ssh root@192.168.52.20 "python node_server_mqtt.py"
ssh root@192.168.52.30 "python node_server_mqtt.py"

# Terminal 2: Run orchestrator with normal operation
python orchestrator_mqtt.py

# Terminal 3: Run orchestrator with alarm triggering (stress test)
python orchestrator_mqtt_alarm_triggered.py
```

**MQTT Infrastructure**:
- **Broker**: 192.168.52.1:1883
- **Protocol**: MQTT v3.11, QoS 1
- **Devices**: 4 (1 orchestrator + 3 VMs)
- **Metrics**: 40+ telemetry points per image

**Telemetry Points** (per VM per image):
```
- inference_time_ms
- cpu_percent
- memory_percent
- confidence_score
- predicted_class
- timestamp
- device_id
- [+ 33 more metrics]
```

**Alarm System** (with criticality levels):
| Alarm | Threshold | Criticality | Purpose |
|-------|-----------|-------------|---------|
| HIGH_LATENCY | > 400ms | HIGH | Slow inference |
| HIGH_CPU | > 85% | MEDIUM | CPU bottleneck |
| HIGH_RAM | > 450MB | MEDIUM | Memory pressure |
| LOW_CONSENSUS | < 0.95 | HIGH | Voting disagreement |

**Test Scenarios**:

**Scenario 1: Normal Operation** (10 images)
```
✓ Alarms Triggered: 0
✓ Average Latency: 60ms
✓ Average CPU: 45%
✓ Average RAM: 370MB
✓ Consensus: 100%
✓ Accuracy: 100% (10/10)
```

**Scenario 2: Stress Test** (CPU/RAM inflated, 30 images)
```
⚠ Alarms Triggered: 30
  - HIGH_LATENCY: 18
  - HIGH_CPU: 8
  - HIGH_RAM: 4
✓ Average Latency: 450ms
⚠ Average CPU: 88-92%
⚠ Average RAM: 460-512MB
✓ Consensus: 100% (maintained despite resource pressure)
✓ Accuracy: 100% (30/30) ← Inference quality preserved!
```

**Key Finding**: System maintains perfect accuracy even under resource stress. Alarms effectively detect degradation but don't impact decision quality.

---

## 📊 Usage Guide

### Training New Baseline
```bash
python train_baseline.py --epochs 10 --batch-size 32 --lr 0.001
```

### Running Full Pipeline
```bash
# Sequential execution of all phases
python train_baseline.py              # Phase 1
python phase2_optimize.py             # Phase 2
python phase3_test.py                 # Phase 3
python phase4_model_selection.py      # Phase 4
# Phases 5-6 require VM setup with SSH
```

### Testing Individual Models
```python
import torch
from pathlib import Path

# Load optimized model
model = torch.load("models/optimized/q5_mixed_fp16.pt")
model.eval()

# Run inference on single image
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image = Image.open("path/to/image.jpg")
input_tensor = transform(image).unsqueeze(0)
output = model(input_tensor)
confidence = torch.softmax(output, dim=1)
```

### Viewing Results
```bash
# Phase 1 baseline scores
cat Progress/baseline_results.json

# Phase 2 optimization comparison
cat Progress/phase2_optimization_results.csv

# Phase 3 hardware profiling
cat Progress/phase3_metrics.csv

# Phase 4 model selection scores
cat Progress/remaining_models_results.csv
```

---

## 📈 Results Summary

### Performance Metrics
| Metric | Value | Phase |
|--------|-------|-------|
| Baseline Accuracy | 99.88% | P1 |
| Best Size Reduction | 48.09% (Q5) | P2 |
| Worst Size Reduction | 25% (P3) | P2 |
| Best Speed Improvement | 6.1× (P3) | P2 |
| Collective Accuracy | 100% | P5 |
| Average Confidence | 99.98% | P5 |
| MQTT Latency (normal) | 60ms | P6 |
| MQTT Latency (stress) | 450ms | P6 |
| Alarms in Stress Test | 30 | P6 |

### Model Comparison Matrix
```
Model  │ Size(MB) │ Speed(ms) │ Accuracy │ RAM Peak│ Selected
───────┼──────────┼───────────┼──────────┼─────────┼──────────
Base   │ 8.51     │ 204       │ 99.88%   │ 800MB   │ -
Q5     │ 4.43     │ 103       │ 99.88%   │ 540MB   │ VM1, VM2 ✓
P3     │ 6.38     │ 33.5      │ 99.88%   │ 400MB   │ VM3 ✓
Q1     │ 2.13     │ 105       │ 99.88%   │ 520MB   │ -
Q2     │ 2.13     │ 107       │ 99.88%   │ 525MB   │ -
Q3     │ 1.06     │ 108       │ 99.88%   │ 530MB   │ -
```

---

## 🌐 IoT Integration (MQTT)

### ThingsBoard Setup
1. Deploy container: `docker-compose up -d` (in Thingsboard/)
2. Access dashboard: http://localhost:8080
3. Default credentials: admin / admin
4. Configure device tokens for each VM

### MQTT Publishing Topics
```
devices/vm1/telemetry
devices/vm2/telemetry
devices/vm3/telemetry
orchestrator/voting_results
orchestrator/alarms
orchestrator/consensus_score
```

### Real-time Monitoring
- Dashboard shows live metrics for all 3 VMs
- Alarm panel displays triggered alerts
- Consensus voting results with confidence scores
- Historical data with time-series graphs

---

## 👥 Team & Attribution

**MASTER'S PROGRAM**: Systèmes Intelligents Distribués & Intelligence Collective

**AUTHORS**:
- Ayoub Mrani Alaoui
- Jihane El Aouni
- Douae Chater
- Najlae Sebbar

**SUPERVISOR**: Mr. Said Ohamouddou

**INSTITUTION**: Université Abdelmalek Essaâdi, Tétouan, Morocco

**PROJECT DATE**: Academic Year 2024-2025

---

## 📄 Complete Documentation

For detailed phase-by-phase analysis, see the comprehensive thesis report:
- **myreport.pdf** - Complete 6-phase analysis with narrative, tables, and screenshots
- **myreport.tex** - LaTeX source for report generation

---

## 📞 Support & Troubleshooting

### Common Issues

**Q: CUDA out of memory?**
A: Set `DEVICE = torch.device("cpu")` in scripts or reduce batch size in configuration.

**Q: MQTT connection refused?**
A: Ensure ThingsBoard is running: `docker ps | grep thingsboard`

**Q: Models not found?**
A: Run `python download_model.py` and `python phase2_optimize.py` first.

**Q: SSH timeout to VMs?**
A: Check VM availability: `ping 192.168.52.X` and verify SSH key setup.

---

## 📝 License

This project is part of academic coursework at Université Abdelmalek Essaâdi.

---

## 🔗 Related Files

- **Thesis Report**: myreport.pdf (comprehensive documentation)
- **Presentation**: projekt_iot.pdf (visual overview)
- **.gitignore**: Excludes Data/, Progress/, and model artifacts (large files)

---

**Last Updated**: April 2026  
**Repository Status**: ✅ Complete and Production-Ready
