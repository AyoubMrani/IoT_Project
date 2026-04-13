import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import time
import json
import csv
from pathlib import Path
from tqdm import tqdm
import numpy as np
import psutil
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("Data")
MODEL_SAVE_DIR = Path("models/baseline")
RESULTS_DIR = Path("results")
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 5

# Create directories
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Class names
CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

class MetricsTracker:
    """Track training metrics and resource usage"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_times = []
        self.best_val_accuracy = 0
        self.best_model_epoch = 0
        
    def update_train_loss(self, loss):
        self.train_losses.append(loss)
    
    def update_val_metrics(self, loss, accuracy):
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
            self.best_model_epoch = len(self.val_accuracies)

def get_data_loaders(batch_size=BATCH_SIZE):
    """Create data loaders for train, val, test"""
    print("📂 Loading datasets...")
    
    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = ImageFolder(DATA_DIR / "val", transform=val_test_transform)
    test_dataset = ImageFolder(DATA_DIR / "test", transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader

def create_model():
    """Create and return MobileNetV2 model for 5 classes"""
    print("🧠 Creating MobileNetV2 model...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Modify classifier for 5 classes
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    
    model = model.to(DEVICE)
    print(f"  Device: {DEVICE}")
    print(f"  Classes: {NUM_CLASSES}")
    
    return model

def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

def evaluate_on_test(model, test_loader):
    """Evaluate model on test set"""
    print("\n📊 Evaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.extend([inference_time / len(images)] * len(images))
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_inference_time = np.mean(inference_times)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_inference_time_ms': avg_inference_time
    }
    
    return results

def get_model_size(model):
    """Get model size in MB"""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    return size_mb

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def save_results(model, metrics, results):
    """Save model and results"""
    print("\n💾 Saving results...")
    
    # Save model
    model_path = MODEL_SAVE_DIR / "baseline_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")
    
    # Calculate model size
    model_size = get_model_size(model)
    
    # Prepare results dictionary
    final_results = {
        'model': 'MobileNetV2',
        'dataset': 'LC25000',
        'num_classes': NUM_CLASSES,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'device': str(DEVICE),
        'model_size_mb': round(model_size, 2),
        'train_time_minutes': round(sum(metrics.train_times) / 60, 2),
        'best_validation_accuracy': round(metrics.best_val_accuracy, 4),
        'best_model_epoch': metrics.best_model_epoch,
        **{k: round(v, 4) for k, v in results.items()}
    }
    
    # Save as JSON
    json_path = RESULTS_DIR / "baseline_results.json"
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"  Results saved: {json_path}")
    
    # Save as CSV for the PDF table
    csv_path = RESULTS_DIR / "phase1_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Baseline Model'])
        writer.writerow(['Model', 'MobileNetV2'])
        writer.writerow(['Precision', final_results['precision']])
        writer.writerow(['Size (MB)', final_results['model_size_mb']])
        writer.writerow(['Inference Time (ms)', final_results['avg_inference_time_ms']])
        writer.writerow(['Accuracy', final_results['accuracy']])
        writer.writerow(['F1 Score', final_results['f1_score']])
    print(f"  CSV saved: {csv_path}")
    
    return final_results

def main():
    print("="*70)
    print("🚀 PHASE 1 — BASELINE MODEL TRAINING")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Create model
    model = create_model()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    # Training loop
    metrics = MetricsTracker()
    
    print("\n" + "="*70)
    print("🔄 TRAINING LOOP")
    print("="*70)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        # Record memory before training
        mem_start = get_memory_usage()
        time_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        metrics.update_train_loss(train_loss)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        metrics.update_val_metrics(val_loss, val_acc)
        
        # Record metrics
        epoch_time = time.time() - time_start
        metrics.train_times.append(epoch_time)
        mem_used = get_memory_usage() - mem_start
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s | Memory: {mem_used:.2f}MB")
        
        scheduler.step()
    
    # Evaluate on test set
    test_results = evaluate_on_test(model, test_loader)
    
    # Save results
    final_results = save_results(model, metrics, test_results)
    
    # Print summary
    print("\n" + "="*70)
    print("📋 BASELINE MODEL SUMMARY")
    print("="*70)
    for key, value in final_results.items():
        print(f"{key:.<40} {value}")
    
    print("\n✅ PHASE 1 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📂 Model saved: {MODEL_SAVE_DIR / 'baseline_model.pt'}")
    print(f"📊 Results saved: {RESULTS_DIR / 'baseline_results.json'}")
    print(f"📈 Metrics for PDF: {RESULTS_DIR / 'phase1_metrics.csv'}")

if __name__ == "__main__":
    main()
