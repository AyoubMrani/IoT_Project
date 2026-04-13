"""
Phase 3: VM Performance Testing
Tests all 8 optimized models on a single VM and measures:
- Inference time
- CPU usage
- RAM consumption
- Model accuracy
"""

import torch
import torch.nn as nn
import os
import csv
import json
import psutil
import time
from pathlib import Path
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np

# ==================== CONFIGURATION ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("models")
TEST_DATA_DIR = Path("Data/test")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5
BATCH_SIZE = 16
NUM_INFERENCES = 10

# Model details
MODELS = {
    'Q1': 'Dynamic Quantization',
    'Q2': 'Static PTQ',
    'Q3': 'QAT',
    'Q4': 'Weight-Only',
    'Q5': 'Mixed Precision (FP16)',
    'P1': 'Unstructured Pruning',
    'P2': 'Structured Pruning',
    'P3': 'Magnitude Pruning',
}

# ==================== UTILITY FUNCTIONS ====================

def get_vm_info():
    """Get VM hardware information"""
    vm_info = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_ram_mb': psutil.virtual_memory().total / (1024 * 1024),
        'cpu_count': psutil.cpu_count(),
        'device': str(DEVICE)
    }
    return vm_info

def get_available_ram():
    """Get available RAM in MB"""
    return psutil.virtual_memory().available / (1024 * 1024)

def load_baseline_model():
    """Load baseline MobileNetV2 architecture"""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.eval()
    return model

def load_test_data():
    """Load test dataset"""
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    try:
        test_dataset = ImageFolder(TEST_DATA_DIR, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"✅ Loaded test dataset: {len(test_dataset)} images")
        return test_loader
    except Exception as e:
        print(f"⚠️  Could not load test data: {e}")
        return None

def test_model(model_id, model_path, test_loader):
    """
    Test a single model and collect metrics
    Returns dict with metrics or error info
    """
    result = {
        'Model_ID': model_id,
        'Technique': MODELS.get(model_id, 'Unknown'),
        'Model_File': model_path.name,
        'File_Size_MB': model_path.stat().st_size / (1024 * 1024),
    }
    
    try:
        # Check available RAM before loading
        ram_before = get_available_ram()
        result['RAM_Before_Load_MB'] = round(ram_before, 2)
        
        # Load model
        print(f"\n📂 Loading {model_id}: {MODELS.get(model_id)}...")
        model = load_baseline_model()
        
        # Load weights
        try:
            weights = torch.load(model_path, map_location='cpu')
            # Use strict=False to handle quantized models with different state dict structure
            model.load_state_dict(weights, strict=False)
        except Exception as e:
            print(f"  ⚠️  Could not load state dict: {e}")
            result['Status'] = 'Weight_Load_Failed'
            return result
        
        model.eval()
        model = model.to(DEVICE)
        
        # Check RAM after loading
        ram_after = get_available_ram()
        ram_used = ram_before - ram_after
        result['RAM_Used_MB'] = round(ram_used, 2)
        result['RAM_After_Load_MB'] = round(ram_after, 2)
        
        print(f"  ✓ Loaded successfully")
        print(f"  RAM used: {result['RAM_Used_MB']} MB")
        
        # Test inferences if we have test data
        if test_loader:
            inference_times = []
            all_preds = []
            all_labels = []
            cpu_usages = []
            
            print(f"  Running {NUM_INFERENCES} inferences...")
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    if batch_idx >= NUM_INFERENCES:
                        break
                    
                    # Prepare inputs
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    # Check model dtype for FP16
                    model_dtype = next(model.parameters()).dtype
                    if model_dtype == torch.float16:
                        images = images.half()
                    
                    # Measure CPU usage
                    cpu_before = psutil.cpu_percent()
                    
                    # Inference timing
                    start_time = time.perf_counter()
                    outputs = model(images)
                    end_time = time.perf_counter()
                    
                    cpu_after = psutil.cpu_percent()
                    
                    # Record metrics
                    inference_time = (end_time - start_time) / len(images) * 1000  # ms per image
                    inference_times.append(inference_time)
                    cpu_usages.append((cpu_before + cpu_after) / 2)
                    
                    # Predictions
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate averages
            if inference_times:
                result['Avg_Inference_Time_ms'] = round(np.mean(inference_times), 4)
                result['Min_Inference_Time_ms'] = round(np.min(inference_times), 4)
                result['Max_Inference_Time_ms'] = round(np.max(inference_times), 4)
            
            if cpu_usages:
                result['Avg_CPU_Usage_%'] = round(np.mean(cpu_usages), 2)
                result['Max_CPU_Usage_%'] = round(np.max(cpu_usages), 2)
            
            if all_preds and all_labels:
                all_preds_arr = np.array(all_preds)
                all_labels_arr = np.array(all_labels)
                accuracy = np.mean(all_preds_arr == all_labels_arr)
                result['Accuracy_%'] = round(accuracy * 100, 2)
        else:
            print(f"  ⚠️  No test data - skipping accuracy metrics")
        
        result['Status'] = 'Success'
        print(f"  ✅ Test complete")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result['Status'] = 'OOM'
            result['Error'] = 'Out of Memory'
            print(f"  ❌ Out of Memory - model too large for this VM")
        else:
            result['Status'] = 'Runtime_Error'
            result['Error'] = str(e)[:100]
            print(f"  ❌ Runtime error: {result['Error']}")
    
    except Exception as e:
        result['Status'] = 'Error'
        result['Error'] = str(e)[:100]
        print(f"  ❌ Error: {result['Error']}")
    
    return result

# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 80)
    print("🚀 PHASE 3: VM PERFORMANCE TESTING")
    print("=" * 80)
    
    # Get VM info
    vm_info = get_vm_info()
    print(f"\n📊 VM Information:")
    print(f"  Device: {vm_info['device']}")
    print(f"  Total RAM: {vm_info['total_ram_mb']:.0f} MB")
    print(f"  CPU Cores: {vm_info['cpu_count']}")
    print(f"  Timestamp: {vm_info['timestamp']}")
    
    # Load test data
    print(f"\n📂 Loading test data...")
    test_loader = load_test_data()
    
    # Test all models
    print(f"\n{'='*80}")
    print(f"Testing all models...")
    print(f"{'='*80}")
    
    results = []
    
    for model_id in sorted(MODELS.keys()):
        model_path = MODELS_DIR / f"{model_id}_model.pt"
        
        if not model_path.exists():
            print(f"\n⚠️  {model_id}: Model file not found: {model_path}")
            results.append({
                'Model_ID': model_id,
                'Technique': MODELS.get(model_id),
                'Status': 'File_Not_Found',
                'Error': f'File not found: {model_path}'
            })
            continue
        
        result = test_model(model_id, model_path, test_loader)
        results.append(result)
    
    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results...")
    print(f"{'='*80}\n")
    
    # Determine VM name from available RAM
    total_ram = vm_info['total_ram_mb']
    if total_ram < 600:
        vm_name = "VM1"
    elif total_ram < 1200:
        vm_name = "VM2"
    else:
        vm_name = "VM3"
    
    # Save to CSV
    csv_filename = f"phase3_results_{vm_name}.csv"
    csv_path = RESULTS_DIR / csv_filename
    
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"📊 Results saved to: {csv_path}")
    
    # Save VM info
    vm_info['results_file'] = csv_filename
    vm_info['total_models_tested'] = len(results)
    vm_info['successful_tests'] = len([r for r in results if r.get('Status') == 'Success'])
    vm_info['oom_models'] = len([r for r in results if r.get('Status') == 'OOM'])
    
    json_path = RESULTS_DIR / f"vm_info_{vm_name}.json"
    with open(json_path, 'w') as f:
        json.dump(vm_info, f, indent=2)
    print(f"📋 VM info saved to: {json_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ TESTING COMPLETE - {vm_name}")
    print(f"{'='*80}")
    print(f"Total models tested: {len(results)}")
    print(f"Successful: {vm_info['successful_tests']}")
    print(f"Out of Memory: {vm_info['oom_models']}")
    print(f"Errors: {len(results) - vm_info['successful_tests'] - vm_info['oom_models']}")
    print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    main()
