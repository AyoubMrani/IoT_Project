"""
Phase 5 + 6: Orchestrator for Distributed Medical Inference with MQTT Collective Telemetry
Runs on Windows Host to coordinate VM1, VM2, VM3 with MQTT reporting

Architecture:
- Orchestrator (Windows Host) -> Sends images to -> Node Servers (VMs)
- VMs perform inference with selected models + publish telemetry
- Orchestrator aggregates results + publishes collective diagnosis

MQTT:
- Each VM publishes individual inference telemetry
- Orchestrator publishes collective voting results to 'Collective_Orchestrator' device
"""

import json
import time
import base64
import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import warnings
from datetime import datetime

try:
    import requests
    from PIL import Image, ImageOps
    import torch
    import torchvision.transforms as transforms
    from tqdm import tqdm
except ImportError as e:
    print(f"Installing required packages...")
    import os
    os.system("pip install requests pillow torch torchvision tqdm paho-mqtt")
    import requests
    from PIL import Image, ImageOps
    import torch
    import torchvision.transforms as transforms
    from tqdm import tqdm

# MQTT Import
try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Installing paho-mqtt...")
    import os
    os.system("pip install paho-mqtt")
    import paho.mqtt.client as mqtt

warnings.filterwarnings('ignore')

# ==================== MQTT CONFIGURATION ====================

THINGSBOARD_HOST = "192.168.52.1"  # Windows Host IP
THINGSBOARD_PORT = 1883
THINGSBOARD_TOPIC = "v1/devices/me/telemetry"
ORCHESTRATOR_TOKEN = "gUheyAJH0JShYPj38LHn"

mqtt_client = None
mqtt_connected = False

# ==================== CONFIGURATION ====================

# VM Configuration with their selected models from Phase 4
VMS = {
    "VM1": {
        "ip": "192.168.52.10",
        "port": 5001,
        "model": "Q5",
        "accuracy_weight": 0.68,
        "token": "dKaHKHs9z21v3mJX77K4"
    },
    "VM2": {
        "ip": "192.168.52.20",
        "port": 5002,
        "model": "Q5",
        "accuracy_weight": 0.96,
        "token": "dZpLloEJH3oxL3bAleLu"
    },
    "VM3": {
        "ip": "192.168.52.30",
        "port": 5003,
        "model": "P3",
        "accuracy_weight": 0.97,
        "token": "jSL4UKDGmjPU6LVqmZCf"
    }
}

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.70
MAX_CPU_USAGE = 85
MAX_RAM_USAGE = 90

# Paths
DATA_DIR = Path("Data/test")
PROGRESS_DIR = Path("Progress")
PROGRESS_DIR.mkdir(exist_ok=True)

CLASS_LABELS = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

# ==================== MQTT HANDLERS ====================

def on_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    global mqtt_connected
    if rc == 0:
        print(f"✅ MQTT Connected (Orchestrator)")
        mqtt_connected = True
    else:
        print(f"❌ MQTT Connection failed: {rc}")
        mqtt_connected = False

def on_disconnect(client, userdata, rc):
    """MQTT disconnection callback"""
    global mqtt_connected
    mqtt_connected = False

def initialize_mqtt():
    """Initialize MQTT client for orchestrator"""
    global mqtt_client, mqtt_connected
    
    try:
        mqtt_client = mqtt.Client(client_id="Collective_Orchestrator")
        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect
        
        # Set username as access token
        mqtt_client.username_pw_set(ORCHESTRATOR_TOKEN)
        
        # Connect to ThingsBoard
        mqtt_client.connect(THINGSBOARD_HOST, THINGSBOARD_PORT, keepalive=60)
        mqtt_client.loop_start()
        
        # Wait for connection
        for _ in range(10):
            if mqtt_connected:
                break
            time.sleep(0.5)
        
        return mqtt_client
    except Exception as e:
        print(f"⚠️  MQTT initialization warning: {e}")
        return None

def publish_collective_telemetry(collective_result: Dict[str, Any]):
    """Publish collective diagnosis to ThingsBoard"""
    if not mqtt_client or not mqtt_connected:
        return
    
    try:
        # Calculate consensus rate (% of VMs that agreed with collective prediction)
        individual_preds = collective_result.get("individual_predictions", {})
        collective_pred = collective_result.get("collective_prediction", "unknown")
        
        if individual_preds:
            agreeing_vms = sum(1 for vm_data in individual_preds.values() 
                              if vm_data.get("prediction") == collective_pred and vm_data.get("status") == "success")
            total_healthy_vms = sum(1 for vm_data in individual_preds.values() 
                                   if vm_data.get("status") == "success")
            consensus_rate = int((agreeing_vms / total_healthy_vms * 100) if total_healthy_vms > 0 else 0)
        else:
            consensus_rate = 0
        
        telemetry = {
            "device_id": "Collective_Orchestrator",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "image_name": collective_result.get("image_name", "unknown"),
            "collective_prediction": collective_result.get("collective_prediction", "unknown"),
            "collective_confidence": round(collective_result.get("collective_confidence", 0), 4),
            "ground_truth": collective_result.get("ground_truth", "unknown"),
            "correct": collective_result.get("correct", False),
            "consensus_rate": consensus_rate,
            "vm1_healthy": collective_result.get("individual_predictions", {}).get("VM1", {}).get("status") == "success",
            "vm2_healthy": collective_result.get("individual_predictions", {}).get("VM2", {}).get("status") == "success",
            "vm3_healthy": collective_result.get("individual_predictions", {}).get("VM3", {}).get("status") == "success"
        }
        
        payload = json.dumps(telemetry)
        result = mqtt_client.publish(THINGSBOARD_TOPIC, payload, qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"   📤 MQTT: Published collective result to ThingsBoard")
    
    except Exception as e:
        print(f"   ⚠️  MQTT publish error: {e}")

# ==================== DATA AUGMENTATION ====================

def augment_image(image_path: str, augmentation_type: str = "rotate") -> Image.Image:
    """Apply data augmentation to image for re-evaluation"""
    image = Image.open(image_path).convert("RGB")
    
    if augmentation_type == "rotate":
        image = image.rotate(90)
    elif augmentation_type == "crop":
        w, h = image.size
        crop_size = int(0.1 * min(w, h))
        image = image.crop((crop_size, crop_size, w - crop_size, h - crop_size))
        image = image.resize((224, 224))
    elif augmentation_type == "flip":
        image = ImageOps.mirror(image)
    
    return image

# ==================== COMMUNICATION WITH VMS ====================

def get_vm_health(vm_name: str) -> Dict[str, Any]:
    """Check health and metrics of a VM"""
    try:
        vm_info = VMS[vm_name]
        url = f"http://{vm_info['ip']}:{vm_info['port']}/health"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "http_code": response.status_code}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

def send_inference_request(vm_name: str, image_bytes: bytes, retries: int = 2) -> Dict[str, Any]:
    """Send inference request to a VM with retries"""
    vm_info = VMS[vm_name]
    url = f"http://{vm_info['ip']}:{vm_info['port']}/inference"
    
    for attempt in range(retries):
        try:
            response = requests.post(url, data=image_bytes, timeout=15, headers={'Content-Type': 'application/octet-stream'})
            
            if response.status_code == 200:
                result = response.json()
                result["vm_id"] = vm_name
                if "status" not in result:
                    result["status"] = "success"
                return result
            else:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return {
                    "vm_id": vm_name,
                    "status": "error",
                    "http_code": response.status_code,
                    "response_text": response.text[:200]
                }
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return {
                "vm_id": vm_name,
                "status": "error",
                "message": f"Timeout after {retries} attempts"
            }
        except requests.exceptions.ConnectionError as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return {
                "vm_id": vm_name,
                "status": "error",
                "message": f"Connection failed: {str(e)}"
            }
        except requests.exceptions.RequestException as e:
            return {
                "vm_id": vm_name,
                "status": "error",
                "message": str(e)
            }
    
    return {
        "vm_id": vm_name,
        "status": "error",
        "message": f"Failed after {retries} retries"
    }

def select_best_vm(exclude_vms: List[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Select best available VM based on load"""
    if exclude_vms is None:
        exclude_vms = []
    
    best_vm = None
    best_load = float('inf')
    
    for vm_name in VMS:
        if vm_name in exclude_vms:
            continue
        
        health = get_vm_health(vm_name)
        if health.get("status") != "ok":
            continue
        
        cpu = health.get("cpu_usage_pct", 100)
        ram = health.get("ram_percent", 100)
        
        if cpu > MAX_CPU_USAGE or ram > MAX_RAM_USAGE:
            continue
        
        load_score = (cpu * 0.6 + ram * 0.4)
        
        if load_score < best_load:
            best_load = load_score
            best_vm = vm_name
    
    if best_vm:
        return best_vm, get_vm_health(best_vm)
    return None, None

# ==================== WEIGHTED VOTING ====================

def normalize_weights(vm_scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize VM weights to sum to 1"""
    total = sum(vm_scores.values())
    if total == 0:
        return {vm: 1/len(vm_scores) for vm in vm_scores}
    return {vm: score / total for vm in vm_scores for score in [vm_scores[vm]]}

def weighted_vote(predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate predictions using weighted voting"""
    weights = {vm: VMS[vm]["accuracy_weight"] for vm in predictions if vm in VMS}
    norm_weights = normalize_weights(weights)
    
    class_scores = defaultdict(float)
    
    for vm_name, prediction in predictions.items():
        if prediction.get("status") != "success" or "all_confidences" not in prediction:
            continue
        
        vm_weight = norm_weights.get(vm_name, 0)
        
        for class_name, confidence in prediction["all_confidences"].items():
            class_scores[class_name] += confidence * vm_weight
    
    if not class_scores:
        return {
            "collective_prediction": "unknown",
            "collective_confidence": 0,
            "status": "error",
            "message": "No valid predictions received"
        }
    
    best_class = max(class_scores, key=class_scores.get)
    collective_confidence = class_scores[best_class]
    
    return {
        "collective_prediction": best_class,
        "collective_confidence": round(collective_confidence, 4),
        "class_scores": {k: round(v, 4) for k, v in sorted(class_scores.items(), 
                                                            key=lambda x: x[1], 
                                                            reverse=True)},
        "voting_breakdown": {
            vm: {
                "prediction": predictions[vm].get("prediction", "N/A"),
                "confidence": predictions[vm].get("confidence", 0),
                "weight": round(norm_weights.get(vm, 0), 4)
            }
            for vm in predictions if vm in VMS
        }
    }

# ==================== ORCHESTRATION ====================

def orchestrate_inference(image_path: str, image_name: str, 
                         augmentation_count: int = 0,
                         max_augmentations: int = 2,
                         aug_image_bytes: bytes = None) -> Dict[str, Any]:
    """Orchestrate collective inference with voting"""
    print(f"\n{'='*70}")
    print(f"📷 Processing: {image_name}")
    print(f"{'='*70}")
    
    try:
        if aug_image_bytes is None:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        else:
            image_bytes = aug_image_bytes
        
        ground_truth = None
        for label in CLASS_LABELS:
            if label in image_path.lower():
                ground_truth = label
                break
        
        print(f"\n[Attempt {augmentation_count + 1}] Sending to all VMs...")
        predictions = {}
        
        for vm_name in VMS:
            health = get_vm_health(vm_name)
            if health.get("status") != "ok":
                print(f"   ⚠️  {vm_name}: Not responding")
                continue
            
            cpu = health.get("cpu_usage_pct", 0)
            ram = health.get("ram_percent", 0)
            
            if cpu > MAX_CPU_USAGE or ram > MAX_RAM_USAGE:
                print(f"   ⚠️  {vm_name}: Overloaded (CPU:{cpu:.1f}%, RAM:{ram:.1f}%)")
                alt_vm, _ = select_best_vm(exclude_vms=[vm_name])
                if alt_vm and alt_vm not in predictions:
                    print(f"   ℹ️  Redirecting to {alt_vm}")
                    result = send_inference_request(alt_vm, image_bytes)
                    predictions[alt_vm] = result
                continue
            
            print(f"   📡 {vm_name}...", end="", flush=True)
            result = send_inference_request(vm_name, image_bytes)
            predictions[vm_name] = result
            
            if result.get("status") == "success":
                print(f" ✅ {result['prediction']} ({result['confidence']:.2%})")
            else:
                error_msg = result.get("message", "Unknown error")
                print(f" ❌ {error_msg}")
        
        voting_result = weighted_vote(predictions)
        collective_pred = voting_result.get("collective_prediction")
        collective_conf = voting_result.get("collective_confidence", 0)
        
        print(f"\n📊 Collective Vote Results:")
        print(f"   Prediction: {collective_pred} ({collective_conf:.2%} confidence)")
        
        if collective_conf < MIN_CONFIDENCE_THRESHOLD and augmentation_count < max_augmentations:
            print(f"\n⚠️  Confidence {collective_conf:.2%} < {MIN_CONFIDENCE_THRESHOLD:.0%} threshold")
            print(f"   Triggering re-evaluation with data augmentation...")
            
            aug_type = ["rotate", "crop", "flip"][augmentation_count]
            augmented_image = augment_image(image_path, aug_type)
            
            aug_bytes_io = io.BytesIO()
            augmented_image.save(aug_bytes_io, format='JPEG')
            new_aug_bytes = aug_bytes_io.getvalue()
            
            return orchestrate_inference(image_path, image_name, 
                                        augmentation_count + 1,
                                        max_augmentations,
                                        aug_image_bytes=new_aug_bytes)
        
        result_dict = {
            "image_name": image_name,
            "ground_truth": ground_truth,
            "collective_prediction": collective_pred,
            "collective_confidence": collective_conf,
            "individual_predictions": {
                vm: {
                    "prediction": predictions.get(vm, {}).get("prediction", "N/A"),
                    "confidence": predictions.get(vm, {}).get("confidence", 0),
                    "model": VMS[vm]["model"],
                    "status": predictions.get(vm, {}).get("status", "not_responding"),
                    "cpu_usage": predictions.get(vm, {}).get("cpu_usage_pct", 0),
                    "ram_usage_mb": predictions.get(vm, {}).get("ram_usage_mb", 0)
                }
                for vm in VMS
            },
            "voting_details": voting_result,
            "augmentation_attempts": augmentation_count,
            "correct": collective_pred == ground_truth if ground_truth else None
        }
        
        # Publish to ThingsBoard
        publish_collective_telemetry(result_dict)
        
        return result_dict
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "image_name": image_name,
            "status": "error",
            "message": str(e)
        }

# ==================== EVALUATION ====================

def evaluate_collective_inference(num_test_images: int = 10) -> Dict[str, Any]:
    """Evaluate collective inference system"""
    print("\n" + "="*80)
    print("🚀 PHASE 5 + 6: COLLECTIVE INTELLIGENCE WITH MQTT TELEMETRY")
    print("="*80)
    
    print("\n📡 Checking VM Health...")
    for vm_name in VMS:
        health = get_vm_health(vm_name)
        if health.get("status") == "ok":
            print(f"   ✅ {vm_name}: Ready (CPU: {health.get('cpu_usage_pct', 0):.1f}%, RAM: {health.get('ram_percent', 0):.1f}%, MQTT: {'✅' if health.get('mqtt_connected') else '⏳'})")
        else:
            print(f"   ⚠️  {vm_name}: {health.get('message', 'Not responding')}")
    
    print(f"\n📂 Gathering test images from {DATA_DIR}...")
    test_images = []
    for class_dir in DATA_DIR.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            test_images.extend(images[:num_test_images // 5])
    
    test_images = test_images[:num_test_images]
    print(f"   Found {len(test_images)} test images")
    
    if not test_images:
        print(f"❌ No test images found in {DATA_DIR}")
        return {"status": "error", "message": "No test images found"}
    
    print(f"\n🔍 Running {len(test_images)} inferences with collective voting...\n")
    
    results = []
    for i, image_path in enumerate(tqdm(test_images, desc="Processing images")):
        result = orchestrate_inference(str(image_path), image_path.name)
        results.append(result)
    
    print("\n" + "="*80)
    print("📊 ANALYSIS")
    print("="*80)
    
    successful_results = [r for r in results if r.get("status") != "error"]
    
    if not successful_results:
        print("❌ No successful inferences")
        return {"status": "error", "message": "No successful inferences"}
    
    collective_correct = sum(1 for r in successful_results if r.get("correct") == True)
    collective_accuracy = collective_correct / len(successful_results) if successful_results else 0
    
    vm_accuracies = {}
    for vm_name in VMS:
        vm_correct = sum(1 for r in successful_results 
                        if r["individual_predictions"][vm_name]["prediction"] == r.get("ground_truth"))
        vm_accuracies[vm_name] = vm_correct / len(successful_results) if successful_results else 0
    
    collective_confidences = [r.get("collective_confidence", 0) for r in successful_results]
    avg_confidence = np.mean(collective_confidences)
    min_confidence = np.min(collective_confidences)
    max_confidence = np.max(collective_confidences)
    
    augmentation_attempts = [r.get("augmentation_attempts", 0) for r in successful_results]
    num_revaluations = sum(1 for a in augmentation_attempts if a > 0)
    
    print(f"\n✅ Collective Intelligence Results:")
    print(f"   Collective Accuracy: {collective_accuracy:.2%} ({collective_correct}/{len(successful_results)})")
    print(f"   Collective Confidence: {avg_confidence:.2%} (min: {min_confidence:.2%}, max: {max_confidence:.2%})")
    print(f"   Re-evaluations Triggered: {num_revaluations}/{len(successful_results)}")
    
    print(f"\n📊 Individual VM Accuracies:")
    for vm_name, vm_model in [(vm, VMS[vm]["model"]) for vm in VMS]:
        accuracy = vm_accuracies.get(vm_name, 0)
        print(f"   {vm_name} ({vm_model}): {accuracy:.2%}")
    
    best_individual_accuracy = max(vm_accuracies.values()) if vm_accuracies else 0
    improvement = collective_accuracy - best_individual_accuracy
    improvement_pct = (improvement / best_individual_accuracy * 100) if best_individual_accuracy > 0 else 0
    
    if improvement > 0:
        print(f"\n✨ Collective Intelligence Benefit:")
        print(f"   Improvement over best individual: +{improvement:.2%} ({improvement_pct:+.1f}%)")
    else:
        print(f"\n⚠️  Individual VM Performance:")
        print(f"   Best individual exceeds collective: {improvement:.2%}")
    
    unanimous_votes = sum(1 for r in successful_results 
                         if len(set([
                             r["individual_predictions"][vm]["prediction"] 
                             for vm in VMS
                         ])) == 1)
    
    print(f"\n🗳️  Voting Analysis:")
    print(f"   Unanimous predictions: {unanimous_votes}/{len(successful_results)} ({unanimous_votes/len(successful_results):.1%})")
    print(f"   Weighted voting influenced decisions: {len(successful_results) - unanimous_votes}")
    
    print(f"\n📤 MQTT Status:")
    print(f"   Connected: {'✅ Yes' if mqtt_connected else '❌ No'}")
    print(f"   Collective Results Published: {sum(1 for r in successful_results if r.get('status') != 'error')}")
    
    evaluation_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "phase": "5+6_mqtt",
        "num_tests": len(successful_results),
        "collective_accuracy": round(collective_accuracy, 4),
        "collective_confidence": {
            "average": round(avg_confidence, 4),
            "min": round(min_confidence, 4),
            "max": round(max_confidence, 4)
        },
        "individual_accuracies": {vm: round(acc, 4) for vm, acc in vm_accuracies.items()},
        "improvement_over_best": round(improvement, 4),
        "re_evaluations": num_revaluations,
        "unanimous_votes": unanimous_votes,
        "mqtt_enabled": True,
        "mqtt_connected": mqtt_connected,
        "mqtt_host": THINGSBOARD_HOST,
        "mqtt_port": THINGSBOARD_PORT,
        "detailed_results": results
    }
    
    return evaluation_report

# ==================== MAIN ====================

def main():
    """Main execution"""
    print("📡 Initializing MQTT for collective telemetry...")
    initialize_mqtt()
    
    evaluation_report = evaluate_collective_inference(num_test_images=10)
    
    results_path = PROGRESS_DIR / "PHASE5_6_EVALUATION_RESULTS.json"
    with open(results_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    
    return evaluation_report

if __name__ == "__main__":
    main()
