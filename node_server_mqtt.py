"""
Phase 5 + 6: Node Server for Distributed Medical Inference with MQTT Telemetry
Uses Python built-in http.server (NO external dependencies beyond PyTorch)
Publishes telemetry to ThingsBoard via MQTT

Each VM provides:
- VM1: Q5 model (Mixed Precision FP16)
- VM2: Q5 model (Mixed Precision FP16)
- VM3: P3 model (Magnitude Pruning)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import psutil
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse
from datetime import datetime
import time

# MQTT Import
try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Installing paho-mqtt...")
    os.system("pip install paho-mqtt")
    import paho.mqtt.client as mqtt

# ==================== MQTT CONFIGURATION ====================

# ThingsBoard MQTT Settings
THINGSBOARD_HOST = "192.168.52.1"  # Windows Host IP
THINGSBOARD_PORT = 1883
THINGSBOARD_TOPIC = "v1/devices/me/telemetry"

# Device Tokens (update these with your tokens from ThingsBoard)
DEVICE_TOKENS = {
    "VM1": "dKaHKHs9z21v3mJX77K4",
    "VM2": "dZpLloEJH3oxL3bAleLu",
    "VM3": "jSL4UKDGmjPU6LVqmZCf"
}

# ==================== CONFIGURATION ====================

# Detect VM ID from hostname or environment
def detect_vm_id():
    """Detect which VM this is based on hostname, IP, or environment variable"""
    vm_id = os.environ.get("VM_ID", None)
    if vm_id:
        return vm_id
    
    hostname = os.environ.get("HOSTNAME", "").lower()
    
    # Try to detect from hostname
    if "vm1" in hostname or "192.168.52.10" in hostname:
        return "VM1"
    elif "vm2" in hostname or "192.168.52.20" in hostname:
        return "VM2"
    elif "vm3" in hostname or "192.168.52.30" in hostname:
        return "VM3"
    
    # Try to detect from IP address
    try:
        import socket
        hostname_ip = socket.gethostbyname(socket.gethostname())
        if "192.168.52.10" in hostname_ip:
            return "VM1"
        elif "192.168.52.20" in hostname_ip:
            return "VM2"
        elif "192.168.52.30" in hostname_ip:
            return "VM3"
    except:
        pass
    
    # Fallback: ask user
    print("\n" + "="*60)
    print("Could not auto-detect VM ID. Which VM is this running on?")
    print("1) VM1 (192.168.52.10) - Q5 Model")
    print("2) VM2 (192.168.52.20) - Q5 Model")
    print("3) VM3 (192.168.52.30) - P3 Model")
    choice = input("Enter choice (1-3): ").strip()
    
    vm_mapping = {"1": "VM1", "2": "VM2", "3": "VM3"}
    selected = vm_mapping.get(choice, "VM1")
    print(f"✓ Selected: {selected}\n")
    return selected

VM_ID = detect_vm_id()

# Model configuration per VM
MODEL_CONFIG = {
    "VM1": {"port": 5001, "model": "Q5"},
    "VM2": {"port": 5002, "model": "Q5"},
    "VM3": {"port": 5003, "model": "P3"}
}

PORT = MODEL_CONFIG[VM_ID]["port"]
MODEL_TYPE = MODEL_CONFIG[VM_ID]["model"]

# Paths - try multiple locations
def find_model_path():
    """Search for model in common locations"""
    model_filenames = [
        f"{MODEL_TYPE}_model.pt",
        f"{MODEL_TYPE.lower()}_model.pt",
        f"{MODEL_TYPE}_optimization_model.pt",
        f"{MODEL_TYPE.lower()}_optimization_model.pt"
    ]
    
    search_base_dirs = [
        Path.cwd() / "models" / "optimized",
        Path.cwd(),
        Path("/root/models/optimized"),
        Path("/root"),
        Path.home() / "models" / "optimized",
    ]
    
    for base_dir in search_base_dirs:
        if base_dir.exists():
            for filename in model_filenames:
                full_path = base_dir / filename
                if full_path.exists():
                    return full_path
    
    return None

MODEL_PATH = find_model_path()

# ==================== MQTT CLIENT ====================

mqtt_client = None
mqtt_connected = False

def on_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    global mqtt_connected
    if rc == 0:
        print(f"✅ MQTT Connected to {THINGSBOARD_HOST}:{THINGSBOARD_PORT}")
        mqtt_connected = True
    else:
        print(f"❌ MQTT Connection failed with code {rc}")
        mqtt_connected = False

def on_disconnect(client, userdata, rc):
    """MQTT disconnection callback"""
    global mqtt_connected
    mqtt_connected = False
    if rc != 0:
        print(f"⚠️  MQTT Disconnected unexpectedly with code {rc}")

def on_publish(client, userdata, mid):
    """MQTT publish callback"""
    pass

def initialize_mqtt():
    """Initialize MQTT client"""
    global mqtt_client, mqtt_connected
    
    try:
        mqtt_client = mqtt.Client(client_id=f"VM_{VM_ID}")
        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect
        mqtt_client.on_publish = on_publish
        
        # Set username as access token
        mqtt_client.username_pw_set(DEVICE_TOKENS[VM_ID])
        
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
        print(f"❌ MQTT initialization failed: {e}")
        return None

def publish_telemetry(inference_result: Dict[str, Any], inference_time_ms: float):
    """Publish inference telemetry to ThingsBoard"""
    if not mqtt_client or not mqtt_connected:
        return
    
    try:
        # Build telemetry payload matching PDF specification
        telemetry = {
            "vm_id": VM_ID,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "technique": MODEL_TYPE,
            "prediction": inference_result.get("prediction", "unknown"),
            "confidence": round(inference_result.get("confidence", 0), 4),
            "inference_time_ms": round(inference_time_ms, 2),
            "cpu_usage_pct": inference_result.get("cpu_usage_pct", 0),
            "ram_usage_mb": inference_result.get("ram_usage_mb", 0),
            "patient_id": f"P-{VM_ID}-{int(time.time())}"
        }
        
        # Publish to ThingsBoard
        payload = json.dumps(telemetry)
        result = mqtt_client.publish(THINGSBOARD_TOPIC, payload, qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"   📤 MQTT: Published telemetry to ThingsBoard")
        else:
            print(f"   ⚠️  MQTT: Publish failed with code {result.rc}")
    
    except Exception as e:
        print(f"   ❌ MQTT publish error: {e}")

# ==================== MODEL LOADING ====================

def load_model():
    """Load the appropriate model for this VM"""
    if MODEL_PATH is None or not MODEL_PATH.exists():
        print(f"❌ Model not found for {MODEL_TYPE}")
        print(f"\n📋 Model Setup Instructions:")
        print(f"   Possible model names:")
        for fname in [f"{MODEL_TYPE}_model.pt", f"{MODEL_TYPE.lower()}_model.pt"]:
            print(f"   - {fname}")
        print(f"\n   Copy from your project to this VM:")
        vm_last_octet = [10, 20, 30][['VM1', 'VM2', 'VM3'].index(VM_ID)]
        model_file = f"{MODEL_TYPE}_model.pt"
        print(f"   scp models/optimized/{model_file} root@192.168.52.{vm_last_octet}:/root/")
        print(f"\n   Searched locations:")
        for base in [Path.cwd() / "models" / "optimized", Path.cwd(), Path("/root")]:
            print(f"   - {base}/")
        return None
    
    try:
        print(f"📥 Loading {MODEL_TYPE} model from {MODEL_PATH}...")
        device = torch.device("cpu")
        
        loaded = torch.load(MODEL_PATH, map_location=device)
        
        if isinstance(loaded, dict):
            # It's a state dict - rebuild the model
            print(f"   ℹ️  Loaded as state dict, rebuilding model architecture...")
            from torchvision.models import mobilenet_v2
            
            model = mobilenet_v2(pretrained=False)
            model.classifier[1] = torch.nn.Linear(1280, 5)
            
            try:
                model.load_state_dict(loaded, strict=False)
            except Exception as e:
                print(f"   ⚠️  State dict loading with strict=False: {e}")
                model.load_state_dict(loaded, strict=False)
            
            model = model.to(device)
            model.eval()
        else:
            model = loaded
            if hasattr(model, 'to'):
                model = model.to(device)
            if hasattr(model, 'eval'):
                model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

model = load_model()

# Class labels for medical imaging
CLASS_LABELS = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================== INFERENCE ====================

def get_system_metrics():
    """Get current CPU and RAM usage"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram_info = psutil.virtual_memory()
    ram_mb = ram_info.used / (1024 * 1024)
    
    return {
        "cpu_usage_pct": round(cpu_percent, 2),
        "ram_usage_mb": round(ram_mb, 2),
        "ram_percent": round(ram_info.percent, 2)
    }

def run_inference(image_data: bytes) -> Dict[str, Any]:
    """
    Run inference on a single image with MQTT telemetry
    
    Args:
        image_data: Raw image bytes
    
    Returns:
        {
            "vm_id": str,
            "prediction": str,
            "confidence": float,
            "all_confidences": dict,
            "cpu_usage_pct": float,
            "ram_usage_mb": float,
            "status": "success" or "error"
        }
    """
    inference_start = time.time()
    
    try:
        if model is None:
            return {
                "vm_id": VM_ID,
                "status": "error",
                "message": "Model not loaded"
            }
        
        # Load image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        
        # CPU doesn't support FP16 convolution operations
        device = torch.device("cpu")
        image_tensor = image_tensor.to(device).float()
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction_idx = torch.max(probabilities, 1)
        
        prediction_idx = prediction_idx.item()
        confidence = confidence.item()
        prediction_class = CLASS_LABELS[prediction_idx]
        
        # Get all confidences
        all_probs = probabilities[0].cpu().numpy()
        all_confidences = {
            CLASS_LABELS[i]: round(float(all_probs[i]), 4)
            for i in range(len(CLASS_LABELS))
        }
        
        # Get system metrics
        metrics = get_system_metrics()
        
        inference_time_ms = (time.time() - inference_start) * 1000
        
        result = {
            "vm_id": VM_ID,
            "model_type": MODEL_TYPE,
            "prediction": prediction_class,
            "confidence": round(confidence, 4),
            "all_confidences": all_confidences,
            "cpu_usage_pct": metrics["cpu_usage_pct"],
            "ram_usage_mb": metrics["ram_usage_mb"],
            "status": "success"
        }
        
        # Publish telemetry to ThingsBoard
        publish_telemetry(result, inference_time_ms)
        
        return result
    
    except Exception as e:
        return {
            "vm_id": VM_ID,
            "status": "error",
            "message": str(e),
            "cpu_usage_pct": get_system_metrics()["cpu_usage_pct"],
            "ram_usage_mb": get_system_metrics()["ram_usage_mb"]
        }

# ==================== HTTP SERVER ====================

class InferenceRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request handler for distributed inference"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.send_health_response()
        elif self.path == '/metrics':
            self.send_metrics_response()
        else:
            self.send_error_response(404, "Endpoint not found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/inference':
            self.send_inference_response()
        else:
            self.send_error_response(404, "Endpoint not found")
    
    def send_inference_response(self):
        """Process inference request and send response"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error_response(400, "No image data provided")
                return
            
            # Read image data
            image_data = self.rfile.read(content_length)
            
            # Run inference (includes MQTT publish)
            result = run_inference(image_data)
            
            # Send response
            self.send_json_response(200, result)
        
        except Exception as e:
            error_response = {
                "vm_id": VM_ID,
                "status": "error",
                "message": str(e)
            }
            self.send_json_response(500, error_response)
    
    def send_health_response(self):
        """Send health check response"""
        metrics = get_system_metrics()
        response = {
            "status": "ok",
            "vm_id": VM_ID,
            "model": MODEL_TYPE,
            "model_ready": model is not None,
            "mqtt_connected": mqtt_connected,
            **metrics
        }
        self.send_json_response(200, response)
    
    def send_metrics_response(self):
        """Send system metrics response"""
        response = {
            "vm_id": VM_ID,
            "model": MODEL_TYPE,
            **get_system_metrics()
        }
        self.send_json_response(200, response)
    
    def send_json_response(self, status_code, data):
        """Send JSON response"""
        response_data = json.dumps(data, default=str).encode('utf-8')
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response_data))
        self.end_headers()
        self.wfile.write(response_data)
    
    def send_error_response(self, status_code, message):
        """Send error response"""
        error_data = {
            "status": "error",
            "message": message
        }
        self.send_json_response(status_code, error_data)
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

# ==================== MAIN ====================

def run_server():
    """Start HTTP server"""
    print("\n" + "="*80)
    print(f"🚀 NODE SERVER - {VM_ID} (Phase 5 + 6: HTTP + MQTT)")
    print("="*80)
    print(f"VM ID: {VM_ID}")
    print(f"Model: {MODEL_TYPE}")
    print(f"Port: {PORT}")
    print(f"Model Ready: {model is not None}")
    print(f"Class Labels: {CLASS_LABELS}")
    print(f"\n📡 MQTT Configuration:")
    print(f"   Host: {THINGSBOARD_HOST}:{THINGSBOARD_PORT}")
    print(f"   Token: {DEVICE_TOKENS[VM_ID]}")
    print(f"   Status: {'✅ Connected' if mqtt_connected else '⏳ Connecting...'}")
    print(f"\n📡 HTTP Endpoints:")
    print(f"   Health Check: http://localhost:{PORT}/health")
    print(f"   Inference: POST http://localhost:{PORT}/inference")
    print(f"   Metrics: http://localhost:{PORT}/metrics")
    print("="*80)
    print(f"\n⏳ Server running... Press Ctrl+C to stop\n")
    
    try:
        server = HTTPServer(('0.0.0.0', PORT), InferenceRequestHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped by user")
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Server error: {e}")
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        sys.exit(1)

if __name__ == "__main__":
    # Initialize MQTT first
    print("📡 Initializing MQTT connection...")
    initialize_mqtt()
    
    # Then start HTTP server
    run_server()
