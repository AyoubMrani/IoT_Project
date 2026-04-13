"""
Phase 4 (CORRECT): Weighted Model Selection
Calculates weighted scores for each optimization technique based on VM-specific criteria.
Uses Phase 3 results to select the best model per VM.

Weighting Criteria (from project PDF):
- VM1 (500 Mo IoT): RAM (0.40) + CPU (0.40) + Precision (0.20)
- VM2 (1 Go Edge): RAM (0.30) + Speed (0.30) + Precision (0.40)
- VM3 (2 Go Production): Precision (0.60) + Speed (0.25) + RAM (0.15)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# ==================== CONFIGURATION ====================
RESULTS_DIR = Path("Results")
PROGRESS_DIR = Path("Progress")
PROGRESS_DIR.mkdir(exist_ok=True)

MODELS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'P1', 'P2', 'P3']
VMS = ['VM1', 'VM2', 'VM3']

# Weighting criteria per VM (from PDF)
WEIGHTS = {
    'VM1': {'RAM': 0.40, 'CPU': 0.40, 'Precision': 0.20},
    'VM2': {'RAM': 0.30, 'Speed': 0.30, 'Precision': 0.40},
    'VM3': {'Precision': 0.60, 'Speed': 0.25, 'RAM': 0.15}
}

# ==================== DATA LOADING ====================

def load_phase3_results():
    """Load all Phase 3 results from Results/VM*/"""
    all_results = {}
    
    for vm_name in VMS:
        csv_file = RESULTS_DIR / vm_name / f"phase3_results_{vm_name}.csv"
        
        if not csv_file.exists():
            print(f"❌ File not found: {csv_file}")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            all_results[vm_name] = df
            print(f"✅ Loaded {vm_name}: {len(df)} models")
        except Exception as e:
            print(f"❌ Error loading {vm_name}: {e}")
    
    return all_results

# ==================== METRICS NORMALIZATION ====================

def normalize_metric(value, metric_type, vm_name=None):
    """
    Normalize metrics to 0-100 scale
    - Accuracy/Precision: higher is better
    - RAM/Speed: lower is better
    """
    if pd.isna(value):
        return 0
    
    if metric_type == 'Accuracy':
        return value  # Already 0-100
    elif metric_type == 'RAM':
        # Lower RAM is better - but invert (less RAM = higher score)
        # Cap at reasonable values to normalize
        if value > 100:
            return 0  # Excessive RAM usage = 0 score
        return 100 - (value / 100 * 100)  # Scale: 0-100MB maps to 100-0 score
    elif metric_type == 'Speed':
        # Lower inference time is better
        # Speed = 1/Inference_Time (inverted for better = higher score)
        # Cap outliers
        if value > 300:
            value = 300
        # Normalize: slower (300ms) = low score, faster (30ms) = high score
        return 100 * (1 - (value / 300))
    elif metric_type == 'CPU':
        # CPU usage: lower is better (0-100%)
        # But we normalize use 100-value to invert
        return 100 - value if value <= 100 else 0
    
    return value

def calculate_scores(results_dict):
    """Calculate weighted scores for each model on each VM"""
    scores = {}
    detailed_scores = {}
    
    for vm_name in VMS:
        if vm_name not in results_dict:
            continue
        
        df = results_dict[vm_name]
        vm_scores = {}
        vm_detailed = {}
        
        print(f"\n{'='*60}")
        print(f"🖥️  {vm_name} ({WEIGHTS[vm_name]})")
        print(f"{'='*60}")
        
        for idx, row in df.iterrows():
            model_id = row['Model_ID']
            
            # Extract metrics
            accuracy = row['Accuracy_%'] if pd.notna(row['Accuracy_%']) else 0
            ram_used = row['RAM_Used_MB'] if pd.notna(row['RAM_Used_MB']) else 0
            avg_inference = row['Avg_Inference_Time_ms'] if pd.notna(row['Avg_Inference_Time_ms']) else 300
            avg_cpu = row['Avg_CPU_Usage_%'] if pd.notna(row['Avg_CPU_Usage_%']) else 50
            
            # Normalize metrics
            norm_accuracy = normalize_metric(accuracy, 'Accuracy')
            norm_ram = normalize_metric(ram_used, 'RAM')
            norm_speed = normalize_metric(avg_inference, 'Speed')
            norm_cpu = normalize_metric(avg_cpu, 'CPU')
            
            # Calculate weighted score based on VM weights
            score = 0
            component_scores = {}
            
            if 'Precision' in WEIGHTS[vm_name]:
                precision_score = norm_accuracy * WEIGHTS[vm_name]['Precision'] / 100
                score += precision_score
                component_scores['Precision'] = precision_score
            
            if 'RAM' in WEIGHTS[vm_name]:
                ram_score = norm_ram * WEIGHTS[vm_name]['RAM'] / 100
                score += ram_score
                component_scores['RAM'] = ram_score
            
            if 'CPU' in WEIGHTS[vm_name]:
                cpu_score = norm_cpu * WEIGHTS[vm_name]['CPU'] / 100
                score += cpu_score
                component_scores['CPU'] = cpu_score
            
            if 'Speed' in WEIGHTS[vm_name]:
                speed_score = norm_speed * WEIGHTS[vm_name]['Speed'] / 100
                score += speed_score
                component_scores['Speed'] = speed_score
            
            vm_scores[model_id] = round(score, 2)
            vm_detailed[model_id] = {
                'score': round(score, 2),
                'components': component_scores,
                'metrics': {
                    'accuracy_%': accuracy,
                    'ram_mb': ram_used,
                    'inference_ms': avg_inference,
                    'cpu_%': avg_cpu
                }
            }
            
            # Print scores
            status = "✅" if model_id in ['P3', 'Q5'] else "⚠️ " if accuracy > 80 else "❌"
            print(f"  {model_id}: {score:.2f}/100 {status} (Acc: {accuracy:.1f}%, RAM: {ram_used:.1f}MB, Speed: {avg_inference:.1f}ms)")
        
        scores[vm_name] = vm_scores
        detailed_scores[vm_name] = vm_detailed
    
    return scores, detailed_scores

def select_best_models(scores):
    """Select the best model per VM"""
    best_models = {}
    
    print(f"\n{'='*60}")
    print("🏆 BEST MODEL SELECTION PER VM")
    print(f"{'='*60}\n")
    
    for vm_name, vm_scores in scores.items():
        best_model = max(vm_scores, key=vm_scores.get)
        best_score = vm_scores[best_model]
        
        best_models[vm_name] = {
            'model_id': best_model,
            'score': best_score
        }
        
        print(f"  {vm_name}: {best_model} (Score: {best_score:.2f}/100)")
    
    return best_models

def generate_justifications(best_models, detailed_scores):
    """Generate detailed justifications for each VM selection"""
    justifications = {}
    
    for vm_name, selection in best_models.items():
        model_id = selection['model_id']
        score = selection['score']
        details = detailed_scores[vm_name][model_id]
        metrics = details['metrics']
        
        weights_str = ", ".join([f"{k} ({v:.0%})" for k, v in WEIGHTS[vm_name].items()])
        
        justification = f"""
### {vm_name} – Selection: {model_id} (Score: {score:.2f}/100)

**VM Profile:** 
- RAM: {metrics['ram_mb']:.1f} MB usage
- CPU Usage: {metrics['cpu_%']:.1f}%
- Inference Speed: {metrics['inference_ms']:.1f} ms/image
- Precision: {metrics['accuracy_%']:.1f}%

**Criteria Weighting:** {weights_str}

**Justification:**
The {model_id} model was selected for {vm_name} because:

1. **Accuracy ({metrics['accuracy_%']:.1f}%):** {self._accuracy_justification(metrics['accuracy_%'])}

2. **Resource Efficiency:** 
   - RAM Usage: {self._ram_justification(vm_name, metrics['ram_mb'])}
   - CPU Usage: {metrics['cpu_%']:.1f}% (acceptable utilization)

3. **Inference Performance:** 
   - Inference Time: {metrics['inference_ms']:.1f}ms/image {self._speed_justification(metrics['inference_ms'])}

4. **Best Match for {vm_name}:**
   - Prioritizes {', '.join(list(WEIGHTS[vm_name].keys())[:len(WEIGHTS[vm_name])-1])} 
   - Meets the production requirements for this deployment scenario
"""
        justifications[vm_name] = justification
    
    return justifications

def _accuracy_justification(accuracy):
    if accuracy >= 99:
        return "Excellent - maintains near-perfect classification accuracy"
    elif accuracy >= 90:
        return "Very good - strong accuracy with consistent performance"
    elif accuracy >= 80:
        return "Good - reliable accuracy for most classifications"
    else:
        return "Acceptable - balances accuracy with other constraints"

def _ram_justification(vm_name, ram_mb):
    if vm_name == 'VM1' and ram_mb < 50:
        return f"{ram_mb:.1f}MB is efficient for IoT device with 500MB total"
    elif vm_name == 'VM2' and ram_mb < 100:
        return f"{ram_mb:.1f}MB is acceptable for edge server with 1GB total"
    elif vm_name == 'VM3' and ram_mb < 200:
        return f"{ram_mb:.1f}MB is minimal for production server with 2GB total"
    return f"{ram_mb:.1f}MB is reasonable for this deployment"

def _speed_justification(speed_ms):
    if speed_ms < 50:
        return "- Very fast, enabling real-time inference"
    elif speed_ms < 150:
        return "- Acceptable latency for most applications"
    elif speed_ms < 250:
        return "- Suitable for batch processing"
    return "- Adequate for non-critical applications"

# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 80)
    print("🚀 PHASE 4: WEIGHTED MODEL SELECTION (CORRECT)")
    print("=" * 80)
    
    # Load Phase 3 results
    print(f"\n📥 Loading Phase 3 results...")
    results = load_phase3_results()
    
    if not results:
        print("❌ No Phase 3 results found!")
        return
    
    # Calculate weighted scores
    print(f"\n{'='*60}")
    print("📊 CALCULATING WEIGHTED SCORES")
    print(f"{'='*60}")
    
    scores, detailed_scores = calculate_scores(results)
    
    # Select best models
    best_models = select_best_models(scores)
    
    # Generate justifications
    print(f"\n{'='*60}")
    print("📝 GENERATING JUSTIFICATIONS")
    print(f"{'='*60}\n")
    
    # Create detailed report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'methodology': 'Weighted scoring based on VM-specific criteria',
        'weights': WEIGHTS,
        'all_scores': scores,
        'detailed_scores': detailed_scores,
        'best_models': best_models
    }
    
    # Save JSON report
    json_path = PROGRESS_DIR / "PHASE4_MODEL_SELECTION.json"
    with open(json_path, 'w') as f:
        # Convert nested dicts to JSON-serializable format
        json_report = {
            'timestamp': report['timestamp'],
            'methodology': report['methodology'],
            'weights': report['weights'],
            'best_models': report['best_models'],
            'all_scores': {vm: {k: v for k, v in vs.items()} for vm, vs in report['all_scores'].items()}
        }
        json.dump(json_report, f, indent=2)
    
    print(f"✅ JSON report saved to: {json_path}")
    
    # Save markdown report with justifications
    md_path = PROGRESS_DIR / "PHASE4_MODEL_SELECTION_REPORT.md"
    with open(md_path, 'w') as f:
        f.write("# Phase 4: Weighted Model Selection Report\n\n")
        f.write("**Project:** Medical Imaging Model Optimization\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Selection Methodology\n\n")
        f.write("Each VM has specific optimization priorities defined by weighted criteria:\n\n")
        
        for vm_name, weights in WEIGHTS.items():
            f.write(f"- **{vm_name}:** {', '.join([f'{k} ({v:.0%})' for k, v in weights.items()])}\n")
        
        f.write("\n## Selected Models\n\n")
        
        for vm_name, selection in best_models.items():
            model_id = selection['model_id']
            score = selection['score']
            f.write(f"- **{vm_name}:** {model_id} (Weighted Score: {score:.2f}/100)\n")
        
        f.write("\n## Detailed Justifications\n\n")
        
        for vm_name in VMS:
            model_id = best_models[vm_name]['model_id']
            score = best_models[vm_name]['score']
            details = detailed_scores[vm_name][model_id]
            metrics = details['metrics']
            
            f.write(f"### {vm_name} – Selection: **{model_id}** (Score: {score:.2f}/100)\n\n")
            f.write(f"**Performance Metrics:**\n")
            f.write(f"- Accuracy: {metrics['accuracy_%']:.1f}%\n")
            f.write(f"- RAM Usage: {metrics['ram_mb']:.1f} MB\n")
            f.write(f"- Inference Time: {metrics['inference_ms']:.1f} ms/image\n")
            f.write(f"- CPU Usage: {metrics['cpu_%']:.1f}%\n\n")
            
            f.write(f"**Weighting:** {', '.join([f'{k} ({v:.0%})' for k, v in WEIGHTS[vm_name].items()])}\n\n")
            
            f.write(f"**Justification:** \n")
            f.write(f"The {model_id} model was selected for {vm_name} because:\n\n")
            
            if metrics['accuracy_%'] >= 99:
                f.write(f"1. **Excellent Accuracy ({metrics['accuracy_%']:.1f}%):** Maintains near-perfect classification accuracy across all test cases.\n\n")
            else:
                f.write(f"1. **Strong Accuracy ({metrics['accuracy_%']:.1f}%):** Reliable performance for medical imaging classification.\n\n")
            
            f.write(f"2. **Resource Constraints:** ")
            if vm_name == 'VM1':
                f.write(f"Uses only {metrics['ram_mb']:.1f}MB RAM, well-suited for IoT device with 500MB total capacity.\n\n")
            elif vm_name == 'VM2':
                f.write(f"Uses {metrics['ram_mb']:.1f}MB RAM with {metrics['inference_ms']:.1f}ms inference time, balanced for edge computing.\n\n")
            else:
                f.write(f"Optimal balance with {metrics['inference_ms']:.1f}ms inference and {metrics['ram_mb']:.1f}MB RAM for production deployment.\n\n")
            
            f.write(f"3. **Weighted Score: {score:.2f}/100** reflects prioritization of:\n")
            for criterion, weight in WEIGHTS[vm_name].items():
                f.write(f"   - {criterion}: {weight:.0%}\n")
            f.write(f"\n")
    
    print(f"📄 Markdown report saved to: {md_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ PHASE 4 COMPLETE - MODEL SELECTION")
    print(f"{'='*60}\n")
    
    print("📊 Selected Models:")
    for vm_name, selection in best_models.items():
        print(f"  {vm_name}: {selection['model_id']} (Score: {selection['score']:.2f}/100)")
    
    print(f"\n📁 Deliverables saved to Progress/:")
    print(f"   - PHASE4_MODEL_SELECTION.json")
    print(f"   - PHASE4_MODEL_SELECTION_REPORT.md")
    
    return best_models

if __name__ == "__main__":
    main()
