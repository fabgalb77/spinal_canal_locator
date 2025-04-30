"""
Image processing utilities for spine classification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2


def get_dcm_stats(dicom_path):
    """Get statistics about a DICOM file to debug preprocessing."""
    try:
        # Load DICOM
        dicom = pydicom.dcmread(dicom_path)
        
        # Get pixel array
        image_array = dicom.pixel_array
        
        # Calculate stats
        stats = {
            "shape": image_array.shape,
            "dtype": image_array.dtype,
            "min": float(np.min(image_array)),
            "max": float(np.max(image_array)),
            "mean": float(np.mean(image_array)),
            "std": float(np.std(image_array)),
            "has_zeros": bool(np.any(image_array == 0)),
            "has_negatives": bool(np.any(image_array < 0)),
        }
        
        # Check additional DICOM attributes for windowing
        if hasattr(dicom, "WindowCenter") and hasattr(dicom, "WindowWidth"):
            stats["window_center"] = float(dicom.WindowCenter) if not isinstance(dicom.WindowCenter, list) else float(dicom.WindowCenter[0])
            stats["window_width"] = float(dicom.WindowWidth) if not isinstance(dicom.WindowWidth, list) else float(dicom.WindowWidth[0])
        
        # Check for rescale attributes
        if hasattr(dicom, "RescaleIntercept") and hasattr(dicom, "RescaleSlope"):
            stats["rescale_intercept"] = float(dicom.RescaleIntercept)
            stats["rescale_slope"] = float(dicom.RescaleSlope)
            
        return stats
    except Exception as e:
        print(f"Error analyzing DICOM {dicom_path}: {e}")
        return {"error": str(e)}


def apply_windowing(image, center, width):
    """Apply windowing to an image."""
    min_value = center - width/2
    max_value = center + width/2
    
    windowed = np.clip(image, min_value, max_value)
    windowed = (windowed - min_value) / (max_value - min_value) * 255.0
    return windowed.astype(np.uint8)


def plot_preprocessing_steps(dicom_path, save_path):
    """Plot all preprocessing steps to debug issues."""
    try:
        # Load DICOM
        dicom = pydicom.dcmread(dicom_path)
        
        # Get raw pixel array
        raw_pixels = dicom.pixel_array
        
        # Get DICOM metadata
        rescale_slope = getattr(dicom, "RescaleSlope", 1.0)
        rescale_intercept = getattr(dicom, "RescaleIntercept", 0.0)
        
        # Apply rescale if available
        rescaled = raw_pixels * float(rescale_slope) + float(rescale_intercept)
        
        # Simple normalization (min-max)
        normalized = rescaled.copy()
        normalized = normalized - np.min(normalized)
        if np.max(normalized) > 0:
            normalized = normalized / np.max(normalized) * 255
        normalized = normalized.astype(np.uint8)
        
        # Apply custom windowing if available
        if hasattr(dicom, "WindowCenter") and hasattr(dicom, "WindowWidth"):
            window_center = float(dicom.WindowCenter) if not isinstance(dicom.WindowCenter, list) else float(dicom.WindowCenter[0])
            window_width = float(dicom.WindowWidth) if not isinstance(dicom.WindowWidth, list) else float(dicom.WindowWidth[0])
            windowed = apply_windowing(rescaled, window_center, window_width)
        else:
            windowed = None
        
        # Create figure
        if windowed is not None:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(raw_pixels, cmap='gray')
            axes[0].set_title(f"Raw Pixels\nRange: [{np.min(raw_pixels)}, {np.max(raw_pixels)}]")
            
            axes[1].imshow(rescaled, cmap='gray')
            axes[1].set_title(f"Rescaled\nRange: [{np.min(rescaled)}, {np.max(rescaled)}]")
            
            axes[2].imshow(normalized, cmap='gray')
            axes[2].set_title(f"Min-Max Normalized\nRange: [0, 255]")
            
            axes[3].imshow(windowed, cmap='gray')
            axes[3].set_title(f"Windowed\nCenter: {window_center}, Width: {window_width}")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(raw_pixels, cmap='gray')
            axes[0].set_title(f"Raw Pixels\nRange: [{np.min(raw_pixels)}, {np.max(raw_pixels)}]")
            
            axes[1].imshow(rescaled, cmap='gray')
            axes[1].set_title(f"Rescaled\nRange: [{np.min(rescaled)}, {np.max(rescaled)}]")
            
            axes[2].imshow(normalized, cmap='gray')
            axes[2].set_title(f"Min-Max Normalized\nRange: [0, 255]")
        
        plt.suptitle(f"Preprocessing Steps for {os.path.basename(dicom_path)}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return {
            "raw_min": float(np.min(raw_pixels)),
            "raw_max": float(np.max(raw_pixels)),
            "rescaled_min": float(np.min(rescaled)),
            "rescaled_max": float(np.max(rescaled)),
            "window_center": window_center if windowed is not None else None,
            "window_width": window_width if windowed is not None else None
        }
    except Exception as e:
        print(f"Error plotting preprocessing for {dicom_path}: {e}")
        return {"error": str(e)}


def visualize_results_with_debug(results, output_dir, study_id, series_id):
    """Create enhanced visualizations of the prediction results."""
    if not results:
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create debug subfolder
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Extract instance numbers and predictions
    instance_numbers = [r["instance_number"] for r in results]
    level_preds = {level: [r["predictions"][level] for r in results] for level in results[0]["predictions"]}
    level_logits = {level: [r["logits"][level] for r in results] for level in results[0]["logits"]}
    max_preds = [r["max_pred"] for r in results]
    avg_preds = [r["avg_pred"] for r in results]
    
    # 1. Plot of logits (pre-sigmoid) - this helps debug saturation
    plt.figure(figsize=(12, 8))
    for level in level_logits:
        plt.plot(instance_numbers, level_logits[level], marker='o', label=f"{level} (logits)")
    plt.xlabel('Instance Number')
    plt.ylabel('Logit Value (pre-sigmoid)')
    plt.title(f'Raw Logits for Study {study_id}, Series {series_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(debug_dir, f"{study_id}_{series_id}_logits.png"))
    plt.close()
    
    # 2. Line plot of predictions for all levels (probabilities)
    plt.figure(figsize=(12, 8))
    for level in level_preds:
        plt.plot(instance_numbers, level_preds[level], marker='o', label=level)
    plt.plot(instance_numbers, max_preds, 'k--', marker='x', label='Max Prediction')
    plt.plot(instance_numbers, avg_preds, 'k:', marker='+', label='Avg Prediction')
    plt.xlabel('Instance Number')
    plt.ylabel('Prediction Probability')
    plt.title(f'Spine Level Predictions for Study {study_id}, Series {series_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{study_id}_{series_id}_predictions.png"))
    plt.close()
    
    # 3. Find best slice for each level
    best_instances = {}
    for level in level_preds:
        level_scores = level_preds[level]
        best_idx = np.argmax(level_scores)
        best_instances[level] = {
            "instance": instance_numbers[best_idx],
            "probability": level_scores[best_idx],
            "logit": level_logits[level][best_idx],
            "image": results[best_idx]["pixel_array"]
        }
    
    # 4. Montage of best slices for each level
    fig, axes = plt.subplots(1, len(level_preds), figsize=(15, 5))
    for i, level in enumerate(level_preds):
        best = best_instances[level]
        axes[i].imshow(best["image"], cmap='gray')
        axes[i].set_title(f"{level}\nInst: {best['instance']}\nProb: {best['probability']:.4f}\nLogit: {best['logit']:.2f}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{study_id}_{series_id}_best_slices.png"))
    plt.close()
    
    # 5. Heatmap of predictions
    plt.figure(figsize=(10, 8))
    heatmap_data = np.array([[level_preds[level][i] for i in range(len(instance_numbers))] 
                             for level in level_preds])
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Prediction Probability')
    plt.yticks(range(len(level_preds)), list(level_preds.keys()))
    plt.xticks(range(len(instance_numbers)), instance_numbers, rotation=90)
    plt.xlabel('Instance Number')
    plt.ylabel('Spine Level')
    plt.title(f'Prediction Heatmap for Study {study_id}, Series {series_id}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{study_id}_{series_id}_heatmap.png"))
    plt.close()
    
    # 6. Heatmap of logits
    plt.figure(figsize=(10, 8))
    logit_data = np.array([[level_logits[level][i] for i in range(len(instance_numbers))] 
                             for level in level_logits])
    plt.imshow(logit_data, aspect='auto', cmap='RdBu_r', vmin=-10, vmax=10)
    plt.colorbar(label='Logit Value')
    plt.yticks(range(len(level_logits)), list(level_logits.keys()))
    plt.xticks(range(len(instance_numbers)), instance_numbers, rotation=90)
    plt.xlabel('Instance Number')
    plt.ylabel('Spine Level')
    plt.title(f'Logit Heatmap for Study {study_id}, Series {series_id}')
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, f"{study_id}_{series_id}_logit_heatmap.png"))
    plt.close()
    
    # 7. Image browser - show all slices with their predictions
    num_cols = 4
    num_rows = (len(results) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3*num_rows))
    axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes
    
    for i, result in enumerate(results):
        if i < len(axes):
            img = result["pixel_array"]
            axes[i].imshow(img, cmap='gray')
            best_level = max(result["predictions"].items(), key=lambda x: x[1])[0]
            best_prob = result["predictions"][best_level]
            best_logit = result["logits"][best_level]
            axes[i].set_title(f"Instance {result['instance_number']}\n{best_level}: {best_prob:.2f}\nLogit: {best_logit:.1f}")
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{study_id}_{series_id}_all_slices.png"))
    plt.close()
    
    # 8. Save numerical results as CSV
    import pandas as pd
    results_df = pd.DataFrame({
        "instance_number": instance_numbers,
        **{f"{level}_logit": level_logits[level] for level in level_logits},
        **{f"{level}_prob": level_preds[level] for level in level_preds},
        "max_prediction": max_preds,
        "avg_prediction": avg_preds
    })
    results_df.to_csv(os.path.join(output_dir, f"{study_id}_{series_id}_results.csv"), index=False)
