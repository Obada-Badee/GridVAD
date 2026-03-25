import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score

def calculate_pixel_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Computes AUROC, AP, AUPRO, F1 for a single frame or flattended video.
    """
    if gt_mask.sum() == 0:
        # None implies no positive class, can't calculate ROC optimally or it's undefined
        return None, None, None, None
    
    # Flatten
    gt_flat = gt_mask.ravel()
    pred_flat = pred_mask.ravel()
    
    pixel_auroc = roc_auc_score(gt_flat, pred_flat)
    pixel_ap = average_precision_score(gt_flat, pred_flat)
    
    precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
    pixel_aupro = auc(recall, precision) # This is technically PR-AUC, not AUPRO (which is Region-based)
    # Original code called it AUPRO but calculated PR-AUC?
    # "pixel_aupro = auc(recall, precision)" -> This is Area Under Precision-Recall Curve.
    # AUPRO typically means Area Under Per-Region Overlap.
    # We will stick to the name used in the reference code but note it is PR-AUC.
    
    # F1 at 0.5
    # F1 at 0.5
    pred_binary = (pred_flat > 0.5).astype(np.uint8)
    pixel_f1 = f1_score(gt_flat, pred_binary)
    
    return pixel_auroc, pixel_ap, pixel_aupro, pixel_f1
