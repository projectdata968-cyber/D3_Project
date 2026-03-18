import os
import argparse
import torch
import numpy as np
import random
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, precision_recall_curve, 
    PrecisionRecallDisplay
)

from data.datasets import D3_dataset_AP, read_video, set_preprocessing
from models import D3_model

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_pr_curve(y_true, y_scores, encoder, timestamp):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    display = PrecisionRecallDisplay(recall=recall, precision=precision, average_precision=ap_score)
    display.plot(color='darkorange', lw=2)
    plt.title(f"Forensic PR Curve: {encoder}\nAverage Precision: {ap_score:.4f}")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = f"results/pr_curve_{timestamp}.png"
    plt.savefig(save_path)
    print(f"[*] PR Curve saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=str, default="0")
    parser.add_argument('--loss', type=str, default='l2', choices=['l2', 'cos'])
    parser.add_argument('--encoder', type=str, default='XCLIP-16')
    parser.add_argument('--real-csv', type=str, required=True)
    parser.add_argument('--fake-csv', type=str, required=True)
    parser.add_argument('--segments', type=int, default=3, help='Segments per video to average')
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs("results", exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Model & Transform
    model = D3_model(encoder_type=args.encoder, loss_type=args.loss).to(device)
    model.eval()
    trans = set_preprocessing()

    # 2. Load Metadata (We use raw CSV data to control segment sampling)
    import pandas as pd
    df_real = pd.read_csv(args.real_csv)
    df_fake = pd.read_csv(args.fake_csv)
    df_real['label'] = 0  # Assuming 0 is Real
    df_fake['label'] = 1  # Assuming 1 is Fake
    eval_df = pd.concat([df_real, df_fake], axis=0, ignore_index=True)

    y_true, y_scores, video_names = [], [], []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[*] Starting Forensic Eval: {args.encoder} | Segments: {args.segments}")
    
    # 3. Evaluation Loop
    with torch.no_grad():
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
            video_path = row['content_path']
            label = row['label']
            v_name = os.path.basename(video_path)

            segment_volatilities = []
            for s in range(args.segments):
                try:
                    # Uses 'consecutive' sampling for real physics extraction
                    frames = read_video(video_path, trans, num_frames=16, sampling_mode='consecutive')
                    frames = frames.unsqueeze(0).to(device)
                    
                    # D3 Output: _, avg_accel, std_volatility
                    _, _, vol_tensor = model(frames)
                    segment_volatilities.append(vol_tensor.cpu().item())
                except Exception as e:
                    continue

            if segment_volatilities:
                avg_volatility = np.mean(segment_volatilities)
                y_scores.append(avg_volatility)
                y_true.append(label) # 0=Real, 1=Fake
                video_names.append(v_name)

    # 4. Metrics & Thresholding
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Forensic logic: Real is Positive (1), Fake is Negative (0)
    # We invert y_true because in D3, Real=0 and Fake=1 normally
    y_true_binary = 1 - y_true 
    
    # Find Optimal Threshold (Maximizing TPR - FPR)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred_binary = (y_scores >= best_thresh).astype(int)

    # 5. Reporting
    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary)
    rec = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    ap = average_precision_score(y_true_binary, y_scores)
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    result_str = (
        f"--- Evaluation Results ---\n"
        f"Encoder: {args.encoder} | Loss: {args.loss}\n"
        f"Total Samples: {len(y_true_binary)}\n"
        f"Optimal Threshold: {best_thresh:.6f}\n"
        f"--------------------------\n"
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {prec:.4f} (Reliability of 'Real' verdict)\n"
        f"Recall:    {rec:.4f} (Ability to catch all 'Real' videos)\n"
        f"F1-Score:  {f1:.4f}\n"
        f"AP Score:  {ap:.4f}\n"
        f"--------------------------\n"
        f"CONFUSION MATRIX STATS:\n"
        f"True Positives (Real correctly identified): {tp}\n"
        f"True Negatives (Fake correctly identified): {tn}\n"
        f"False Positives (AI labeled as REAL):       {fp}\n"
        f"False Negatives (REAL labeled as AI):       {fn}\n"
    )

    print("\n" + "="*50)
    print(result_str.strip())
    print("="*50)

    # --- Identify & Log Specific Errors ---
    print("\n[!] FORENSIC ERROR LOG:")
    print("-" * 30)
    
    error_count = 0
    for name, true, pred in zip(video_names, y_true_binary, y_pred_binary):
        if true == 0 and pred == 1:
            print(f"FAILED [FP - AI Labeled as Real]: {name}")
            error_count += 1
        elif true == 1 and pred == 0:
            print(f"FAILED [FN - Real Labeled as AI]: {name}")
            error_count += 1
            
    if error_count == 0:
        print("PERFECT CLASSIFICATION: No errors found.")
    print("-" * 30)

    # --- Save Text Report ---
    with open(f"results/report_{timestamp}.txt", "w") as f:
        f.write(result_str)
        f.write("\n\nMISCLASSIFIED LIST:\n")
        for name, true, pred in zip(video_names, y_true_binary, y_pred_binary):
            if true != pred:
                error_type = "FP" if pred == 1 else "FN"
                f.write(f"{error_type}: {name}\n")

    # 6. Save Artifacts
    plot_pr_curve(y_true_binary, y_scores, args.encoder, timestamp)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Fake', 'Pred Real'], 
                yticklabels=['Actual Fake', 'Actual Real'])
    plt.savefig(f"results/cm_{timestamp}.png")
