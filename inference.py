import os
import torch
import argparse
import shutil
import random
import numpy as np
import datetime
from models import D3_model
from data.datasets import read_video, set_preprocessing

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything(42)

    parser = argparse.ArgumentParser(description='D3 Forensic Video Detector')
    parser.add_argument('--video', type=str, required=True, help='Path to test video')
    parser.add_argument('--threshold', type=float, default=3.2) # Updated to your eval threshold
    parser.add_argument('--encoder', type=str, default='XCLIP-16')
    parser.add_argument('--loss', type=str, default='l2')
    parser.add_argument('--segments', type=int, default=3, help='Number of segments to average')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_name = os.path.splitext(os.path.basename(args.video))[0]

    # 1. Setup workspace
    # Handle input (single / batch)
    video_list = []

    if args.video:
        if not os.path.exists(args.video):
            print(f"[!] Error: File not found -> {args.video}")
            return
        video_list.append(args.video)

    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"[!] Error: Folder not found -> {args.folder}")
            return
        video_list = [os.path.join(args.folder, f)
                      for f in os.listdir(args.folder)
                      if f.endswith(('.mp4', '.avi', '.mov'))]

    else:
        print("[!] Please provide --video or --folder")
        return

    # # 2. Extract Frames
    # print(f"[*] Extracting frames from {os.path.basename(args.video)}...")
    # os.system(f"python utils/video2frame.py --dataset-path {temp_dir} > /dev/null 2>&1")

    # # 3. Preprocessing
    # trans = set_preprocessing()
    # frames_path = os.path.join(temp_dir, "frames", "input", video_name)

    # 4. Initialize Model
    print("[*] Loading D3 model...")
    model = D3_model(encoder_type=args.encoder, loss_type=args.loss).to(device)
    model.eval()

    trans = set_preprocessing()

    # Process each video
    for video_path in video_list:

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"\n[*] Processing: {video_name}")

        temp_dir = f"forensic_temp_{video_name}"
        video_in = os.path.join(temp_dir, "video", "input")
        os.makedirs(video_in, exist_ok=True)

        shutil.copy(video_path, video_in)

        # Frame extraction
        os.system(f"python utils/video2frame.py --dataset-path {temp_dir} > /dev/null 2>&1")

        frames_path = os.path.join(temp_dir, "frames", "input", video_name)

        # 5. Multi-Segment Inference
        scores = []
        print(f"[*] Running multi-segment analysis ({args.segments} segments)...")
        
        with torch.no_grad():
            for i in range(args.segments):
                try:
                    # read_video now uses 'consecutive' sampling from datasets.py
                    video_tensor = read_video(frames_path, trans, sampling_mode='consecutive')
                    video_tensor = video_tensor.unsqueeze(0).to(device)
                    
                    # Model returns: (features, acceleration_avg, volatility_std)
                    _, _, score_tensor = model(video_tensor)
                    scores.append(score_tensor.cpu().item())
                except Exception as e:
                    print(f"[!] Warning: Segment {i} failed: {e}")

        if not scores:
            print("[!] Skipped (no valid segments)")
            shutil.rmtree(temp_dir)
            continue

        final_score = np.mean(scores)
        is_real = final_score >= args.threshold
        verdict = "REAL" if is_real else "AI-GENERATED"

        # 6. Forensic Report
        print("-" * 45)
        print(" D3 FORENSIC ANALYSIS RESULT")
        print("-" * 45)
        print(f" File:        {os.path.basename(video_path)}")
        print(f" Segments:    {len(scores)}")
        print(f" Volatility:  {final_score:.6f}")
        print(f" Threshold:   {args.threshold:.6f}")
        print(f" Verdict:     {verdict}")
        print("-" * 45)

        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
