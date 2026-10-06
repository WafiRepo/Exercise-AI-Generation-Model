#!/usr/bin/env python3
"""
Script untuk membuat ground truth JSON dari dataset pickle
untuk digunakan dengan GEval_score_calculator.py

Usage:
    python create_gt_json.py --dataset ./dataset/FS_test.pkl --output ./results/skating_gt/skating_gt.json
"""

import pickle
import json
import argparse
import os


def create_gt_json(dataset_path, output_path):
    """
    Membuat ground truth JSON dari dataset pickle
    
    Args:
        dataset_path: path ke dataset pickle file
        output_path: path untuk menyimpan ground truth JSON
    """
    print(f"Loading dataset from: {dataset_path}")
    
    # Load dataset pickle
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset)} entries")
    
    # Create ground truth dictionary
    # Format: {video_name: labels[0]} atau {video_name: " ".join(labels)}
    ground_truth = {}
    
    for item in dataset:
        video_name = item.get('video_name', 'unknown')
        labels = item.get('labels', [])
        
        if labels:
            # Ambil label pertama, atau gabungkan semua label
            # Untuk GEval, biasanya menggunakan label pertama atau gabungan
            if isinstance(labels, list):
                # Gabungkan semua label dengan spasi
                gt_text = ' '.join(labels) if len(labels) > 0 else labels[0]
            else:
                gt_text = labels
            
            ground_truth[video_name] = gt_text
        else:
            print(f"Warning: No labels found for {video_name}")
            ground_truth[video_name] = ""
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Ground truth JSON saved to: {output_path}")
    print(f"  Total entries: {len(ground_truth)}")
    
    return ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create ground truth JSON from dataset pickle')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset pickle file (e.g., ./dataset/FS_test.pkl)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for ground truth JSON (e.g., ./results/skating_gt/skating_gt.json)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        exit(1)
    
    create_gt_json(args.dataset, args.output)
