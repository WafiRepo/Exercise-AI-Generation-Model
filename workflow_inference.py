"""
Workflow Lengkap: Video User -> Skeleton Extraction -> Inference -> Visualization
"""
import torch
import torch.distributed as dist
import numpy as np
import cv2
import pickle
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from models.CoachMe import CoachMe
from utils.parser import load_config
from easydict import EasyDict
from dataloaders.Dataset import get_coords
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# Initialize distributed for single-process inference (if not already initialized)
# This is needed because the model uses dist.get_rank() for attention visualization
if not dist.is_initialized():
    try:
        # Try NCCL first (for CUDA), fallback to gloo if needed
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        # Use a random port to avoid conflicts
        import random
        port = random.randint(20000, 30000)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1)
        print(f"✓ Initialized distributed (backend={backend}, port={port}) for single-process inference")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize distributed: {e}")
        print("  Attempting to patch dist.get_rank() to return 0...")
        # Patch dist.get_rank to return 0 if not initialized
        original_get_rank = dist.get_rank
        def patched_get_rank():
            try:
                return original_get_rank()
            except (ValueError, RuntimeError):
                return 0
        dist.get_rank = patched_get_rank

class CoachMeWorkflow:
    def __init__(self, config_path='./results/skating_gt/skating_gt.yaml',
                 checkpoint_path='./results/skating_gt/checkpoints/checkpoint_epoch_00050.pth'):
        """Initialize workflow with model and config."""
        print("="*60)
        print("Initializing CoachMe Workflow...")
        print("="*60)
        
        # Load config
        args = EasyDict(cfg_file=config_path)
        self.cfg = load_config(args)
        
        # Load model
        print("Loading model...")
        self.model = CoachMe(self.cfg).to(torch.float32)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"✓ Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"⚠ Warning: Checkpoint not found at {checkpoint_path}")
            print("  Model will use random weights (not recommended for inference)")
        
        self.model = self.model.cuda().eval()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
        self.prompt = "Motion Instruction : "
        
        print("✓ Model initialized successfully!")
        print("="*60)
    
    def extract_skeleton_from_video(self, video_path=None, skeleton_pkl_path=None):
        """
        Step 1: Extract skeleton from video using HybrIK or load from pickle
        
        Args:
            video_path: path to user video (optional if skeleton_pkl_path provided)
            skeleton_pkl_path: path to pre-extracted skeleton pickle file
        
        Returns:
            skeleton_coords: numpy array [num_frames, 66]
            uvd_coords: numpy array [num_frames, 66] or None (UVD coordinates for accurate 2D overlay)
            video_frames: list of video frames for visualization
        """
        print(f"\n[Step 1/5] Extracting skeleton...")
        
        # Auto-correct skeleton_pkl_path if file not found (handle underscore vs space)
        if skeleton_pkl_path and not os.path.exists(skeleton_pkl_path):
            # Try to find similar file name (replace underscore with space or vice versa)
            dir_path = os.path.dirname(skeleton_pkl_path)
            base_name = os.path.basename(skeleton_pkl_path)
            
            # Try replacing underscore with space
            corrected_name1 = base_name.replace('_', ' ')
            corrected_path1 = os.path.join(dir_path, corrected_name1)
            
            # Try replacing space with underscore
            corrected_name2 = base_name.replace(' ', '_')
            corrected_path2 = os.path.join(dir_path, corrected_name2)
            
            if os.path.exists(corrected_path1):
                print(f"  ⚠ File not found: {skeleton_pkl_path}")
                print(f"  ✓ Auto-corrected to: {corrected_path1}")
                skeleton_pkl_path = corrected_path1
            elif os.path.exists(corrected_path2):
                print(f"  ⚠ File not found: {skeleton_pkl_path}")
                print(f"  ✓ Auto-corrected to: {corrected_path2}")
                skeleton_pkl_path = corrected_path2
            else:
                # Try case-insensitive search in directory
                if os.path.exists(dir_path):
                    dir_files = os.listdir(dir_path)
                    base_lower = base_name.lower()
                    for f in dir_files:
                        if f.lower() == base_lower:
                            corrected_path3 = os.path.join(dir_path, f)
                            print(f"  ⚠ File not found: {skeleton_pkl_path}")
                            print(f"  ✓ Auto-corrected to: {corrected_path3}")
                            skeleton_pkl_path = corrected_path3
                            break
        
        # Option 1: Load from existing pickle file (recommended)
        if skeleton_pkl_path and os.path.exists(skeleton_pkl_path):
            print(f"  Loading skeleton from {skeleton_pkl_path}...")
            with open(skeleton_pkl_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    # If it's a list, take first item
                    if len(data) > 0 and 'coordinates' in data[0]:
                        skeleton_coords = data[0]['coordinates']  # [num_frames, 66]
                        # Check for UVD coordinates
                        uvd_coords = data[0].get('uvd_coords', None) if isinstance(data[0], dict) else None
                    else:
                        skeleton_coords = data[0]  # Assume first item is coordinates
                        uvd_coords = None
                elif isinstance(data, dict):
                    skeleton_coords = data['coordinates']  # [num_frames, 66]
                    uvd_coords = data.get('uvd_coords', None)
                else:
                    skeleton_coords = data  # Assume it's the coordinates directly
                    uvd_coords = None
            print(f"  ✓ Loaded skeleton: {skeleton_coords.shape}")
            if uvd_coords is not None:
                print(f"  ✓ Loaded UVD coordinates: {uvd_coords.shape} (for accurate 2D overlay)")
            else:
                print(f"  ⚠ UVD coordinates not found in pickle (will use 3D projection if needed)")
        elif video_path and os.path.exists(video_path):
            # Option 2: Extract from video using HybrIK
            # Note: extract_skeleton_hybrik now uses preprocessing format (same as BX_handle_frame_features.py)
            # This ensures consistency with training data preprocessing
            print(f"  Extracting skeleton from video using HybrIK (with preprocessing format)...")
            try:
                from hybrik_extractor import extract_skeleton_hybrik
                skeleton_coords, uvd_coords = extract_skeleton_hybrik(video_path)
                print(f"  ✓ Extracted skeleton: {skeleton_coords.shape}")
                if uvd_coords is not None:
                    print(f"  ✓ Extracted UVD coordinates: {uvd_coords.shape} (for accurate 2D overlay)")
                else:
                    print(f"  ⚠ UVD coordinates not available (will use 3D projection)")
                print(f"  ✓ Using preprocessing format (same as training data)")
                
                # Normalize skeleton to match pickle format (normalize to similar scale)
                # Skeleton from pickle typically has values in range [-1, 1] or small values
                # HybrIK skeleton is in raw 3D coordinates, need normalization
                # Note: Preprocessing extracts coordinates, but normalization is still needed for accuracy
                max_abs_value = np.abs(skeleton_coords).max()
                if max_abs_value > 2.0:  # If not normalized, normalize it
                    print(f"  Normalizing skeleton to match pickle format (max_abs_value: {max_abs_value:.4f})...")
                    # Center each frame at root joint (joint 0)
                    for frame_idx in range(len(skeleton_coords)):
                        root_coords = skeleton_coords[frame_idx, 0:3].copy()
                        for joint_idx in range(22):
                            skeleton_coords[frame_idx, joint_idx*3:(joint_idx+1)*3] -= root_coords
                    
                    # Normalize to similar scale as pickle skeletons
                    # Find global scale across all frames
                    all_coords = skeleton_coords.reshape(-1)
                    coord_range = all_coords.max() - all_coords.min()
                    if coord_range > 0:
                        # Normalize to roughly [-1, 1] range
                        skeleton_coords = (skeleton_coords - all_coords.min()) / coord_range * 2 - 1
                    print(f"  ✓ Normalized skeleton (new range: [{skeleton_coords.min():.4f}, {skeleton_coords.max():.4f}])")
            except ImportError:
                print(f"  ❌ hybrik_extractor module not found")
                raise ValueError(
                    "HybrIK extraction requires hybrik_extractor module.\n"
                    "Please ensure hybrik_extractor.py is in the project directory."
                )
            except Exception as e:
                print(f"  ❌ HybrIK extraction failed: {e}")
                print(f"  Please provide skeleton_pkl_path or check HybrIK installation:")
                print(f"    1. Install HybrIK: git clone https://github.com/Jeff-sjtu/HybrIK.git")
                print(f"    2. Set environment variable: export HYBRIK_PATH=/path/to/HybrIK")
                raise
        else:
            # Provide detailed error message
            error_msg = "Either skeleton_pkl_path or video_path must be provided.\n"
            if skeleton_pkl_path:
                if not os.path.exists(skeleton_pkl_path):
                    error_msg += f"  ❌ skeleton_pkl_path provided but file not found: {skeleton_pkl_path}\n"
                else:
                    error_msg += f"  ⚠ skeleton_pkl_path exists but failed to load: {skeleton_pkl_path}\n"
            else:
                error_msg += f"  ⚠ skeleton_pkl_path not provided (None)\n"
            
            if video_path:
                if not os.path.exists(video_path):
                    error_msg += f"  ❌ video_path provided but file not found: {video_path}\n"
                else:
                    error_msg += f"  ⚠ video_path exists but extraction failed: {video_path}\n"
            else:
                error_msg += f"  ⚠ video_path not provided (None)\n"
            
            error_msg += "\nFor video extraction, ensure HybrIK is installed and HYBRIK_PATH is set:\n"
            error_msg += "  1. Install HybrIK: git clone https://github.com/Jeff-sjtu/HybrIK.git\n"
            error_msg += "  2. Set environment variable: export HYBRIK_PATH=/path/to/HybrIK\n"
            error_msg += "\nOr provide a valid skeleton pickle file using --user_skeleton"
            raise ValueError(error_msg)
        
        # Load video frames for visualization
        video_frames = []
        if video_path and os.path.exists(video_path):
            print(f"  Loading video frames from {video_path}...")
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            print(f"  ✓ Loaded {len(video_frames)} video frames")
        else:
            print(f"  ⚠ Video not found, will create skeleton-only visualization")
            # Don't set uvd_coords to None here - it might already be loaded from pickle
        
        return skeleton_coords, uvd_coords, video_frames
    
    def load_reference_video(self, reference_path, motion_type=None):
        """
        Step 2: Load reference video skeleton
        
        Args:
            reference_path: path to reference video or skeleton pickle
            motion_type: optional motion type/class (e.g., Single_Axel, Double_Axel, Loop, Lutz for Skating, 
                       or Jab, Cross for Boxing). If provided, will search for matching reference in dataset.
        
        Returns:
            ref_skeleton: numpy array [num_frames, 66]
            ref_uvd_coords: numpy array [num_frames, 66] or None (UVD coordinates for accurate 2D overlay)
            ref_frames: list of reference video frames (if available)
        """
        print(f"\n[Step 2/5] Loading reference...")
        
        # If it's a pickle file with skeleton
        if reference_path.endswith('.pkl'):
            print(f"  Loading reference skeleton from {reference_path}...")
            with open(reference_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    ref_skeleton = None
                    # If motion_type specified, search for matching reference
                    if motion_type:
                        print(f"  Searching for motion_type: {motion_type}...")
                        for item in data:
                            if isinstance(item, dict):
                                # Check both video_name and motion_type fields
                                item_name = item.get('video_name', '')
                                item_type = item.get('motion_type', '')
                                if item_name == motion_type or item_type == motion_type:
                                    if 'coordinates' in item:
                                        ref_skeleton = item['coordinates']
                                        print(f"  ✓ Found reference for motion_type: {motion_type}")
                                        break
                        if ref_skeleton is None:
                            print(f"  ⚠ Motion type '{motion_type}' not found, using first available reference")
                            # Fallback to first item if not found
                            if len(data) > 0:
                                if isinstance(data[0], dict) and 'coordinates' in data[0]:
                                    ref_skeleton = data[0]['coordinates']
                                else:
                                    ref_skeleton = data[0]
                            else:
                                raise ValueError("Empty reference data")
                    else:
                        # Use first item if no motion_type specified
                        if len(data) > 0:
                            if isinstance(data[0], dict) and 'coordinates' in data[0]:
                                ref_skeleton = data[0]['coordinates']
                            else:
                                ref_skeleton = data[0]
                        else:
                            raise ValueError("Empty reference data")
                elif isinstance(data, dict):
                    if 'coordinates' in data:
                        ref_skeleton = data['coordinates']
                    else:
                        # Try to find a standard reference
                        ref_skeleton = None
                        for key in data.keys():
                            if isinstance(data[key], dict) and 'coordinates' in data[key]:
                                ref_skeleton = data[key]['coordinates']
                                break
                        if ref_skeleton is None:
                            raise ValueError("Could not find coordinates in reference data")
                else:
                    ref_skeleton = data
            ref_frames = None  # No video frames in pickle
            ref_uvd_coords = None  # No UVD coordinates in pickle
            print(f"  ✓ Loaded reference skeleton: {ref_skeleton.shape}")
        else:
            # If it's a video, extract skeleton (needs HybrIK)
            print(f"  Extracting reference skeleton from video...")
            ref_skeleton, ref_uvd_coords, ref_frames = self.extract_skeleton_from_video(reference_path)
            print(f"  ✓ Loaded reference skeleton: {ref_skeleton.shape}")
            if ref_uvd_coords is not None:
                print(f"  ✓ Loaded reference UVD coordinates: {ref_uvd_coords.shape}")
        
        return ref_skeleton, ref_uvd_coords, ref_frames
    
    def run_inference(self, learner_skeleton, reference_skeleton, video_name='user_video'):
        """
        Step 3: Run inference to generate instruction
        
        Args:
            learner_skeleton: [num_frames, 66]
            reference_skeleton: [num_frames, 66]
            video_name: name for output files
        
        Returns:
            instruction: generated text
            attention_data: attention weights for visualization
        """
        print(f"\n[Step 3/5] Running inference...")
        
        # Preprocess skeletons
        if isinstance(learner_skeleton, torch.Tensor):
            learner_skeleton = learner_skeleton.cpu().numpy()
        if isinstance(reference_skeleton, torch.Tensor):
            reference_skeleton = reference_skeleton.cpu().numpy()
        
        # Resample to same length if different
        learner_len = len(learner_skeleton)
        ref_len = len(reference_skeleton)
        
        if learner_len != ref_len:
            print(f"  ⚠ Sequence length mismatch: learner={learner_len}, reference={ref_len}")
            print(f"  Resampling to match lengths...")
            from scipy.interpolate import interp1d
            
            # Use the shorter length to avoid information loss
            target_length = min(learner_len, ref_len)
            
            if learner_len != target_length:
                # Resample learner skeleton
                original_indices = np.linspace(0, learner_len - 1, learner_len)
                target_indices = np.linspace(0, learner_len - 1, target_length)
                resampled_learner = np.zeros((target_length, learner_skeleton.shape[1]))
                for i in range(learner_skeleton.shape[1]):
                    f = interp1d(original_indices, learner_skeleton[:, i], kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
                    resampled_learner[:, i] = f(target_indices)
                learner_skeleton = resampled_learner
                print(f"  ✓ Resampled learner skeleton: {learner_len} -> {target_length}")
            
            if ref_len != target_length:
                # Resample reference skeleton
                original_indices = np.linspace(0, ref_len - 1, ref_len)
                target_indices = np.linspace(0, ref_len - 1, target_length)
                resampled_ref = np.zeros((target_length, reference_skeleton.shape[1]))
                for i in range(reference_skeleton.shape[1]):
                    f = interp1d(original_indices, reference_skeleton[:, i], kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
                    resampled_ref[:, i] = f(target_indices)
                reference_skeleton = resampled_ref
                print(f"  ✓ Resampled reference skeleton: {ref_len} -> {target_length}")
        
        # Convert to [6, frames, 22] format
        skeleton_coords = get_coords(learner_skeleton)  # [6, frames, 22]
        std_coords = get_coords(reference_skeleton)     # [6, frames, 22]
        
        # Convert to tensors
        skeleton_coords = torch.FloatTensor(skeleton_coords).unsqueeze(0).cuda()
        std_coords = torch.FloatTensor(std_coords).unsqueeze(0).cuda()
        seq_len = skeleton_coords.shape[2]
        frame_mask = torch.ones(1, seq_len).cuda()
        
        # Prepare decoder input
        decoder_input_ids = self.tokenizer(
            [self.prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=160,
            add_special_tokens=False
        )['input_ids'].cuda()
        
        # Run inference
        inputs = {
            "video_name": [video_name],
            "skeleton_coords": skeleton_coords,
            "frame_mask": frame_mask,
            "seq_len": [seq_len],
            "std_coords": std_coords,
            "decoder_input_ids": decoder_input_ids,
            "subtraction": torch.empty(0),
            "tokenizer": self.tokenizer,
            "result_dir": self.cfg.LOGDIR,
            "epoch": "inference"
        }
        
        with torch.no_grad():
            generated_ids, att_node, att_graph, max_indices = self.model.generate(**inputs)
        
        # Decode instruction
        decoded = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        instruction = decoded.split(self.prompt)[1].strip() if self.prompt in decoded else decoded.strip()
        
        # Extract attention data
        # att_node shape might be [num_heads, frames, 22] or [frames, 22]
        if isinstance(att_node, torch.Tensor):
            att_node_np = att_node[0].cpu().numpy() if len(att_node.shape) > 2 else att_node.cpu().numpy()
        else:
            att_node_np = att_node[0] if isinstance(att_node, (list, tuple)) else att_node
        
        # Ensure att_node_np is 2D: [frames, 22]
        if len(att_node_np.shape) == 3:
            # Average over heads if multiple heads
            att_node_np = att_node_np.mean(axis=0)
        elif len(att_node_np.shape) == 1:
            # If it's 1D, reshape to [1, 22] or [frames, 1]
            if att_node_np.shape[0] == 22:
                att_node_np = att_node_np.reshape(1, 22)
            else:
                att_node_np = att_node_np.reshape(-1, 1)
        
        # Resample attention to match skeleton length if needed
        att_frames = att_node_np.shape[0]
        if att_frames != seq_len:
            print(f"  ⚠ Attention frames ({att_frames}) != skeleton frames ({seq_len}), resampling attention...")
            from scipy.interpolate import interp1d
            original_indices = np.linspace(0, att_frames - 1, att_frames)
            target_indices = np.linspace(0, att_frames - 1, seq_len)
            resampled_att = np.zeros((seq_len, att_node_np.shape[1]))
            for j in range(att_node_np.shape[1]):  # For each joint
                f = interp1d(original_indices, att_node_np[:, j], kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
                resampled_att[:, j] = f(target_indices)
            att_node_np = resampled_att
            print(f"  ✓ Resampled attention: {att_frames} -> {seq_len} frames")
        
        attention_data = {
            'attention_node': att_node_np,      # [frames, 22] - now matches skeleton length
            'attention_graph': att_graph[0].cpu().numpy() if isinstance(att_graph, torch.Tensor) else att_graph[0],
            'max_indices': max_indices[0].cpu().numpy() if isinstance(max_indices, torch.Tensor) else max_indices[0]
        }
        
        print(f"  ✓ Generated instruction: {instruction}")
        print(f"  ✓ Attention shape: {att_node_np.shape}, Skeleton frames: {seq_len}")
        return instruction, attention_data
    
    def skeleton_to_joints_2d(self, skeleton_frame, frame_img_shape=None, uvd_frame=None):
        """
        Convert skeleton [66] to 22 joint positions for 2D visualization
        
        Args:
            skeleton_frame: [66] array (x0,y0,z0, x1,y1,z1, ..., x21,y21,z21)
            frame_img_shape: (height, width) of frame for normalization
            uvd_frame: [66] array (u0,v0,d0, u1,v1,d1, ..., u21,v21,d21) - UVD coordinates from HybrIK (more accurate for video overlay)
        
        Returns:
            joints: [22, 2] array of (x, y) positions in image coordinates
        """
        # If UVD coordinates available, use them (more accurate for video overlay)
        if uvd_frame is not None and frame_img_shape is not None:
            h, w = frame_img_shape[:2]
            
            # Extract UV coordinates (first 2 values per joint)
            joints_2d = []
            for i in range(0, 66, 3):
                u, v, d = uvd_frame[i], uvd_frame[i+1], uvd_frame[i+2]
                
                # UVD coordinates from HybrIK are already transformed to pixel coordinates
                # (transformed using bbox in hybrik_extractor.py)
                # Use directly as pixel coordinates
                x = u
                y = v
                
                # Clamp to image bounds
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                
                joints_2d.append([x, y])
            
            joints_2d = np.array(joints_2d)
            
            # UVD coordinates from HybrIK are already in pixel coordinates with origin at top-left
            # matplotlib imshow also uses origin='upper' by default, so no Y-axis inversion needed
            # Inversion would cause skeleton to be flipped vertically
            # joints_2d[:, 1] = h - joints_2d[:, 1]  # REMOVED: causes skeleton to be flipped
            
            return joints_2d
        
        # Fallback to 3D projection (existing code)
        # Extract 3D coordinates
        joints_3d = []
        for i in range(0, 66, 3):
            x, y, z = skeleton_frame[i], skeleton_frame[i+1], skeleton_frame[i+2]
            joints_3d.append([x, y, z])
        joints_3d = np.array(joints_3d)
        
        # Check if skeleton is already normalized (like from pickle files)
        # Normalized skeletons typically have values in range [-1, 1] or small values
        max_abs_value = np.abs(joints_3d).max()
        is_normalized = max_abs_value < 2.0  # If max value is less than 2, likely normalized
        
        if is_normalized:
            # Skeleton from pickle: already normalized, use directly
            # Center at root joint (joint 0) for consistent view
            root_joint = joints_3d[0].copy()
            joints_3d_centered = joints_3d - root_joint
            joints_2d = joints_3d_centered[:, :2].copy()
        else:
            # Skeleton from HybrIK: raw 3D coordinates, need normalization
            # Center skeleton at root joint (relative to root joint, which is typically joint 0)
            root_joint = joints_3d[0].copy()
            joints_3d_centered = joints_3d - root_joint
            
            # Project to 2D: use X and Y
            joints_2d = joints_3d_centered[:, :2].copy()
            
            # Normalize to similar scale as pickle skeletons (roughly [-1, 1] range)
            # Find bounding box of skeleton
            x_min, x_max = joints_2d[:, 0].min(), joints_2d[:, 0].max()
            y_min, y_max = joints_2d[:, 1].min(), joints_2d[:, 1].max()
            
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0
            
            # Normalize to [-1, 1] range (preserve aspect ratio)
            if x_range > 0:
                joints_2d[:, 0] = (joints_2d[:, 0] - x_min) / x_range * 2 - 1
            if y_range > 0:
                joints_2d[:, 1] = (joints_2d[:, 1] - y_min) / y_range * 2 - 1
        
        # Scale to image coordinates if frame shape provided
        if frame_img_shape is not None:
            h, w = frame_img_shape[:2]
            # Scale to fit in image while preserving aspect ratio
            # Use larger scale for better visibility
            scale = min(w, h) * 0.45  # Use 45% of smaller dimension
            center_x, center_y = w / 2, h / 2
            
            joints_2d[:, 0] = joints_2d[:, 0] * scale + center_x
            joints_2d[:, 1] = joints_2d[:, 1] * scale + center_y
            
            # Invert Y-axis for image coordinates (origin at top-left)
            joints_2d[:, 1] = h - joints_2d[:, 1]
        
        return joints_2d
    
    def visualize_skeleton_overlay(self, video_frames, skeleton_coords, 
                                   ref_frames, ref_skeleton_coords,
                                   attention_weights, output_path='output_comparison.mp4',
                                   uvd_coords=None, ref_uvd_coords=None):
        """
        Step 4: Create side-by-side video with skeleton overlay and attention visualization
        
        Args:
            video_frames: list of learner video frames
            skeleton_coords: [num_frames, 66] learner skeleton
            ref_frames: list of reference video frames
            ref_skeleton_coords: [num_frames, 66] reference skeleton
            attention_weights: attention per joint per frame [frames, 22]
            output_path: path to save output video
            uvd_coords: [num_frames, 66] UVD coordinates for learner (for accurate 2D overlay)
            ref_uvd_coords: [num_frames, 66] UVD coordinates for reference (for accurate 2D overlay)
        """
        print(f"\n[Step 4/5] Creating visualization video...")
        
        # Skeleton connections (bone structure) - 22 joints
        connections = [
            (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
            (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
            (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
            (16, 18), (17, 19), (18, 20), (19, 21)
        ]
        
        # Normalize attention weights for color mapping
        # Ensure attention is always visible and red when present
        if attention_weights is not None and attention_weights.size > 0:
            # Check if attention_weights is valid
            if isinstance(attention_weights, np.ndarray):
                att_min = attention_weights.min()
                att_max = attention_weights.max()
                att_range = att_max - att_min + 1e-8
                att_norm = (attention_weights - att_min) / att_range
                # Ensure minimum visibility: if range is too small, scale it up
                if att_range < 0.1:
                    att_norm = attention_weights / (att_max + 1e-8)  # Normalize to [0, 1]
                print(f"  Attention weights range: [{att_min:.4f}, {att_max:.4f}], normalized to [0, 1]")
            else:
                print(f"  ⚠ Warning: attention_weights is not numpy array: {type(attention_weights)}")
                att_norm = None
        else:
            print(f"  ⚠ Warning: attention_weights is None or empty")
            att_norm = None
        
        # Determine number of frames
        num_frames_learner = len(skeleton_coords)
        num_frames_ref = len(ref_skeleton_coords) if ref_skeleton_coords is not None else 0
        
        # Use attention length as ground truth since skeletons should match attention after resampling
        if attention_weights is not None and len(attention_weights) > 0:
            num_frames = len(attention_weights)
        else:
            num_frames = max(num_frames_learner, num_frames_ref) if num_frames_ref > 0 else num_frames_learner
        
        # Calculate unified view bounds from both skeletons for consistent camera view
        # This ensures learner and reference are viewed from the same angle/scale
        all_joints_learner = []
        all_joints_ref = []
        for frame_idx in range(min(10, num_frames_learner)):  # Sample first 10 frames
            joints_temp = self.skeleton_to_joints_2d(skeleton_coords[frame_idx], None)
            all_joints_learner.append(joints_temp)
        if ref_skeleton_coords is not None:
            for frame_idx in range(min(10, num_frames_ref)):
                joints_temp = self.skeleton_to_joints_2d(ref_skeleton_coords[frame_idx], None)
                all_joints_ref.append(joints_temp)
        
        # Get unified bounds
        if all_joints_learner:
            all_learner = np.vstack(all_joints_learner)
            x_min_learner, x_max_learner = all_learner[:, 0].min(), all_learner[:, 0].max()
            y_min_learner, y_max_learner = all_learner[:, 1].min(), all_learner[:, 1].max()
        else:
            x_min_learner = x_max_learner = y_min_learner = y_max_learner = 0
        
        if all_joints_ref:
            all_ref = np.vstack(all_joints_ref)
            x_min_ref, x_max_ref = all_ref[:, 0].min(), all_ref[:, 0].max()
            y_min_ref, y_max_ref = all_ref[:, 1].min(), all_ref[:, 1].max()
        else:
            x_min_ref = x_max_ref = y_min_ref = y_max_ref = 0
        
        # Use unified bounds (min of mins, max of maxs) for consistent view
        unified_x_min = min(x_min_learner, x_min_ref) if all_joints_ref else x_min_learner
        unified_x_max = max(x_max_learner, x_max_ref) if all_joints_ref else x_max_learner
        unified_y_min = min(y_min_learner, y_min_ref) if all_joints_ref else y_min_learner
        unified_y_max = max(y_min_learner, y_max_ref) if all_joints_ref else y_max_learner
        
        # Calculate unified range and padding
        unified_x_range = unified_x_max - unified_x_min if unified_x_max > unified_x_min else 1.0
        unified_y_range = unified_y_max - unified_y_min if unified_y_max > unified_y_min else 1.0
        unified_padding = max(unified_x_range, unified_y_range) * 0.25
        
        # Store unified view bounds for use in draw_skeleton
        unified_view_bounds = {
            'x_min': unified_x_min - unified_padding,
            'x_max': unified_x_max + unified_padding,
            'y_min': unified_y_min - unified_padding,
            'y_max': unified_y_max + unified_padding
        }
        
        # Create figure with 2 subplots (side by side) + space for legend
        # Use 16:9 aspect ratio: width = height * 16/9
        # Using height=10.8 gives width=19.2 for 16:9 ratio
        fig = plt.figure(figsize=(19.2, 10.8), facecolor='white')
        # More generous margins: left, right, top, bottom - prevent clipping
        # Increased legend height ratio to prevent text overlap
        # Increased top margin since we removed suptitle - more space for subplot titles
        # Wider left/right margins for 16:9 aspect ratio to prevent legend text cutoff
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.12], width_ratios=[1, 1], 
                             hspace=0.45, wspace=0.35, left=0.08, right=0.92, top=0.92, bottom=0.12)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[1, :])
        
        # Removed suptitle to prevent overlap with subplot titles
        # Titles are now only in individual subplots (learner and reference)
        
        def draw_skeleton(ax, frame_img, skeleton_frame, attention_frame=None, title="", is_learner=True, unified_bounds=None, uvd_frame=None):
            """Draw skeleton overlay on frame - NO OVERLAP, separate panels
            
            Args:
                unified_bounds: dict with 'x_min', 'x_max', 'y_min', 'y_max' for consistent view
                uvd_frame: [66] UVD coordinates for accurate 2D overlay (if available)
            """
            ax.clear()
            ax.set_facecolor('#ffffff')
            
            if frame_img is not None:
                # Use origin='upper' explicitly to ensure Y=0 is at top (default for imshow)
                # This matches the UVD coordinates which have origin at top-left
                # UVD coordinates: Y=0 at top, Y=height at bottom (pixel coordinates)
                # imshow with origin='upper': Y=0 at top, Y=height at bottom
                # So they match perfectly - no inversion needed
                ax.imshow(frame_img, aspect='auto', origin='upper', extent=[0, frame_img.shape[1], frame_img.shape[0], 0])
                # extent=[x_min, x_max, y_min, y_max] where y_min=height (bottom), y_max=0 (top)
                # This explicitly sets the coordinate system to match origin='upper'
                ax.set_xlim(0, frame_img.shape[1])
                ax.set_ylim(frame_img.shape[0], 0)  # Y=height at bottom, Y=0 at top (matches origin='upper' and extent)
            else:
                # Create blank canvas with unified view bounds for consistent camera angle
                # Use unified bounds if provided to ensure same view for learner and reference
                if unified_bounds is not None:
                    ax.set_xlim(unified_bounds['x_min'], unified_bounds['x_max'])
                    ax.set_ylim(unified_bounds['y_max'], unified_bounds['y_min'])  # Invert Y-axis
                else:
                    # Fallback: normalize skeleton to fit in a standard view
                    joints_temp = self.skeleton_to_joints_2d(skeleton_frame, None)
                    if len(joints_temp) > 0 and joints_temp.size > 0:
                        x_min, x_max = joints_temp[:, 0].min(), joints_temp[:, 0].max()
                        y_min, y_max = joints_temp[:, 1].min(), joints_temp[:, 1].max()
                        x_range = x_max - x_min if x_max > x_min else 1.0
                        y_range = y_max - y_min if y_max > y_min else 1.0
                        
                        # Add generous padding to prevent clipping
                        padding = max(x_range, y_range) * 0.25
                        ax.set_xlim(x_min - padding, x_max + padding)
                        ax.set_ylim(y_max + padding, y_min - padding)  # Invert Y-axis
                    else:
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(1, -1)  # Invert Y-axis
                ax.set_aspect('equal')
                ax.set_facecolor('#f9f9f9')
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
            # Static title in corner - NO title update per frame
            # Title will be set once outside this function
            ax.axis('off')
            
            # Get joint positions (use UVD coordinates if available for accurate overlay)
            joints = self.skeleton_to_joints_2d(
                skeleton_frame, 
                frame_img.shape if frame_img is not None else None,
                uvd_frame=uvd_frame
            )
            
            # Draw bones first (so joints appear on top)
            for (i, j) in connections:
                if i < len(joints) and j < len(joints):
                    if attention_frame is not None and len(attention_frame) > 0 and i < len(attention_frame) and j < len(attention_frame):
                        # Color based on attention using Reds colormap (red = high attention)
                        att_val = float((attention_frame[i] + attention_frame[j]) / 2.0)
                        # Ensure att_val is in [0, 1] range
                        att_val = max(0.0, min(1.0, att_val))
                        # Map to [0.3, 1.0] range so even low attention is visible as light red
                        color = plt.cm.Reds(0.3 + att_val * 0.7)  # Range from light red to dark red
                        linewidth = 2.5 + att_val * 4  # Thicker line for higher attention
                    else:
                        # Default colors when no attention
                        if is_learner:
                            color = '#ff9999'  # Light red for learner (should have attention)
                        else:
                            color = '#81c784'  # Green for reference (no attention overlay)
                        linewidth = 2.5
                    ax.plot([joints[i, 0], joints[j, 0]],
                           [joints[i, 1], joints[j, 1]],
                           color=color, linewidth=linewidth, alpha=0.9, zorder=5)
            
            # Draw joints
            for i, joint in enumerate(joints):
                if attention_frame is not None and i < len(attention_frame) and len(attention_frame) > 0:
                    color_val = float(attention_frame[i])
                    # Ensure color_val is in [0, 1] range
                    color_val = max(0.0, min(1.0, color_val))
                    # Use Reds colormap - ensure always visible red color
                    # Map to [0.3, 1.0] range so even low attention is visible as light red
                    color = plt.cm.Reds(0.3 + color_val * 0.7)  # Range from light red (0.3) to dark red (1.0)
                    size = 60 + color_val * 120
                    # Edge color: dark red for high attention, dark gray for low
                    edge_color = '#8b0000' if color_val > 0.6 else '#333333'
                    edge_width = 2.5 if color_val > 0.6 else 1.5
                else:
                    # Default colors when no attention (only for reference or when attention is missing)
                    if is_learner:
                        # For learner without attention, use light red to indicate attention should be there
                        color = '#ff9999'  # Light red
                    else:
                        # For reference, use green (no attention overlay)
                        color = '#66bb6a'  # Green
                    size = 50
                    edge_color = '#333333'
                    edge_width = 1.5
                ax.scatter(joint[0], joint[1], c=[color], s=size, 
                          edgecolors=edge_color, linewidths=edge_width, zorder=10, alpha=0.9)
        
        def draw_legend():
            """Draw legend for attention weights - clean, no dashed lines, no text clipping"""
            ax_legend.clear()
            ax_legend.axis('off')
            ax_legend.set_facecolor('white')
            
            # Set limits with safe margins to prevent overflow - increased vertical space
            # Wider limits to prevent text from being cut off at edges
            ax_legend.set_xlim(0, 100)  # Clean limits
            ax_legend.set_ylim(0, 1.5)  # Increased vertical space to prevent overlap
            
            # Main label at top - centered with safe margins to prevent frame cutoff
            # Position more towards center with wider safe zone
            ax_legend.text(50, 1.25, 'Attention: Redder = Higher Focus', 
                          ha='center', va='top', fontsize=11, fontweight='bold', 
                          color='#333')  # Simple text, no bbox
            
            # Create colorbar gradient using Reds colormap - positioned lower, centered
            # Use wider margins (10-90 instead of 5-95) to prevent cutoff
            gradient = np.linspace(0, 1, 100).reshape(1, -1)
            im = ax_legend.imshow(gradient, aspect='auto', cmap=plt.cm.Reds, 
                                 extent=[10, 90, 0.5, 0.7], interpolation='nearest')
            
            # Add labels below colorbar with safe spacing - NO overlap, centered
            ax_legend.text(15, 0.35, 'Low', ha='left', va='center', fontsize=9, 
                         color='#333', fontweight='bold')
            ax_legend.text(50, 0.35, 'Medium', ha='center', va='center', fontsize=9, 
                         color='#d32f2f', fontweight='bold')
            ax_legend.text(85, 0.35, 'High', ha='right', va='center', fontsize=9, 
                         color='#b71c1c', fontweight='bold')
        
        def animate(frame_idx):
            """Animation update function - NO OVERLAP, separate panels, no text overlap"""
            # Learner video with skeleton - LEFT PANEL ONLY
            learner_frame = video_frames[frame_idx % len(video_frames)] if video_frames else None
            learner_skeleton = skeleton_coords[frame_idx % len(skeleton_coords)]
            
            # Get attention frame for this specific frame
            att_frame = None
            if att_norm is not None and att_norm.size > 0:
                if len(att_norm.shape) == 2:
                    # Use modulo to handle frame_idx beyond attention length
                    att_idx = frame_idx % att_norm.shape[0]
                    att_frame = att_norm[att_idx]
                    # Ensure att_frame is 1D array with 22 elements
                    if len(att_frame.shape) > 1:
                        att_frame = att_frame.flatten()
                    # Debug: print attention values for first and problematic frames
                    if frame_idx == 0:
                        print(f"  Frame 0 attention range: [{att_frame.min():.4f}, {att_frame.max():.4f}]")
                    elif frame_idx == 70:
                        print(f"  Frame 70 attention range: [{att_frame.min():.4f}, {att_frame.max():.4f}], shape: {att_frame.shape}")
                        print(f"  Attention array shape: {att_norm.shape}, frame_idx: {frame_idx}, att_idx: {att_idx}")
                elif len(att_norm.shape) == 1 and att_norm.shape[0] == 22:
                    # If attention is same for all frames, use it
                    att_frame = att_norm
                else:
                    # Fallback: if shape doesn't match, try to use first frame's attention
                    if frame_idx == 70:
                        print(f"  ⚠ Frame 70: att_norm shape {att_norm.shape} doesn't match expected format")
            else:
                if frame_idx == 70:
                    print(f"  ⚠ Frame 70: att_norm is None or empty")
            
            # Get UVD coordinates for learner if available
            learner_uvd = None
            if uvd_coords is not None:
                learner_uvd = uvd_coords[frame_idx % len(uvd_coords)]
            
            # Draw ONLY learner skeleton in left panel - NO title update (static title set outside)
            # Use unified bounds to ensure same camera view as reference
            draw_skeleton(ax1, learner_frame, learner_skeleton, att_frame, 
                         "", is_learner=True, unified_bounds=unified_view_bounds, uvd_frame=learner_uvd)
            
            # Add title in top left corner (re-added after ax.clear())
            # Format: Learner [current_frame/total_frames] - updates per frame
            # Color: Orange to match learner landmark color
            # Use transform to position relative to axes (0,0 = bottom-left, 1,1 = top-right)
            ax1.text(0.02, 0.98, f"Learner [{frame_idx + 1}/{num_frames}]", 
                     fontsize=11, fontweight='bold', color='#ff9800',
                     ha='left', va='top', antialiased=True, zorder=100,
                     transform=ax1.transAxes)
            
            # Reference video with skeleton - RIGHT PANEL ONLY (NO OVERLAP)
            if ref_skeleton_coords is not None:
                # Use modulo to handle frame index safely
                ref_frame = ref_frames[frame_idx % len(ref_frames)] if ref_frames else None
                ref_skeleton = ref_skeleton_coords[frame_idx % len(ref_skeleton_coords)]
                
                # Get UVD coordinates for reference if available
                ref_uvd = None
                if ref_uvd_coords is not None:
                    ref_uvd = ref_uvd_coords[frame_idx % len(ref_uvd_coords)]
                
                # Draw ONLY reference skeleton in right panel (no attention overlay) - NO title update
                # Use unified bounds to ensure same camera view as learner
                draw_skeleton(ax2, ref_frame, ref_skeleton, None,
                             "", is_learner=False, unified_bounds=unified_view_bounds, uvd_frame=ref_uvd)
                
                # Add title in top right corner (re-added after ax.clear())
                # Format: Reference [current_frame/total_frames] - updates per frame
                # Color: Green to match reference landmark color
                # Use transform to position relative to axes (0,0 = bottom-left, 1,1 = top-right)
                ax2.text(0.98, 0.98, f"Reference [{frame_idx + 1}/{num_frames}]",
                         fontsize=11, fontweight='bold', color='#66bb6a',
                         ha='right', va='top', antialiased=True, zorder=100,
                         transform=ax2.transAxes)
            else:
                ax2.clear()
                ax2.set_facecolor('#f9f9f9')
                # Simple text, no bbox
                ax2.text(0.5, 0.5, 'No Reference', ha='center', va='center', 
                        fontsize=12, color='#666', fontweight='bold')
                ax2.axis('off')
            
            # Update legend
            draw_legend()
            
            # Ensure tight layout to prevent overflow - with safe margins
            # Adjusted for new margins (left=0.08, right=0.92, top=0.92, bottom=0.12) for 16:9 aspect ratio
            plt.tight_layout(rect=[0.08, 0.12, 0.92, 0.92])
        
        # Create animation
        print(f"  Creating animation with {num_frames} frames...")
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=100, repeat=True)
        
        # Save video using reliable method: save frames as images then combine with ffmpeg
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        frame_pattern = os.path.join(temp_dir, 'frame_%06d.png')
        
        try:
            # Render and save each frame as PNG
            print(f"  Rendering {num_frames} frames to temporary images...")
            for frame_idx in range(num_frames):
                animate(frame_idx)
                fig.canvas.draw()
                # Save frame as PNG
                frame_path = os.path.join(temp_dir, f'frame_{frame_idx+1:06d}.png')
                fig.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
                if (frame_idx + 1) % 10 == 0:
                    print(f"    Rendered {frame_idx + 1}/{num_frames} frames...")
            
            # Get actual image dimensions
            first_frame = cv2.imread(os.path.join(temp_dir, 'frame_000001.png'))
            if first_frame is None:
                raise Exception("Failed to read first frame")
            h, w = first_frame.shape[:2]
            # Ensure even dimensions for H.264
            w = w if w % 2 == 0 else w + 1
            h = h if h % 2 == 0 else h + 1
            
            print(f"  Combining frames into video ({w}x{h})...")
            
            # Try using ffmpeg directly (most reliable)
            try:
                import subprocess
                ffmpeg_cmd = [
                    'ffmpeg', '-y',  # Overwrite output
                    '-framerate', '10',  # 10 fps
                    '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                    '-c:v', 'libx264',  # H.264 codec
                    '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                    '-crf', '23',  # Quality (lower = better, 18-28 is good range)
                    '-preset', 'medium',  # Encoding speed
                    output_path
                ]
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(f"  ✓ Visualization saved to {output_path} (using ffmpeg)")
                else:
                    raise Exception(f"ffmpeg failed: {result.stderr}")
            except FileNotFoundError:
                # ffmpeg not available, try cv2.VideoWriter
                print(f"  ffmpeg not found, using cv2.VideoWriter...")
                
                # Use AVI format with MJPG codec (most compatible, always works)
                # MP4 with OpenCV VideoWriter often has codec compatibility issues
                avi_path = output_path.replace('.mp4', '.avi')
                print(f"  Using AVI format with Motion JPEG codec for maximum compatibility...")
                
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(avi_path, fourcc, 10.0, (w, h))
                
                if not out.isOpened():
                    # Try alternative: use XVID codec with AVI
                    print(f"  ⚠ MJPG failed, trying XVID codec with AVI format...")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(avi_path, fourcc, 10.0, (w, h))
                    used_codec = 'XVID (AVI)'
                else:
                    used_codec = 'Motion JPEG (AVI)'
                
                if not out.isOpened():
                    raise Exception("Cannot open VideoWriter with any codec")
                
                output_path = avi_path  # Update output path to AVI
                print(f"  ✓ Using {used_codec} codec")
                
                # Read and write frames
                for frame_idx in range(num_frames):
                    frame_path = os.path.join(temp_dir, f'frame_{frame_idx+1:06d}.png')
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        # Resize if dimensions don't match
                        if frame.shape[:2] != (h, w):
                            frame = cv2.resize(frame, (w, h))
                        out.write(frame)
                
                out.release()
                file_format = os.path.splitext(output_path)[1].upper().replace('.', '')
                print(f"  ✓ Visualization saved to {output_path} (using cv2 with {used_codec}, {w}x{h}, format: {file_format})")
                # Return actual output path (may be .avi instead of .mp4)
                return output_path
                
        except Exception as e:
            print(f"  ⚠ Video saving failed: {e}")
            # Last resort: save as GIF
            try:
                gif_path = output_path.replace('.mp4', '.gif')
                print(f"  Trying to save as GIF instead...")
                # Save frames as GIF using imageio or pillow
                try:
                    import imageio
                    frames = []
                    for frame_idx in range(num_frames):
                        frame_path = os.path.join(temp_dir, f'frame_{frame_idx+1:06d}.png')
                        if os.path.exists(frame_path):
                            frames.append(imageio.imread(frame_path))
                    if frames:
                        imageio.mimsave(gif_path, frames, fps=10)
                        print(f"  ✓ Visualization saved to {gif_path} (GIF format)")
                except ImportError:
                    # Fallback to pillow
                    from PIL import Image
                    frames = []
                    for frame_idx in range(num_frames):
                        frame_path = os.path.join(temp_dir, f'frame_{frame_idx+1:06d}.png')
                        if os.path.exists(frame_path):
                            frames.append(Image.open(frame_path))
                    if frames:
                        frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                                     duration=100, loop=0)  # 100ms = 10fps
                        print(f"  ✓ Visualization saved to {gif_path} (GIF format)")
                        return gif_path
            except Exception as e3:
                print(f"  ❌ All video saving methods failed:")
                print(f"     - Main method: {e}")
                print(f"     - GIF fallback: {e3}")
                print(f"  Skipping video output. Attention timeline will still be saved.")
                return None
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"  ✓ Cleaned up temporary files")
                except Exception as e_cleanup:
                    print(f"  ⚠ Warning: Could not clean up temp directory: {e_cleanup}")
                    print(f"  ❌ All video saving methods failed:")
                    print(f"     - ffmpeg: {e1}")
                    print(f"     - cv2: {e2}")
                    print(f"     - GIF: {e3}")
                    print(f"  Skipping video output. Attention timeline will still be saved.")
        
        plt.close()
    
    def visualize_attention_timeline(self, attention_data, output_path='attention_timeline.png'):
        """
        Step 5: Visualize attention weights over time for each joint
        
        Args:
            attention_data: dict with 'attention_node' [frames, 22]
            output_path: path to save plot
        """
        print(f"\n[Step 5/5] Creating attention timeline visualization...")
        
        att_node = attention_data['attention_node']
        
        # Ensure 2D shape [frames, 22]
        if len(att_node.shape) == 1:
            if att_node.shape[0] == 22:
                att_node = att_node.reshape(1, 22)
            else:
                att_node = att_node.reshape(-1, 1)
        elif len(att_node.shape) == 3:
            # Average over first dimension if needed
            att_node = att_node.mean(axis=0)
        
        # If still not 2D, try to reshape
        if len(att_node.shape) != 2:
            print(f"  ⚠ Warning: Unexpected attention shape {att_node.shape}, attempting to reshape...")
            att_node = att_node.reshape(-1, 22) if att_node.size % 22 == 0 else att_node.reshape(1, -1)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(att_node.T, aspect='auto', cmap='Reds', interpolation='nearest')
        ax.set_xlabel('Frame', fontsize=12, fontweight='bold')
        ax.set_ylabel('Joint Index', fontsize=12, fontweight='bold')
        ax.set_title('Attention Weights Over Time (per Joint)\nRed = High Attention, White = Low Attention', 
                     fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=10)
        
        # Joint labels (simplified - 22 joints)
        joint_names = [f'J{i}' for i in range(22)]
        ax.set_yticks(range(22))
        ax.set_yticklabels(joint_names)
        
        # Add frame ticks (every 10 frames)
        num_frames = att_node.shape[0]
        frame_ticks = list(range(0, num_frames, max(1, num_frames // 10)))
        ax.set_xticks(frame_ticks)
        ax.set_xticklabels([str(f) for f in frame_ticks])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Attention timeline saved to {output_path}")
    
    def run_full_workflow(self, user_video_path=None, user_skeleton_path=None,
                         reference_path=None, output_dir='./workflow_output',
                         video_name='user_video', motion_type=None):
        """
        Complete workflow: Video -> Skeleton -> Inference -> Visualization
        
        Args:
            user_video_path: path to user video (optional)
            user_skeleton_path: path to user skeleton pickle (required if no video)
            reference_path: path to reference video or skeleton pickle (required)
            output_dir: directory to save all outputs
            video_name: name for output files
            motion_type: optional motion type/class (e.g., Single_Axel, Double_Axel, Loop, Lutz for Skating,
                       or Jab, Cross for Boxing). If provided, will search for matching reference in dataset.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("COACHME WORKFLOW: Complete Inference Pipeline")
        print("="*60)
        
        # Step 1: Extract skeleton from user video
        # Debug: Print paths for troubleshooting
        if user_skeleton_path:
            print(f"  DEBUG: user_skeleton_path = {user_skeleton_path}")
            print(f"  DEBUG: user_skeleton_path exists = {os.path.exists(user_skeleton_path)}")
        if user_video_path:
            print(f"  DEBUG: user_video_path = {user_video_path}")
            print(f"  DEBUG: user_video_path exists = {os.path.exists(user_video_path) if user_video_path else False}")
        
        learner_skeleton, learner_uvd, learner_frames = self.extract_skeleton_from_video(
            video_path=user_video_path,
            skeleton_pkl_path=user_skeleton_path
        )
        
        # Step 2: Load reference
        ref_skeleton, ref_uvd, ref_frames = self.load_reference_video(reference_path, motion_type=motion_type)
        
        if ref_skeleton is None:
            raise ValueError("Reference skeleton is required for inference")
        
        # Step 3: Run inference
        instruction, attention_data = self.run_inference(
            learner_skeleton, ref_skeleton, video_name=video_name
        )
        
        # Save instruction
        instruction_path = f'{output_dir}/instruction.txt'
        with open(instruction_path, 'w') as f:
            f.write(instruction)
        print(f"\n✓ Instruction saved to {instruction_path}")
        print(f"  Instruction: {instruction}")
        
        # Resample skeletons and frames to match attention length for visualization
        # Attention weights are already resampled to match inference skeleton length
        att_node = attention_data.get('attention_node')
        if att_node is not None and len(att_node) > 0:
            target_length = len(att_node)
            learner_len = len(learner_skeleton)
            ref_len = len(ref_skeleton)
            
            # Resample learner skeleton and frames if needed
            if learner_len != target_length:
                print(f"  Resampling learner skeleton for visualization: {learner_len} -> {target_length}")
                from scipy.interpolate import interp1d
                original_indices = np.linspace(0, learner_len - 1, learner_len)
                target_indices = np.linspace(0, learner_len - 1, target_length)
                resampled_learner = np.zeros((target_length, learner_skeleton.shape[1]))
                for i in range(learner_skeleton.shape[1]):
                    f = interp1d(original_indices, learner_skeleton[:, i], kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
                    resampled_learner[:, i] = f(target_indices)
                learner_skeleton = resampled_learner
                
                # Resample frames (select frames evenly)
                if learner_frames and len(learner_frames) > 0:
                    frame_indices = np.linspace(0, len(learner_frames) - 1, target_length, dtype=int)
                    learner_frames = [learner_frames[i] for i in frame_indices]
            
            # Resample reference skeleton and frames if needed
            if ref_len != target_length:
                print(f"  Resampling reference skeleton for visualization: {ref_len} -> {target_length}")
                from scipy.interpolate import interp1d
                original_indices = np.linspace(0, ref_len - 1, ref_len)
                target_indices = np.linspace(0, ref_len - 1, target_length)
                resampled_ref = np.zeros((target_length, ref_skeleton.shape[1]))
                for i in range(ref_skeleton.shape[1]):
                    f = interp1d(original_indices, ref_skeleton[:, i], kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
                    resampled_ref[:, i] = f(target_indices)
                ref_skeleton = resampled_ref
                
                # Resample frames (select frames evenly)
                if ref_frames and len(ref_frames) > 0:
                    frame_indices = np.linspace(0, len(ref_frames) - 1, target_length, dtype=int)
                    ref_frames = [ref_frames[i] for i in frame_indices]
            
            # Resample UVD coordinates if available
            if learner_uvd is not None and len(learner_uvd) != target_length:
                print(f"  Resampling learner UVD coordinates for visualization: {len(learner_uvd)} -> {target_length}")
                from scipy.interpolate import interp1d
                original_indices = np.linspace(0, len(learner_uvd) - 1, len(learner_uvd))
                target_indices = np.linspace(0, len(learner_uvd) - 1, target_length)
                resampled_learner_uvd = np.zeros((target_length, learner_uvd.shape[1]))
                for i in range(learner_uvd.shape[1]):
                    f = interp1d(original_indices, learner_uvd[:, i], kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
                    resampled_learner_uvd[:, i] = f(target_indices)
                learner_uvd = resampled_learner_uvd
            
            if ref_uvd is not None and len(ref_uvd) != target_length:
                print(f"  Resampling reference UVD coordinates for visualization: {len(ref_uvd)} -> {target_length}")
                from scipy.interpolate import interp1d
                original_indices = np.linspace(0, len(ref_uvd) - 1, len(ref_uvd))
                target_indices = np.linspace(0, len(ref_uvd) - 1, target_length)
                resampled_ref_uvd = np.zeros((target_length, ref_uvd.shape[1]))
                for i in range(ref_uvd.shape[1]):
                    f = interp1d(original_indices, ref_uvd[:, i], kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
                    resampled_ref_uvd[:, i] = f(target_indices)
                ref_uvd = resampled_ref_uvd
        
        # Step 4: Create side-by-side visualization
        # Note: output may be .avi if MP4 codecs are not available
        comparison_path = f'{output_dir}/comparison_video.mp4'
        actual_output_path = self.visualize_skeleton_overlay(
            learner_frames, learner_skeleton,
            ref_frames, ref_skeleton,
            attention_data.get('attention_node'),
            output_path=comparison_path,
            uvd_coords=learner_uvd,
            ref_uvd_coords=ref_uvd
        )
        # Update comparison_path if format changed (e.g., to .avi)
        if actual_output_path and actual_output_path != comparison_path:
            comparison_path = actual_output_path
        
        # Step 5: Visualize attention timeline
        timeline_path = f'{output_dir}/attention_timeline.png'
        self.visualize_attention_timeline(
            attention_data,
            output_path=timeline_path
        )
        
        # Save attention data as JSON (convert numpy to list)
        attention_json_path = f'{output_dir}/attention_data.json'
        with open(attention_json_path, 'w') as f:
            json.dump({
                'attention_node': attention_data['attention_node'].tolist() if isinstance(attention_data['attention_node'], np.ndarray) else attention_data['attention_node'],
                'max_indices': attention_data['max_indices'].tolist() if isinstance(attention_data['max_indices'], np.ndarray) else attention_data['max_indices']
            }, f, indent=2)
        
        print("\n" + "="*60)
        print("WORKFLOW COMPLETE!")
        print("="*60)
        print(f"\nOutputs saved in: {output_dir}/")
        print("  - instruction.txt: Generated coaching instruction")
        # Show actual video filename (may be .avi or .gif instead of .mp4)
        if comparison_path:
            video_filename = os.path.basename(comparison_path)
            print(f"  - {video_filename}: Side-by-side video with skeleton overlay")
        else:
            print("  - comparison_video.*: Side-by-side video (format may vary: .mp4, .avi, or .gif)")
        print("  - attention_timeline.png: Attention weights over time")
        print("  - attention_data.json: Raw attention data")
        print("\n" + "="*60)
        
        return instruction, attention_data


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CoachMe Workflow: Complete Inference Pipeline')
    parser.add_argument('--config', type=str, default='./results/skating_gt/skating_gt.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, 
                       default='./results/skating_gt/checkpoints/checkpoint_epoch_00050.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--user_video', type=str, default=None,
                       help='Path to user video (optional if --user_skeleton provided)')
    parser.add_argument('--user_skeleton', type=str, default=None,
                       help='Path to user skeleton pickle file (required if no video)')
    parser.add_argument('--reference', type=str, required=True,
                       help='Path to reference skeleton pickle or video')
    parser.add_argument('--output_dir', type=str, default='./workflow_output',
                       help='Output directory for results')
    parser.add_argument('--video_name', type=str, default='user_video',
                       help='Name for output files')
    parser.add_argument('--motion_type', type=str, default=None,
                       help='Motion type/class (e.g., Single_Axel, Double_Axel, Loop, Lutz for Skating, or Jab, Cross for Boxing). If provided, will search for matching reference in standard dataset.')
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = CoachMeWorkflow(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    # Run full workflow
    try:
        instruction, attention = workflow.run_full_workflow(
            user_video_path=args.user_video,
            user_skeleton_path=args.user_skeleton,
            reference_path=args.reference,
            output_dir=args.output_dir,
            video_name=args.video_name,
            motion_type=args.motion_type
        )
        
        print(f"\n{'='*60}")
        print("FINAL INSTRUCTION:")
        print(f"{'='*60}")
        print(instruction)
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup distributed if we initialized it
        if dist.is_initialized():
            dist.destroy_process_group()