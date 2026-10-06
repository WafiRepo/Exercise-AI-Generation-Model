"""
HybrIK Skeleton Extraction Wrapper
Install HybrIK: https://github.com/Jeff-sjtu/HybrIK

This implementation uses HybrIK demo script to extract 3D skeleton from video.
Make sure HybrIK is installed and HYBRIK_PATH environment variable is set.

Preprocessing Integration:
- Uses the same preprocessing logic as BX_handle_frame_features.py (deal() function)
- Ensures consistency with training data preprocessing format
- Output format: torch.tensor [num_frames, 66] (converted to numpy for compatibility)
- Format: [x0, y0, z0, x1, y1, z1, ..., x21, y21, z21] (22 joints * 3 coords = 66 values)
"""
import torch
import numpy as np
import cv2
import os
import subprocess
import tempfile
import shutil
import pickle


def process_hybrik_output_preprocessing(res_pk_path):
    """
    Process HybrIK output file (res.pk) using the same preprocessing logic as BX_handle_frame_features.py
    This function matches the 'deal()' function from preprocessing scripts.
    
    Args:
        res_pk_path: path to HybrIK res.pk file
    
    Returns:
        coordinates: torch.tensor [num_frames, 66] (same format as preprocessing)
    """
    with open(res_pk_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'pred_xyz_24_struct' not in data:
        raise ValueError(
            f"HybrIK output file does not contain 'pred_xyz_24_struct'\n"
            f"Available keys: {list(data.keys())}"
        )
    
    processed = []
    for frame in data['pred_xyz_24_struct']:
        # 22 joints × 3 coords = 66
        # Same logic as deal() function in BX_handle_frame_features.py
        flattened = [coord for joint in frame[0:22] for coord in joint]
        processed.append(flattened)
    
    # Return torch.tensor (same format as preprocessing)
    coordinates = torch.tensor(processed, dtype=torch.float32)
    return coordinates


def _process_hybrik_output(res_pk_path, use_preprocessing_format=True):
    """
    Process HybrIK output file (res.pk) to extract skeleton coordinates
    Uses preprocessing logic for consistency with training data format.
    
    Args:
        res_pk_path: path to HybrIK res.pk file
        use_preprocessing_format: if True, uses preprocessing logic (same as deal() function)
    
    Returns:
        skeleton_coords: numpy array [num_frames, 66] (converted from torch.tensor for compatibility)
        uvd_coords: numpy array [num_frames, 66] (optional, for accurate 2D overlay)
    """
    # Use preprocessing format (same as BX_handle_frame_features.py)
    if use_preprocessing_format:
        coordinates_tensor = process_hybrik_output_preprocessing(res_pk_path)
        skeleton_coords = coordinates_tensor.numpy()  # Convert to numpy for compatibility
        print(f"✓ Processed {len(skeleton_coords)} frames from HybrIK output (using preprocessing format)")
    else:
        # Legacy format (for backward compatibility)
        with open(res_pk_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'pred_xyz_24_struct' not in data:
            raise ValueError(
                f"HybrIK output file does not contain 'pred_xyz_24_struct'\n"
                f"Available keys: {list(data.keys())}"
            )
        
        processed = []
        for frame_idx in range(len(data['pred_xyz_24_struct'])):
            frame_xyz = data['pred_xyz_24_struct'][frame_idx]
            flattened = [coord for joint in frame_xyz[0:22] for coord in joint]
            processed.append(flattened)
        
        skeleton_coords = np.array(processed, dtype=np.float32)
        print(f"✓ Processed {len(processed)} frames from HybrIK output")
    
    # Extract UVD coordinates if available (for accurate 2D overlay)
    with open(res_pk_path, 'rb') as f:
        data = pickle.load(f)
    
    uvd_processed = []
    bbox_list = []
    has_uvd = 'pred_uvd' in data and len(data['pred_uvd']) > 0
    has_bbox = 'bbox' in data and len(data['bbox']) > 0
    has_height = 'height' in data and len(data['height']) > 0
    has_width = 'width' in data and len(data['width']) > 0
    
    if has_uvd:
        for frame_idx in range(len(data['pred_xyz_24_struct'])):
            if frame_idx < len(data['pred_uvd']):
                frame_uvd = data['pred_uvd'][frame_idx]
                if len(frame_uvd) >= 22:
                    # Transform UVD coordinates to image pixel coordinates using bbox
                    # Format: pred_uvd is [29, 3], we need first 22 joints
                    uv_coords = frame_uvd[0:22, :2]  # [22, 2] - UV coordinates only
                    
                    # Get bbox for transformation
                    if has_bbox and frame_idx < len(data['bbox']):
                        bbox = data['bbox'][frame_idx]  # [x1, y1, x2, y2] format - transformed bbox from transformation.test_transform()
                        # Convert to xywh format (center x, center y, width, height) - same as demo_video_no_render.py line 278
                        # IMPORTANT: bbox ini adalah hasil dari transformation.test_transform(), bukan tight_bbox (original detection)
                        # Transformasi UVD harus menggunakan bbox yang sudah di-transform ini (sama seperti demo_video_no_render.py line 319-321)
                        bbox_xywh = np.array([
                            (bbox[0] + bbox[2]) / 2,  # cx
                            (bbox[1] + bbox[3]) / 2,  # cy
                            bbox[2] - bbox[0],        # w
                            bbox[3] - bbox[1]         # h
                        ])
                        
                        # Transform UV coordinates to image pixel coordinates
                        # Formula from demo_video_no_render.py line 319-321 dan demo_image_no_render.py line 168-170:
                        # uv_29 is normalized coordinates (range typically [-0.5, 0.5] or [0, 1])
                        # pts = uv_29 * bbox_xywh[2]  # Scale by bbox width
                        # pts[:, 0] = pts[:, 0] + bbox_xywh[0]  # Add bbox center x
                        # pts[:, 1] = pts[:, 1] + bbox_xywh[1]  # Add bbox center y
                        # Note: bbox_xywh[0] and bbox_xywh[1] are center coordinates dari transformed bbox
                        # The transformation assumes uv is relative to bbox center
                        pts = uv_coords * bbox_xywh[2]  # Scale normalized UV by bbox width
                        pts[:, 0] = pts[:, 0] + bbox_xywh[0]  # Add bbox center x
                        pts[:, 1] = pts[:, 1] + bbox_xywh[1]  # Add bbox center y
                        
                        # Flatten: [u0, v0, d0, u1, v1, d1, ..., u21, v21, d21]
                        uvd_flattened = []
                        for j in range(22):
                            uvd_flattened.extend([pts[j, 0], pts[j, 1], frame_uvd[j, 2]])  # Use transformed UV, keep original D
                        uvd_processed.append(uvd_flattened)
                        bbox_list.append(bbox_xywh)
                    else:
                        # Fallback: use UVD coordinates as-is (may need transformation later)
                        uvd_flattened = [coord for joint in frame_uvd[0:22] for coord in joint]
                        uvd_processed.append(uvd_flattened)
        
        if len(uvd_processed) == len(skeleton_coords):
            uvd_coords = np.array(uvd_processed, dtype=np.float32)
            print(f"✓ Extracted UVD coordinates for accurate 2D overlay")
            if has_bbox:
                print(f"✓ Transformed UVD coordinates using bbox information")
            return skeleton_coords, uvd_coords
    
    return skeleton_coords, None


def extract_skeleton_hybrik(video_path, hybrik_model=None, device='cuda', 
                           hybrik_output_dir=None, hybrik_config_path=None, 
                           hybrik_checkpoint_path=None):
    """
    Extract 3D skeleton from video using HybrIK
    
    Args:
        video_path: path to video file
        hybrik_model: loaded HybrIK model (if None, will run HybrIK demo script)
        device: 'cuda' or 'cpu'
        hybrik_output_dir: directory where HybrIK saves output (if None, uses temp dir)
        hybrik_config_path: path to HybrIK config file
        hybrik_checkpoint_path: path to HybrIK checkpoint
    
    Returns:
        skeleton_coords: numpy array [num_frames, 66]
                        Format: [x0, y0, z0, x1, y1, z1, ..., x21, y21, z21]
                        (22 joints * 3 coordinates = 66 values)
        uvd_coords: numpy array [num_frames, 66] or None
                    Format: [u0, v0, d0, u1, v1, d1, ..., u21, v21, d21]
                    UVD coordinates for accurate 2D overlay on video frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Option 1: Use HybrIK demo script (recommended if HybrIK is installed separately)
    if hybrik_model is None:
        # Run HybrIK demo_video.py script
        if hybrik_output_dir is None:
            hybrik_output_dir = tempfile.mkdtemp(prefix='hybrik_output_')
        
        print(f"Running HybrIK extraction on {video_path}...")
        print(f"Output directory: {hybrik_output_dir}")
        
        # Construct HybrIK command
        # Use demo_video_no_render.py to avoid pytorch3d dependency issues
        hybrik_script = "scripts/demo_video_no_render.py"  # Path to HybrIK demo script (no render version)
        hybrik_repo_path = os.environ.get('HYBRIK_PATH', './HybrIK/HybrIK')  # Set HYBRIK_PATH env var
        
        # Resolve to absolute path to avoid issues with relative paths
        if not os.path.isabs(hybrik_repo_path):
            # Get absolute path relative to current working directory
            hybrik_repo_path = os.path.abspath(hybrik_repo_path)
        
        # Fallback to original script if no_render version doesn't exist
        if not os.path.exists(os.path.join(hybrik_repo_path, hybrik_script)):
            hybrik_script = "scripts/demo_video.py"
            print(f"⚠ demo_video_no_render.py not found, using demo_video.py (may require pytorch3d)")
        
        if not os.path.exists(os.path.join(hybrik_repo_path, hybrik_script)):
            raise FileNotFoundError(
                f"HybrIK script not found at {os.path.join(hybrik_repo_path, hybrik_script)}\n"
                f"Please set HYBRIK_PATH environment variable or install HybrIK:\n"
                f"  export HYBRIK_PATH=/path/to/HybrIK\n"
                f"Or clone HybrIK: git clone https://github.com/Jeff-sjtu/HybrIK.git"
            )
        
        # Run HybrIK demo script
        # Use relative path for script (since cwd will be set to hybrik_repo_path)
        # Use absolute paths for video and output to ensure they work from any directory
        cmd = [
            'python', 
            hybrik_script,  # Path relatif (karena cwd akan di-set ke hybrik_repo_path)
            '--video-name', os.path.abspath(video_path),  # Path absolut untuk video
            '--out-dir', os.path.abspath(hybrik_output_dir),  # Path absolut untuk output
            '--save-pk'
        ]
        
        if hybrik_config_path:
            cmd.extend(['--config', os.path.abspath(hybrik_config_path) if not os.path.isabs(hybrik_config_path) else hybrik_config_path])
        if hybrik_checkpoint_path:
            cmd.extend(['--checkpoint', os.path.abspath(hybrik_checkpoint_path) if not os.path.isabs(hybrik_checkpoint_path) else hybrik_checkpoint_path])
        
        try:
            # Set PYTHONPATH to include HybrIK directory so 'hybrik' module can be imported
            env = os.environ.copy()
            # Add HybrIK directory to PYTHONPATH
            pythonpath = env.get('PYTHONPATH', '')
            if pythonpath:
                env['PYTHONPATH'] = f"{hybrik_repo_path}:{pythonpath}"
            else:
                env['PYTHONPATH'] = hybrik_repo_path
            
            # Use absolute path for cwd to avoid path duplication issues
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, 
                                 cwd=hybrik_repo_path, env=env)
            print(f"✓ HybrIK extraction completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ HybrIK extraction failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise RuntimeError(f"HybrIK extraction failed. Make sure HybrIK is properly installed.")
        
        # Find the generated res.pk file
        # HybrIK saves it as: output_dir/res.pk (for demo_video_no_render.py)
        # Or: output_dir/video_name/res.pk (for demo_video.py)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join(hybrik_output_dir, 'res.pk'),  # demo_video_no_render.py saves here
            os.path.join(hybrik_output_dir, video_name, 'res.pk'),  # demo_video.py saves here
        ]
        
        res_pk_path = None
        for path in possible_paths:
            if os.path.exists(path):
                res_pk_path = path
                break
        
        if res_pk_path is None:
            # Try searching recursively
            for root, dirs, files in os.walk(hybrik_output_dir):
                if 'res.pk' in files:
                    res_pk_path = os.path.join(root, 'res.pk')
                    break
            
            if res_pk_path is None:
                # Cleanup temp directory
                if hybrik_output_dir.startswith(tempfile.gettempdir()):
                    shutil.rmtree(hybrik_output_dir, ignore_errors=True)
                raise FileNotFoundError(
                    f"HybrIK output file res.pk not found in {hybrik_output_dir}\n"
                    f"Please check HybrIK output directory. Available files:\n"
                    f"{list(os.walk(hybrik_output_dir))}"
                )
        
        # Load and process HybrIK output
        skeleton_coords, uvd_coords = _process_hybrik_output(res_pk_path)
        # uvd_coords can be used for more accurate 2D overlay on video frames
        
        # Cleanup temp directory if we created it
        if hybrik_output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(hybrik_output_dir, ignore_errors=True)
        
        return skeleton_coords, uvd_coords
    
    # Option 2: Use loaded HybrIK model directly (if you have model loaded)
    else:
        print(f"Using loaded HybrIK model for extraction...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        skeleton_list = []
        hybrik_model.eval()
        
        with torch.no_grad():
            for frame in frames:
                # Run HybrIK inference on frame
                # This is a placeholder - adjust based on your HybrIK model API
                pred_output = hybrik_model.predict(frame)  # Adjust method name as needed
                
                # Extract 3D joints (22 joints from 24)
                if 'pred_xyz_24_struct' in pred_output:
                    joints_3d = pred_output['pred_xyz_24_struct'][0][:22]  # Take first 22 joints
                elif 'pred_joints_3d' in pred_output:
                    joints_3d = pred_output['pred_joints_3d'][:22]
                else:
                    raise ValueError("Unknown HybrIK output format")
                
                # Flatten to [66] format
                skeleton_frame = [coord for joint in joints_3d for coord in joint]
                skeleton_list.append(skeleton_frame)
        
        return np.array(skeleton_list, dtype=np.float32)


def process_res_pk_file(res_pk_path, return_tensor=False):
    """
    Process HybrIK res.pk file using preprocessing format (same as BX_handle_frame_features.py)
    This is a convenience function for direct preprocessing of res.pk files.
    
    Args:
        res_pk_path: path to HybrIK res.pk file
        return_tensor: if True, returns torch.tensor (same as preprocessing), else numpy array
    
    Returns:
        coordinates: torch.tensor or numpy array [num_frames, 66]
    """
    coordinates_tensor = process_hybrik_output_preprocessing(res_pk_path)
    
    if return_tensor:
        return coordinates_tensor
    else:
        return coordinates_tensor.numpy()


def save_skeleton_to_pickle(skeleton_coords, output_path, video_name="extracted_video", uvd_coords=None):
    """
    Save extracted skeleton to pickle file in the format expected by CoachMe
    
    Args:
        skeleton_coords: [num_frames, 66] numpy array or torch.tensor
        output_path: path to save pickle file
        video_name: name for the video
        uvd_coords: [num_frames, 66] UVD coordinates (optional, for accurate 2D overlay)
    """
    import pickle
    
    # Convert to numpy if torch.tensor
    if isinstance(skeleton_coords, torch.Tensor):
        skeleton_coords = skeleton_coords.numpy()
    if uvd_coords is not None and isinstance(uvd_coords, torch.Tensor):
        uvd_coords = uvd_coords.numpy()
    
    # Format: list of dicts (compatible with CoachMe dataset format)
    data_dict = {
        'video_name': video_name,
        'coordinates': skeleton_coords  # [num_frames, 66]
    }
    
    # Add UVD coordinates if available
    if uvd_coords is not None:
        data_dict['uvd_coords'] = uvd_coords
        print(f"  ✓ Including UVD coordinates in pickle file")
    
    data = [data_dict]
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved skeleton to {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract skeleton from video using HybrIK')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Path to output pickle file')
    parser.add_argument('--video_name', type=str, default='extracted_video', help='Video name')
    parser.add_argument('--hybrik_path', type=str, default=None, 
                       help='Path to HybrIK repository (or set HYBRIK_PATH env var)')
    parser.add_argument('--hybrik_config', type=str, default=None, 
                       help='Path to HybrIK config file')
    parser.add_argument('--hybrik_checkpoint', type=str, default=None, 
                       help='Path to HybrIK checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='HybrIK output directory (default: temp directory)')
    
    args = parser.parse_args()
    
    # Set HYBRIK_PATH if provided
    if args.hybrik_path:
        os.environ['HYBRIK_PATH'] = args.hybrik_path
    
    # Extract skeleton
    try:
        skeleton = extract_skeleton_hybrik(
            args.video,
            hybrik_output_dir=args.output_dir,
            hybrik_config_path=args.hybrik_config,
            hybrik_checkpoint_path=args.hybrik_checkpoint
        )
        
        # Save to pickle
        save_skeleton_to_pickle(skeleton, args.output, args.video_name)
        
        print(f"✓ Extraction complete: {skeleton.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
