"""
Script sederhana untuk merekonstruksi dan memvisualisasikan skeleton dari pickle file
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import cv2

# Bone connections (parent-child relationships)
BONE_LINKS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
    (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21)
]

# Joint names (22 joints)
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist"
]


def get_coords(joint_coords):
    """
    Rekonstruksi skeleton dari format [num_frames, 66] ke [6, frames, 22]
    Sama seperti fungsi di dataloaders/Dataset.py
    """
    # Initialize the coordinates (x, y, z) for 22 joints and 22 bones
    joint = np.zeros((3, len(joint_coords), 22))
    bone = np.zeros((3, len(joint_coords), 22))
    
    for i in range(len(joint_coords)):
        for j in range(0, len(joint_coords[i]), 3):
            joint[:, i, j // 3] = joint_coords[i, j : j + 3]
    
    # Calculate bone vectors
    for v1, v2 in BONE_LINKS:
        bone[:, :, v2] = joint[:, :, v1] - joint[:, :, v2]
    
    skeleton_coords = np.concatenate((joint, bone), axis=0)
    return skeleton_coords


def load_skeleton_from_pickle(pickle_path):
    """
    Load skeleton dari pickle file
    
    Returns:
        skeleton_coords: numpy array [num_frames, 66]
        video_name: string
        uvd_coords: numpy array [num_frames, 66] or None (UVD coordinates for accurate overlay)
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    uvd_coords = None
    
    if isinstance(data, list):
        if len(data) > 0 and 'coordinates' in data[0]:
            skeleton_coords = data[0]['coordinates']
            video_name = data[0].get('video_name', 'unknown')
            # Check for UVD coordinates
            if 'uvd_coords' in data[0]:
                uvd_coords = data[0]['uvd_coords']
        else:
            skeleton_coords = data[0]
            video_name = 'unknown'
    elif isinstance(data, dict):
        skeleton_coords = data['coordinates']
        video_name = data.get('video_name', 'unknown')
        # Check for UVD coordinates
        if 'uvd_coords' in data:
            uvd_coords = data['uvd_coords']
    else:
        skeleton_coords = data
        video_name = 'unknown'
    
    # Convert to numpy if torch tensor
    if hasattr(skeleton_coords, 'numpy'):
        skeleton_coords = skeleton_coords.numpy()
    if uvd_coords is not None and hasattr(uvd_coords, 'numpy'):
        uvd_coords = uvd_coords.numpy()
    
    return skeleton_coords, video_name, uvd_coords


def load_uvd_from_respk(respk_path):
    """
    Load UVD coordinates dari HybrIK res.pk file untuk overlay yang lebih akurat
    
    Returns:
        uvd_coords: numpy array [num_frames, 66] or None
    """
    if not os.path.exists(respk_path):
        return None
    
    try:
        with open(respk_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'pred_uvd' not in data or len(data['pred_uvd']) == 0:
            return None
        
        uvd_processed = []
        has_bbox = 'bbox' in data and len(data['bbox']) > 0
        
        for frame_idx in range(len(data['pred_uvd'])):
            frame_uvd = data['pred_uvd'][frame_idx]
            if len(frame_uvd) >= 22:
                uv_coords = frame_uvd[0:22, :2]  # [22, 2] - UV coordinates
                
                # Transform menggunakan bbox jika tersedia
                if has_bbox and frame_idx < len(data['bbox']):
                    bbox = data['bbox'][frame_idx]  # [x1, y1, x2, y2] - transformed bbox dari transformation.test_transform()
                    # IMPORTANT: bbox ini adalah hasil dari transformation.test_transform(), bukan tight_bbox (original detection)
                    # Transformasi UVD harus menggunakan bbox yang sudah di-transform ini (sama seperti demo_video_no_render.py line 319-321)
                    bbox_xywh = np.array([
                        (bbox[0] + bbox[2]) / 2,  # cx
                        (bbox[1] + bbox[3]) / 2,  # cy
                        bbox[2] - bbox[0],        # w
                        bbox[3] - bbox[1]         # h
                    ])
                    
                    # Transform UV ke pixel coordinates
                    # Formula dari demo_video_no_render.py line 319-321 dan demo_image_no_render.py line 168-170:
                    pts = uv_coords * bbox_xywh[2]  # Scale by width
                    pts[:, 0] = pts[:, 0] + bbox_xywh[0]  # Add center x
                    pts[:, 1] = pts[:, 1] + bbox_xywh[1]  # Add center y
                    
                    # Flatten: [u0, v0, d0, u1, v1, d1, ..., u21, v21, d21]
                    uvd_flattened = []
                    for j in range(22):
                        uvd_flattened.extend([pts[j, 0], pts[j, 1], frame_uvd[j, 2]])
                    uvd_processed.append(uvd_flattened)
                else:
                    # Fallback: use as-is
                    uvd_flattened = [coord for joint in frame_uvd[0:22] for coord in joint]
                    uvd_processed.append(uvd_flattened)
        
        if len(uvd_processed) > 0:
            return np.array(uvd_processed, dtype=np.float32)
    except Exception as e:
        print(f"  ⚠ Warning: Could not load UVD from res.pk: {e}")
    
    return None


def load_frame_from_video(video_path, frame_idx):
    """
    Load frame tertentu dari video
    
    Args:
        video_path: path ke video file
        frame_idx: index frame yang ingin di-load
    
    Returns:
        frame: numpy array (RGB) atau None jika gagal
    """
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None


def skeleton_to_joints_2d(skeleton_frame, frame_img_shape=None, uvd_frame=None):
    """
    Convert skeleton [66] ke 2D joint positions untuk overlay di frame image
    Menggunakan UVD coordinates jika tersedia (lebih akurat), atau proyeksi 3D
    
    Args:
        skeleton_frame: [66] array (x0,y0,z0, x1,y1,z1, ..., x21,y21,z21)
        frame_img_shape: (height, width) dari frame image
        uvd_frame: [66] array (u0,v0,d0, u1,v1,d1, ..., u21,v21,d21) - UVD coordinates dari HybrIK
    
    Returns:
        joints_2d: [22, 2] array of (x, y) positions dalam pixel coordinates
    """
    # Jika UVD coordinates tersedia, gunakan langsung (paling akurat)
    if uvd_frame is not None and frame_img_shape is not None:
        h, w = frame_img_shape[:2]
        
        # Extract UV coordinates (first 2 values per joint)
        u_values = []
        v_values = []
        for i in range(0, 66, 3):
            u_values.append(uvd_frame[i])
            v_values.append(uvd_frame[i+1])
        
        u_values = np.array(u_values)
        v_values = np.array(v_values)
        
        # Cek apakah UVD sudah dalam pixel coordinates atau masih normalized
        u_min, u_max = u_values.min(), u_values.max()
        v_min, v_max = v_values.min(), v_values.max()
        u_range = u_max - u_min
        v_range = v_max - v_min
        
        # Jika range masih kecil (< 100), kemungkinan masih normalized
        # Jika range sudah besar (> 50), kemungkinan sudah pixel coordinates
        is_normalized = (u_range < 100) and (v_range < 100) and (abs(u_min) < 10) and (abs(v_min) < 10)
        
        joints_2d = []
        for i in range(22):
            u = u_values[i]
            v = v_values[i]
            
            if is_normalized:
                # UVD masih normalized (range [-0.5, 0.5] atau [-1, 1])
                # Transform ke pixel coordinates
                # UVD dari HybrIK relatif ke bbox center, perlu transform ke image coordinates
                center_x, center_y = w / 2, h / 2
                # Scale berdasarkan image size (konservatif)
                # Gunakan 50% dari image size untuk scaling
                scale = min(w, h) * 0.5
                x = u * scale + center_x
                y = v * scale + center_y
            else:
                # UVD sudah dalam pixel coordinates (sudah ditransformasi dengan bbox)
                x = u
                y = v
            
            # Clamp to image bounds
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            
            joints_2d.append([x, y])
        
        joints_2d = np.array(joints_2d)
        
        # UVD coordinates dari HybrIK sudah dalam format image coordinates
        # (origin di top-left, Y ke bawah), jadi tidak perlu invert Y-axis
        # Tapi jika masih tidak akurat, bisa coba dengan invert:
        # joints_2d[:, 1] = h - joints_2d[:, 1]
        
        return joints_2d
    
    # Fallback: Project 3D to 2D (kurang akurat tapi masih bisa digunakan)
    # Extract 3D coordinates
    joints_3d = []
    for i in range(0, 66, 3):
        x, y, z = skeleton_frame[i], skeleton_frame[i+1], skeleton_frame[i+2]
        joints_3d.append([x, y, z])
    joints_3d = np.array(joints_3d)
    
    # Center at root joint (pelvis)
    root_joint = joints_3d[0].copy()
    joints_3d_centered = joints_3d - root_joint
    
    # Project to 2D: use X and Y (ignore Z for now)
    joints_2d = joints_3d_centered[:, :2].copy()
    
    # Scale to image coordinates jika frame shape diberikan
    if frame_img_shape is not None:
        h, w = frame_img_shape[:2]
        
        # Normalize to [-1, 1] range first
        x_min, x_max = joints_2d[:, 0].min(), joints_2d[:, 0].max()
        y_min, y_max = joints_2d[:, 1].min(), joints_2d[:, 1].max()
        
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        if x_range > 0:
            joints_2d[:, 0] = (joints_2d[:, 0] - x_min) / x_range * 2 - 1
        if y_range > 0:
            joints_2d[:, 1] = (joints_2d[:, 1] - y_min) / y_range * 2 - 1
        
        # Scale to image coordinates (center di tengah frame)
        scale = min(w, h) * 0.4  # 40% of smaller dimension
        center_x, center_y = w / 2, h / 2
        
        joints_2d[:, 0] = joints_2d[:, 0] * scale + center_x
        joints_2d[:, 1] = joints_2d[:, 1] * scale + center_y
        
        # Invert Y-axis untuk image coordinates
        joints_2d[:, 1] = h - joints_2d[:, 1]
    else:
        # Normalize to [-1, 1] range jika tidak ada frame shape
        x_min, x_max = joints_2d[:, 0].min(), joints_2d[:, 0].max()
        y_min, y_max = joints_2d[:, 1].min(), joints_2d[:, 1].max()
        
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        if x_range > 0:
            joints_2d[:, 0] = (joints_2d[:, 0] - x_min) / x_range * 2 - 1
        if y_range > 0:
            joints_2d[:, 1] = (joints_2d[:, 1] - y_min) / y_range * 2 - 1
    
    return joints_2d


def visualize_skeleton_3d(skeleton_coords, video_name="skeleton", frame_idx=0, save_path=None, 
                          frame_image=None, video_path=None, uvd_coords=None):
    """
    Visualisasi skeleton 3D untuk satu frame dengan opsi menampilkan frame image dari video
    
    Args:
        skeleton_coords: [num_frames, 66] atau [6, frames, 22]
        video_name: nama video
        frame_idx: index frame yang akan divisualisasikan
        save_path: path untuk menyimpan gambar (optional)
        frame_image: numpy array (RGB) frame image dari video (optional)
        video_path: path ke video file untuk load frame (optional, jika frame_image None)
        uvd_coords: [num_frames, 66] UVD coordinates untuk overlay yang lebih akurat (optional)
    """
    # Load frame image jika video_path diberikan
    if frame_image is None and video_path is not None:
        frame_image = load_frame_from_video(video_path, frame_idx)
        if frame_image is not None:
            print(f"  ✓ Loaded frame {frame_idx} from video")
    
    # Rekonstruksi jika masih dalam format [num_frames, 66]
    if len(skeleton_coords.shape) == 2:
        skeleton_reconstructed = get_coords(skeleton_coords)
    else:
        skeleton_reconstructed = skeleton_coords
    
    # Extract joint coordinates untuk frame tertentu
    # skeleton_reconstructed shape: [6, frames, 22]
    # joint coords ada di channel 0-2
    joints = skeleton_reconstructed[0:3, frame_idx, :].copy()  # [3, 22]
    
    # IMPORTANT: Center skeleton di root joint (pelvis = joint 0)
    # Ini memastikan skeleton terpusat di origin untuk visualisasi yang benar
    root_joint = joints[:, 0:1].copy()  # [3, 1] - pelvis coordinates
    joints_centered = joints - root_joint  # Center semua joints relative ke pelvis
    
    # Create figure: side-by-side jika ada frame image, atau hanya 3D plot
    if frame_image is not None:
        fig = plt.figure(figsize=(20, 10))
        
        # Left: Frame image with skeleton overlay
        ax1 = fig.add_subplot(121)
        ax1.imshow(frame_image)
        ax1.axis('off')
        ax1.set_title(f'Frame {frame_idx} with Skeleton Overlay', fontsize=14)
        
        # Overlay skeleton 2D pada frame image
        skeleton_frame = skeleton_coords[frame_idx] if len(skeleton_coords.shape) == 2 else None
        uvd_frame = uvd_coords[frame_idx] if uvd_coords is not None and frame_idx < len(uvd_coords) else None
        
        if skeleton_frame is not None:
            h, w = frame_image.shape[:2]
            joints_2d = skeleton_to_joints_2d(skeleton_frame, frame_img_shape=(h, w), uvd_frame=uvd_frame)
            
            # Plot joints
            ax1.scatter(joints_2d[:, 0], joints_2d[:, 1], c='red', s=30, alpha=0.8, zorder=10)
            
            # Plot bones
            for v1, v2 in BONE_LINKS:
                ax1.plot([joints_2d[v1, 0], joints_2d[v2, 0]],
                        [joints_2d[v1, 1], joints_2d[v2, 1]],
                        'b-', linewidth=2, alpha=0.7, zorder=5)
        
        # Right: 3D skeleton plot
        ax2 = fig.add_subplot(122, projection='3d')
    else:
        fig = plt.figure(figsize=(12, 10))
        ax2 = fig.add_subplot(111, projection='3d')
    
    # Plot 3D skeleton
    ax2.scatter(joints_centered[0, :], joints_centered[1, :], joints_centered[2, :], 
               c='red', s=50, alpha=0.8, label='Joints')
    
    # Plot bones (connections)
    for v1, v2 in BONE_LINKS:
        ax2.plot([joints_centered[0, v1], joints_centered[0, v2]],
                [joints_centered[1, v1], joints_centered[1, v2]],
                [joints_centered[2, v1], joints_centered[2, v2]],
                'b-', linewidth=2, alpha=0.6)
    
    # Label root joint (pelvis) - sekarang di origin (0,0,0)
    ax2.text(0, 0, 0, 'pelvis', fontsize=8)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Skeleton 3D - {video_name} (Frame {frame_idx})')
    ax2.legend()
    
    # Set equal aspect ratio dengan centering di origin
    max_range = np.array([joints_centered[0, :].max() - joints_centered[0, :].min(),
                          joints_centered[1, :].max() - joints_centered[1, :].min(),
                          joints_centered[2, :].max() - joints_centered[2, :].min()]).max() / 2.0
    
    # Center di origin (0,0,0) karena sudah di-center
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    ax2.set_zlim(-max_range, max_range)
    
    # Set view angle yang lebih baik untuk melihat pose
    ax2.view_init(elev=10, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_skeleton_sequence(skeleton_coords, video_name="skeleton", 
                                num_frames=5, save_dir=None, video_path=None, uvd_coords=None):
    """
    Visualisasi beberapa frame dari sequence
    
    Args:
        skeleton_coords: [num_frames, 66]
        video_name: nama video
        num_frames: jumlah frame yang akan divisualisasikan
        save_dir: direktori untuk menyimpan gambar (optional)
        video_path: path ke video file untuk load frames (optional)
        uvd_coords: [num_frames, 66] UVD coordinates untuk overlay yang lebih akurat (optional)
    """
    num_total_frames = len(skeleton_coords)
    
    # Jika num_frames sama dengan total frames, visualisasi semua frame secara berurutan
    if num_frames >= num_total_frames:
        frame_indices = np.arange(num_total_frames)
        print(f"Visualizing ALL {num_total_frames} frames (consecutive)")
    else:
        # Sampling: pilih frame secara merata
        frame_indices = np.linspace(0, num_total_frames - 1, num_frames, dtype=int)
        print(f"Visualizing {num_frames} frames (sampled): {frame_indices}")
    
    for idx, frame_idx in enumerate(frame_indices):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{video_name}_frame_{frame_idx:03d}.png")
        else:
            save_path = None
        
        visualize_skeleton_3d(skeleton_coords, video_name, frame_idx, save_path, 
                            video_path=video_path, uvd_coords=uvd_coords)
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or (idx + 1) == len(frame_indices):
            print(f"  Progress: {idx + 1}/{len(frame_indices)} frames processed")


def print_skeleton_info(skeleton_coords, video_name):
    """
    Print informasi tentang skeleton
    """
    print("=" * 60)
    print(f"Skeleton Information: {video_name}")
    print("=" * 60)
    print(f"Shape: {skeleton_coords.shape}")
    print(f"Number of frames: {len(skeleton_coords)}")
    print(f"Number of joints: 22")
    print(f"Coordinates per joint: 3 (x, y, z)")
    print(f"Total values per frame: 66 (22 joints × 3 coords)")
    print()
    
    # Rekonstruksi untuk melihat format [6, frames, 22]
    skeleton_reconstructed = get_coords(skeleton_coords)
    print(f"Reconstructed shape: {skeleton_reconstructed.shape} [6 channels, {len(skeleton_coords)} frames, 22 joints]")
    print(f"  - Channels 0-2: Joint coordinates (x, y, z)")
    print(f"  - Channels 3-5: Bone vectors (dx, dy, dz)")
    print()
    
    # Statistik untuk frame pertama
    frame_0 = skeleton_coords[0]
    print(f"Frame 0 statistics:")
    print(f"  Min: {frame_0.min():.4f}, Max: {frame_0.max():.4f}")
    print(f"  Mean: {frame_0.mean():.4f}, Std: {frame_0.std():.4f}")
    print()
    
    # Sample joint positions (pelvis, left hip, right hip)
    print(f"Sample joint positions (Frame 0):")
    print(f"  Pelvis (joint 0):     [{frame_0[0]:.4f}, {frame_0[1]:.4f}, {frame_0[2]:.4f}]")
    print(f"  Left hip (joint 1):  [{frame_0[3]:.4f}, {frame_0[4]:.4f}, {frame_0[5]:.4f}]")
    print(f"  Right hip (joint 2): [{frame_0[6]:.4f}, {frame_0[7]:.4f}, {frame_0[8]:.4f}]")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Rekonstruksi dan visualisasi skeleton dari pickle file')
    parser.add_argument('--pickle', type=str, required=True,
                       help='Path ke pickle file')
    parser.add_argument('--frame', type=int, default=0,
                       help='Index frame yang akan divisualisasikan (default: 0)')
    parser.add_argument('--frames', type=int, default=None,
                       help='Jumlah frame yang akan divisualisasikan (default: None, hanya frame yang dipilih)')
    parser.add_argument('--all', action='store_true',
                       help='Visualisasi semua frame (default: False)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path untuk menyimpan gambar (default: None, tampilkan di layar)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Direktori untuk menyimpan multiple frames (default: None)')
    parser.add_argument('--info', action='store_true',
                       help='Hanya tampilkan informasi skeleton, tidak visualisasi')
    parser.add_argument('--video', type=str, default=None,
                       help='Path ke video file untuk menampilkan frame image (optional)')
    
    args = parser.parse_args()
    
    # Load skeleton
    print(f"Loading skeleton from {args.pickle}...")
    skeleton_coords, video_name, uvd_coords = load_skeleton_from_pickle(args.pickle)
    print(f"✓ Loaded skeleton: {skeleton_coords.shape}")
    if uvd_coords is not None:
        print(f"✓ Loaded UVD coordinates: {uvd_coords.shape} (for accurate overlay)")
    else:
        print(f"  ⚠ UVD coordinates not found in pickle file")
        # Coba load dari res.pk jika tersedia
        pickle_dir = os.path.dirname(args.pickle)
        pickle_basename = os.path.splitext(os.path.basename(args.pickle))[0]
        
        # Cari di berbagai lokasi yang mungkin
        possible_respk_paths = [
            # Lokasi relatif dari pickle file
            os.path.join(pickle_dir, '..', 'HybrIK', 'output', pickle_basename, 'res.pk'),
            os.path.join(pickle_dir, '..', 'HybrIK', pickle_basename, 'res.pk'),
            os.path.join(pickle_dir, pickle_basename, 'res.pk'),
            # Lokasi absolut
            os.path.join(os.path.expanduser('~'), 'tmp', pickle_basename, 'res.pk'),
            # Cari di temp directory dengan pattern
        ]
        
        # Cari di temp directory dengan pattern video name
        import tempfile
        temp_base = tempfile.gettempdir()
        for root, dirs, files in os.walk(temp_base):
            if 'res.pk' in files and pickle_basename.lower() in root.lower():
                possible_respk_paths.append(os.path.join(root, 'res.pk'))
                break
        
        # Cari di seluruh sistem dengan nama yang cocok (terbatas)
        search_dirs = [
            os.path.join(os.path.dirname(pickle_dir), '..'),
            os.path.join(os.path.dirname(pickle_dir)),
            './',
            '../',
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    # Batasi depth untuk menghindari pencarian terlalu lama
                    depth = root[len(search_dir):].count(os.sep)
                    if depth > 3:  # Max 3 levels deep
                        dirs[:] = []  # Don't recurse deeper
                        continue
                    
                    if 'res.pk' in files:
                        respk_path = os.path.join(root, 'res.pk')
                        # Cek apakah nama video cocok dengan path
                        if pickle_basename.lower() in root.lower() or pickle_basename.lower() in respk_path.lower():
                            possible_respk_paths.append(respk_path)
        
        # Coba semua path yang mungkin
        for respk_path in possible_respk_paths:
            if os.path.exists(respk_path):
                print(f"  Trying to load UVD from: {respk_path}")
                uvd_coords = load_uvd_from_respk(respk_path)
                if uvd_coords is not None and len(uvd_coords) == len(skeleton_coords):
                    print(f"  ✓ Loaded UVD coordinates from res.pk: {uvd_coords.shape}")
                    break
    print()
    
    # Print info
    print_skeleton_info(skeleton_coords, video_name)
    
    if args.info:
        return
    
    # Cari video path jika tidak diberikan
    video_path = args.video
    if video_path is None:
        # Coba cari video di folder my_videos dengan nama yang sama
        pickle_basename = os.path.splitext(os.path.basename(args.pickle))[0]
        possible_video_paths = [
            f'./my_videos/{pickle_basename}.mp4',
            f'./my_videos/{pickle_basename}.avi',
            f'./my_videos/{pickle_basename}.mov',
            f'./my_videos/{video_name}.mp4',
            f'./my_videos/{video_name}.avi',
            f'./my_videos/{video_name}.mov',
        ]
        for path in possible_video_paths:
            if os.path.exists(path):
                video_path = path
                print(f"  ✓ Found video: {video_path}")
                break
    
    # Visualisasi
    if args.all:
        # Visualisasi semua frame
        num_total_frames = len(skeleton_coords)
        print(f"\nVisualizing ALL {num_total_frames} frames...")
        if args.save_dir is None:
            args.save_dir = f'./{video_name}_all_frames'
            print(f"  Using default save directory: {args.save_dir}")
        visualize_skeleton_sequence(skeleton_coords, video_name, 
                                   num_frames=num_total_frames, 
                                   save_dir=args.save_dir,
                                   video_path=video_path,
                                   uvd_coords=uvd_coords)
    elif args.frames:
        # Visualisasi multiple frames (sampling)
        visualize_skeleton_sequence(skeleton_coords, video_name, 
                                   num_frames=args.frames, 
                                   save_dir=args.save_dir,
                                   video_path=video_path,
                                   uvd_coords=uvd_coords)
    else:
        # Visualisasi single frame
        frame_idx = args.frame
        if frame_idx >= len(skeleton_coords):
            print(f"⚠ Warning: frame_idx {frame_idx} >= num_frames {len(skeleton_coords)}, using frame 0")
            frame_idx = 0
        
        visualize_skeleton_3d(skeleton_coords, video_name, frame_idx, args.save, 
                           video_path=video_path, uvd_coords=uvd_coords)
    
    print("\n✓ Reconstruction complete!")


if __name__ == "__main__":
    main()
