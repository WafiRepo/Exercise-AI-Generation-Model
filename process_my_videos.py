"""
Process all videos in my_videos folder:
1. Extract skeleton using HybrIK (with preprocessing format)
2. Save to pickle files (same format as training dataset)
"""
import os
import glob
from hybrik_extractor import extract_skeleton_hybrik, save_skeleton_to_pickle

def process_videos_in_folder(video_folder='./my_videos', output_folder='./my_videos_pkl'):
    """
    Process all videos in the specified folder
    
    Args:
        video_folder: Path to folder containing videos
        output_folder: Path to save pickle files
    """
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))
        video_files.extend(glob.glob(os.path.join(video_folder, ext.upper())))
    
    if not video_files:
        print(f"❌ No video files found in {video_folder}")
        return
    
    print(f"Found {len(video_files)} video(s) to process:")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")
    print()
    
    # Process each video
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_pkl = os.path.join(output_folder, f"{video_name}.pkl")
        
        print(f"=" * 60)
        print(f"Processing: {video_name}")
        print(f"  Video: {video_path}")
        print(f"  Output: {output_pkl}")
        print()
        
        try:
            # Extract skeleton using HybrIK (with preprocessing format)
            print(f"  [1/2] Extracting skeleton from video...")
            skeleton_coords, uvd_coords = extract_skeleton_hybrik(video_path)
            print(f"  ✓ Extracted skeleton: {skeleton_coords.shape}")
            if uvd_coords is not None:
                print(f"  ✓ Extracted UVD coordinates: {uvd_coords.shape}")
            
            # Save to pickle file (same format as training dataset)
            print(f"  [2/2] Saving to pickle file...")
            save_skeleton_to_pickle(skeleton_coords, output_pkl, video_name, uvd_coords=uvd_coords)
            
            print(f"  ✓ Successfully processed: {video_name}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    print("=" * 60)
    print(f"✓ Processing complete!")
    print(f"  Processed videos: {len(video_files)}")
    print(f"  Output folder: {output_folder}")
    print()
    print("You can now use these pickle files with workflow_inference.py:")
    print("  python workflow_inference.py \\")
    print("      --user_skeleton ./my_videos_pkl/Single_Axel_Test.pkl \\")
    print("      --reference ./dataset/FS_standard.pkl \\")
    print("      --motion_type Single_Axel \\")
    print("      --output_dir ./workflow_output")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process all videos in my_videos folder')
    parser.add_argument('--video_folder', type=str, default='./my_videos',
                       help='Path to folder containing videos (default: ./my_videos)')
    parser.add_argument('--output_folder', type=str, default='./my_videos_pkl',
                       help='Path to save pickle files (default: ./my_videos_pkl)')
    
    args = parser.parse_args()
    
    process_videos_in_folder(args.video_folder, args.output_folder)
