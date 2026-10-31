"""
Fix dataset imbalance by adjusting classification thresholds
"""

import os
import shutil
from pathlib import Path
import numpy as np
import cv2
from collections import Counter

def calculate_white_coverage(image_path):
    """Calculate white coverage percentage"""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    value = img_hsv[:, :, 2]
    saturation = img_hsv[:, :, 1]
    
    bright_pixels = np.sum(value > 180)
    low_sat_pixels = np.sum(saturation < 50)
    total_pixels = value.size
    
    white_score = ((bright_pixels + low_sat_pixels) / (2 * total_pixels)) * 100
    return white_score


def analyze_and_suggest_thresholds(white_scale_folder):
    """
    Analyze all images and suggest optimal thresholds
    """
    print("="*70)
    print("📊 ANALYZING ALL WHITE SCALE IMAGES")
    print("="*70)
    
    source_path = Path(white_scale_folder)
    image_files = (list(source_path.glob('*.jpg')) + 
                   list(source_path.glob('*.jpeg')) + 
                   list(source_path.glob('*.png')))
    
    print(f"\nAnalyzing {len(image_files)} images...")
    
    white_scores = []
    for idx, img_file in enumerate(image_files):
        try:
            score = calculate_white_coverage(img_file)
            white_scores.append(score)
            
            if (idx + 1) % 100 == 0:
                print(f"   Analyzed {idx + 1}/{len(image_files)}...")
        except Exception as e:
            print(f"   Error: {img_file.name}: {e}")
    
    white_scores = np.array(white_scores)
    
    print("\n" + "="*70)
    print("📈 DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print(f"\nWhite Coverage Statistics:")
    print(f"   Min:     {np.min(white_scores):.1f}%")
    print(f"   Max:     {np.max(white_scores):.1f}%")
    print(f"   Mean:    {np.mean(white_scores):.1f}%")
    print(f"   Median:  {np.median(white_scores):.1f}%")
    
    # Calculate percentiles
    p33 = np.percentile(white_scores, 33.33)
    p66 = np.percentile(white_scores, 66.67)
    
    print(f"\n💡 RECOMMENDED THRESHOLDS (Equal Distribution):")
    print(f"   Stage 1 (Low):    < {p33:.1f}%")
    print(f"   Stage 2 (Medium): {p33:.1f}% - {p66:.1f}%")
    print(f"   Stage 3 (High):   > {p66:.1f}%")
    
    stage1_count = len(white_scores[white_scores < p33])
    stage2_count = len(white_scores[(white_scores >= p33) & (white_scores < p66)])
    stage3_count = len(white_scores[white_scores >= p66])
    
    print(f"\n📊 Predicted Distribution:")
    print(f"   Stage 1: ~{stage1_count} images ({stage1_count/len(white_scores)*100:.1f}%)")
    print(f"   Stage 2: ~{stage2_count} images ({stage2_count/len(white_scores)*100:.1f}%)")
    print(f"   Stage 3: ~{stage3_count} images ({stage3_count/len(white_scores)*100:.1f}%)")
    
    return p33, p66


def reclassify_images(source_folder, output_folder, threshold1, threshold2):
    """
    Reclassify all White Scale images with new thresholds
    """
    print("\n" + "="*70)
    print("🔄 RECLASSIFYING WITH NEW THRESHOLDS")
    print("="*70)
    
    print(f"\nThresholds:")
    print(f"   Stage 1 (Low):    < {threshold1:.1f}%")
    print(f"   Stage 2 (Medium): {threshold1:.1f}% - {threshold2:.1f}%")
    print(f"   Stage 3 (High):   > {threshold2:.1f}%")
    
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    
    # Clear and recreate stage folders
    print("\n🗑️  Clearing old classifications...")
    for stage in [1, 2, 3]:
        stage_folder = output_path / f'Stage{stage}'
        if stage_folder.exists():
            shutil.rmtree(stage_folder)
        stage_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = (list(source_path.glob('*.jpg')) + 
                   list(source_path.glob('*.jpeg')) + 
                   list(source_path.glob('*.png')))
    
    print(f"\n📋 Reclassifying {len(image_files)} images...")
    
    stage_counts = Counter()
    
    for idx, img_file in enumerate(image_files):
        try:
            score = calculate_white_coverage(img_file)
            
            # Classify with new thresholds
            if score < threshold1:
                stage = 1
            elif score < threshold2:
                stage = 2
            else:
                stage = 3
            
            # Copy to new folder
            dest_folder = output_path / f'Stage{stage}'
            dest_path = dest_folder / img_file.name
            shutil.copy2(img_file, dest_path)
            
            stage_counts[stage] += 1
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(image_files)}...")
        
        except Exception as e:
            print(f"   Error: {img_file.name}: {e}")
    
    print("\n✅ Reclassification complete!")
    
    print("\n" + "="*70)
    print("📊 NEW DISTRIBUTION")
    print("="*70)
    
    total = sum(stage_counts.values())
    print(f"\n   Stage 1 (Low):    {stage_counts[1]:4d} images ({stage_counts[1]/total*100:5.1f}%)")
    print(f"   Stage 2 (Medium): {stage_counts[2]:4d} images ({stage_counts[2]/total*100:5.1f}%)")
    print(f"   Stage 3 (High):   {stage_counts[3]:4d} images ({stage_counts[3]/total*100:5.1f}%)")
    
    return stage_counts


def main():
    print("="*70)
    print("🔧 DATASET REBALANCING TOOL")
    print("   Fix class imbalance issue")
    print("="*70)
    
    # ============================================
    # 🔧 CONFIGURE PATHS
    # ============================================
    
    WHITE_SCALE_FOLDER = r"C:\My_projects\date_palm_disease\data\white scale"
    OUTPUT_FOLDER = r"C:\My_projects\date_palm_disease\data\raw"
    
    # ============================================
    
    print(f"\n📂 Paths:")
    print(f"   Source: {WHITE_SCALE_FOLDER}")
    print(f"   Output: {OUTPUT_FOLDER}")
    
    # Check if source exists
    if not os.path.exists(WHITE_SCALE_FOLDER):
        print(f"\n❌ Error: Source folder not found!")
        return
    
    # Step 1: Analyze and get recommended thresholds
    threshold1, threshold2 = analyze_and_suggest_thresholds(WHITE_SCALE_FOLDER)
    
    # Step 2: Ask user confirmation
    print("\n" + "="*70)
    print("⚠️  IMPORTANT")
    print("="*70)
    print("\nThis will:")
    print("   1. DELETE current Stage1, Stage2, Stage3 folders")
    print("   2. RECLASSIFY all 958 White Scale images")
    print("   3. CREATE new balanced distribution")
    print("\nHealthy folder will NOT be affected.")
    
    response = input("\n⚡ Proceed with rebalancing? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\n❌ Cancelled. No changes made.")
        return
    
    # Step 3: Reclassify
    stage_counts = reclassify_images(WHITE_SCALE_FOLDER, OUTPUT_FOLDER, threshold1, threshold2)
    
    # Step 4: Final summary
    print("\n" + "="*70)
    print("✅ REBALANCING COMPLETE")
    print("="*70)
    
    # Count healthy images
    healthy_folder = Path(OUTPUT_FOLDER) / "Healthy"
    healthy_count = 0
    if healthy_folder.exists():
        healthy_images = (list(healthy_folder.glob('*.jpg')) + 
                         list(healthy_folder.glob('*.jpeg')) + 
                         list(healthy_folder.glob('*.png')))
        healthy_count = len(healthy_images)
    
    total = healthy_count + sum(stage_counts.values())
    
    print(f"\n📊 Final Dataset:")
    print(f"   Healthy:  {healthy_count:4d} images ({healthy_count/total*100:5.1f}%)")
    print(f"   Stage 1:  {stage_counts[1]:4d} images ({stage_counts[1]/total*100:5.1f}%)")
    print(f"   Stage 2:  {stage_counts[2]:4d} images ({stage_counts[2]/total*100:5.1f}%)")
    print(f"   Stage 3:  {stage_counts[3]:4d} images ({stage_counts[3]/total*100:5.1f}%)")
    print(f"   Total:    {total:4d} images")
    
    print(f"\n📁 Dataset location: {OUTPUT_FOLDER}")
    
    print("\n🚀 Next steps:")
    print("   1. Verify distribution looks better")
    print("   2. Run: python prepare_dataset_paper_exact.py")
    print("   3. Run: python train_models_paper_exact.py")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()