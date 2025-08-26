#!/usr/bin/env python3
"""
Script to run comprehensive analysis for all protocol CSV files.
Runs individual analysis for all protocols and comparisons against AARF.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_analysis_for_protocol(protocol_name):
    """Run individual analysis for a specific protocol using command-line arguments."""
    
    print(f"\n{'='*80}")
    print(f"RUNNING INDIVIDUAL ANALYSIS FOR: {protocol_name}")
    print(f"{'='*80}")
    
    # Run the analysis using command-line arguments
    try:
        result = subprocess.run([
            'bash', '-c', 
            f'cd /home/ahmedjk34/smart-wifi-manager && source ~/myenv/bin/activate && python python_files/protocol_analysis.py --protocol {protocol_name}'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: Individual analysis completed for {protocol_name}")
            print(result.stdout)
        else:
            print(f"❌ ERROR: Individual analysis failed for {protocol_name}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"❌ EXCEPTION: Failed to run individual analysis for {protocol_name}: {e}")

def run_comparison_against_aarf(protocol_name):
    """Run comparison analysis against AARF for a specific protocol using command-line arguments."""
    
    # Skip if it's AARF itself
    if protocol_name == "aarf":
        return
    
    print(f"\n{'='*80}")
    print(f"RUNNING COMPARISON: AARF vs {protocol_name}")
    print(f"{'='*80}")
    
    # Run the comparison using command-line arguments
    try:
        result = subprocess.run([
            'bash', '-c', 
            f'cd /home/ahmedjk34/smart-wifi-manager && source ~/myenv/bin/activate && python python_files/protocol_comparision.py --protocol1 aarf --protocol2 {protocol_name}'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: Comparison completed for AARF vs {protocol_name}")
            print(result.stdout)
        else:
            print(f"❌ ERROR: Comparison failed for AARF vs {protocol_name}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"❌ EXCEPTION: Failed to run comparison for AARF vs {protocol_name}: {e}")

def main():
    """Run analysis for all protocols and comparisons against AARF."""
    
    # List of ALL protocols to analyze (using short names)
    protocols = [
        "aarf",
        "smartv1", 
        "smartv2",
        "smartv3",
        "smartrf",
        "smartrfv3"
    ]
    
    print("🚀 STARTING COMPREHENSIVE ANALYSIS FOR ALL PROTOCOLS")
    print("=" * 80)
    print("Protocols to analyze:")
    for protocol in protocols:
        print(f"  • {protocol}")
    print("=" * 80)
    print("This will run:")
    print("  1. Individual analysis for each protocol")
    print("  2. Comparison analysis against AARF for each protocol")
    print("=" * 80)
    
    # Check if CSV files exist
    csv_files = [
        "aarf-benchmark.csv",
        "smartv1-benchmark.csv", 
        "smartv2-benchmark.csv",
        "smartv3-benchmark.csv",
        "smartrf-benchmark-oracle.csv",
        "smartrf-benchmark-v3.csv"
    ]
    
    missing_files = []
    for csv_file in csv_files:
        csv_path = Path(__file__).parent.parent / csv_file
        if not csv_path.exists():
            missing_files.append(csv_file)
    
    if missing_files:
        print("❌ ERROR: The following CSV files are missing:")
        for file in missing_files:
            print(f"  • {file}")
        print("\nPlease ensure all CSV files are in the parent directory.")
        return False
    
    print("✅ All CSV files found. Starting analysis...\n")
    
    # PHASE 1: Run individual analysis for each protocol
    print("📊 PHASE 1: INDIVIDUAL PROTOCOL ANALYSIS")
    print("=" * 80)
    for i, protocol in enumerate(protocols, 1):
        print(f"\n📊 INDIVIDUAL ANALYSIS {i}/{len(protocols)}: {protocol}")
        run_analysis_for_protocol(protocol)
    
    # PHASE 2: Run comparison analysis against AARF
    print("\n📊 PHASE 2: COMPARISON ANALYSIS AGAINST AARF")
    print("=" * 80)
    comparison_count = 0
    for protocol in protocols:
        if protocol != "aarf":  # Skip AARF vs AARF
            comparison_count += 1
            print(f"\n📊 COMPARISON {comparison_count}/5: AARF vs {protocol}")
            run_comparison_against_aarf(protocol)
    
    print(f"\n{'='*80}")
    print("🎉 ALL ANALYSES COMPLETED!")
    print("=" * 80)
    print("Results are saved in:")
    print("  python_files/test_results/")
    print("    ├── AARF/")
    print("    ├── SmartV1/")
    print("    ├── SmartV2/")
    print("    ├── SmartV3/")
    print("    ├── SmartRF/")
    print("    └── SmartRFV3/")
    print("\n  python_files/enhanced_comparison/")
    print("    ├── AARF_vs_SmartV1/")
    print("    ├── AARF_vs_SmartV2/")
    print("    ├── AARF_vs_SmartV3/")
    print("    ├── AARF_vs_SmartRF/")
    print("    └── AARF_vs_SmartRFV3/")
    print("\nEach folder contains:")
    print("  • 14 comprehensive analysis plots (.png files)")
    print("  • Detailed Excel report with multiple sheets")
    print("  • Comprehensive text summary report")
    print("  • Analysis log file")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    main()
