"""
OPTIMAL: Hybrid CSV Balancer with Power Law Sampling
Implements mathematically optimal reservoir sampling with configurable realism

Balancing Strategies:
- power=1.0: No balancing (original distribution)
- power=0.7: Gentle (50x imbalance - very realistic for WiFi)
- power=0.5: RECOMMENDED (20x imbalance - balanced but realistic)
- power=0.3: Aggressive (3x imbalance - almost flat)
- 'balanced': Fully balanced (1x imbalance - all classes equal)

Quality Guarantee: IDENTICAL to loading all data and using stratified sampling

PHASE 1A UPDATE (2025-10-03):
- Now expects 25 columns (was 19)
- Added 6 new features: retryRate, frameErrorRate, channelBusyRatio,
  recentRateAvg, rateStability, sinceLastChange

Author: ahmedjk34
Date: 2025-10-03 08:45:00 UTC (PHASE 1A UPDATE)
Version: 5.1 (PHASE 1A COMPATIBLE)
"""

import csv
import random
import math
import os
import sys

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
INPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-ALL.csv")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-BALANCED.csv")

RANDOM_SEED = 42
NUM_RATES = 8

# BALANCING STRATEGY - Choose one:
STRATEGY = 'power'  # 'power', 'balanced', or 'tiered'
POWER = 0.5         # Only used if STRATEGY='power' (0.3-0.7 recommended)
TARGET_TOTAL = 1_250_000  # Total samples in final dataset

# Tiered strategy percentages (only used if STRATEGY='tiered')
TIERED_PERCENTAGES = {
    0: 8,   # 8% of final dataset
    1: 8,
    2: 8,
    3: 7,
    4: 7,
    5: 12,
    6: 15,
    7: 35,  # Still dominant, but not overwhelming
}

# FIXED: Phase 1A - 25 columns (4 metadata + 20 features + 1 scenario)
EXPECTED_COLUMNS = [
    # Metadata (4)
    'time', 'stationId', 'rateIdx', 'phyRate',
    
    # SNR features (7)
    'lastSnr', 'snrFast', 'snrSlow', 'snrTrendShort',
    'snrStabilityIndex', 'snrPredictionConfidence', 'snrVariance',
    
    # Previous window success (2)
    'shortSuccRatio', 'medSuccRatio',
    
    # Previous window loss (1)
    'packetLossRate',
    
    # Network state (2)
    'channelWidth', 'mobilityMetric',
    
    # Assessment (2)
    'severity', 'confidence',
    
    # PHASE 1A: New features (6)
    'retryRate', 'frameErrorRate', 'channelBusyRatio',
    'recentRateAvg', 'rateStability', 'sinceLastChange',
    
    # Scenario identifier (1)
    'scenario_file'
]  # TOTAL: 25 columns

# ==================== PASS 1: COUNT RATES ====================
def count_rates_fast(filepath):
    """
    PASS 1: Count samples per rate (ultra-fast, minimal memory)
    """
    print("🔍 PASS 1/2: Counting samples per rate...")
    print("="*60)
    
    rate_counts = [0] * NUM_RATES
    total_rows = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            
            if total_rows % 1_000_000 == 0:
                print(f"  Scanned {total_rows:,} rows...", end='\r')
            
            try:
                rate = int(row['rateIdx'])
                if 0 <= rate < NUM_RATES:
                    rate_counts[rate] += 1
            except (ValueError, KeyError):
                continue
    
    print(f"\n\n📊 Original Distribution ({total_rows:,} total rows):")
    max_count = max(rate_counts)
    min_count = min(c for c in rate_counts if c > 0)
    
    for rate in range(NUM_RATES):
        pct = (rate_counts[rate] / total_rows * 100) if total_rows > 0 else 0
        bar_length = int(pct / 2)  # Scale for display
        bar = '█' * bar_length
        print(f"   Rate {rate}: {rate_counts[rate]:,} samples ({pct:.1f}%) {bar}")
    
    imbalance = max_count / min_count if min_count > 0 else 0
    print(f"\n   Imbalance ratio: {imbalance:.0f}x (max/min)")
    
    return rate_counts, total_rows


# ==================== STRATEGY CALCULATORS ====================
def calculate_targets_power(rate_counts, power, target_total):
    """
    Power Law Balancing: Most flexible and recommended
    
    Formula: target[i] = count[i]^power * scaling_factor
    
    power=1.0 → Original distribution (no balancing)
    power=0.7 → Gentle (50x imbalance, very realistic)
    power=0.5 → RECOMMENDED (20x imbalance, balanced but realistic)
    power=0.3 → Aggressive (3x imbalance, almost flat)
    """
    # Apply power transformation
    powered_counts = [count ** power for count in rate_counts]
    
    # Calculate scaling factor
    current_total = sum(powered_counts)
    scale_factor = target_total / current_total if current_total > 0 else 1
    
    # Calculate targets
    targets = [int(pc * scale_factor) for pc in powered_counts]
    
    # Ensure we don't exceed available samples
    targets = [min(targets[i], rate_counts[i]) for i in range(NUM_RATES)]
    
    # Ensure minimum per class (avoid zeros)
    MIN_PER_CLASS = 5_000
    targets = [max(t, MIN_PER_CLASS) if rate_counts[i] >= MIN_PER_CLASS else rate_counts[i] 
               for i, t in enumerate(targets)]
    
    return targets


def calculate_targets_balanced(rate_counts, target_total):
    """
    Fully Balanced: All classes get equal representation
    """
    valid_counts = [c for c in rate_counts if c > 0]
    
    if not valid_counts:
        return [0] * NUM_RATES
    
    # Equal samples per rate
    per_rate = target_total // len(valid_counts)
    
    # Don't exceed available samples
    targets = [min(per_rate, count) if count > 0 else 0 for count in rate_counts]
    
    return targets


def calculate_targets_tiered(rate_counts, target_percentages, target_total):
    """
    Tiered: Custom percentages for each rate
    """
    targets = []
    
    for rate in range(NUM_RATES):
        target_pct = target_percentages.get(rate, 0)
        target_count = int(target_total * target_pct / 100)
        
        # Don't exceed available samples
        target_count = min(target_count, rate_counts[rate])
        
        targets.append(target_count)
    
    return targets


# ==================== MAIN TARGET CALCULATOR ====================
def calculate_targets(rate_counts, strategy, **kwargs):
    """
    Calculate optimal targets based on chosen strategy
    """
    print(f"\n🎯 Balancing Strategy: {strategy.upper()}")
    
    if strategy == 'power':
        power = kwargs.get('power', 0.5)
        target_total = kwargs.get('target_total', 500_000)
        print(f"   Power: {power} (0.0=fully balanced, 1.0=no balancing)")
        print(f"   Target total: {target_total:,} samples")
        targets = calculate_targets_power(rate_counts, power, target_total)
        
    elif strategy == 'balanced':
        target_total = kwargs.get('target_total', 500_000)
        print(f"   Mode: Fully balanced (all classes equal)")
        print(f"   Target total: {target_total:,} samples")
        targets = calculate_targets_balanced(rate_counts, target_total)
        
    elif strategy == 'tiered':
        target_pcts = kwargs.get('target_percentages', TIERED_PERCENTAGES)
        target_total = kwargs.get('target_total', 500_000)
        print(f"   Mode: Custom percentages")
        print(f"   Target total: {target_total:,} samples")
        targets = calculate_targets_tiered(rate_counts, target_pcts, target_total)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Calculate statistics
    actual_total = sum(targets)
    max_target = max(t for t in targets if t > 0)
    min_target = min(t for t in targets if t > 0)
    final_imbalance = max_target / min_target if min_target > 0 else 0
    
    print(f"\n📊 Target Distribution:")
    
    for rate in range(NUM_RATES):
        if rate_counts[rate] == 0:
            print(f"   Rate {rate}: NO DATA (skipping)")
            continue
        
        target = targets[rate]
        pct = (target / actual_total * 100) if actual_total > 0 else 0
        reduction_pct = (1 - target / rate_counts[rate]) * 100 if rate_counts[rate] > 0 else 0
        
        bar_length = int(pct / 2)
        bar = '█' * bar_length
        
        if rate_counts[rate] <= target:
            status = "KEEP ALL"
            print(f"   Rate {rate}: {target:,} samples ({pct:.1f}%) {bar} [{status}]")
        else:
            status = f"{reduction_pct:.0f}% reduced"
            print(f"   Rate {rate}: {target:,} samples ({pct:.1f}%) {bar} [{status}]")
    
    print(f"\n   Final imbalance ratio: {final_imbalance:.1f}x")
    print(f"   Actual total: {actual_total:,} samples")
    
    return targets


# ==================== PASS 2: RESERVOIR SAMPLING ====================
def reservoir_sample_stratified(filepath, rate_targets):
    """
    PASS 2: Stratified Reservoir Sampling
    
    Mathematical guarantee: Every sample has EXACTLY probability k/n
    Produces IDENTICAL distribution to loading all data and sampling
    """
    print("="*60)
    print("🔬 PASS 2/2: Reservoir Sampling")
    print("="*60)
    
    random.seed(RANDOM_SEED)
    
    # Initialize reservoirs and counters
    reservoirs = [[] for _ in range(NUM_RATES)]
    seen_counts = [0] * NUM_RATES
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        total_processed = 0
        
        for row in reader:
            total_processed += 1
            
            if total_processed % 1_000_000 == 0:
                total_in_reservoirs = sum(len(r) for r in reservoirs)
                print(f"  Processed {total_processed:,} rows | "
                      f"Reservoirs: {total_in_reservoirs:,} samples", end='\r')
            
            try:
                rate = int(row['rateIdx'])
                if not (0 <= rate < NUM_RATES):
                    continue
                
                target = rate_targets[rate]
                if target == 0:
                    continue
                
                seen_counts[rate] += 1
                n = seen_counts[rate]
                reservoir = reservoirs[rate]
                
                # RESERVOIR SAMPLING ALGORITHM
                if len(reservoir) < target:
                    # Reservoir not full - always add
                    reservoir.append(row)
                else:
                    # Reservoir full - replace with probability k/n
                    j = random.randint(1, n)
                    if j <= target:
                        reservoir[j - 1] = row
                
            except (ValueError, KeyError):
                continue
    
    print(f"\n\n✅ Reservoir Sampling Complete!")
    print(f"   Total processed: {total_processed:,} rows")
    
    # Report final sizes
    print(f"\n📊 Sampled Distribution:")
    total_sampled = 0
    for rate in range(NUM_RATES):
        count = len(reservoirs[rate])
        total_sampled += count
        if count > 0:
            pct = (count / total_sampled * 100) if total_sampled > 0 else 0
            bar_length = int(pct / 2)
            bar = '█' * bar_length
            print(f"   Rate {rate}: {count:,} samples ({pct:.1f}%) {bar}")
    
    final_imbalance = max(len(r) for r in reservoirs if len(r) > 0) / \
                      min(len(r) for r in reservoirs if len(r) > 0)
    
    print(f"\n   Final imbalance ratio: {final_imbalance:.1f}x")
    print(f"   Total sampled: {total_sampled:,}")
    
    return reservoirs, total_sampled


# ==================== PASS 3: SHUFFLE & WRITE ====================
def shuffle_and_write(reservoirs, output_csv, original_total):
    """
    PASS 3: Shuffle and write final balanced dataset
    """
    print("="*60)
    print("🔀 PASS 3/3: Shuffling and Writing")
    print("="*60)
    
    # Flatten reservoirs
    all_samples = []
    for reservoir in reservoirs:
        all_samples.extend(reservoir)
    
    print(f"   Collected {len(all_samples):,} samples")
    
    # Shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    print(f"   ✓ Shuffled (seed={RANDOM_SEED})")
    
    # Write
    print(f"   Writing to {os.path.basename(output_csv)}...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        
        for row in all_samples:
            clean_row = {col: row.get(col, '') for col in EXPECTED_COLUMNS}
            writer.writerow(clean_row)
    
    print(f"   ✓ Wrote {len(all_samples):,} rows")
    
    # Statistics
    retention_pct = (len(all_samples) / original_total * 100) if original_total > 0 else 0
    
    print(f"\n✅ BALANCING COMPLETE!")
    print("="*60)
    print(f"📊 Final Statistics:")
    print(f"   Original: {original_total:,} rows")
    print(f"   Balanced: {len(all_samples):,} rows")
    print(f"   Retention: {retention_pct:.1f}%")
    print(f"   Speed up: {original_total/len(all_samples):.0f}x faster training")
    print(f"\n📁 Output: {output_csv}")
    print(f"💾 Size: {os.path.getsize(output_csv) / (1024**2):.1f} MB")
    print("="*60)


# ==================== MAIN ====================
def main():
    """
    OPTIMAL HYBRID APPROACH
    """
    print("="*60)
    print("OPTIMAL CSV BALANCER - POWER LAW STRATEGY")
    print("="*60)
    print(f"📖 Algorithm: Stratified Reservoir Sampling")
    print(f"🎯 Strategy: {STRATEGY}")
    if STRATEGY == 'power':
        print(f"⚡ Power: {POWER}")
    print(f"💾 Memory: O(target_samples) only")
    print(f"🎲 Random seed: {RANDOM_SEED}")
    print("="*60 + "\n")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file not found: {INPUT_CSV}")
        sys.exit(1)
    
    print(f"📂 Input: {INPUT_CSV}")
    print(f"📂 Output: {OUTPUT_CSV}\n")
    
    # PASS 1: Count
    rate_counts, total_rows = count_rates_fast(INPUT_CSV)
    
    # Calculate targets
    rate_targets = calculate_targets(
        rate_counts,
        strategy=STRATEGY,
        power=POWER,
        target_total=TARGET_TOTAL,
        target_percentages=TIERED_PERCENTAGES
    )
    
    # PASS 2: Sample
    reservoirs, total_sampled = reservoir_sample_stratified(INPUT_CSV, rate_targets)
    
    # PASS 3: Write
    shuffle_and_write(reservoirs, OUTPUT_CSV, total_rows)
    
    print("\n🎉 SUCCESS! Your balanced dataset is ready.")
    print("\n💡 To change strategy, edit the script:")
    print("   STRATEGY = 'power'  # or 'balanced', 'tiered'")
    print("   POWER = 0.5         # 0.3-0.7 recommended")
    print("   TARGET_TOTAL = 500_000")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)