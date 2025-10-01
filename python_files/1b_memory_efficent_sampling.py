"""
OPTIMAL: Hybrid CSV Balancer with Pure Reservoir Sampling
Combines the best of both approaches for MAXIMUM quality + MINIMAL memory

Key Features:
- Pure CSV reader (no pandas overhead during sampling)
- Mathematically perfect reservoir sampling (every sample has equal probability)
- Final shuffle to eliminate sequential bias
- Memory usage: Only stores final balanced samples (~120K rows max)

Quality Guarantee: IDENTICAL to loading all 38M rows and using df.sample()

Author: ahmedjk34
Date: 2025-10-02
Version: 4.0 (OPTIMAL HYBRID)
"""

import csv
import random
import os
import sys

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
INPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-ALL.csv")
OUTPUT_CSV = os.path.join(PARENT_DIR, "smart-v3-logged-ALLB.csv")
RANDOM_SEED = 42
NUM_RATES = 8

EXPECTED_COLUMNS = [
    'time', 'stationId', 'rateIdx', 'phyRate',
    'lastSnr', 'snrFast', 'snrSlow', 'snrTrendShort',
    'snrStabilityIndex', 'snrPredictionConfidence', 'snrVariance',
    'shortSuccRatio', 'medSuccRatio', 'packetLossRate',
    'channelWidth', 'mobilityMetric',
    'severity', 'confidence', 'scenario_file'
]


def count_rates_fast(filepath):
    """
    PASS 1: Count samples per rate (ultra-fast, minimal memory)
    Uses pure CSV reader for maximum speed
    """
    print("🔍 PASS 1/2: Counting samples per rate...")
    print("="*60)
    
    rate_counts = [0] * NUM_RATES
    total_rows = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            
            # Progress indicator every 1M rows
            if total_rows % 1_000_000 == 0:
                print(f"  Scanned {total_rows:,} rows...", end='\r')
            
            try:
                rate = int(row['rateIdx'])
                if 0 <= rate < NUM_RATES:
                    rate_counts[rate] += 1
            except (ValueError, KeyError):
                continue
    
    print(f"\n\n📊 Distribution Analysis ({total_rows:,} total rows):")
    for rate in range(NUM_RATES):
        pct = (rate_counts[rate] / total_rows * 100) if total_rows > 0 else 0
        print(f"   Rate {rate}: {rate_counts[rate]:,} samples ({pct:.1f}%)")
    
    return rate_counts, total_rows


def calculate_targets(rate_counts):
    """
    Calculate optimal target samples per rate
    Strategy: 3x minimum, capped at 15K (or 10K minimum if dataset is small)
    """
    valid_counts = [c for c in rate_counts if c > 0]
    
    if not valid_counts:
        return [0] * NUM_RATES
    
    min_samples = min(valid_counts)
    max_samples = max(valid_counts)
    
    # Target: 3x minimum, capped at 15K
    target_per_rate = min(min_samples * 3, 15000)
    
    # If minimum is too low, use 10K as absolute minimum target
    if target_per_rate < 5000:
        target_per_rate = min(10000, max_samples)
    
    # Calculate actual targets (never exceed available samples)
    rate_targets = [min(target_per_rate, count) for count in rate_counts]
    
    print(f"\n🎯 Balancing Strategy:")
    print(f"   Target per rate: {target_per_rate:,}")
    print(f"   Min in dataset: {min_samples:,}")
    print(f"   Max in dataset: {max_samples:,}")
    print(f"   Original imbalance: {max_samples/min_samples:.1f}x")
    print(f"\n📊 Per-rate targets:")
    
    for rate in range(NUM_RATES):
        if rate_counts[rate] == 0:
            print(f"   Rate {rate}: NO DATA (skipping)")
        elif rate_counts[rate] <= rate_targets[rate]:
            print(f"   Rate {rate}: KEEP ALL {rate_counts[rate]:,} samples")
        else:
            reduction_pct = (1 - rate_targets[rate]/rate_counts[rate]) * 100
            print(f"   Rate {rate}: SAMPLE {rate_targets[rate]:,} from {rate_counts[rate]:,} "
                  f"({reduction_pct:.0f}% reduction)")
    
    return rate_targets


def reservoir_sample_stratified(filepath, rate_targets):
    """
    PASS 2: Stratified Reservoir Sampling (THE OPTIMAL ALGORITHM)
    
    Mathematical Guarantee:
    - Every sample has EXACTLY probability k/n of being selected
    - k = target size, n = total samples seen for that rate
    - Produces IDENTICAL distribution to loading all data and sampling
    
    Algorithm per rate:
    1. Fill reservoir with first k samples
    2. For each subsequent sample i (where i > k):
       - Generate random j in [1, i]
       - If j <= k: replace reservoir[j-1] with sample i
    
    This ensures uniform random selection across the entire stream!
    """
    print("="*60)
    print("🔬 PASS 2/2: Reservoir Sampling (Optimal Algorithm)")
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
            
            # Progress indicator
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
    
    # Report final reservoir sizes
    print(f"\n📊 Sampled Distribution:")
    total_sampled = 0
    for rate in range(NUM_RATES):
        count = len(reservoirs[rate])
        total_sampled += count
        if count > 0:
            pct = (count / total_sampled * 100) if total_sampled > 0 else 0
            print(f"   Rate {rate}: {count:,} samples")
    
    final_imbalance = max(len(r) for r in reservoirs) / min(len(r) for r in reservoirs if len(r) > 0)
    print(f"\n   Final imbalance ratio: {final_imbalance:.2f}x")
    print(f"   Total balanced samples: {total_sampled:,}")
    
    return reservoirs, total_sampled


def shuffle_and_write(reservoirs, output_csv, original_total):
    """
    PASS 3: Shuffle and write final balanced dataset
    
    Critical: Shuffle to eliminate sequential bias (all rate 0, then rate 1, etc.)
    This ensures training data has random rate distribution
    """
    print("="*60)
    print("🔀 PASS 3/3: Shuffling and Writing Output")
    print("="*60)
    
    # Flatten all reservoirs into single list
    all_samples = []
    for rate, reservoir in enumerate(reservoirs):
        for row in reservoir:
            all_samples.append(row)
    
    print(f"   Collected {len(all_samples):,} samples from reservoirs")
    
    # Shuffle to eliminate sequential bias
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    print(f"   ✓ Shuffled dataset (seed={RANDOM_SEED})")
    
    # Write to output CSV
    print(f"   Writing to {os.path.basename(output_csv)}...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        
        for row in all_samples:
            # Ensure all columns are present (fill missing with empty string)
            clean_row = {col: row.get(col, '') for col in EXPECTED_COLUMNS}
            writer.writerow(clean_row)
    
    print(f"   ✓ Wrote {len(all_samples):,} rows")
    
    # Final statistics
    retention_pct = (len(all_samples) / original_total * 100) if original_total > 0 else 0
    
    print(f"\n✅ BALANCING COMPLETE!")
    print("="*60)
    print(f"📊 Final Statistics:")
    print(f"   Original dataset: {original_total:,} rows")
    print(f"   Balanced dataset: {len(all_samples):,} rows")
    print(f"   Retention rate: {retention_pct:.1f}%")
    print(f"\n📁 Output file: {output_csv}")
    print(f"💾 File size: {os.path.getsize(output_csv) / (1024**2):.1f} MB")
    print("="*60)


def main():
    """
    OPTIMAL HYBRID APPROACH - Best of Both Worlds
    
    Combines:
    1. Pure CSV reading (fast, no pandas overhead)
    2. Mathematically perfect reservoir sampling
    3. Final shuffle for training quality
    
    Result: IDENTICAL quality to full-load sampling, MINIMAL memory
    """
    print("="*60)
    print("OPTIMAL CSV BALANCER - HYBRID APPROACH")
    print("="*60)
    print("📖 Algorithm: Stratified Reservoir Sampling")
    print("💾 Memory: Minimal (only final samples stored)")
    print("🎯 Quality: IDENTICAL to full-load pd.DataFrame.sample()")
    print("⚡ Speed: Pure CSV (no pandas overhead during sampling)")
    print("="*60 + "\n")
    
    # Validate input file
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file not found: {INPUT_CSV}")
        sys.exit(1)
    
    print(f"📂 Input: {INPUT_CSV}")
    print(f"📂 Output: {OUTPUT_CSV}")
    print(f"🎲 Random seed: {RANDOM_SEED}\n")
    
    # PASS 1: Count samples per rate
    rate_counts, total_rows = count_rates_fast(INPUT_CSV)
    
    # Calculate optimal targets
    rate_targets = calculate_targets(rate_counts)
    
    # PASS 2: Reservoir sampling
    reservoirs, total_sampled = reservoir_sample_stratified(INPUT_CSV, rate_targets)
    
    # PASS 3: Shuffle and write
    shuffle_and_write(reservoirs, OUTPUT_CSV, total_rows)
    
    print("\n🎉 SUCCESS! Your balanced dataset is ready for training.")
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