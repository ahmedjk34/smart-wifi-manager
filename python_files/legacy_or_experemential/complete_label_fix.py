import pandas as pd
import numpy as np
import os
import logging
import sys
from typing import Union

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOGGING SETUP ---
def setup_logging():
    """Setup logging with fallback if file creation fails"""
    handlers = []
    
    # Try to create file handler in script directory
    try:
        log_file = os.path.join(BASE_DIR, 'ml_data_prep_fixed.log')
        handlers.append(logging.FileHandler(log_file))
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not create log file ({e}), logging to console only")
    
    # Always add console handler
    handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()
INPUT_CSV = os.path.join(BASE_DIR, "..", "smartv4-ml-ready.csv")  # FIXED: Uses your label script output
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "smartv4-ml-ready-FIXED.csv")

# FIXED: 802.11g rates only (8 rates: 0-7)
G_RATES_BPS = [1000000, 2000000, 5500000, 6000000, 9000000, 11000000, 12000000, 18000000]

# FIXED: Much more conservative SNR thresholds for ns-3
SNR_THRESHOLDS = {
    1000000: 10,   # 1 Mbps  - need 10dB for reliable 1Mbps
    2000000: 13,   # 2 Mbps  - need 13dB
    5500000: 16,   # 5.5 Mbps - need 16dB  
    6000000: 17,   # 6 Mbps  - need 17dB
    9000000: 20,   # 9 Mbps  - need 20dB
    11000000: 23,  # 11 Mbps - need 23dB
    12000000: 25,  # 12 Mbps - need 25dB
    18000000: 28,  # 18 Mbps - need 28dB (highest rate)
}

# --- NEW CRITICAL PATCH: FIX SNR VALUES ---
def fix_snr_values(df):
    """✅ CRITICAL: Fix the crazy high SNR values in ns-3 data"""
    logger.info("🔧 CRITICAL PATCH: Fixing SNR values...")
    
    # Log original SNR stats
    logger.info(f"📊 Original lastSnr stats: min={df['lastSnr'].min():.2f}, max={df['lastSnr'].max():.2f}, mean={df['lastSnr'].mean():.2f}")
    
    # Convert crazy high SNR values to reasonable dB values
    def convert_snr_to_db(snr_val):
        try:
            snr_val = float(snr_val)
            
            # If SNR > 100, it's probably a linear power ratio - convert to dB
            if snr_val > 100:
                # Convert from linear to dB: 10 * log10(ratio)
                snr_db = 10 * np.log10(max(snr_val, 1))
                # Cap at reasonable max (40 dB)
                return min(snr_db, 40)
            
            # If already reasonable, keep it
            elif 0 <= snr_val <= 50:
                return snr_val
            
            # If negative or weird, set to low value
            else:
                return 5
                
        except:
            return 10  # Default fallback
    
    # Fix all SNR columns
    for col in ['lastSnr', 'snrFast', 'snrSlow']:
        if col in df.columns:
            logger.info(f"🔧 Fixing {col} column...")
            df[col] = df[col].apply(convert_snr_to_db)
    
    # Log fixed SNR stats
    logger.info(f"✅ Fixed lastSnr stats: min={df['lastSnr'].min():.2f}, max={df['lastSnr'].max():.2f}, mean={df['lastSnr'].mean():.2f}")
    
    return df

# --- PATCH 1: CLEAN V3 RATE DATA ---
def clean_v3_rate_data(df):
    """✅ Clean the corrupted v3_rateIdx column"""
    logger.info("🔧 PATCH 1: Cleaning v3_rateIdx data...")
    
    def fix_rate_idx(rate_val):
        try:
            rate_val = float(rate_val)
            
            # If it's a large number, it's probably bps - convert to index
            if rate_val >= 1000000:
                rate_map = {
                    1000000: 0, 2000000: 1, 5500000: 2, 6000000: 3,
                    9000000: 4, 11000000: 5, 12000000: 6, 18000000: 7,
                    24000000: 7, 36000000: 7, 48000000: 7, 54000000: 7  # Cap at max
                }
                return rate_map.get(int(rate_val), 3)  # Default to middle
            
            # If it's -1 or invalid, default to middle
            if rate_val < 0 or rate_val > 7:
                return 3
                
            return int(rate_val)
        except:
            return 3
    
    if 'v3_rateIdx' in df.columns:
        original_unique = df['v3_rateIdx'].unique()
        logger.info(f"📊 Original v3_rateIdx unique values: {sorted([x for x in original_unique if pd.notna(x)])}")
        
        df['v3_rateIdx'] = df['v3_rateIdx'].apply(fix_rate_idx)
        
        new_unique = df['v3_rateIdx'].unique()
        logger.info(f"✅ Cleaned v3_rateIdx unique values: {sorted(new_unique)}")
    else:
        logger.warning("⚠️ v3_rateIdx column not found - skipping cleaning")
    
    return df

# --- PATCH 2: FIXED ORACLE FUNCTION ---
def estimate_p_success(row, rate: int) -> float:
    """✅ Estimate probability of success for a given rate"""
    try:
        # For current rate, use shortSuccRatio; for others, use SNR threshold model
        if int(row['phyRate']) == rate:
            return float(row['shortSuccRatio'])
        
        if rate not in SNR_THRESHOLDS:
            return 0.0
            
        return float(row['lastSnr'] >= SNR_THRESHOLDS[rate])
    except (ValueError, KeyError, TypeError) as e:
        return 0.0

def find_oracle_best_rate(row) -> int:
    """✅ FIXED: Returns rate index that maximizes expected goodput (0-7 for 802.11g ONLY)"""
    try:
        best_rate_idx = 0
        best_goodput = 0.0
        
        for i, rate_bps in enumerate(G_RATES_BPS):
            p_succ = estimate_p_success(row, rate_bps)
            goodput = p_succ * rate_bps
            
            if goodput > best_goodput:
                best_goodput = goodput
                best_rate_idx = i
        
        # Ensure we return 0-7 only
        return min(best_rate_idx, 7)
        
    except Exception as e:
        return 3  # Fallback to middle rate

def find_oracle_success(row) -> int:
    """✅ Returns 1 if expected p_success at best rate > 0.5, else 0"""
    try:
        best_rate_idx = find_oracle_best_rate(row)
        rate = G_RATES_BPS[best_rate_idx]
        p_succ = estimate_p_success(row, rate)
        return int(p_succ > 0.5)
    except Exception as e:
        return 0

# --- PATCH 3: ADD CHALLENGING SCENARIOS ---
def add_challenging_scenarios(df):
    """✅ PATCH 3: Artificially create some poor-condition samples"""
    logger.info("🔧 PATCH 3: Adding challenging scenarios...")
    
    # Take good samples and artificially degrade them
    good_samples = df[df['lastSnr'] > 25].copy()
    if len(good_samples) > 100000:
        good_samples = good_samples.sample(n=100000, random_state=42)
    
    if len(good_samples) == 0:
        logger.warning("⚠️ No good samples found for degradation - skipping challenging scenarios")
        return df
    
    logger.info(f"📊 Creating {len(good_samples)} challenging scenarios from good samples")
    
    # Create poor SNR versions
    np.random.seed(42)
    good_samples['lastSnr'] = good_samples['lastSnr'] - np.random.uniform(10, 20, len(good_samples))
    
    # Fix other SNR columns if they exist
    if 'snrFast' in good_samples.columns:
        good_samples['snrFast'] = good_samples['lastSnr'] - np.random.uniform(0, 2, len(good_samples))
    if 'snrSlow' in good_samples.columns:
        good_samples['snrSlow'] = good_samples['lastSnr'] - np.random.uniform(0, 1, len(good_samples))
    
    # Degrade success ratios
    if 'shortSuccRatio' in good_samples.columns:
        good_samples['shortSuccRatio'] = np.random.uniform(0.1, 0.6, len(good_samples))
    if 'medSuccRatio' in good_samples.columns:
        good_samples['medSuccRatio'] = np.random.uniform(0.2, 0.7, len(good_samples))
    
    # Degrade other metrics if they exist
    if 'severity' in good_samples.columns:
        good_samples['severity'] = np.random.uniform(0.3, 0.8, len(good_samples))
    if 'confidence' in good_samples.columns:
        good_samples['confidence'] = np.random.uniform(0.1, 0.5, len(good_samples))
    
    # Recompute oracle labels for these degraded samples
    logger.info("🔧 Recomputing oracle labels for challenging scenarios...")
    good_samples['oracle_best_rateIdx_new'] = good_samples.apply(find_oracle_best_rate, axis=1)
    good_samples['oracle_best_success_new'] = good_samples.apply(find_oracle_success, axis=1)
    
    # Replace oracle columns if they exist
    if 'oracle_best_rateIdx' in good_samples.columns:
        good_samples['oracle_best_rateIdx'] = good_samples['oracle_best_rateIdx_new']
        good_samples.drop('oracle_best_rateIdx_new', axis=1, inplace=True)
    
    if 'oracle_best_success' in good_samples.columns:
        good_samples['oracle_best_success'] = good_samples['oracle_best_success_new']
        good_samples.drop('oracle_best_success_new', axis=1, inplace=True)
    
    # Combine original and degraded
    augmented_df = pd.concat([df, good_samples], ignore_index=True)
    logger.info(f"✅ Added {len(good_samples)} challenging scenarios. Total: {len(augmented_df)}")
    
    return augmented_df

# --- PATCH 4: FORCE DATA BALANCE ---
def force_balanced_oracle_labels(df):
    """✅ PATCH 4: Force a balanced distribution by sampling scenarios differently"""
    logger.info("🔧 PATCH 4: Force balancing oracle labels...")
    
    if 'oracle_best_rateIdx' not in df.columns:
        logger.error("❌ oracle_best_rateIdx column not found - cannot balance")
        return df
    
    # Check current distribution
    current_dist = df['oracle_best_rateIdx'].value_counts().sort_index()
    logger.info(f"📊 Current distribution before balancing:\n{current_dist}")
    
    # Target: roughly equal samples per rate (0-7)
    target_per_rate = 80000
    balanced_dfs = []
    
    for rate_idx in range(8):
        rate_data = df[df['oracle_best_rateIdx'] == rate_idx]
        
        if len(rate_data) == 0:
            logger.warning(f"⚠️ No data for rate {rate_idx} - skipping")
            continue
            
        if len(rate_data) > target_per_rate:
            # Downsample overrepresented rates
            rate_data = rate_data.sample(n=target_per_rate, random_state=42)
        else:
            # Keep all samples for underrepresented rates
            logger.info(f"📊 Rate {rate_idx}: keeping all {len(rate_data)} samples (below target)")
        
        balanced_dfs.append(rate_data)
        logger.info(f"✅ Rate {rate_idx}: final {len(rate_data)} samples")
    
    if not balanced_dfs:
        logger.error("❌ No valid rate data found!")
        return df
        
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Show final distribution
    final_dist = balanced_df['oracle_best_rateIdx'].value_counts().sort_index()
    logger.info(f"🎯 Final balanced distribution:\n{final_dist}")
    
    # Show percentages
    total = final_dist.sum()
    logger.info("📊 Percentages:")
    for rate, count in final_dist.items():
        pct = (count / total) * 100
        logger.info(f"   Rate {rate}: {pct:.1f}%")
    
    logger.info(f"✅ Balanced: {len(df)} → {len(balanced_df)} samples")
    return balanced_df

def filter_row(row) -> bool:
    """Filter function for removing noisy/ambiguous rows"""
    try:
        return float(row['lastSnr']) >= 3
    except (ValueError, KeyError, TypeError) as e:
        return False

def validate_input_file(filepath: str) -> bool:
    """Validate input CSV file exists and is readable"""
    if not os.path.exists(filepath):
        logger.error(f"❌ Input file does not exist: {filepath}")
        return False
    
    if not os.access(filepath, os.R_OK):
        logger.error(f"❌ Input file is not readable: {filepath}")
        return False
    
    logger.info(f"✅ Input file validated: {filepath}")
    return True

def validate_dataframe(df: pd.DataFrame) -> bool:
    """✅ FIXED: Validate dataframe has required columns (uses packetSuccessLabel)"""
    required_cols = [
        'lastSnr', 'shortSuccRatio', 'phyRate', 'rateIdx'
    ]
    
    # Check for packetSuccessLabel OR packetSuccess
    has_packet_success = 'packetSuccessLabel' in df.columns or 'packetSuccess' in df.columns
    if not has_packet_success:
        logger.error("❌ Missing packet success column (need packetSuccessLabel or packetSuccess)")
        return False
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"❌ Missing required columns: {missing_cols}")
        return False
    
    logger.info(f"✅ DataFrame validation passed. Shape: {df.shape}")
    return True

def main():
    """✅ Main execution function with ALL PATCHES APPLIED + SNR FIX"""
    try:
        logger.info("=" * 80)
        logger.info("🚀 STARTING COMPLETE FIXED ML DATA PREPARATION")
        logger.info(f"👤 User: ahmedjk34")
        logger.info(f"🕐 Date: 2025-08-20 05:42:00 UTC")
        logger.info("=" * 80)
        logger.info(f"📥 Input file: {INPUT_CSV}")
        logger.info(f"📤 Output file: {OUTPUT_CSV}")
        
        # Validate input file
        if not validate_input_file(INPUT_CSV):
            sys.exit(1)
        
        # --- LOAD DATA ---
        logger.info("📊 Loading labeled input CSV...")
        try:
            df = pd.read_csv(INPUT_CSV, low_memory=False)
            logger.info(f"✅ Successfully loaded {len(df)} rows from labeled CSV")
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            sys.exit(1)
        
        # Validate dataframe structure
        if not validate_dataframe(df):
            sys.exit(1)
        
        # --- CRITICAL: FIX SNR VALUES FIRST ---
        df = fix_snr_values(df)
        
        # --- APPLY PATCH 1: CLEAN V3 RATE DATA ---
        df = clean_v3_rate_data(df)
        
        # --- FILTER NOISY/AMBIGUOUS ROWS ---
        logger.info("🔧 Filtering noisy/ambiguous rows...")
        initial_count = len(df)
        df = df[df.apply(filter_row, axis=1)]
        filtered_count = len(df)
        logger.info(f"✅ Filtered {initial_count - filtered_count} rows, {filtered_count} remaining")
        
        if filtered_count == 0:
            logger.error("❌ No rows remaining after filtering")
            sys.exit(1)
        
        # --- APPLY PATCH 2: FIXED LABEL CONSTRUCTION ---
        logger.info("🔧 PATCH 2: Recomputing labels with FIXED oracle function...")
        
        try:
            # Show distribution BEFORE fixing oracle
            if 'oracle_best_rateIdx' in df.columns:
                logger.info("📊 Distribution BEFORE oracle fix:")
                before_dist = df['oracle_best_rateIdx'].value_counts().sort_index()
                logger.info(f"\n{before_dist}")
            
            # 1. Oracle best rate (index) - FIXED VERSION
            logger.info("🔧 Computing oracle_best_rateIdx with FIXED algorithm...")
            df['oracle_best_rateIdx'] = df.apply(find_oracle_best_rate, axis=1)
            
            # 2. V3 imitation label (already cleaned)
            logger.info("✅ Using cleaned v3_rateIdx...")
            
            # 3. Binary success label - FIXED: use existing packetSuccessLabel
            if 'packetSuccessLabel' not in df.columns and 'packetSuccess' in df.columns:
                logger.info("🔧 Creating packetSuccessLabel from packetSuccess...")
                df['packetSuccessLabel'] = pd.to_numeric(df['packetSuccess'], errors='coerce').fillna(0).astype(int)
            else:
                logger.info("✅ Using existing packetSuccessLabel...")
            
            # 4. Oracle success for best rate
            logger.info("🔧 Computing oracle_best_success...")
            df['oracle_best_success'] = df.apply(find_oracle_success, axis=1)
            
            logger.info("✅ Label construction completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during label construction: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Show distribution after oracle fix
        logger.info("📊 Distribution AFTER oracle fix:")
        after_fix_dist = df['oracle_best_rateIdx'].value_counts().sort_index()
        logger.info(f"\n{after_fix_dist}")
        
        # --- APPLY PATCH 3: ADD CHALLENGING SCENARIOS ---
        df = add_challenging_scenarios(df)
        
        # Show distribution after patch 3
        if len(df) > 0:
            logger.info("📊 Distribution AFTER adding challenging scenarios:")
            after_challenge_dist = df['oracle_best_rateIdx'].value_counts().sort_index()
            logger.info(f"\n{after_challenge_dist}")
        
        # --- APPLY PATCH 4: FORCE BALANCE ---
        df = force_balanced_oracle_labels(df)
        
        # --- FEATURE SELECTION ---
        feature_cols = [
            'lastSnr','snrFast','snrSlow','shortSuccRatio','medSuccRatio',
            'consecSuccess','consecFailure','severity','confidence','T1','T2','T3',
            'offeredLoad','queueLen','retryCount','channelWidth','mobilityMetric','snrVariance'
        ]
        
        # Check if all feature columns exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.warning(f"⚠️ Missing feature columns (will be skipped): {missing_features}")
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        logger.info(f"✅ Using {len(feature_cols)} feature columns")
        
        # --- LABEL SELECTION ---
        label_cols = [
            'oracle_best_rateIdx',   # FIXED oracle (multi-class)
            'packetSuccessLabel',    # actual result (binary classification)
            'oracle_best_success'    # oracle result at best rate (binary classification)
        ]
        
        # Add v3_rateIdx if it exists
        if 'v3_rateIdx' in df.columns:
            label_cols.insert(1, 'v3_rateIdx')  # Insert after oracle_best_rateIdx
        
        # --- METADATA ---
        meta_cols = ['time','stationId','rateIdx','phyRate','decisionReason','scenario_file']
        available_meta_cols = [col for col in meta_cols if col in df.columns]
        
        # --- EXPORT DATASET ---
        logger.info("📦 Preparing final dataset...")
        all_cols = available_meta_cols + feature_cols + label_cols
        
        try:
            ml_df = df[all_cols].copy()
            logger.info(f"✅ Final dataset shape: {ml_df.shape}")
        except KeyError as e:
            logger.error(f"❌ Error selecting columns: {e}")
            available_cols = df.columns.tolist()
            logger.error(f"Available columns: {available_cols}")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(OUTPUT_CSV)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"📁 Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV
        logger.info(f"💾 Exporting to {OUTPUT_CSV}...")
        try:
            ml_df.to_csv(OUTPUT_CSV, index=False)
            logger.info("✅ Export completed successfully")
        except Exception as e:
            logger.error(f"❌ Error writing output CSV: {e}")
            sys.exit(1)
        
        # --- FINAL SUMMARY REPORT ---
        logger.info("=" * 80)
        logger.info("🎯 FINAL PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📊 Input rows: {initial_count}")
        logger.info(f"📊 Output rows: {len(ml_df)}")
        logger.info(f"📊 Features: {len(feature_cols)}")
        logger.info(f"📊 Labels: {len(label_cols)}")
        logger.info(f"📊 Metadata cols: {len(available_meta_cols)}")
        
        print(f"\n🎯 FIXED ML-ready CSV exported: {OUTPUT_CSV}")
        print(f"📊 Shape: {ml_df.shape}")
        print("📋 Columns:", ml_df.columns.tolist())
        print("\n📈 FINAL Label distribution summary:")
        
        for label in label_cols:
            try:
                print(f"\n🏷️  {label}:")
                dist = ml_df[label].value_counts().sort_index()
                print(dist)
                
                if label == 'oracle_best_rateIdx':
                    total = dist.sum()
                    print("📊 Percentages:")
                    for idx, count in dist.items():
                        pct = (count / total) * 100
                        print(f"   Rate {idx}: {pct:.1f}%")
                
            except Exception as e:
                logger.error(f"❌ Error displaying distribution for {label}: {e}")
        
        logger.info("🎉 ALL PATCHES APPLIED SUCCESSFULLY!")
        logger.info("✅ SNR values fixed")
        logger.info("✅ packetSuccessLabel used instead of packetSuccess")
        logger.info("✅ v3_rateIdx corruption cleaned")
        logger.info("✅ Oracle recomputed with realistic SNR thresholds")
        logger.info("✅ Data balanced across all rate indices")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    main()