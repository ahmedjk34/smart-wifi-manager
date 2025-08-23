#!/usr/bin/env python3
"""
High-Quality WiFi Dataset Generator
Generates realistic WiFi rate adaptation data based on physics models
Addresses poor ML performance from manually gathered data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import argparse
from pathlib import Path
import json

class WiFiChannelModel:
    """Physics-based WiFi channel modeling"""
    
    def __init__(self, frequency_ghz=2.4):
        self.frequency = frequency_ghz * 1e9  # Convert to Hz
        self.c = 3e8  # Speed of light
        self.rates_80211g = {
            0: 6e6,   # 6 Mbps
            1: 9e6,   # 9 Mbps  
            2: 12e6,  # 12 Mbps
            3: 18e6,  # 18 Mbps
            4: 24e6,  # 24 Mbps
            5: 36e6,  # 36 Mbps
            6: 48e6,  # 48 Mbps
            7: 54e6   # 54 Mbps
        }
        
    def path_loss_model(self, distance, environment='indoor'):
        """Calculate path loss based on environment"""
        if environment == 'indoor':
            # ITU-R P.1238 indoor model
            L0 = 20 * np.log10(self.frequency) + 20 * np.log10(distance) - 28
            n = 2.8  # Path loss exponent for indoor
            L_floor = 15  # Floor penetration loss
            return L0 + 10 * n * np.log10(distance/1) + L_floor
        elif environment == 'outdoor':
            # Free space + additional losses
            L_fs = 20 * np.log10(4 * np.pi * distance * self.frequency / self.c)
            return L_fs + np.random.normal(0, 4)  # Shadow fading
        else:  # mixed
            return 0.6 * self.path_loss_model(distance, 'indoor') + \
                   0.4 * self.path_loss_model(distance, 'outdoor')
    
    def fading_model(self, coherence_time=50):
        """Generate realistic fading patterns"""
        # Rayleigh fading for NLOS, Rician for LOS
        los_prob = np.random.random()
        if los_prob > 0.3:  # 70% NLOS probability
            # Rayleigh fading
            return np.random.rayleigh(1.0)
        else:
            # Rician fading (K-factor = 3 dB)
            K = 2  # Linear K-factor
            return np.sqrt((K/(K+1)) + np.random.rayleigh(1.0)/(K+1))
    
    def interference_model(self, num_interferers, distance_interferers):
        """Model interference from other APs/devices"""
        interference_power = 0
        for i in range(num_interferers):
            # Random interferer distance and power
            int_distance = distance_interferers[i] if i < len(distance_interferers) else np.random.uniform(20, 100)
            int_power = 20 - self.path_loss_model(int_distance, 'indoor')  # dBm
            interference_power += 10**(int_power/10)  # Convert to linear
        
        return 10 * np.log10(interference_power + 1e-12)  # Convert back to dB
    
    def calculate_snr(self, tx_power, distance, environment, num_interferers=0, interferer_distances=None):
        """Calculate realistic SNR"""
        # Signal power
        path_loss = self.path_loss_model(distance, environment)
        fading_loss = 20 * np.log10(self.fading_model())
        signal_power = tx_power - path_loss - fading_loss

    # Noise power (thermal noise + interference)
        thermal_noise = -94  # dBm for 20MHz bandwidth

        if num_interferers > 0 and interferer_distances is not None and len(interferer_distances) > 0:
            interference = self.interference_model(num_interferers, interferer_distances)
            noise_power = 10 * np.log10(10**(thermal_noise/10) + 10**(interference/10))
        else:
            noise_power = thermal_noise

        snr = signal_power - noise_power
        return max(0, min(50, snr))  # Realistic SNR bounds

class WiFiDatasetGenerator:
    """Generate comprehensive WiFi dataset"""
    
    def __init__(self, n_samples=2000000):
        self.n_samples = n_samples
        self.channel_model = WiFiChannelModel()
        self.environments = ['indoor', 'outdoor', 'mixed']
        self.scenarios = self._define_scenarios()
        
    def _define_scenarios(self):
        """Define realistic WiFi scenarios"""
        return {
            'static_good': {
                'distance_range': (5, 30),
                'speed_range': (0, 0.5),
                'interference_prob': 0.1,
                'environment': 'indoor',
                'weight': 0.25
            },
            'static_medium': {
                'distance_range': (20, 80),
                'speed_range': (0, 1),
                'interference_prob': 0.3,
                'environment': 'mixed',
                'weight': 0.25
            },
            'mobile_indoor': {
                'distance_range': (10, 50),
                'speed_range': (1, 5),
                'interference_prob': 0.4,
                'environment': 'indoor',
                'weight': 0.2
            },
            'mobile_outdoor': {
                'distance_range': (20, 150),
                'speed_range': (2, 15),
                'interference_prob': 0.2,
                'environment': 'outdoor',
                'weight': 0.15
            },
            'high_interference': {
                'distance_range': (15, 60),
                'speed_range': (0, 3),
                'interference_prob': 0.8,
                'environment': 'indoor',
                'weight': 0.15
            }
        }
    
    def generate_time_series_features(self, base_snr, duration_steps=100):
        """Generate realistic time-series SNR patterns"""
        # Create correlated SNR sequence
        snr_sequence = []
        current_snr = base_snr
        
        for i in range(duration_steps):
            # Add temporal correlation
            alpha = 0.8  # Correlation factor
            noise = np.random.normal(0, 1.5)
            current_snr = alpha * current_snr + (1-alpha) * base_snr + noise
            current_snr = max(0, min(50, current_snr))
            snr_sequence.append(current_snr)
        
        # Calculate derived features
        snr_fast = current_snr  # Most recent
        snr_slow = np.mean(snr_sequence[-20:])  # 20-step average
        snr_variance = np.var(snr_sequence[-10:])  # Recent variance
        
        return snr_fast, snr_slow, snr_variance, snr_sequence
    
    def calculate_success_ratio(self, snr, rate_idx, packet_size=1500):
        """Calculate packet success ratio based on SNR and rate"""
        rate = self.channel_model.rates_80211g[rate_idx]
        
        # SNR thresholds for different rates (empirical values)
        snr_thresholds = {
            0: 6,   # 6 Mbps needs 6 dB
            1: 8,   # 9 Mbps needs 8 dB
            2: 10,  # 12 Mbps needs 10 dB
            3: 12,  # 18 Mbps needs 12 dB
            4: 15,  # 24 Mbps needs 15 dB
            5: 18,  # 36 Mbps needs 18 dB
            6: 22,  # 48 Mbps needs 22 dB
            7: 25   # 54 Mbps needs 25 dB
        }
        
        required_snr = snr_thresholds[rate_idx]
        snr_margin = snr - required_snr
        
        # Sigmoid function for success probability
        success_prob = 1 / (1 + np.exp(-0.5 * snr_margin))
        
        # Adjust for packet size (longer packets more likely to fail)
        size_penalty = 1500 / packet_size  # Baseline 1500 bytes
        success_prob *= size_penalty
        
        return max(0.01, min(0.99, success_prob))
    
    def select_optimal_rate(self, snr, scenario_type='balanced'):
        """Select optimal rate based on SNR and scenario"""
        if scenario_type == 'conservative':
        # Conservative: choose rate with >95% success
            optimal_rate = 0  # default fallback
            for rate_idx in range(8):
                if self.calculate_success_ratio(snr, rate_idx) > 0.95:
                    optimal_rate = rate_idx
                    break
        elif scenario_type == 'aggressive':
        # Aggressive: choose highest rate with >70% success
            optimal_rate = 0
            for rate_idx in range(7, -1, -1):
                if self.calculate_success_ratio(snr, rate_idx) > 0.70:
                    optimal_rate = rate_idx
                    break
        else:  # balanced
        # Balanced: maximize throughput * success_ratio
            best_throughput = 0
            optimal_rate = 0
            for rate_idx in range(8):
                rate = self.channel_model.rates_80211g[rate_idx]
                success = self.calculate_success_ratio(snr, rate_idx)
                effective_throughput = rate * success
                if effective_throughput > best_throughput:
                    best_throughput = effective_throughput
                    optimal_rate = rate_idx

        return optimal_rate

    def generate_samples(self):
        """Generate complete dataset"""
        print("🔄 Generating high-quality WiFi dataset...")
        
        samples = []
        scenario_counts = {}
        
        # Calculate samples per scenario
        for scenario_name, scenario in self.scenarios.items():
            count = int(self.n_samples * scenario['weight'])
            scenario_counts[scenario_name] = count
        
        print(f"📊 Scenario distribution: {scenario_counts}")
        
        for scenario_name, scenario in tqdm(self.scenarios.items(), desc="Scenarios"):
            n_scenario_samples = scenario_counts[scenario_name]
            
            for i in tqdm(range(n_scenario_samples), desc=f"{scenario_name}", leave=False):
                # Generate scenario parameters
                distance = np.random.uniform(*scenario['distance_range'])
                speed = np.random.uniform(*scenario['speed_range'])
                environment = scenario['environment']
                
                # Interference
                if np.random.random() < scenario['interference_prob']:
                    num_interferers = np.random.randint(1, 4)
                    interferer_distances = np.random.uniform(20, 100, num_interferers)
                else:
                    num_interferers = 0
                    interferer_distances = []
                
                # Calculate base SNR
                tx_power = 20  # 20 dBm
                base_snr = self.channel_model.calculate_snr(
                    tx_power, distance, environment, num_interferers, interferer_distances
                )
                
                # Generate time-series features
                snr_fast, snr_slow, snr_variance, snr_history = self.generate_time_series_features(base_snr)
                
                # Calculate success ratios for different windows
                short_successes = []
                medium_successes = []
                
                # Simulate recent transmission history
                for j in range(20):  # Medium window
                    hist_snr = snr_history[-(j+1)] if j < len(snr_history) else base_snr
                    # Use current rate estimate for simulation
                    temp_rate = min(7, max(0, int((hist_snr - 5) / 3)))
                    success = np.random.random() < self.calculate_success_ratio(hist_snr, temp_rate)
                    medium_successes.append(success)
                    
                    if j < 10:  # Short window
                        short_successes.append(success)
                
                short_succ_ratio = sum(short_successes) / len(short_successes)
                medium_succ_ratio = sum(medium_successes) / len(medium_successes)
                
                # Calculate other features
                consec_success = np.random.randint(0, 15)
                consec_failure = np.random.randint(0, 8)
                severity = max(0, min(1, consec_failure / 10))
                confidence = max(0.1, min(1, 1 - severity))
                
                # Timing features (ms)
                T1 = np.random.uniform(1, 50)
                T2 = T1 * np.random.uniform(1.5, 2.5)
                T3 = T2 * np.random.uniform(1.2, 2.0)
                
                # Network features
                offered_load = np.random.uniform(1, 50)  # Mbps
                queue_length = np.random.randint(0, 100)
                retry_count = np.random.randint(0, 7)
                channel_width = 20  # MHz for 802.11g
                
                # Mobility metric based on speed and SNR variance
                mobility_metric = min(1.0, (speed / 10.0) + (snr_variance / 20.0))
                
                # Generate optimal rates for different strategies
                oracle_rate = self.select_optimal_rate(snr_fast, 'balanced')
                v3_rate = self.select_optimal_rate(snr_fast, 'conservative')
                
                # Create sample
                sample = {
                    'lastSnr': snr_fast,
                    'snrFast': snr_fast,
                    'snrSlow': snr_slow,
                    'shortSuccRatio': short_succ_ratio,
                    'medSuccRatio': medium_succ_ratio,
                    'consecSuccess': consec_success,
                    'consecFailure': consec_failure,
                    'severity': severity,
                    'confidence': confidence,
                    'T1': T1,
                    'T2': T2,
                    'T3': T3,
                    'offeredLoad': offered_load,
                    'queueLen': queue_length,
                    'retryCount': retry_count,
                    'channelWidth': channel_width,
                    'mobilityMetric': mobility_metric,
                    'snrVariance': snr_variance,
                    'oracle_best_rateIdx': oracle_rate,
                    'v3_rateIdx': v3_rate,
                    'distance': distance,
                    'speed': speed,
                    'environment': environment,
                    'scenario': scenario_name,
                    'num_interferers': num_interferers
                }
                
                samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def save_dataset(self, df, filename='realistic_wifi_dataset.csv'):
        """Save dataset and metadata"""
        print(f"💾 Saving dataset to {filename}")
        df.to_csv(filename, index=False)
        
        # Save metadata
        metadata = {
            'total_samples': len(df),
            'features': list(df.columns),
            'scenarios': self.scenarios,
            'rate_distribution': df['oracle_best_rateIdx'].value_counts().to_dict(),
            'v3_rate_distribution': df['v3_rateIdx'].value_counts().to_dict(),
            'snr_stats': {
                'mean': float(df['lastSnr'].mean()),
                'std': float(df['lastSnr'].std()),
                'min': float(df['lastSnr'].min()),
                'max': float(df['lastSnr'].max())
            }
        }
        
        with open(f"{filename.replace('.csv', '_metadata.json')}", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate summary plots
        # self.plot_dataset_summary(df, filename.replace('.csv', ''))
        
        print(f"✅ Dataset saved with {len(df)} samples")
        print(f"📊 Oracle rate distribution: {df['oracle_best_rateIdx'].value_counts().sort_index().to_dict()}")
        print(f"📊 V3 rate distribution: {df['v3_rateIdx'].value_counts().sort_index().to_dict()}")
    
    def plot_dataset_summary(self, df, base_filename):
        """Generate summary plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('WiFi Dataset Summary', fontsize=16)
        
        # Rate distributions
        axes[0,0].hist(df['oracle_best_rateIdx'], bins=8, alpha=0.7, label='Oracle')
        axes[0,0].hist(df['v3_rateIdx'], bins=8, alpha=0.7, label='V3')
        axes[0,0].set_xlabel('Rate Index')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_title('Rate Distribution')
        axes[0,0].legend()
        
        # SNR distribution
        axes[0,1].hist(df['lastSnr'], bins=50, alpha=0.7)
        axes[0,1].set_xlabel('SNR (dB)')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_title('SNR Distribution')
        
        # Success ratio vs SNR
        axes[0,2].scatter(df['lastSnr'], df['shortSuccRatio'], alpha=0.1, s=1)
        axes[0,2].set_xlabel('SNR (dB)')
        axes[0,2].set_ylabel('Success Ratio')
        axes[0,2].set_title('Success Ratio vs SNR')
        
        # Environment distribution
        env_counts = df['environment'].value_counts()
        axes[1,0].pie(env_counts.values, labels=env_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Environment Distribution')
        
        # Speed vs Distance
        axes[1,1].scatter(df['distance'], df['speed'], alpha=0.1, s=1)
        axes[1,1].set_xlabel('Distance (m)')
        axes[1,1].set_ylabel('Speed (m/s)')
        axes[1,1].set_title('Mobility Pattern')
        
        # Rate vs SNR
        for rate in range(8):
            rate_data = df[df['oracle_best_rateIdx'] == rate]
            if len(rate_data) > 0:
                axes[1,2].scatter(rate_data['lastSnr'], [rate]*len(rate_data), 
                                alpha=0.1, s=1, label=f'Rate {rate}')
        axes[1,2].set_xlabel('SNR (dB)')
        axes[1,2].set_ylabel('Selected Rate')
        axes[1,2].set_title('Optimal Rate Selection')
        
        plt.tight_layout()
        plt.savefig(f'{base_filename}_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate realistic WiFi dataset')
    parser.add_argument('--samples', type=int, default=2000000,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='realistic_wifi_dataset.csv',
                       help='Output filename')
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = WiFiDatasetGenerator(n_samples=args.samples)
    df = generator.generate_samples()
    generator.save_dataset(df, args.output)
    
    print(f"\n🎉 High-quality WiFi dataset generated successfully!")
    print(f"📁 Dataset: {args.output}")
    print(f"📊 Samples: {len(df)}")
    print(f"📈 Features: {len(df.columns) - 5}")  # Subtract metadata columns

if __name__ == "__main__":
    main()