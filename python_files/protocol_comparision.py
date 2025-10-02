"""
Enhanced Protocol Comparison Analyzer - NEW ML PIPELINE VERSION (COMPLETE)
Compatible with AARF vs SmartRF-v7.0 (9 features, fully fixed)

Author: ahmedjk34 (https://github.com/ahmedjk34)
Date: 2025-10-02 19:10:33 UTC
Version: 2.0 (NEW PIPELINE - COMPLETE WITH ALL METHODS)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedProtocolComparisonAnalyzer:
    """Enhanced side-by-side comparison analyzer with ML-specific metrics."""
    
    def __init__(self, protocol1_csv: str, protocol2_csv: str, 
                 protocol1_name: str = None, protocol2_name: str = None):
        """Initialize the comparison analyzer."""
        self.protocol1_csv = Path(__file__).parent.parent / protocol1_csv
        self.protocol2_csv = Path(__file__).parent.parent / protocol2_csv
        
        self.protocol1_name = protocol1_name or self._extract_protocol_name(protocol1_csv)
        self.protocol2_name = protocol2_name or self._extract_protocol_name(protocol2_csv)
        
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / 'enhanced_comparison' / f'{self.protocol1_name}_vs_{self.protocol2_name}'
        self._create_results_directory()
        
        self._setup_logging()
        
        self.df1: Optional[pd.DataFrame] = None
        self.df2: Optional[pd.DataFrame] = None
        self.combined_df: Optional[pd.DataFrame] = None
        self.plots_generated: List[str] = []
        self.comparison_results: Dict = {}
        
    def _extract_protocol_name(self, csv_path: str) -> str:
        """Extract protocol name from CSV filename (NEW PIPELINE VERSION)."""
        filename = Path(csv_path).stem.lower()
        
        if 'aarf' in filename:
            if 'fixed' in filename or 'expanded' in filename:
                return 'AARF-v2.0'
            return 'AARF'
        elif 'smart' in filename:
            if 'fixed' in filename and 'expanded' in filename:
                return 'SmartRF-v7.0'
            elif 'newpipeline' in filename or 'new-pipeline' in filename:
                return 'SmartRF-v6.0'
            elif 'v1' in filename:
                return 'SmartV1'
            elif 'v2' in filename:
                return 'SmartV2'
            elif 'v3' in filename:
                return 'SmartV3'
            elif 'v4' in filename:
                return 'SmartV4'
            elif 'xgb' in filename:
                return 'SmartXGB'
            elif 'rf' in filename:
                if 'v3' in filename:
                    return 'SmartRFV3'
                else:
                    return 'SmartRF'
            else:
                return 'Smart'
        else:
            return filename.replace('-benchmark', '').replace('_benchmark', '').upper()
    
    def _create_results_directory(self) -> None:
        """Create the results directory structure."""
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            print(f"Results directory created: {self.results_dir}")
        except Exception as e:
            print(f"Failed to create results directory: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = self.results_dir / 'enhanced_comparison_analysis.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True
        )
        
        logger.info(f"=== Enhanced Protocol Comparison: {self.protocol1_name} vs {self.protocol2_name} ===")
        logger.info(f"NEW PIPELINE Version 2.0 (9 Features, ML Metrics)")
        logger.info(f"Log file: {log_file}")
    
    def load_data(self) -> bool:
        """Load both CSV files with error handling."""
        try:
            if not self.protocol1_csv.exists():
                logger.error(f"Protocol 1 CSV not found: {self.protocol1_csv}")
                return False
            
            self.df1 = pd.read_csv(self.protocol1_csv)
            self.df1['Protocol'] = self.protocol1_name
            logger.info(f"Loaded {len(self.df1)} rows for {self.protocol1_name}")
            
            if not self.protocol2_csv.exists():
                logger.error(f"Protocol 2 CSV not found: {self.protocol2_csv}")
                return False
            
            self.df2 = pd.read_csv(self.protocol2_csv)
            self.df2['Protocol'] = self.protocol2_name
            logger.info(f"Loaded {len(self.df2)} rows for {self.protocol2_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self) -> None:
        """Preprocess and combine data for analysis (NEW PIPELINE VERSION)."""
        if self.df1 is None or self.df2 is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        column_mappings = {
            'Distance': 'distance',
            'Speed': 'speed',
            'Interferers': 'interferers',
            'PacketSize': 'packet_size',
            'TrafficRate': 'traffic_rate',
            'Throughput(Mbps)': 'throughput',
            'PacketLoss(%)': 'packet_loss',
            'AvgDelay(ms)': 'avg_delay',
            'AvgDelay(s)': 'avg_delay',
            'Jitter(ms)': 'jitter',
            'Jitter(s)': 'jitter',
            'RxPackets': 'rx_packets',
            'TxPackets': 'tx_packets',
            'MLInferences': 'ml_inferences',
            'MLFailures': 'ml_failures',
            'AvgMLConfidence': 'ml_confidence',
            'RateChanges': 'rate_changes',
            'AvgSNR': 'avg_snr',
            'FinalContext': 'final_context',
            'StatsValid': 'stats_valid'
        }
        
        self.df1 = self.df1.rename(columns=column_mappings)
        self.df2 = self.df2.rename(columns=column_mappings)
        
        for df in [self.df1, self.df2]:
            if 'avg_delay' in df.columns and df['avg_delay'].max() < 10:
                df['avg_delay'] = df['avg_delay'] * 1000
            if 'jitter' in df.columns and df['jitter'].max() < 1:
                df['jitter'] = df['jitter'] * 1000
        
        self.combined_df = pd.concat([self.df1, self.df2], ignore_index=True)
        
        if 'traffic_rate' in self.combined_df.columns:
            self.combined_df['traffic_rate_num'] = (
                self.combined_df['traffic_rate']
                .astype(str)
                .str.replace('Mbps', '', case=False)
                .str.replace(' ', '')
                .astype(float)
            )
        
        self.combined_df['scenario_group'] = (
            self.combined_df['distance'].astype(str) + 'm_' +
            self.combined_df['speed'].astype(str) + 'mps_' +
            self.combined_df['interferers'].astype(str) + 'intf'
        )
        
        logger.info(f"Combined data: {len(self.combined_df)} total rows")
        logger.info(f"Scenarios: {self.combined_df['scenario_group'].nunique()} unique scenarios")
        
        ml_columns = ['ml_inferences', 'ml_failures', 'ml_confidence', 'rate_changes', 'avg_snr']
        has_ml = any(col in self.combined_df.columns for col in ml_columns)
        logger.info(f"ML metrics detected: {has_ml}")
    
    def create_enhanced_comparison_plots(self) -> None:
        """Create enhanced comparison plots with ML-specific analysis."""
        if self.combined_df is None:
            logger.warning("No data available for plotting")
            return
        
        logger.info("Creating enhanced comparison plots (NEW PIPELINE)...")
        
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # NEW ML-SPECIFIC PLOTS (6 new plots):
        self._create_ml_success_rate_analysis()
        self._create_ml_confidence_vs_snr()
        self._create_rate_change_stability_analysis()
        self._create_mobility_fix_validation()
        self._create_interference_resilience_comparison()
        self._create_snr_analysis_plots()
        
        logger.info(f"Created {len(self.plots_generated)} ML-specific plots")
    
    def _create_ml_success_rate_analysis(self) -> None:
        """NEW: Create ML success rate analysis plots."""
        try:
            if 'ml_inferences' not in self.combined_df.columns or 'ml_failures' not in self.combined_df.columns:
                logger.info("ML metrics not available, skipping ML success rate analysis")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            self.combined_df['ml_success_rate'] = (
                (self.combined_df['ml_inferences'] - self.combined_df['ml_failures']) / 
                self.combined_df['ml_inferences'].replace(0, 1) * 100
            )
            
            # ML Success Rate vs Distance
            ax1 = axes[0, 0]
            ml_dist = (
                self.combined_df[self.combined_df['ml_inferences'] > 0]
                .groupby(['distance', 'Protocol'])['ml_success_rate']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = ml_dist[ml_dist['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('distance')
                    ax1.plot(protocol_data['distance'], 
                            protocol_data['ml_success_rate'],
                            marker='o', linewidth=2, markersize=6, label=protocol)
            
            ax1.set_xlabel('Distance (m)')
            ax1.set_ylabel('ML Success Rate (%)')
            ax1.set_title('ML Inference Success Rate vs Distance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=85, color='r', linestyle='--', alpha=0.5, label='Target (85%)')
            
            # ML Success Rate vs Interferers
            ax2 = axes[0, 1]
            ml_intf = (
                self.combined_df[self.combined_df['ml_inferences'] > 0]
                .groupby(['interferers', 'Protocol'])['ml_success_rate']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = ml_intf[ml_intf['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('interferers')
                    ax2.plot(protocol_data['interferers'], 
                            protocol_data['ml_success_rate'],
                            marker='s', linewidth=2, markersize=6, label=protocol)
            
            ax2.set_xlabel('Number of Interferers')
            ax2.set_ylabel('ML Success Rate (%)')
            ax2.set_title('ML Inference Success Rate vs Interference')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=85, color='r', linestyle='--', alpha=0.5, label='Target (85%)')
            
            # ML Success Rate vs Traffic Rate
            ax3 = axes[1, 0]
            ml_rate = (
                self.combined_df[self.combined_df['ml_inferences'] > 0]
                .groupby(['traffic_rate_num', 'Protocol'])['ml_success_rate']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = ml_rate[ml_rate['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('traffic_rate_num')
                    ax3.plot(protocol_data['traffic_rate_num'], 
                            protocol_data['ml_success_rate'],
                            marker='^', linewidth=2, markersize=6, label=protocol)
            
            ax3.set_xlabel('Traffic Rate (Mbps)')
            ax3.set_ylabel('ML Success Rate (%)')
            ax3.set_title('ML Inference Success Rate vs Traffic Rate')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log')
            ax3.axhline(y=85, color='r', linestyle='--', alpha=0.5, label='Target (85%)')
            
            # ML Failure Distribution
            ax4 = axes[1, 1]
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = self.combined_df[
                    (self.combined_df['Protocol'] == protocol) & 
                    (self.combined_df['ml_inferences'] > 0)
                ]
                
                if len(protocol_data) > 0:
                    failure_rate = (protocol_data['ml_failures'] / protocol_data['ml_inferences'] * 100)
                    ax4.hist(failure_rate, bins=20, alpha=0.6, label=protocol, edgecolor='black')
            
            ax4.set_xlabel('ML Failure Rate (%)')
            ax4.set_ylabel('Number of Scenarios')
            ax4.set_title('ML Failure Rate Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'ml_success_rate_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('ml_success_rate_analysis.png')
            logger.info("Created ML success rate analysis")
            
        except Exception as e:
            logger.error(f"Error creating ML success rate analysis: {e}")
    
    def _create_ml_confidence_vs_snr(self) -> None:
        """NEW: Create ML confidence vs SNR analysis."""
        try:
            if 'ml_confidence' not in self.combined_df.columns or 'avg_snr' not in self.combined_df.columns:
                logger.info("ML confidence or SNR data not available")
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # ML Confidence vs SNR (scatter plot)
            ax1 = axes[0]
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = self.combined_df[
                    (self.combined_df['Protocol'] == protocol) &
                    (self.combined_df['ml_confidence'] > 0)
                ]
                
                if len(protocol_data) > 0:
                    ax1.scatter(protocol_data['avg_snr'], 
                               protocol_data['ml_confidence'],
                               alpha=0.5, s=30, label=protocol)
            
            ax1.set_xlabel('Average SNR (dB)')
            ax1.set_ylabel('ML Confidence')
            ax1.set_title('ML Confidence vs SNR')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0.15, color='r', linestyle='--', alpha=0.5, label='Threshold (0.15)')
            
            # Average ML Confidence by SNR Bucket
            ax2 = axes[1]
            
            snr_bins = [-30, -10, 5, 15, 25, 45]
            snr_labels = ['<-10dB', '-10-5dB', '5-15dB', '15-25dB', '>25dB']
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = self.combined_df[
                    (self.combined_df['Protocol'] == protocol) &
                    (self.combined_df['ml_confidence'] > 0)
                ].copy()
                
                if len(protocol_data) > 0:
                    protocol_data['snr_bucket'] = pd.cut(protocol_data['avg_snr'], 
                                                         bins=snr_bins, 
                                                         labels=snr_labels)
                    
                    snr_conf = (
                        protocol_data.groupby('snr_bucket')['ml_confidence']
                        .mean()
                        .reindex(snr_labels, fill_value=0)
                    )
                    
                    ax2.plot(snr_labels, snr_conf.values, 
                            marker='o', linewidth=2, markersize=8, label=protocol)
            
            ax2.set_xlabel('SNR Range')
            ax2.set_ylabel('Average ML Confidence')
            ax2.set_title('ML Confidence by SNR Range')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'ml_confidence_vs_snr.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('ml_confidence_vs_snr.png')
            logger.info("Created ML confidence vs SNR analysis")
            
        except Exception as e:
            logger.error(f"Error creating ML confidence vs SNR plot: {e}")
    def _create_rate_change_stability_analysis(self) -> None:
        """NEW: Create rate change stability analysis (validates FIX #4/#6)."""
        try:
            if 'rate_changes' not in self.combined_df.columns:
                logger.info("Rate changes data not available")
                return
            
            # Check if we have valid data
            valid_data = self.combined_df[self.combined_df['rate_changes'].notna()]
            if len(valid_data) == 0:
                logger.info("No valid rate_changes data")
                return
            
            # Check which protocols have rate_changes
            protocols_with_data = []
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = valid_data[valid_data['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocols_with_data.append(protocol)
            
            if len(protocols_with_data) == 0:
                logger.info("No protocols with rate_changes data")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Rate Changes vs Distance
            ax1 = axes[0, 0]
            rate_dist = (
                valid_data.groupby(['distance', 'Protocol'])['rate_changes']
                .mean()
                .reset_index()
            )
            
            for protocol in protocols_with_data:
                protocol_data = rate_dist[rate_dist['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('distance')
                    ax1.plot(protocol_data['distance'], 
                            protocol_data['rate_changes'],
                            marker='o', linewidth=2, markersize=6, label=protocol)
            
            ax1.set_xlabel('Distance (m)')
            ax1.set_ylabel('Average Rate Changes')
            title_suffix = ' (Smart-RF only)' if len(protocols_with_data) == 1 else ''
            ax1.set_title(f'Rate Change Frequency vs Distance{title_suffix}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Rate Changes vs Speed (Mobility Test)
            ax2 = axes[0, 1]
            rate_speed = (
                valid_data.groupby(['speed', 'Protocol'])['rate_changes']
                .mean()
                .reset_index()
            )
            
            for protocol in protocols_with_data:
                protocol_data = rate_speed[rate_speed['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('speed')
                    ax2.plot(protocol_data['speed'], 
                            protocol_data['rate_changes'],
                            marker='s', linewidth=2, markersize=6, label=protocol)
            
            ax2.set_xlabel('Speed (m/s)')
            ax2.set_ylabel('Average Rate Changes')
            ax2.set_title(f'Rate Change Frequency vs Mobility{title_suffix}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Rate Change Distribution
            ax3 = axes[1, 0]
            for protocol in protocols_with_data:
                protocol_data = valid_data[valid_data['Protocol'] == protocol]
                
                if len(protocol_data) > 0:
                    ax3.hist(protocol_data['rate_changes'], bins=30, alpha=0.6, 
                            label=protocol, edgecolor='black')
            
            ax3.set_xlabel('Rate Changes per Test')
            ax3.set_ylabel('Number of Scenarios')
            ax3.set_title('Rate Change Distribution (Lower = More Stable)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Rate Changes vs Interferers
            ax4 = axes[1, 1]
            rate_intf = (
                valid_data.groupby(['interferers', 'Protocol'])['rate_changes']
                .mean()
                .reset_index()
            )
            
            for protocol in protocols_with_data:
                protocol_data = rate_intf[rate_intf['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('interferers')
                    ax4.plot(protocol_data['interferers'], 
                            protocol_data['rate_changes'],
                            marker='^', linewidth=2, markersize=6, label=protocol)
            
            ax4.set_xlabel('Number of Interferers')
            ax4.set_ylabel('Average Rate Changes')
            ax4.set_title(f'Rate Change Frequency vs Interference{title_suffix}')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'rate_change_stability_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('rate_change_stability_analysis.png')
            logger.info("Created rate change stability analysis")
            
        except Exception as e:
            logger.error(f"Error creating rate change stability analysis: {e}")

    def _create_mobility_fix_validation(self) -> None:
        """NEW: Validate mobility fix (FIX #1) - Speed vs Performance."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Throughput vs Speed
            ax1 = axes[0, 0]
            speed_th = (
                self.combined_df.groupby(['speed', 'Protocol'])['throughput']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = speed_th[speed_th['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('speed')
                    ax1.plot(protocol_data['speed'], 
                            protocol_data['throughput'],
                            marker='o', linewidth=2, markersize=8, label=protocol)
            
            ax1.set_xlabel('Speed (m/s)')
            ax1.set_ylabel('Average Throughput (Mbps)')
            ax1.set_title('MOBILITY FIX VALIDATION: Throughput vs Speed')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=0, color='g', linestyle='--', alpha=0.3, label='Stationary')
            
            # PDR vs Speed (FIXED FutureWarning)
            ax2 = axes[0, 1]
            speed_pdr_data = []
            for (speed, protocol), group in self.combined_df.groupby(['speed', 'Protocol']):
                pdr = 100 - group['packet_loss'].mean()
                speed_pdr_data.append({'speed': speed, 'Protocol': protocol, 'pdr': pdr})
            
            speed_pdr = pd.DataFrame(speed_pdr_data)
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = speed_pdr[speed_pdr['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('speed')
                    ax2.plot(protocol_data['speed'], 
                            protocol_data['pdr'],
                            marker='s', linewidth=2, markersize=8, label=protocol)
            
            ax2.set_xlabel('Speed (m/s)')
            ax2.set_ylabel('Packet Delivery Ratio (%)')
            ax2.set_title('MOBILITY FIX VALIDATION: PDR vs Speed')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=0, color='g', linestyle='--', alpha=0.3, label='Stationary')
            
            # Performance Retention with Mobility
            ax3 = axes[1, 0]
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = self.combined_df[self.combined_df['Protocol'] == protocol]
                speed_perf = (
                    protocol_data.groupby('speed')['throughput']
                    .mean()
                    .reset_index()
                    .sort_values('speed')
                )
                
                if len(speed_perf) > 0:
                    baseline = speed_perf[speed_perf['speed'] == 0]['throughput'].values
                    if len(baseline) > 0:
                        baseline = baseline[0]
                        retention = (speed_perf['throughput'] / baseline * 100)
                        ax3.plot(speed_perf['speed'], retention,
                                marker='D', linewidth=2, markersize=6, 
                                label=f'{protocol} Retention')
            
            ax3.set_xlabel('Speed (m/s)')
            ax3.set_ylabel('Performance Retention (%)')
            ax3.set_title('MOBILITY FIX: Performance Retention vs Speed')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Threshold')
            
            # Mobility Impact by Distance
            ax4 = axes[1, 1]
            distances = sorted(self.combined_df['distance'].unique())[:3]
            colors = plt.cm.Set1(np.linspace(0, 1, len(distances)))
            
            for i, dist in enumerate(distances):
                dist_data = self.combined_df[self.combined_df['distance'] == dist]
                dist_speed = (
                    dist_data.groupby(['speed', 'Protocol'])['throughput']
                    .mean()
                    .reset_index()
                )
                
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    protocol_data = dist_speed[dist_speed['Protocol'] == protocol]
                    if len(protocol_data) > 0:
                        protocol_data = protocol_data.sort_values('speed')
                        linestyle = '-' if protocol == self.protocol1_name else '--'
                        ax4.plot(protocol_data['speed'], 
                                protocol_data['throughput'],
                                color=colors[i], linestyle=linestyle,
                                marker='o', linewidth=2, markersize=4, 
                                label=f'{protocol} @ {dist}m')
            
            ax4.set_xlabel('Speed (m/s)')
            ax4.set_ylabel('Throughput (Mbps)')
            ax4.set_title('Mobility Impact at Different Distances')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'mobility_fix_validation.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('mobility_fix_validation.png')
            logger.info("Created mobility fix validation plots")
            
        except Exception as e:
            logger.error(f"Error creating mobility fix validation: {e}")

    def _create_interference_resilience_comparison(self) -> None:
        """NEW: Create interference resilience comparison (Smart-RF advantage)."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Throughput vs Interferers
            ax1 = axes[0, 0]
            intf_th = (
                self.combined_df.groupby(['interferers', 'Protocol'])['throughput']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = intf_th[intf_th['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('interferers')
                    ax1.plot(protocol_data['interferers'], 
                            protocol_data['throughput'],
                            marker='o', linewidth=3, markersize=8, label=protocol)
            
            ax1.set_xlabel('Number of Interferers')
            ax1.set_ylabel('Average Throughput (Mbps)')
            ax1.set_title('INTERFERENCE RESILIENCE: Throughput vs Interferers')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # PDR vs Interferers (FIXED FutureWarning)
            ax2 = axes[0, 1]
            intf_pdr_data = []
            for (interferers, protocol), group in self.combined_df.groupby(['interferers', 'Protocol']):
                pdr = 100 - group['packet_loss'].mean()
                intf_pdr_data.append({'interferers': interferers, 'Protocol': protocol, 'pdr': pdr})
            
            intf_pdr = pd.DataFrame(intf_pdr_data)
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = intf_pdr[intf_pdr['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('interferers')
                    ax2.plot(protocol_data['interferers'], 
                            protocol_data['pdr'],
                            marker='s', linewidth=3, markersize=8, label=protocol)
            
            ax2.set_xlabel('Number of Interferers')
            ax2.set_ylabel('Packet Delivery Ratio (%)')
            ax2.set_title('INTERFERENCE RESILIENCE: PDR vs Interferers')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Degradation with Interference
            ax3 = axes[1, 0]
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = self.combined_df[self.combined_df['Protocol'] == protocol]
                intf_perf = (
                    protocol_data.groupby('interferers')['throughput']
                    .mean()
                    .reset_index()
                    .sort_values('interferers')
                )
                
                if len(intf_perf) > 0:
                    baseline = intf_perf[intf_perf['interferers'] == 0]['throughput'].values
                    if len(baseline) > 0:
                        baseline = baseline[0]
                        degradation = ((baseline - intf_perf['throughput']) / baseline * 100)
                        ax3.plot(intf_perf['interferers'], degradation,
                                marker='^', linewidth=2, markersize=6, 
                                label=f'{protocol} Degradation')
            
            ax3.set_xlabel('Number of Interferers')
            ax3.set_ylabel('Throughput Degradation (%)')
            ax3.set_title('Performance Degradation with Interference')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Interference Impact at Different Traffic Rates
            ax4 = axes[1, 1]
            traffic_rates = sorted(self.combined_df['traffic_rate_num'].unique())[:3]
            colors = plt.cm.Set1(np.linspace(0, 1, len(traffic_rates)))
            
            for i, rate in enumerate(traffic_rates):
                rate_data = self.combined_df[self.combined_df['traffic_rate_num'] == rate]
                rate_intf = (
                    rate_data.groupby(['interferers', 'Protocol'])['throughput']
                    .mean()
                    .reset_index()
                )
                
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    protocol_data = rate_intf[rate_intf['Protocol'] == protocol]
                    if len(protocol_data) > 0:
                        protocol_data = protocol_data.sort_values('interferers')
                        linestyle = '-' if protocol == self.protocol1_name else '--'
                        ax4.plot(protocol_data['interferers'], 
                                protocol_data['throughput'],
                                color=colors[i], linestyle=linestyle,
                                marker='o', linewidth=2, markersize=4, 
                                label=f'{protocol} @ {rate}Mbps')
            
            ax4.set_xlabel('Number of Interferers')
            ax4.set_ylabel('Throughput (Mbps)')
            ax4.set_title('Interference Impact at Different Traffic Rates')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'interference_resilience_comparison.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('interference_resilience_comparison.png')
            logger.info("Created interference resilience comparison")
            
        except Exception as e:
            logger.error(f"Error creating interference resilience comparison: {e}")
    
    def _create_snr_analysis_plots(self) -> None:
        """NEW: Create SNR analysis plots (realistic SNR conversion)."""
        try:
            if 'avg_snr' not in self.combined_df.columns:
                logger.info("SNR data not available")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # SNR Distribution
            ax1 = axes[0, 0]
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = self.combined_df[self.combined_df['Protocol'] == protocol]
                
                if len(protocol_data) > 0 and 'avg_snr' in protocol_data.columns:
                    ax1.hist(protocol_data['avg_snr'], bins=30, alpha=0.6, 
                            label=protocol, edgecolor='black')
            
            ax1.set_xlabel('Average SNR (dB)')
            ax1.set_ylabel('Number of Scenarios')
            ax1.set_title('SNR Distribution (Realistic Conversion)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # SNR vs Distance
            ax2 = axes[0, 1]
            snr_dist = (
                self.combined_df.groupby(['distance', 'Protocol'])['avg_snr']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = snr_dist[snr_dist['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('distance')
                    ax2.plot(protocol_data['distance'], 
                            protocol_data['avg_snr'],
                            marker='o', linewidth=2, markersize=6, label=protocol)
            
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Average SNR (dB)')
            ax2.set_title('SNR vs Distance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # SNR vs Interferers
            ax3 = axes[1, 0]
            snr_intf = (
                self.combined_df.groupby(['interferers', 'Protocol'])['avg_snr']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = snr_intf[snr_intf['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('interferers')
                    ax3.plot(protocol_data['interferers'], 
                            protocol_data['avg_snr'],
                            marker='s', linewidth=2, markersize=6, label=protocol)
            
            ax3.set_xlabel('Number of Interferers')
            ax3.set_ylabel('Average SNR (dB)')
            ax3.set_title('SNR vs Interference')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Throughput vs SNR (correlation)
            ax4 = axes[1, 1]
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = self.combined_df[self.combined_df['Protocol'] == protocol]
                
                if len(protocol_data) > 0:
                    ax4.scatter(protocol_data['avg_snr'], 
                               protocol_data['throughput'],
                               alpha=0.5, s=30, label=protocol)
            
            ax4.set_xlabel('Average SNR (dB)')
            ax4.set_ylabel('Throughput (Mbps)')
            ax4.set_title('Throughput vs SNR Correlation')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'snr_analysis_plots.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('snr_analysis_plots.png')
            logger.info("Created SNR analysis plots")
            
        except Exception as e:
            logger.error(f"Error creating SNR analysis plots: {e}")
    
    def calculate_statistical_comparison(self) -> Dict:
        """Calculate comprehensive statistical comparisons."""
        if self.combined_df is None:
            logger.error("No combined data available.")
            return {}
        
        logger.info("Calculating statistical comparisons...")
        
        results = {}
        metrics = ['throughput', 'packet_loss', 'avg_delay']
        
        # Overall comparison
        overall_stats = {}
        for metric in metrics:
            if metric in self.combined_df.columns:
                stats = (
                    self.combined_df.groupby('Protocol')[metric]
                    .agg(['mean', 'std', 'median', 'min', 'max', 'count'])
                    .round(4)
                )
                overall_stats[metric] = stats
        
        results['overall'] = overall_stats
        
        self.comparison_results = results
        return results
    
    def generate_summary(self) -> str:
        """Generate COMPREHENSIVE summary with detailed breakdown."""
        if self.combined_df is None or len(self.combined_df) == 0:
            return "No comparison data available."
        
        summary = []
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        p1, p2 = self.protocol1_name, self.protocol2_name
        
        # Get unique scenarios
        unique_scenarios = self.combined_df['scenario_group'].nunique()
        n1 = len(self.df1) if self.df1 is not None else 0
        n2 = len(self.df2) if self.df2 is not None else 0
        
        summary.append(f"\n{'='*100}")
        summary.append(f"COMPREHENSIVE PROTOCOL COMPARISON ANALYSIS")
        summary.append(f"{p1} vs {p2}")
        summary.append(f"{'='*100}")
        summary.append(f"Analysis Date: {now}")
        summary.append(f"Total Scenarios Analyzed: {unique_scenarios}")
        summary.append(f"Total Data Points: {len(self.combined_df)} ({n1} vs {n2})\n")
        
        # Safe stat helpers
        def safe_mean(arr): 
            arr = arr.dropna() if hasattr(arr, 'dropna') else arr
            return float(np.nan_to_num(np.mean(arr), nan=0.0))
        
        def safe_min(arr): 
            arr = arr.dropna() if hasattr(arr, 'dropna') else arr
            return float(np.nan_to_num(np.min(arr), nan=0.0))
        
        def safe_max(arr): 
            arr = arr.dropna() if hasattr(arr, 'dropna') else arr
            return float(np.nan_to_num(np.max(arr), nan=0.0))
        
        def safe_std(arr): 
            arr = arr.dropna() if hasattr(arr, 'dropna') else arr
            return float(np.nan_to_num(np.std(arr), nan=0.0))
        
        p1_data = self.combined_df[self.combined_df['Protocol'] == p1]
        p2_data = self.combined_df[self.combined_df['Protocol'] == p2]
        
        # Overall metrics
        p1_th = safe_mean(p1_data['throughput']) if 'throughput' in p1_data else 0
        p2_th = safe_mean(p2_data['throughput']) if 'throughput' in p2_data else 0
        p1_loss = safe_mean(p1_data['packet_loss']) if 'packet_loss' in p1_data else 0
        p2_loss = safe_mean(p2_data['packet_loss']) if 'packet_loss' in p2_data else 0
        p1_delay = safe_mean(p1_data['avg_delay']) if 'avg_delay' in p1_data else 0
        p2_delay = safe_mean(p2_data['avg_delay']) if 'avg_delay' in p2_data else 0
        
        # --- EXECUTIVE SUMMARY ---
        summary.append("EXECUTIVE SUMMARY")
        summary.append("-" * 60)
        
        # Category winners
        throughput_winner = p1 if p1_th > p2_th else p2
        reliability_winner = p1 if p1_loss < p2_loss else p2
        responsiveness_winner = p1 if p1_delay < p2_delay else p2
        category_wins_p1 = sum([throughput_winner == p1, reliability_winner == p1, responsiveness_winner == p1])
        category_wins_p2 = 3 - category_wins_p1
        
        # Improvements
        min_th = min(p1_th, p2_th)
        throughput_improvement = abs((p1_th - p2_th) / max(min_th, 0.001)) * 100
        max_loss = max(p1_loss, p2_loss)
        reliability_improvement = abs((p1_loss - p2_loss) / max(max_loss, 0.001)) * 100 if max_loss > 0 else 0
        max_delay = max(p1_delay, p2_delay)
        responsiveness_improvement = abs((p1_delay - p2_delay) / max(max_delay, 0.001)) * 100 if max_delay > 0 else 0
        
        # Composite scores
        p1_score = (p1_th * 0.5) + ((100 - p1_loss) * 0.3) + ((1000 - min(p1_delay, 1000)) * 0.2)
        p2_score = (p2_th * 0.5) + ((100 - p2_loss) * 0.3) + ((1000 - min(p2_delay, 1000)) * 0.2)
        overall_winner = p1 if p1_score > p2_score else p2
        
        summary.append(f"OVERALL WINNER: {overall_winner}")
        summary.append(f"Category Wins: {p1} ({category_wins_p1}) vs {p2} ({category_wins_p2})")
        summary.append(f"Composite Score: {p1} ({p1_score:.2f}) vs {p2} ({p2_score:.2f})\n")
        
        # --- DETAILED PERFORMANCE ANALYSIS ---
        summary.append("DETAILED PERFORMANCE ANALYSIS")
        summary.append("-" * 60)
        
        # Throughput
        summary.append("THROUGHPUT PERFORMANCE:")
        summary.append(f"  {p1}:")
        summary.append(f"    - Average: {p1_th:.3f} Mbps")
        summary.append(f"    - Range: {safe_min(p1_data['throughput']):.3f} - {safe_max(p1_data['throughput']):.3f} Mbps")
        summary.append(f"    - Std Dev: {safe_std(p1_data['throughput']):.3f} Mbps")
        summary.append(f"  {p2}:")
        summary.append(f"    - Average: {p2_th:.3f} Mbps")
        summary.append(f"    - Range: {safe_min(p2_data['throughput']):.3f} - {safe_max(p2_data['throughput']):.3f} Mbps")
        summary.append(f"    - Std Dev: {safe_std(p2_data['throughput']):.3f} Mbps")
        summary.append(f"  Winner: {throughput_winner} by {throughput_improvement:.1f}%\n")
        
        # Reliability
        summary.append("RELIABILITY PERFORMANCE (Packet Loss):")
        summary.append(f"  {p1}:")
        summary.append(f"    - Average Loss: {p1_loss:.3f}%")
        summary.append(f"    - Loss Range: {safe_min(p1_data['packet_loss']):.3f}% - {safe_max(p1_data['packet_loss']):.3f}%")
        summary.append(f"    - Success Rate: {100-p1_loss:.1f}%")
        summary.append(f"  {p2}:")
        summary.append(f"    - Average Loss: {p2_loss:.3f}%")
        summary.append(f"    - Loss Range: {safe_min(p2_data['packet_loss']):.3f}% - {safe_max(p2_data['packet_loss']):.3f}%")
        summary.append(f"    - Success Rate: {100-p2_loss:.1f}%")
        summary.append(f"  Winner: {reliability_winner} by {reliability_improvement:.1f}%\n")
        
        # Responsiveness
        summary.append("RESPONSIVENESS PERFORMANCE (Delay):")
        summary.append(f"  {p1}:")
        summary.append(f"    - Average Delay: {p1_delay:.3f} ms")
        summary.append(f"    - Delay Range: {safe_min(p1_data['avg_delay']):.3f} - {safe_max(p1_data['avg_delay']):.3f} ms")
        summary.append(f"  {p2}:")
        summary.append(f"    - Average Delay: {p2_delay:.3f} ms")
        summary.append(f"    - Delay Range: {safe_min(p2_data['avg_delay']):.3f} - {safe_max(p2_data['avg_delay']):.3f} ms")
        summary.append(f"  Winner: {responsiveness_winner} by {responsiveness_improvement:.1f}%\n")
        
        # --- DISTANCE BREAKDOWN ---
        summary.append("PERFORMANCE BY DISTANCE")
        summary.append("-" * 80)
        distances = sorted(self.combined_df['distance'].unique())
        dist_wins = {p1: 0, p2: 0}
        
        for dist in distances:
            dist_data = self.combined_df[self.combined_df['distance'] == dist]
            d1 = dist_data[dist_data['Protocol'] == p1]
            d2 = dist_data[dist_data['Protocol'] == p2]
            
            if len(d1) == 0 or len(d2) == 0:
                continue
            
            th1, th2 = safe_mean(d1['throughput']), safe_mean(d2['throughput'])
            loss1, loss2 = safe_mean(d1['packet_loss']), safe_mean(d2['packet_loss'])
            delay1, delay2 = safe_mean(d1['avg_delay']), safe_mean(d2['avg_delay'])
            
            eff1 = th1 * (100 - loss1) / 100
            eff2 = th2 * (100 - loss2) / 100
            winner = p1 if eff1 > eff2 else p2
            dist_wins[winner] += 1
            
            summary.append(f"\nAt Distance {dist}m:")
            summary.append(f"  {p1}: {th1:.2f} Mbps, {loss1:.2f}% loss, {delay1:.2f} ms delay")
            summary.append(f"  {p2}: {th2:.2f} Mbps, {loss2:.2f}% loss, {delay2:.2f} ms delay")
            summary.append(f"  Winner: {winner} (effective throughput: {max(eff1, eff2):.2f} Mbps)")
            
            # Degradation from minimum distance
            if dist != min(distances):
                min_dist = min(distances)
                base1 = safe_mean(self.combined_df[(self.combined_df['distance']==min_dist) & 
                                                (self.combined_df['Protocol']==p1)]['throughput'])
                base2 = safe_mean(self.combined_df[(self.combined_df['distance']==min_dist) & 
                                                (self.combined_df['Protocol']==p2)]['throughput'])
                deg1 = ((base1 - th1) / max(base1, 0.001)) * 100 if base1 > 0 else 0
                deg2 = ((base2 - th2) / max(base2, 0.001)) * 100 if base2 > 0 else 0
                summary.append(f"  Performance Degradation: {p1} ({deg1:.1f}%), {p2} ({deg2:.1f}%)")
        
        summary.append(f"\nDistance Performance Summary:")
        summary.append(f"  {p1} wins at {dist_wins[p1]} distances")
        summary.append(f"  {p2} wins at {dist_wins[p2]} distances")
        
        # --- INTERFERENCE BREAKDOWN ---
        summary.append("\n\nPERFORMANCE BY INTERFERENCE LEVEL")
        summary.append("-" * 80)
        interferers = sorted(self.combined_df['interferers'].unique())
        intf_wins = {p1: 0, p2: 0}
        
        for intf in interferers:
            intf_data = self.combined_df[self.combined_df['interferers'] == intf]
            i1 = intf_data[intf_data['Protocol'] == p1]
            i2 = intf_data[intf_data['Protocol'] == p2]
            
            if len(i1) == 0 or len(i2) == 0:
                continue
            
            th1, th2 = safe_mean(i1['throughput']), safe_mean(i2['throughput'])
            loss1, loss2 = safe_mean(i1['packet_loss']), safe_mean(i2['packet_loss'])
            delay1, delay2 = safe_mean(i1['avg_delay']), safe_mean(i2['avg_delay'])
            
            eff1 = th1 * (100 - loss1) / 100
            eff2 = th2 * (100 - loss2) / 100
            winner = p1 if eff1 > eff2 else p2
            intf_wins[winner] += 1
            
            summary.append(f"\nWith {intf} Interferers:")
            summary.append(f"  {p1}: {th1:.2f} Mbps, {loss1:.2f}% loss, {delay1:.2f} ms delay")
            summary.append(f"  {p2}: {th2:.2f} Mbps, {loss2:.2f}% loss, {delay2:.2f} ms delay")
            summary.append(f"  Winner: {winner} (effective throughput: {max(eff1, eff2):.2f} Mbps)")
            
            # Impact from no interference
            if intf > 0:
                no_intf1 = safe_mean(self.combined_df[(self.combined_df['interferers']==0) & 
                                                    (self.combined_df['Protocol']==p1)]['throughput'])
                no_intf2 = safe_mean(self.combined_df[(self.combined_df['interferers']==0) & 
                                                    (self.combined_df['Protocol']==p2)]['throughput'])
                impact1 = ((no_intf1 - th1) / max(no_intf1, 0.001)) * 100 if no_intf1 > 0 else 0
                impact2 = ((no_intf2 - th2) / max(no_intf2, 0.001)) * 100 if no_intf2 > 0 else 0
                summary.append(f"  Interference Impact: {p1} ({impact1:.1f}%), {p2} ({impact2:.1f}%)")
        
        summary.append(f"\nInterference Resilience Summary:")
        summary.append(f"  {p1} wins at {intf_wins[p1]} interference levels")
        summary.append(f"  {p2} wins at {intf_wins[p2]} interference levels")
        
        # --- TRAFFIC RATE BREAKDOWN ---
        summary.append("\n\nPERFORMANCE BY TRAFFIC RATE")
        summary.append("-" * 80)
        traffic_rates = sorted(self.combined_df['traffic_rate_num'].unique())
        rate_wins = {p1: 0, p2: 0}
        
        for rate in traffic_rates:
            rate_data = self.combined_df[self.combined_df['traffic_rate_num'] == rate]
            r1 = rate_data[rate_data['Protocol'] == p1]
            r2 = rate_data[rate_data['Protocol'] == p2]
            
            if len(r1) == 0 or len(r2) == 0:
                continue
            
            th1, th2 = safe_mean(r1['throughput']), safe_mean(r2['throughput'])
            loss1, loss2 = safe_mean(r1['packet_loss']), safe_mean(r2['packet_loss'])
            
            eff1 = (th1 / max(rate, 0.001)) * 100 if th1 > 0 else 0
            eff2 = (th2 / max(rate, 0.001)) * 100 if th2 > 0 else 0
            winner = p1 if th1 > th2 else p2
            rate_wins[winner] += 1
            
            summary.append(f"\nWith Traffic Rate {rate} Mbps:")
            summary.append(f"  {p1}: {th1:.2f} Mbps, {loss1:.2f}% loss, Efficiency: {eff1:.1f}%")
            summary.append(f"  {p2}: {th2:.2f} Mbps, {loss2:.2f}% loss, Efficiency: {eff2:.1f}%")
            summary.append(f"  Winner: {winner}")
        
        summary.append(f"\nTraffic Rate Performance Summary:")
        summary.append(f"  {p1} wins at {rate_wins[p1]} traffic rates")
        summary.append(f"  {p2} wins at {rate_wins[p2]} traffic rates")
        
        # --- SPEED/MOBILITY BREAKDOWN ---
        if 'speed' in self.combined_df.columns:
            summary.append("\n\nPERFORMANCE BY MOBILITY (SPEED)")
            summary.append("-" * 80)
            speeds = sorted(self.combined_df['speed'].unique())
            speed_wins = {p1: 0, p2: 0}
            
            for speed in speeds:
                speed_data = self.combined_df[self.combined_df['speed'] == speed]
                s1 = speed_data[speed_data['Protocol'] == p1]
                s2 = speed_data[speed_data['Protocol'] == p2]
                
                if len(s1) == 0 or len(s2) == 0:
                    continue
                
                th1, th2 = safe_mean(s1['throughput']), safe_mean(s2['throughput'])
                loss1, loss2 = safe_mean(s1['packet_loss']), safe_mean(s2['packet_loss'])
                
                winner = p1 if th1 > th2 else p2
                speed_wins[winner] += 1
                
                summary.append(f"\nAt Speed {speed} m/s:")
                summary.append(f"  {p1}: {th1:.2f} Mbps, {loss1:.2f}% loss")
                summary.append(f"  {p2}: {th2:.2f} Mbps, {loss2:.2f}% loss")
                summary.append(f"  Winner: {winner}")
                
                # Mobility impact
                if speed > 0:
                    stationary1 = safe_mean(self.combined_df[(self.combined_df['speed']==0) & 
                                                            (self.combined_df['Protocol']==p1)]['throughput'])
                    stationary2 = safe_mean(self.combined_df[(self.combined_df['speed']==0) & 
                                                            (self.combined_df['Protocol']==p2)]['throughput'])
                    mob_impact1 = ((stationary1 - th1) / max(stationary1, 0.001)) * 100 if stationary1 > 0 else 0
                    mob_impact2 = ((stationary2 - th2) / max(stationary2, 0.001)) * 100 if stationary2 > 0 else 0
                    summary.append(f"  Mobility Impact: {p1} ({mob_impact1:.1f}%), {p2} ({mob_impact2:.1f}%)")
            
            summary.append(f"\nMobility Performance Summary:")
            summary.append(f"  {p1} wins at {speed_wins[p1]} speed levels")
            summary.append(f"  {p2} wins at {speed_wins[p2]} speed levels")
        
        # --- ML-SPECIFIC METRICS (if available) ---
        if 'ml_inferences' in p2_data.columns:
            summary.append("\n\nML SYSTEM PERFORMANCE (Smart-RF only)")
            summary.append("-" * 80)
            
            ml_data = p2_data[p2_data['ml_inferences'] > 0]
            if len(ml_data) > 0:
                avg_inferences = safe_mean(ml_data['ml_inferences'])
                avg_failures = safe_mean(ml_data['ml_failures'])
                success_rate = ((avg_inferences - avg_failures) / max(avg_inferences, 1)) * 100
                avg_confidence = safe_mean(ml_data['ml_confidence']) if 'ml_confidence' in ml_data else 0
                avg_rate_changes = safe_mean(ml_data['rate_changes']) if 'rate_changes' in ml_data else 0
                
                summary.append(f"  Average ML Inferences per Test: {avg_inferences:.1f}")
                summary.append(f"  Average ML Failures per Test: {avg_failures:.1f}")
                summary.append(f"  ML Success Rate: {success_rate:.1f}%")
                summary.append(f"  Average ML Confidence: {avg_confidence:.3f}")
                summary.append(f"  Average Rate Changes per Test: {avg_rate_changes:.1f}")
                
                summary.append(f"\n  ML System Assessment:")
                if success_rate >= 85:
                    summary.append(f"    ✅ EXCELLENT - ML success rate above 85% target")
                elif success_rate >= 70:
                    summary.append(f"    ⚠️ GOOD - ML success rate acceptable but below target")
                else:
                    summary.append(f"    ❌ POOR - ML success rate below 70%")
                
                if avg_confidence >= 0.40:
                    summary.append(f"    ✅ HIGH CONFIDENCE - Average confidence above 0.40")
                elif avg_confidence >= 0.25:
                    summary.append(f"    ⚠️ MODERATE CONFIDENCE - Average confidence acceptable")
                else:
                    summary.append(f"    ❌ LOW CONFIDENCE - Average confidence below 0.25")
                
                if avg_rate_changes < 100:
                    summary.append(f"    ✅ STABLE - Rate changes under 100 per test")
                elif avg_rate_changes < 150:
                    summary.append(f"    ⚠️ MODERATE - Rate changes acceptable but could be more stable")
                else:
                    summary.append(f"    ❌ UNSTABLE - Excessive rate changes (>150 per test)")
        
        # --- RECOMMENDATIONS ---
        summary.append("\n\nPERFORMANCE RECOMMENDATIONS")
        summary.append("-" * 60)
        summary.append("OPTIMAL USE CASES:\n")
        
        # Best scenarios for each protocol
        p1_best, p2_best = [], []
        for scenario in self.combined_df['scenario_group'].unique():
            sdata = self.combined_df[self.combined_df['scenario_group'] == scenario]
            d1 = sdata[sdata['Protocol'] == p1]
            d2 = sdata[sdata['Protocol'] == p2]
            
            if len(d1)==0 or len(d2)==0:
                continue
            
            perf1 = safe_mean(d1['throughput']) * (100 - safe_mean(d1['packet_loss'])) / 100
            perf2 = safe_mean(d2['throughput']) * (100 - safe_mean(d2['packet_loss'])) / 100
            
            if perf1 > perf2:
                p1_best.append(scenario)
            else:
                p2_best.append(scenario)
        
        total_scenarios = len(p1_best) + len(p2_best)
        p1_pct = (len(p1_best)/max(total_scenarios,1)*100) if total_scenarios > 0 else 0
        p2_pct = (len(p2_best)/max(total_scenarios,1)*100) if total_scenarios > 0 else 0
        
        summary.append(f"{p1} excels in:")
        summary.append(f"  - {len(p1_best)} out of {total_scenarios} scenarios ({p1_pct:.1f}%)")
        if len(p1_best)>0:
            for s in p1_best[:5]:
                summary.append(f"    * {s}")
            if len(p1_best)>5:
                summary.append(f"    * ... and {len(p1_best)-5} more scenarios")
        
        summary.append(f"\n{p2} excels in:")
        summary.append(f"  - {len(p2_best)} out of {total_scenarios} scenarios ({p2_pct:.1f}%)")
        if len(p2_best)>0:
            for s in p2_best[:5]:
                summary.append(f"    * {s}")
            if len(p2_best)>5:
                summary.append(f"    * ... and {len(p2_best)-5} more scenarios")
        
        summary.append(f"\nFINAL RECOMMENDATIONS:")
        if overall_winner == p1:
            summary.append(f"  - {p1} is the overall better choice for general use")
            summary.append(f"  - Consider {p2} for specific scenarios where it excels")
        else:
            summary.append(f"  - {p2} is the overall better choice for general use")
            summary.append(f"  - Consider {p1} for specific scenarios where it excels")
        
        # --- GENERATED FILES ---
        summary.append(f"\nGENERATED ANALYSIS FILES")
        summary.append("-" * 40)
        summary.append("Visual Analysis Plots:")
        for plot in self.plots_generated:
            summary.append(f"  • {plot}")
        summary.append(f"\nDetailed Reports:")
        summary.append(f"  • Excel Report: {p1}_vs_{p2}_report.xlsx")
        summary.append(f"  • This Summary: comparison_summary.txt")
        summary.append(f"  • Analysis Log: enhanced_comparison_analysis.log")
        summary.append(f"\nAll files saved to: {self.results_dir}")
        summary.append("="*100)
        
        return "\n".join(summary)
    def create_excel_report(self) -> None:
        """Create simple Excel report."""
        try:
            logger.info("Creating Excel report...")
            excel_path = self.results_dir / f'{self.protocol1_name}_vs_{self.protocol2_name}_report.xlsx'
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                if self.combined_df is not None:
                    self.df1.to_excel(writer, sheet_name=f'{self.protocol1_name}_Data', index=False)
                    self.df2.to_excel(writer, sheet_name=f'{self.protocol2_name}_Data', index=False)
            
            logger.info(f"Excel report created: {excel_path}")
            
        except Exception as e:
            logger.error(f"Error creating Excel report: {e}")
    
    def run_complete_analysis(self) -> bool:
        """Run the complete enhanced comparison analysis."""
        logger.info("Starting complete enhanced protocol comparison analysis...")
        
        try:
            if not self.load_data():
                logger.error("Failed to load data")
                return False
            
            logger.info("Data loaded successfully")
            self.preprocess_data()
            logger.info("Data preprocessing completed")
            
            self.create_enhanced_comparison_plots()
            logger.info(f"Generated {len(self.plots_generated)} plots")
            
            self.calculate_statistical_comparison()
            logger.info("Statistical analysis completed")
            
            self.create_excel_report()
            logger.info("Excel report generated")
            
            summary = self.generate_summary()
            summary_path = self.results_dir / 'comparison_summary.txt'
            
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            logger.info(f"Summary saved to: {summary_path}")
            print(summary)
            
            logger.info("=" * 80)
            logger.info("ENHANCED COMPARISON ANALYSIS COMPLETED SUCCESSFULLY")
            logger.info(f"Results directory: {self.results_dir}")
            logger.info(f"Total plots generated: {len(self.plots_generated)}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def main():
    """Main function - DEFAULT: AARF vs SmartRF-v7.0 comparison."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Protocol Comparison Analyzer - NEW PIPELINE',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--protocol1', '-p1', type=str, default='aarf')
    parser.add_argument('--protocol2', '-p2', type=str, default='smartrf')
    
    args = parser.parse_args()
    
    protocol_mapping = {
        'aarf': 'aarf-benchmark-fixed-expanded.csv',
        'smartrf': 'smartrf-fixed-expanded-benchmark-results.csv',
        'smartrf-old': 'smartrf-newpipeline-benchmark-results.csv',
    }
    
    protocol1_csv = protocol_mapping.get(args.protocol1.lower())
    protocol2_csv = protocol_mapping.get(args.protocol2.lower())
    
    if not protocol1_csv or not protocol2_csv:
        print(f"ERROR: Unknown protocol. Available: {list(protocol_mapping.keys())}")
        return False
    
    print("=" * 80)
    print("ENHANCED PROTOCOL COMPARISON ANALYZER - NEW PIPELINE v2.0")
    print("=" * 80)
    print(f"DEFAULT COMPARISON: AARF-v2.0 vs SmartRF-v7.0 (Fully Fixed)")
    print(f"Protocol 1 CSV: {protocol1_csv}")
    print(f"Protocol 2 CSV: {protocol2_csv}")
    print()
    
    try:
        analyzer = EnhancedProtocolComparisonAnalyzer(
            protocol1_csv=protocol1_csv,
            protocol2_csv=protocol2_csv
        )
        
        print(f"Analyzing: {analyzer.protocol1_name} vs {analyzer.protocol2_name}")
        print(f"Results will be saved to: {analyzer.results_dir}")
        print()
        
        success = analyzer.run_complete_analysis()
        
        if success:
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Check the results directory: {analyzer.results_dir}")
            print("\nNEW PIPELINE FEATURES:")
            print("• 6 ML-specific comparison plots")
            print("• ML success rate analysis")
            print("• ML confidence vs SNR")
            print("• Rate change stability (validates FIX #4/#6)")
            print("• Mobility fix validation (validates FIX #1)")
            print("• Interference resilience comparison")
            print("• SNR analysis (realistic conversion)")
        else:
            print("\nANALYSIS FAILED! Check log file.")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()