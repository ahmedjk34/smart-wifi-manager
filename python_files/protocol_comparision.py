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
    """Enhanced side-by-side comparison analyzer with meaningful line graphs and plots."""
    
    def __init__(self, protocol1_csv: str, protocol2_csv: str, 
                 protocol1_name: str = None, protocol2_name: str = None):
        """
        Initialize the comparison analyzer.
        
        Args:
            protocol1_csv: Path to first protocol's CSV file
            protocol2_csv: Path to second protocol's CSV file
            protocol1_name: Display name for protocol 1 (auto-detected if None)
            protocol2_name: Display name for protocol 2 (auto-detected if None)
        """
        # Fix: Look in parent directory
        self.protocol1_csv = Path(__file__).parent.parent / protocol1_csv
        self.protocol2_csv = Path(__file__).parent.parent / protocol2_csv
        
        # Auto-detect protocol names from filenames if not provided
        self.protocol1_name = protocol1_name or self._extract_protocol_name(protocol1_csv)
        self.protocol2_name = protocol2_name or self._extract_protocol_name(protocol2_csv)
        
        # Create results directory
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / 'enhanced_comparison' / f'{self.protocol1_name}_vs_{self.protocol2_name}'
        self._create_results_directory()
        
        # Set up logging
        self._setup_logging()
        
        # Data storage
        self.df1: Optional[pd.DataFrame] = None
        self.df2: Optional[pd.DataFrame] = None
        self.combined_df: Optional[pd.DataFrame] = None
        self.plots_generated: List[str] = []
        self.comparison_results: Dict = {}
        
    def _extract_protocol_name(self, csv_path: str) -> str:
        """Extract protocol name from CSV filename."""
        filename = Path(csv_path).stem
        if 'aarf' in filename.lower():
            return 'AARF'
        elif 'smart' in filename.lower():
            if 'v1' in filename.lower():
                return 'SmartV1'
            elif 'v2' in filename.lower():
                return 'SmartV2'
            elif 'v3' in filename.lower():
                return 'SmartV3'
            elif 'v4' in filename.lower():
                return 'SmartV4'
            elif 'xgb' in filename.lower():
                return 'SmartXGD'
            elif 'rf' in filename.lower():
                if 'v3' in filename.lower():
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
        logger.info(f"Log file: {log_file}")
    
    def load_data(self) -> bool:
        """Load both CSV files with error handling."""
        try:
            # Load protocol 1 data
            if not self.protocol1_csv.exists():
                logger.error(f"Protocol 1 CSV not found: {self.protocol1_csv}")
                return False
            
            self.df1 = pd.read_csv(self.protocol1_csv)
            self.df1['Protocol'] = self.protocol1_name
            logger.info(f"Loaded {len(self.df1)} rows for {self.protocol1_name}")
            
            # Load protocol 2 data
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
        """Preprocess and combine data for analysis."""
        if self.df1 is None or self.df2 is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        # Standardize column names
        column_mappings = {
            'Distance': 'distance',
            'Speed': 'speed',
            'Interferers': 'interferers',
            'PacketSize': 'packet_size',
            'TrafficRate': 'traffic_rate',
            'Throughput(Mbps)': 'throughput',
            'PacketLoss(%)': 'packet_loss',
            'AvgDelay(ms)': 'avg_delay',
            'RxPackets': 'rx_packets',
            'TxPackets': 'tx_packets'
        }
        
        self.df1 = self.df1.rename(columns=column_mappings)
        self.df2 = self.df2.rename(columns=column_mappings)
        
        # Combine dataframes
        self.combined_df = pd.concat([self.df1, self.df2], ignore_index=True)
        
        # Convert traffic_rate to numeric for proper sorting
        if 'traffic_rate' in self.combined_df.columns:
            self.combined_df['traffic_rate_num'] = (
                self.combined_df['traffic_rate']
                .astype(str)
                .str.replace('Mbps', '', case=False)
                .str.replace(' ', '')
                .astype(float)
            )
        
        # Create scenario grouping for easier analysis
        self.combined_df['scenario_group'] = (
            self.combined_df['distance'].astype(str) + 'm_' +
            self.combined_df['speed'].astype(str) + 'mps_' +
            self.combined_df['interferers'].astype(str) + 'intf'
        )
        
        logger.info(f"Combined data: {len(self.combined_df)} total rows")
        logger.info(f"Scenarios: {self.combined_df['scenario_group'].nunique()} unique scenarios")
    
    def create_enhanced_comparison_plots(self) -> None:
        """Create enhanced comparison plots with meaningful line graphs."""
        if self.combined_df is None:
            logger.warning("No data available for plotting")
            return
        
        logger.info("Creating enhanced comparison plots...")
        
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # 1. Throughput vs Traffic Rate (Line Plot)
        self._create_throughput_vs_traffic_rate()
        
        # 2. Throughput vs Distance (Line Plot)
        self._create_throughput_vs_distance()
        
        # 3. Performance vs Speed (Line Plot)
        self._create_performance_vs_speed()
        
        # 4. Performance vs Interferers (Line Plot)
        self._create_performance_vs_interferers()
        
        # 5. Multi-metric comparison across scenarios
        self._create_multi_metric_scenario_comparison()
        
        # 6. Traffic Rate Impact Analysis
        self._create_traffic_rate_impact_analysis()
        
        # 7. Distance Performance Comparison
        self._create_distance_performance_comparison()
        
        # 8. Packet Loss vs Traffic Load
        self._create_packet_loss_vs_load()
        
        # 9. Delay vs Traffic Load
        self._create_delay_vs_load()
        
        # 10. Overall Performance Radar Chart
        self._create_performance_radar_chart()
        
        # 11. Scenario-by-Scenario Bar Comparison
        self._create_scenario_bar_comparison()
        
        # 12. Performance Efficiency Analysis
        self._create_efficiency_analysis()
    
    def _create_throughput_vs_traffic_rate(self) -> None:
        """Create throughput vs traffic rate line plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Get unique distances for separate plots
            distances = sorted(self.combined_df['distance'].unique())
            
            for idx, distance in enumerate(distances[:4]):  # Limit to 4 distances
                ax = axes[idx]
                
                # Filter data for this distance
                dist_data = self.combined_df[self.combined_df['distance'] == distance]
                
                # Plot for each protocol
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    protocol_data = dist_data[dist_data['Protocol'] == protocol]
                    
                    if len(protocol_data) > 0:
                        # Sort by traffic rate for proper line connection
                        protocol_data = protocol_data.sort_values('traffic_rate_num')
                        
                        ax.plot(protocol_data['traffic_rate_num'], 
                               protocol_data['throughput'], 
                               marker='o', linewidth=2, markersize=6,
                               label=protocol)
                
                ax.set_xlabel('Traffic Rate (Mbps)')
                ax.set_ylabel('Throughput (Mbps)')
                ax.set_title(f'Throughput vs Traffic Rate - Distance: {distance}m')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'throughput_vs_traffic_rate_lines.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('throughput_vs_traffic_rate_lines.png')
            logger.info("Created throughput vs traffic rate line plots")
            
        except Exception as e:
            logger.error(f"Error creating throughput vs traffic rate plot: {e}")
    
    def _create_throughput_vs_distance(self) -> None:
        """Create throughput vs distance line plots for different traffic rates."""
        try:
            # Get unique traffic rates
            traffic_rates = sorted(self.combined_df['traffic_rate_num'].unique())
            
            # Create subplots for different traffic rates
            n_rates = len(traffic_rates)
            n_cols = 3
            n_rows = (n_rates + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for idx, rate in enumerate(traffic_rates):
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                
                # Filter data for this traffic rate
                rate_data = self.combined_df[self.combined_df['traffic_rate_num'] == rate]
                
                # Plot for each protocol
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    protocol_data = rate_data[rate_data['Protocol'] == protocol]
                    
                    if len(protocol_data) > 0:
                        # Sort by distance for proper line connection
                        protocol_data = protocol_data.sort_values('distance')
                        
                        ax.plot(protocol_data['distance'], 
                               protocol_data['throughput'], 
                               marker='s', linewidth=2, markersize=6,
                               label=protocol)
                
                ax.set_xlabel('Distance (m)')
                ax.set_ylabel('Throughput (Mbps)')
                ax.set_title(f'Throughput vs Distance - Traffic Rate: {rate} Mbps')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(traffic_rates), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'throughput_vs_distance_lines.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('throughput_vs_distance_lines.png')
            logger.info("Created throughput vs distance line plots")
            
        except Exception as e:
            logger.error(f"Error creating throughput vs distance plot: {e}")
    
    def _create_performance_vs_speed(self) -> None:
        """Create performance vs speed line plots."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            metrics = ['throughput', 'packet_loss', 'avg_delay']
            metric_labels = ['Throughput (Mbps)', 'Packet Loss (%)', 'Average Delay (ms)']
            
            for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
                if metric not in self.combined_df.columns:
                    continue
                    
                ax = axes[idx]
                
                # Group by speed and protocol, calculate mean
                speed_performance = (
                    self.combined_df.groupby(['speed', 'Protocol'])[metric]
                    .mean()
                    .reset_index()
                )
                
                # Plot for each protocol
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    protocol_data = speed_performance[speed_performance['Protocol'] == protocol]
                    
                    if len(protocol_data) > 0:
                        protocol_data = protocol_data.sort_values('speed')
                        
                        ax.plot(protocol_data['speed'], 
                               protocol_data[metric], 
                               marker='D', linewidth=2, markersize=6,
                               label=protocol)
                
                ax.set_xlabel('Speed (m/s)')
                ax.set_ylabel(label)
                ax.set_title(f'{label} vs Speed')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'performance_vs_speed_lines.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('performance_vs_speed_lines.png')
            logger.info("Created performance vs speed line plots")
            
        except Exception as e:
            logger.error(f"Error creating performance vs speed plot: {e}")
    
    def _create_performance_vs_interferers(self) -> None:
        """Create performance vs interferers line plots."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            metrics = ['throughput', 'packet_loss', 'avg_delay']
            metric_labels = ['Throughput (Mbps)', 'Packet Loss (%)', 'Average Delay (ms)']
            
            for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
                if metric not in self.combined_df.columns:
                    continue
                    
                ax = axes[idx]
                
                # Group by interferers and protocol, calculate mean
                intf_performance = (
                    self.combined_df.groupby(['interferers', 'Protocol'])[metric]
                    .mean()
                    .reset_index()
                )
                
                # Plot for each protocol
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    protocol_data = intf_performance[intf_performance['Protocol'] == protocol]
                    
                    if len(protocol_data) > 0:
                        protocol_data = protocol_data.sort_values('interferers')
                        
                        ax.plot(protocol_data['interferers'], 
                               protocol_data[metric], 
                               marker='^', linewidth=2, markersize=6,
                               label=protocol)
                
                ax.set_xlabel('Number of Interferers')
                ax.set_ylabel(label)
                ax.set_title(f'{label} vs Interferers')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'performance_vs_interferers_lines.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('performance_vs_interferers_lines.png')
            logger.info("Created performance vs interferers line plots")
            
        except Exception as e:
            logger.error(f"Error creating performance vs interferers plot: {e}")
    
    def _create_multi_metric_scenario_comparison(self) -> None:
        """Create multi-metric comparison across all scenarios."""
        try:
            # Calculate scenario averages
            scenario_avg = (
                self.combined_df.groupby(['scenario_group', 'Protocol'])
                [['throughput', 'packet_loss', 'avg_delay']]
                .mean()
                .reset_index()
            )
            
            fig, axes = plt.subplots(3, 1, figsize=(16, 18))
            metrics = ['throughput', 'packet_loss', 'avg_delay']
            metric_labels = ['Throughput (Mbps)', 'Packet Loss (%)', 'Average Delay (ms)']
            
            for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
                if metric not in scenario_avg.columns:
                    continue
                    
                ax = axes[idx]
                
                # Pivot for easier plotting
                pivot_data = scenario_avg.pivot(index='scenario_group', 
                                              columns='Protocol', 
                                              values=metric).fillna(0)
                
                # Plot lines for each protocol
                scenarios = list(pivot_data.index)
                x_positions = range(len(scenarios))
                
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    if protocol in pivot_data.columns:
                        ax.plot(x_positions, pivot_data[protocol].values, 
                               marker='o', linewidth=2, markersize=4,
                               label=protocol)
                
                ax.set_xlabel('Scenarios')
                ax.set_ylabel(label)
                ax.set_title(f'{label} Across All Scenarios')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Set x-axis labels with rotation
                ax.set_xticks(x_positions[::max(1, len(scenarios)//10)])  # Show every nth label
                ax.set_xticklabels([scenarios[i] for i in x_positions[::max(1, len(scenarios)//10)]], 
                                 rotation=45, ha='right')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'multi_metric_scenario_comparison.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('multi_metric_scenario_comparison.png')
            logger.info("Created multi-metric scenario comparison")
            
        except Exception as e:
            logger.error(f"Error creating multi-metric scenario comparison: {e}")
    
    def _create_traffic_rate_impact_analysis(self) -> None:
        """Create traffic rate impact analysis with subplots for different conditions."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Analysis 1: Average throughput vs traffic rate
            ax1 = axes[0, 0]
            traffic_throughput = (
                self.combined_df.groupby(['traffic_rate_num', 'Protocol'])['throughput']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = traffic_throughput[traffic_throughput['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('traffic_rate_num')
                    ax1.plot(protocol_data['traffic_rate_num'], 
                            protocol_data['throughput'],
                            marker='o', linewidth=2, label=protocol)
            
            ax1.set_xlabel('Traffic Rate (Mbps)')
            ax1.set_ylabel('Average Throughput (Mbps)')
            ax1.set_title('Average Throughput vs Traffic Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            
            # Analysis 2: Efficiency (Throughput/Traffic Rate)
            ax2 = axes[0, 1]
            self.combined_df['efficiency'] = self.combined_df['throughput'] / self.combined_df['traffic_rate_num']
            efficiency_data = (
                self.combined_df.groupby(['traffic_rate_num', 'Protocol'])['efficiency']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = efficiency_data[efficiency_data['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('traffic_rate_num')
                    ax2.plot(protocol_data['traffic_rate_num'], 
                            protocol_data['efficiency'],
                            marker='s', linewidth=2, label=protocol)
            
            ax2.set_xlabel('Traffic Rate (Mbps)')
            ax2.set_ylabel('Efficiency (Throughput/Rate)')
            ax2.set_title('Protocol Efficiency vs Traffic Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            
            # Analysis 3: Packet loss vs traffic rate
            ax3 = axes[1, 0]
            loss_data = (
                self.combined_df.groupby(['traffic_rate_num', 'Protocol'])['packet_loss']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = loss_data[loss_data['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('traffic_rate_num')
                    ax3.plot(protocol_data['traffic_rate_num'], 
                            protocol_data['packet_loss'],
                            marker='^', linewidth=2, label=protocol)
            
            ax3.set_xlabel('Traffic Rate (Mbps)')
            ax3.set_ylabel('Average Packet Loss (%)')
            ax3.set_title('Packet Loss vs Traffic Rate')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log')
            
            # Analysis 4: Delay vs traffic rate
            ax4 = axes[1, 1]
            delay_data = (
                self.combined_df.groupby(['traffic_rate_num', 'Protocol'])['avg_delay']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = delay_data[delay_data['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('traffic_rate_num')
                    ax4.plot(protocol_data['traffic_rate_num'], 
                            protocol_data['avg_delay'],
                            marker='D', linewidth=2, label=protocol)
            
            ax4.set_xlabel('Traffic Rate (Mbps)')
            ax4.set_ylabel('Average Delay (ms)')
            ax4.set_title('Average Delay vs Traffic Rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xscale('log')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'traffic_rate_impact_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('traffic_rate_impact_analysis.png')
            logger.info("Created traffic rate impact analysis")
            
        except Exception as e:
            logger.error(f"Error creating traffic rate impact analysis: {e}")
    
    def _create_distance_performance_comparison(self) -> None:
        """Create distance performance comparison with multiple metrics."""
        try:
            distances = sorted(self.combined_df['distance'].unique())
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Throughput vs Distance
            ax1 = axes[0, 0]
            dist_throughput = (
                self.combined_df.groupby(['distance', 'Protocol'])['throughput']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = dist_throughput[dist_throughput['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('distance')
                    ax1.plot(protocol_data['distance'], 
                            protocol_data['throughput'],
                            marker='o', linewidth=2, markersize=6, label=protocol)
            
            ax1.set_xlabel('Distance (m)')
            ax1.set_ylabel('Average Throughput (Mbps)')
            ax1.set_title('Throughput vs Distance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Packet Loss vs Distance
            ax2 = axes[0, 1]
            dist_loss = (
                self.combined_df.groupby(['distance', 'Protocol'])['packet_loss']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = dist_loss[dist_loss['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('distance')
                    ax2.plot(protocol_data['distance'], 
                            protocol_data['packet_loss'],
                            marker='s', linewidth=2, markersize=6, label=protocol)
            
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Average Packet Loss (%)')
            ax2.set_title('Packet Loss vs Distance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Delay vs Distance
            ax3 = axes[1, 0]
            dist_delay = (
                self.combined_df.groupby(['distance', 'Protocol'])['avg_delay']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = dist_delay[dist_delay['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('distance')
                    ax3.plot(protocol_data['distance'], 
                            protocol_data['avg_delay'],
                            marker='^', linewidth=2, markersize=6, label=protocol)
            
            ax3.set_xlabel('Distance (m)')
            ax3.set_ylabel('Average Delay (ms)')
            ax3.set_title('Delay vs Distance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Performance degradation rate
            ax4 = axes[1, 1]
            # Calculate throughput degradation as percentage from minimum distance
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = dist_throughput[dist_throughput['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('distance')
                    baseline = protocol_data['throughput'].iloc[0]  # First distance performance
                    degradation = ((baseline - protocol_data['throughput']) / baseline) * 100
                    ax4.plot(protocol_data['distance'], 
                            degradation,
                            marker='D', linewidth=2, markersize=6, label=f'{protocol} Degradation')
            
            ax4.set_xlabel('Distance (m)')
            ax4.set_ylabel('Throughput Degradation (%)')
            ax4.set_title('Performance Degradation with Distance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'distance_performance_comparison.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('distance_performance_comparison.png')
            logger.info("Created distance performance comparison")
            
        except Exception as e:
            logger.error(f"Error creating distance performance comparison: {e}")
    
    def _create_packet_loss_vs_load(self) -> None:
        """Create packet loss vs network load analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Packet Loss vs Traffic Rate
            ax1 = axes[0, 0]
            loss_vs_rate = (
                self.combined_df.groupby(['traffic_rate_num', 'Protocol'])['packet_loss']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = loss_vs_rate[loss_vs_rate['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('traffic_rate_num')
                    ax1.plot(protocol_data['traffic_rate_num'], 
                            protocol_data['packet_loss'],
                            marker='o', linewidth=2, markersize=6, label=protocol)
            
            ax1.set_xlabel('Traffic Rate (Mbps)')
            ax1.set_ylabel('Packet Loss (%)')
            ax1.set_title('Packet Loss vs Traffic Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            
            # Packet Loss vs Distance (for different traffic rates)
            ax2 = axes[0, 1]
            traffic_rates = sorted(self.combined_df['traffic_rate_num'].unique())
            colors = plt.cm.Set1(np.linspace(0, 1, len(traffic_rates)))
            
            for i, rate in enumerate(traffic_rates[:3]):  # Show top 3 rates
                rate_data = self.combined_df[self.combined_df['traffic_rate_num'] == rate]
                loss_vs_dist = (
                    rate_data.groupby(['distance', 'Protocol'])['packet_loss']
                    .mean()
                    .reset_index()
                )
                
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    protocol_data = loss_vs_dist[loss_vs_dist['Protocol'] == protocol]
                    if len(protocol_data) > 0:
                        protocol_data = protocol_data.sort_values('distance')
                        linestyle = '-' if protocol == self.protocol1_name else '--'
                        ax2.plot(protocol_data['distance'], 
                                protocol_data['packet_loss'],
                                color=colors[i], linestyle=linestyle,
                                marker='s', linewidth=2, markersize=4, 
                                label=f'{protocol} @ {rate}Mbps')
            
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Packet Loss (%)')
            ax2.set_title('Packet Loss vs Distance (Different Traffic Rates)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Packet Loss vs Interferers
            ax3 = axes[1, 0]
            loss_vs_intf = (
                self.combined_df.groupby(['interferers', 'Protocol'])['packet_loss']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = loss_vs_intf[loss_vs_intf['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('interferers')
                    ax3.plot(protocol_data['interferers'], 
                            protocol_data['packet_loss'],
                            marker='^', linewidth=2, markersize=6, label=protocol)
            
            ax3.set_xlabel('Number of Interferers')
            ax3.set_ylabel('Packet Loss (%)')
            ax3.set_title('Packet Loss vs Interferers')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Combined load effect (Traffic Rate + Interferers)
            ax4 = axes[1, 1]
            # Create combined load metric
            self.combined_df['combined_load'] = (
                self.combined_df['traffic_rate_num'] * (1 + self.combined_df['interferers'] * 0.2)
            )
            
            load_vs_loss = (
                self.combined_df.groupby(['combined_load', 'Protocol'])['packet_loss']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = load_vs_loss[load_vs_loss['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('combined_load')
                    ax4.plot(protocol_data['combined_load'], 
                            protocol_data['packet_loss'],
                            marker='D', linewidth=2, markersize=6, label=protocol)
            
            ax4.set_xlabel('Combined Load Factor')
            ax4.set_ylabel('Packet Loss (%)')
            ax4.set_title('Packet Loss vs Combined Network Load')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xscale('log')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'packet_loss_vs_load.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('packet_loss_vs_load.png')
            logger.info("Created packet loss vs load analysis")
            
        except Exception as e:
            logger.error(f"Error creating packet loss vs load plot: {e}")
    
    def _create_delay_vs_load(self) -> None:
        """Create delay vs network load analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Delay vs Traffic Rate
            ax1 = axes[0, 0]
            delay_vs_rate = (
                self.combined_df.groupby(['traffic_rate_num', 'Protocol'])['avg_delay']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = delay_vs_rate[delay_vs_rate['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('traffic_rate_num')
                    ax1.plot(protocol_data['traffic_rate_num'], 
                            protocol_data['avg_delay'],
                            marker='o', linewidth=2, markersize=6, label=protocol)
            
            ax1.set_xlabel('Traffic Rate (Mbps)')
            ax1.set_ylabel('Average Delay (ms)')
            ax1.set_title('Delay vs Traffic Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            
            # Delay vs Distance
            ax2 = axes[0, 1]
            delay_vs_dist = (
                self.combined_df.groupby(['distance', 'Protocol'])['avg_delay']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = delay_vs_dist[delay_vs_dist['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('distance')
                    ax2.plot(protocol_data['distance'], 
                            protocol_data['avg_delay'],
                            marker='s', linewidth=2, markersize=6, label=protocol)
            
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Average Delay (ms)')
            ax2.set_title('Delay vs Distance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Delay vs Speed
            ax3 = axes[1, 0]
            delay_vs_speed = (
                self.combined_df.groupby(['speed', 'Protocol'])['avg_delay']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = delay_vs_speed[delay_vs_speed['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('speed')
                    ax3.plot(protocol_data['speed'], 
                            protocol_data['avg_delay'],
                            marker='^', linewidth=2, markersize=6, label=protocol)
            
            ax3.set_xlabel('Speed (m/s)')
            ax3.set_ylabel('Average Delay (ms)')
            ax3.set_title('Delay vs Speed')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Delay distribution comparison
            ax4 = axes[1, 1]
            # Create delay buckets for better visualization
            delay_ranges = [(0, 10), (10, 50), (50, 100), (100, float('inf'))]
            range_labels = ['0-10ms', '10-50ms', '50-100ms', '>100ms']
            
            delay_distribution = {}
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = self.combined_df[self.combined_df['Protocol'] == protocol]
                distribution = []
                
                for min_delay, max_delay in delay_ranges:
                    if max_delay == float('inf'):
                        count = len(protocol_data[protocol_data['avg_delay'] >= min_delay])
                    else:
                        count = len(protocol_data[
                            (protocol_data['avg_delay'] >= min_delay) & 
                            (protocol_data['avg_delay'] < max_delay)
                        ])
                    distribution.append(count / len(protocol_data) * 100)
                
                delay_distribution[protocol] = distribution
            
            x = np.arange(len(range_labels))
            width = 0.35
            
            ax4.bar(x - width/2, delay_distribution[self.protocol1_name], 
                   width, label=self.protocol1_name, alpha=0.8)
            ax4.bar(x + width/2, delay_distribution[self.protocol2_name], 
                   width, label=self.protocol2_name, alpha=0.8)
            
            ax4.set_xlabel('Delay Range')
            ax4.set_ylabel('Percentage of Scenarios (%)')
            ax4.set_title('Delay Distribution Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(range_labels)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'delay_vs_load.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('delay_vs_load.png')
            logger.info("Created delay vs load analysis")
            
        except Exception as e:
            logger.error(f"Error creating delay vs load plot: {e}")
    
    def _create_performance_radar_chart(self) -> None:
        """Create radar chart comparing overall performance."""
        try:
            # Calculate average metrics for each protocol
            metrics = []
            protocol1_values = []
            protocol2_values = []
            
            # Throughput (normalize to 0-10 scale)
            if 'throughput' in self.combined_df.columns:
                p1_throughput = self.df1['throughput'].mean()
                p2_throughput = self.df2['throughput'].mean()
                max_throughput = max(p1_throughput, p2_throughput)
                
                metrics.append('Throughput')
                protocol1_values.append((p1_throughput / max_throughput) * 10)
                protocol2_values.append((p2_throughput / max_throughput) * 10)
            
            # Packet Loss (invert - lower is better, scale 0-10)
            if 'packet_loss' in self.combined_df.columns:
                p1_loss = self.df1['packet_loss'].mean()
                p2_loss = self.df2['packet_loss'].mean()
                max_loss = max(p1_loss, p2_loss, 1)  # Avoid division by zero
                
                metrics.append('Reliability\n(Low Loss)')
                protocol1_values.append((1 - p1_loss / 100) * 10)
                protocol2_values.append((1 - p2_loss / 100) * 10)
            
            # Delay (invert - lower is better)
            if 'avg_delay' in self.combined_df.columns:
                p1_delay = self.df1['avg_delay'].mean()
                p2_delay = self.df2['avg_delay'].mean()
                max_delay = max(p1_delay, p2_delay, 1)
                
                metrics.append('Responsiveness\n(Low Delay)')
                protocol1_values.append((1 - p1_delay / max_delay) * 10)
                protocol2_values.append((1 - p2_delay / max_delay) * 10)
            
            # Consistency (invert of standard deviation)
            if 'throughput' in self.combined_df.columns:
                p1_std = self.df1['throughput'].std()
                p2_std = self.df2['throughput'].std()
                max_std = max(p1_std, p2_std, 1)
                
                metrics.append('Consistency\n(Low Variance)')
                protocol1_values.append((1 - p1_std / max_std) * 10)
                protocol2_values.append((1 - p2_std / max_std) * 10)
            
            if len(metrics) < 3:
                logger.warning("Not enough metrics for radar chart")
                return
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            protocol1_values += protocol1_values[:1]
            protocol2_values += protocol2_values[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Plot data
            ax.plot(angles, protocol1_values, 'o-', linewidth=2, 
                   label=self.protocol1_name, color='blue')
            ax.fill(angles, protocol1_values, alpha=0.25, color='blue')
            
            ax.plot(angles, protocol2_values, 's-', linewidth=2, 
                   label=self.protocol2_name, color='red')
            ax.fill(angles, protocol2_values, alpha=0.25, color='red')
            
            # Customize chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=10)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
            ax.grid(True)
            
            plt.title(f'Performance Radar Chart\n{self.protocol1_name} vs {self.protocol2_name}', 
                     size=14, weight='bold', pad=20)
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            
            plt.tight_layout()
            plot_path = self.results_dir / 'performance_radar_chart.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('performance_radar_chart.png')
            logger.info("Created performance radar chart")
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
    
    def _create_scenario_bar_comparison(self) -> None:
        """Create scenario-by-scenario bar comparison for throughput."""
        try:
            # Get scenario comparison data
            scenario_comparison = (
                self.combined_df.groupby(['scenario_group', 'Protocol'])['throughput']
                .mean()
                .reset_index()
            )
            
            # Pivot for easier plotting
            pivot_data = scenario_comparison.pivot(index='scenario_group', 
                                                 columns='Protocol', 
                                                 values='throughput').fillna(0)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(16, 8))
            
            x = np.arange(len(pivot_data.index))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pivot_data[self.protocol1_name], width, 
                          label=self.protocol1_name, alpha=0.8)
            bars2 = ax.bar(x + width/2, pivot_data[self.protocol2_name], width, 
                          label=self.protocol2_name, alpha=0.8)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Scenarios')
            ax.set_ylabel('Throughput (Mbps)')
            ax.set_title(f'Scenario-by-Scenario Throughput Comparison\n{self.protocol1_name} vs {self.protocol2_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'scenario_bar_comparison.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('scenario_bar_comparison.png')
            logger.info("Created scenario bar comparison")
            
        except Exception as e:
            logger.error(f"Error creating scenario bar comparison: {e}")
    
    def _create_efficiency_analysis(self) -> None:
        """Create protocol efficiency analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Throughput Efficiency (Throughput / Traffic Rate)
            ax1 = axes[0, 0]
            efficiency_data = (
                self.combined_df.groupby(['traffic_rate_num', 'Protocol'])['efficiency']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = efficiency_data[efficiency_data['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('traffic_rate_num')
                    ax1.plot(protocol_data['traffic_rate_num'], 
                            protocol_data['efficiency'],
                            marker='o', linewidth=2, markersize=6, label=protocol)
            
            ax1.set_xlabel('Traffic Rate (Mbps)')
            ax1.set_ylabel('Efficiency (Throughput/Traffic Rate)')
            ax1.set_title('Protocol Efficiency vs Traffic Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            
            # Success Rate (1 - Packet Loss Rate)
            ax2 = axes[0, 1]
            self.combined_df['success_rate'] = 100 - self.combined_df['packet_loss']
            success_data = (
                self.combined_df.groupby(['traffic_rate_num', 'Protocol'])['success_rate']
                .mean()
                .reset_index()
            )
            
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = success_data[success_data['Protocol'] == protocol]
                if len(protocol_data) > 0:
                    protocol_data = protocol_data.sort_values('traffic_rate_num')
                    ax2.plot(protocol_data['traffic_rate_num'], 
                            protocol_data['success_rate'],
                            marker='s', linewidth=2, markersize=6, label=protocol)
            
            ax2.set_xlabel('Traffic Rate (Mbps)')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Packet Success Rate vs Traffic Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            
            # Performance vs Distance Efficiency
            ax3 = axes[1, 0]
            # Calculate performance retention over distance
            for protocol in [self.protocol1_name, self.protocol2_name]:
                protocol_data = self.combined_df[self.combined_df['Protocol'] == protocol]
                dist_perf = (
                    protocol_data.groupby('distance')['throughput']
                    .mean()
                    .reset_index()
                    .sort_values('distance')
                )
                
                if len(dist_perf) > 0:
                    # Normalize to first distance performance
                    baseline = dist_perf['throughput'].iloc[0]
                    retention = (dist_perf['throughput'] / baseline) * 100
                    
                    ax3.plot(dist_perf['distance'], retention,
                            marker='^', linewidth=2, markersize=6, 
                            label=f'{protocol} Retention')
            
            ax3.set_xlabel('Distance (m)')
            ax3.set_ylabel('Performance Retention (%)')
            ax3.set_title('Throughput Retention vs Distance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Overall Protocol Score (weighted combination)
            ax4 = axes[1, 1]
            
            # Calculate weighted scores
            protocols = [self.protocol1_name, self.protocol2_name]
            scores = {'Throughput': [], 'Reliability': [], 'Efficiency': [], 'Overall': []}
            
            for protocol in protocols:
                protocol_data = self.combined_df[self.combined_df['Protocol'] == protocol]
                
                # Throughput score (normalized)
                avg_throughput = protocol_data['throughput'].mean()
                max_possible = self.combined_df['throughput'].max()
                throughput_score = (avg_throughput / max_possible) * 100
                scores['Throughput'].append(throughput_score)
                
                # Reliability score (based on packet loss)
                avg_loss = protocol_data['packet_loss'].mean()
                reliability_score = max(0, 100 - avg_loss)
                scores['Reliability'].append(reliability_score)
                
                # Efficiency score
                avg_efficiency = protocol_data['efficiency'].mean()
                max_efficiency = self.combined_df['efficiency'].max()
                efficiency_score = (avg_efficiency / max_efficiency) * 100
                scores['Efficiency'].append(efficiency_score)
                
                # Overall weighted score
                overall_score = (throughput_score * 0.4 + reliability_score * 0.3 + efficiency_score * 0.3)
                scores['Overall'].append(overall_score)
            
            # Create grouped bar chart
            x = np.arange(len(protocols))
            width = 0.2
            
            metrics_to_plot = ['Throughput', 'Reliability', 'Efficiency', 'Overall']
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            
            for i, metric in enumerate(metrics_to_plot):
                ax4.bar(x + i * width, scores[metric], width, 
                       label=metric, color=colors[i], alpha=0.8)
            
            ax4.set_xlabel('Protocols')
            ax4.set_ylabel('Score (0-100)')
            ax4.set_title('Overall Protocol Performance Scores')
            ax4.set_xticks(x + width * 1.5)
            ax4.set_xticklabels(protocols)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, metric in enumerate(metrics_to_plot):
                for j, score in enumerate(scores[metric]):
                    ax4.text(j + i * width, score + 1, f'{score:.1f}', 
                            ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'efficiency_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('efficiency_analysis.png')
            logger.info("Created efficiency analysis")
            
        except Exception as e:
            logger.error(f"Error creating efficiency analysis: {e}")
    
    def calculate_statistical_comparison(self) -> Dict:
        """Calculate comprehensive statistical comparisons."""
        if self.combined_df is None:
            logger.error("No combined data available.")
            return {}
        
        logger.info("Calculating statistical comparisons...")
        
        results = {}
        metrics = ['throughput', 'packet_loss', 'avg_delay']
        grouping_vars = ['distance', 'speed', 'interferers', 'packet_size', 'traffic_rate']
        
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
        
        # Comparison by grouping variables
        for var in grouping_vars:
            if var in self.combined_df.columns:
                var_stats = {}
                for metric in metrics:
                    if metric in self.combined_df.columns:
                        stats = (
                            self.combined_df.groupby(['Protocol', var])[metric]
                            .agg(['mean', 'std', 'count'])
                            .round(4)
                            .reset_index()
                        )
                        var_stats[metric] = stats
                results[f'by_{var}'] = var_stats
        
        # Scenario-by-scenario comparison
        scenario_comparison = {}
        for scenario in self.combined_df['scenario_group'].unique():
            scenario_data = self.combined_df[self.combined_df['scenario_group'] == scenario]
            scenario_stats = {}
            
            for metric in metrics:
                if metric in scenario_data.columns:
                    stats = (
                        scenario_data.groupby('Protocol')[metric]
                        .agg(['mean', 'std', 'count'])
                        .round(4)
                    )
                    scenario_stats[metric] = stats
            
            scenario_comparison[scenario] = scenario_stats
        
        results['by_scenario'] = scenario_comparison
        
        self.comparison_results = results
        return results
    
    def generate_human_readable_summary(self) -> str:
        """Generate comprehensive human-readable comparison summary with detailed scenario analysis."""

        if self.combined_df is None or len(self.combined_df) == 0:
            return "No comparison data available."

        summary = []
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        p1, p2 = self.protocol1_name, self.protocol2_name

        # Scenario count
        scenario_col = 'scenario' if 'scenario' in self.combined_df.columns else None
        unique_scenarios = self.combined_df[scenario_col].unique() if scenario_col else []
        total_scenarios = len(unique_scenarios)
        n1 = len(self.df1) if self.df1 is not None else 0
        n2 = len(self.df2) if self.df2 is not None else 0

        summary.append(f"\n{'='*99}")
        summary.append(f"COMPREHENSIVE PROTOCOL COMPARISON ANALYSIS")
        summary.append(f"{p1} vs {p2}")
        summary.append(f"{'='*99}")
        summary.append(f"Analysis Date: {now}")
        summary.append(f"Total Scenarios Analyzed: {total_scenarios}")
        summary.append(f"Total Data Points: {len(self.combined_df)} ({n1} vs {n2})\n")

        # --- Executive Summary ---
        summary.append("EXECUTIVE SUMMARY")
        summary.append("-" * 60)
        # Safe stat helpers
        def safe_mean(arr): return float(np.nan_to_num(np.mean(arr), nan=0.0))
        def safe_min(arr): return float(np.nan_to_num(np.min(arr), nan=0.0))
        def safe_max(arr): return float(np.nan_to_num(np.max(arr), nan=0.0))
        def safe_std(arr): return float(np.nan_to_num(np.std(arr), nan=0.0))

        p1_data = self.combined_df[self.combined_df['Protocol'] == p1]
        p2_data = self.combined_df[self.combined_df['Protocol'] == p2]

        # Metrics
        p1_th = safe_mean(p1_data['throughput']) if 'throughput' in p1_data else 0
        p2_th = safe_mean(p2_data['throughput']) if 'throughput' in p2_data else 0
        p1_loss = safe_mean(p1_data['packet_loss']) if 'packet_loss' in p1_data else 0
        p2_loss = safe_mean(p2_data['packet_loss']) if 'packet_loss' in p2_data else 0
        p1_delay = safe_mean(p1_data['avg_delay']) if 'avg_delay' in p1_data else 0
        p2_delay = safe_mean(p2_data['avg_delay']) if 'avg_delay' in p2_data else 0

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

        # --- Detailed Performance Analysis ---
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
        summary.append(f"    - Average Delay: {p1_delay:.3f}ms")
        summary.append(f"    - Delay Range: {safe_min(p1_data['avg_delay']):.3f} - {safe_max(p1_data['avg_delay']):.3f}ms")
        summary.append(f"  {p2}:")
        summary.append(f"    - Average Delay: {p2_delay:.3f}ms")
        summary.append(f"    - Delay Range: {safe_min(p2_data['avg_delay']):.3f} - {safe_max(p2_data['avg_delay']):.3f}ms")
        summary.append(f"  Winner: {responsiveness_winner} by {responsiveness_improvement:.1f}%\n")

        # --- Scenario-by-Scenario Analysis ---
        summary.append("COMPREHENSIVE SCENARIO-BY-SCENARIO ANALYSIS")
        summary.append("-" * 80)
        # Helper for per-parameter breakdown
        def param_breakdown(param, label, unit, eff=False):
            if param not in self.combined_df.columns:
                return []
            out = []
            levels = sorted(self.combined_df[param].dropna().unique())
            wins = {p1: 0, p2: 0}
            for lvl in levels:
                lvl_data = self.combined_df[self.combined_df[param] == lvl]
                d1 = lvl_data[lvl_data['Protocol'] == p1]
                d2 = lvl_data[lvl_data['Protocol'] == p2]
                if len(d1) == 0 or len(d2) == 0:
                    continue
                # Compute metrics
                mean1, mean2 = safe_mean(d1['throughput']), safe_mean(d2['throughput'])
                loss1, loss2 = safe_mean(d1['packet_loss']), safe_mean(d2['packet_loss'])
                delay1, delay2 = safe_mean(d1['avg_delay']), safe_mean(d2['avg_delay'])
                # Effective throughput for winner
                eff1 = mean1 * (100 - loss1) / 100
                eff2 = mean2 * (100 - loss2) / 100
                if eff:
                    try:
                        effrate1 = (mean1 / max(float(lvl), 0.001)) * 100 if mean1 > 0 else 0
                        effrate2 = (mean2 / max(float(lvl), 0.001)) * 100 if mean2 > 0 else 0
                    except Exception:
                        effrate1 = effrate2 = 0
                winner = p1 if eff1 > eff2 else p2
                wins[winner] += 1
                out.append(f"\nAt {label} {lvl}{unit}:")
                out.append(f"  {p1}: {mean1:.2f} Mbps, {loss1:.2f}% loss, {delay1:.2f}ms delay")
                out.append(f"  {p2}: {mean2:.2f} Mbps, {loss2:.2f}% loss, {delay2:.2f}ms delay")
                if eff:
                    out.append(f"  Efficiency Winner: {p1 if effrate1 > effrate2 else p2}")
                else:
                    out.append(f"  Winner: {winner} (effective throughput: {max(eff1, eff2):.2f} Mbps)")
                # Degradation and impact (if possible)
                if param == 'distance' and lvl != min(levels):
                    min_lvl = min(levels)
                    base1 = safe_mean(self.combined_df[(self.combined_df[param]==min_lvl) & (self.combined_df['Protocol']==p1)]['throughput'])
                    base2 = safe_mean(self.combined_df[(self.combined_df[param]==min_lvl) & (self.combined_df['Protocol']==p2)]['throughput'])
                    if base1 > 0:
                        deg1 = ((base1 - mean1) / max(base1, 0.001)) * 100
                    else:
                        deg1 = 0
                    if base2 > 0:
                        deg2 = ((base2 - mean2) / max(base2, 0.001)) * 100
                    else:
                        deg2 = 0
                    out.append(f"  Performance Degradation: {p1} ({deg1:-.1f}%), {p2} ({deg2:-.1f}%)")
                if param == 'speed' and lvl > 0:
                    stationary1 = safe_mean(self.combined_df[(self.combined_df[param]==0) & (self.combined_df['Protocol']==p1)]['throughput'])
                    stationary2 = safe_mean(self.combined_df[(self.combined_df[param]==0) & (self.combined_df['Protocol']==p2)]['throughput'])
                    mob_impact1 = ((stationary1 - mean1) / max(stationary1, 0.001)) * 100 if stationary1 > 0 else 0
                    mob_impact2 = ((stationary2 - mean2) / max(stationary2, 0.001)) * 100 if stationary2 > 0 else 0
                    out.append(f"  Mobility Impact: {p1} ({mob_impact1:-.1f}%), {p2} ({mob_impact2:-.1f}%)")
                if param == 'interferers' and lvl > 0:
                    no_intf1 = safe_mean(self.combined_df[(self.combined_df[param]==0) & (self.combined_df['Protocol']==p1)]['throughput'])
                    no_intf2 = safe_mean(self.combined_df[(self.combined_df[param]==0) & (self.combined_df['Protocol']==p2)]['throughput'])
                    intf_impact1 = ((no_intf1 - mean1) / max(no_intf1, 0.001)) * 100 if no_intf1 > 0 else 0
                    intf_impact2 = ((no_intf2 - mean2) / max(no_intf2, 0.001)) * 100 if no_intf2 > 0 else 0
                    out.append(f"  Interference Impact: {p1} ({intf_impact1:-.1f}%), {p2} ({intf_impact2:-.1f}%)")
            out.append(f"\n{label} Performance Summary:")
            out.append(f"  {p1} wins at {wins[p1]} {label.lower()}s")
            out.append(f"  {p2} wins at {wins[p2]} {label.lower()}s")
            return out

        # Distance
        summary.extend(param_breakdown('distance', 'Distance', 'm'))
        # Traffic Rate
        summary.extend(param_breakdown('traffic_rate', 'Traffic Rate', 'Mbps', eff=True))
        # Speed
        summary.extend(param_breakdown('speed', 'Speed', ' m/s'))
        # Interferers
        summary.extend(param_breakdown('interferers', 'Interference Resilience', ''))
        # Packet Size
        if 'packet_size' in self.combined_df.columns:
            pkt_out = []
            pkt_sizes = sorted(self.combined_df['packet_size'].dropna().unique())
            for ps in pkt_sizes:
                d1 = self.combined_df[(self.combined_df['packet_size']==ps)&(self.combined_df['Protocol']==p1)]
                d2 = self.combined_df[(self.combined_df['packet_size']==ps)&(self.combined_df['Protocol']==p2)]
                if len(d1)==0 or len(d2)==0: continue
                th1, th2 = safe_mean(d1['throughput']), safe_mean(d2['throughput'])
                l1, l2 = safe_mean(d1['packet_loss']), safe_mean(d2['packet_loss'])
                winner = p1 if th1 > th2 else p2
                pkt_out.append(f"\nWith Packet Size {ps} bytes:")
                pkt_out.append(f"  {p1}: {th1:.2f} Mbps, {l1:.2f}% loss")
                pkt_out.append(f"  {p2}: {th2:.2f} Mbps, {l2:.2f}% loss")
                pkt_out.append(f"  Winner: {winner}")
            if pkt_out:
                summary.append("\nPACKET SIZE PERFORMANCE BREAKDOWN:")
                summary.extend(pkt_out)

        # --- Recommendations ---
        summary.append("\nPERFORMANCE RECOMMENDATIONS")
        summary.append("-" * 60)
        # Optimal use cases
        summary.append("OPTIMAL USE CASES:\n")
        p1_best, p2_best = [], []
        if scenario_col:
            for scenario in unique_scenarios:
                sdata = self.combined_df[self.combined_df[scenario_col] == scenario]
                d1 = sdata[sdata['Protocol'] == p1]
                d2 = sdata[sdata['Protocol'] == p2]
                if len(d1)==0 or len(d2)==0: continue
                perf1 = safe_mean(d1['throughput']) * (100 - safe_mean(d1['packet_loss'])) / 100
                perf2 = safe_mean(d2['throughput']) * (100 - safe_mean(d2['packet_loss'])) / 100
                if perf1 > perf2:
                    p1_best.append(scenario)
                else:
                    p2_best.append(scenario)
        summary.append(f"{p1} excels in:")
        p1_pct = (len(p1_best)/max(total_scenarios,1)*100) if total_scenarios > 0 else 0
        summary.append(f"  - {len(p1_best)} out of {total_scenarios} scenarios ({p1_pct:.1f}%)")
        if len(p1_best)>0:
            for s in p1_best[:5]:
                summary.append(f"    * {s}")
            if len(p1_best)>5:
                summary.append(f"    * ... and {len(p1_best)-5} more scenarios")
        summary.append(f"\n{p2} excels in:")
        p2_pct = (len(p2_best)/max(total_scenarios,1)*100) if total_scenarios > 0 else 0
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

        # --- Generated files ---
        summary.append(f"\nGENERATED ANALYSIS FILES")
        summary.append("-" * 40)
        summary.append("Visual Analysis Plots:")
        for plot in self.plots_generated:
            summary.append(f"  • {plot}")
        summary.append(f"\nDetailed Reports:")
        summary.append(f"  • Excel Report: {p1}_vs_{p2}_detailed_report.xlsx")
        summary.append(f"  • This Summary: comparison_summary.txt")
        summary.append(f"  • Analysis Log: enhanced_comparison_analysis.log")
        summary.append(f"\nAll files saved to: {self.results_dir}")
        summary.append("="*99)
        return "\n".join(summary)

    def create_excel_report(self) -> None:
        """Create comprehensive Excel report with multiple sheets."""
        try:
            logger.info("Creating Excel report...")
            
            excel_path = self.results_dir / f'{self.protocol1_name}_vs_{self.protocol2_name}_detailed_report.xlsx'
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Overview sheet
                if self.comparison_results and 'overall' in self.comparison_results:
                    overall_df = pd.DataFrame()
                    for metric, stats in self.comparison_results['overall'].items():
                        metric_df = stats.copy()
                        metric_df['Metric'] = metric
                        overall_df = pd.concat([overall_df, metric_df])
                    
                    overall_df.to_excel(writer, sheet_name='Overall_Comparison', index=True)
                
                # Raw data
                if self.combined_df is not None:
                    # Protocol 1 data
                    self.df1.to_excel(writer, sheet_name=f'{self.protocol1_name}_Data', index=False)
                    
                    # Protocol 2 data
                    self.df2.to_excel(writer, sheet_name=f'{self.protocol2_name}_Data', index=False)
                    
                    # Combined summary by scenario
                    scenario_summary = (
                        self.combined_df.groupby(['scenario_group', 'Protocol'])
                        [['throughput', 'packet_loss', 'avg_delay']]
                        .mean()
                        .round(4)
                        .reset_index()
                    )
                    scenario_summary.to_excel(writer, sheet_name='Scenario_Summary', index=False)
                
                # Detailed comparisons by variable
                for var in ['distance', 'speed', 'interferers', 'traffic_rate']:
                    if f'by_{var}' in self.comparison_results:
                        var_data = self.comparison_results[f'by_{var}']
                        if 'throughput' in var_data:
                            var_df = var_data['throughput']
                            var_df.to_excel(writer, sheet_name=f'By_{var.title()}', index=False)
            
            logger.info(f"Excel report created: {excel_path}")
            
        except Exception as e:
            logger.error(f"Error creating Excel report: {e}")
    
    def run_complete_analysis(self) -> bool:
        """Run the complete enhanced comparison analysis."""
        logger.info("Starting complete enhanced protocol comparison analysis...")
        
        try:
            # Load and preprocess data
            if not self.load_data():
                logger.error("Failed to load data")
                return False
            
            logger.info("Data loaded successfully")
            self.preprocess_data()
            logger.info("Data preprocessing completed")
            
            # Create enhanced plots
            self.create_enhanced_comparison_plots()
            logger.info(f"Generated {len(self.plots_generated)} plots")
            
            # Calculate statistical comparisons
            self.calculate_statistical_comparison()
            logger.info("Statistical analysis completed")
            
            # Create Excel report
            self.create_excel_report()
            logger.info("Excel report generated")
            
            # Generate and save summary
            summary = self.generate_human_readable_summary()
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
            return False


def main():
    """Main function to run the enhanced protocol comparison."""
    
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Enhanced Protocol Comparison Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python protocol_comparision.py --protocol1 aarf --protocol2 smartv3
  python protocol_comparision.py --protocol1 smartv1 --protocol2 smartv2 --name1 "SmartV1" --name2 "SmartV2"
  python protocol_comparision.py -p1 aarf -p2 smartv1 -n1 "AARF" -n2 "SmartV1"
        """
    )
    
    parser.add_argument(
        '--protocol1', '-p1',
        type=str,
        default='aarf',
        help='First protocol name (default: aarf)'
    )
    
    parser.add_argument(
        '--protocol2', '-p2', 
        type=str,
        default='smartv3',
        help='Second protocol name (default: smartv3)'
    )
    
    parser.add_argument(
        '--name1', '-n1',
        type=str,
        default=None,
        help='Custom display name for protocol 1 (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--name2', '-n2',
        type=str,
        default=None,
        help='Custom display name for protocol 2 (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--list-protocols', '-l',
        action='store_true',
        help='List available protocols and exit'
    )
    
    args = parser.parse_args()
    
    # List available protocols if requested
    if args.list_protocols:
        print("Available protocols:")
        print("  • aarf (AARF)")
        print("  • smartv1 (SmartV1)")
        print("  • smartv2 (SmartV2)")
        print("  • smartv3 (SmartV3)")
        print("  • smartrf (SmartRF)")
        print("  • smartrfv3 (SmartRFV3)")
        return True
    
    # Convert protocol names to CSV filenames
    protocol_mapping = {
        'aarf': 'aarf-benchmark.csv',
        'smartv1': 'smartv1-benchmark.csv',
        'smartv2': 'smartv2-benchmark.csv', 
        'smartv3': 'smartv3-benchmark.csv',
        'smartrf': 'smartrf-benchmark-results.csv',
        'smartrfv3': 'smartrf-benchmark-v3.csv'
    }
    
    protocol1_csv = protocol_mapping.get(args.protocol1.lower())
    protocol2_csv = protocol_mapping.get(args.protocol2.lower())
    
    if not protocol1_csv:
        print(f"❌ ERROR: Unknown protocol '{args.protocol1}'. Use --list-protocols to see available options.")
        return False
    
    if not protocol2_csv:
        print(f"❌ ERROR: Unknown protocol '{args.protocol2}'. Use --list-protocols to see available options.")
        return False
    
    # Optional: Specify custom names (auto-detected from filename if None)
    protocol1_name = args.name1
    protocol2_name = args.name2
    
    print("=" * 80)
    print("ENHANCED PROTOCOL COMPARISON ANALYZER")
    print("=" * 80)
    print(f"Protocol 1 CSV: {protocol1_csv}")
    print(f"Protocol 2 CSV: {protocol2_csv}")
    print()
    
    # Create analyzer instance
    try:
        analyzer = EnhancedProtocolComparisonAnalyzer(
            protocol1_csv=protocol1_csv,
            protocol2_csv=protocol2_csv,
            protocol1_name=protocol1_name,
            protocol2_name=protocol2_name
        )
        
        print(f"Analyzing: {analyzer.protocol1_name} vs {analyzer.protocol2_name}")
        print(f"Results will be saved to: {analyzer.results_dir}")
        print()
        
        # Run complete analysis
        success = analyzer.run_complete_analysis()
        
        if success:
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Check the results directory: {analyzer.results_dir}")
            print("\nGenerated files include:")
            print("• Multiple comparison plots (.png files)")
            print("• Detailed Excel report")
            print("• Text summary report")
            print("• Analysis log file")
        else:
            print("\n" + "=" * 80)
            print("ANALYSIS FAILED!")
            print("=" * 80)
            print("Check the log file for details about the error.")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPlease check:")
        print("1. CSV file paths are correct")
        print("2. CSV files exist in the specified location")
        print("3. CSV files have the required columns")
        return False
    
    return True


if __name__ == "__main__":
    main()