import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ComprehensiveSingleProtocolAnalyzer:
    """
    Comprehensive single-protocol analysis tool that generates detailed statistics,
    graphs, results, and summaries for WiFi rate adaptation protocols.
    """
    
    def __init__(self, protocol_csv: str, protocol_name: str = None):
        """
        Initialize the single protocol analyzer.
        
        Args:
            protocol_csv: Path to protocol's CSV file
            protocol_name: Display name for protocol (auto-detected if None)
        """
        # Fix: Look in parent directory
        self.protocol_csv = Path(__file__).parent.parent / protocol_csv
        
        # Auto-detect protocol name from filename if not provided
        self.protocol_name = protocol_name or self._extract_protocol_name(protocol_csv)
        
        # Create results directory structure
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / 'test_results' / self.protocol_name
        self._create_results_directory()
        
        # Set up logging
        self._setup_logging()
        
        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.plots_generated: List[str] = []
        self.analysis_results: Dict = {}
        
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
                return 'SmartXGB'
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
        log_file = self.results_dir / 'protocol_analysis.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"=== Comprehensive Single Protocol Analysis: {self.protocol_name} ===")
        self.logger.info(f"Log file: {log_file}")
    
    def load_data(self) -> bool:
        """Load CSV file with error handling."""
        try:
            if not self.protocol_csv.exists():
                self.logger.error(f"Protocol CSV not found: {self.protocol_csv}")
                return False
            
            self.df = pd.read_csv(self.protocol_csv)
            self.df['Protocol'] = self.protocol_name
            
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
            
            self.df = self.df.rename(columns=column_mappings)
            
            # Convert traffic_rate to numeric for proper sorting
            if 'traffic_rate' in self.df.columns:
                self.df['traffic_rate_num'] = (
                    self.df['traffic_rate']
                    .astype(str)
                    .str.replace('Mbps', '', case=False)
                    .str.replace(' ', '')
                    .astype(float)
                )
            
            # Create scenario grouping for easier analysis
            self.df['scenario_group'] = (
                self.df['distance'].astype(str) + 'm_' +
                self.df['speed'].astype(str) + 'mps_' +
                self.df['interferers'].astype(str) + 'intf'
            )
            
            self.logger.info(f"Loaded {len(self.df)} rows for {self.protocol_name}")
            self.logger.info(f"Scenarios: {self.df['scenario_group'].nunique()} unique scenarios")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def generate_comprehensive_plots(self) -> None:
        """Generate comprehensive plots and visualizations."""
        if self.df is None:
            self.logger.warning("No data available for plotting")
            return
        
        self.logger.info("Creating comprehensive plots...")
        
        # 1. Performance Overview Dashboard
        self._create_performance_dashboard()
        
        # 2. Throughput Analysis
        self._create_throughput_analysis()
        
        # 3. Packet Loss Analysis
        self._create_packet_loss_analysis()
        
        # 4. Delay Analysis
        self._create_delay_analysis()
        
        # 5. Distance Impact Analysis
        self._create_distance_impact_analysis()
        
        # 6. Speed Impact Analysis
        self._create_speed_impact_analysis()
        
        # 7. Interference Impact Analysis
        self._create_interference_impact_analysis()
        
        # 8. Packet Size Impact Analysis
        self._create_packet_size_impact_analysis()
        
        # 9. Traffic Rate Impact Analysis
        self._create_traffic_rate_impact_analysis()
        
        # 10. Scenario Performance Heatmap
        self._create_scenario_heatmap()
        
        # 11. Performance Distribution Analysis
        self._create_performance_distributions()
        
        # 12. Correlation Analysis
        self._create_correlation_analysis()
        
        # 13. Performance Efficiency Analysis
        self._create_efficiency_analysis()
        
        # 14. Statistical Summary Plots
        self._create_statistical_summary_plots()
    
    def _create_performance_dashboard(self) -> None:
        """Create a comprehensive performance dashboard."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{self.protocol_name} - Performance Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Throughput vs Distance
            ax1 = axes[0, 0]
            for rate in sorted(self.df['traffic_rate_num'].unique()):
                rate_data = self.df[self.df['traffic_rate_num'] == rate]
                ax1.plot(rate_data['distance'], rate_data['throughput'], 
                        marker='o', label=f'{rate}Mbps')
            ax1.set_xlabel('Distance (m)')
            ax1.set_ylabel('Throughput (Mbps)')
            ax1.set_title('Throughput vs Distance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Packet Loss vs Traffic Rate
            ax2 = axes[0, 1]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax2.plot(dist_data['traffic_rate_num'], dist_data['packet_loss'], 
                        marker='s', label=f'{distance}m')
            ax2.set_xlabel('Traffic Rate (Mbps)')
            ax2.set_ylabel('Packet Loss (%)')
            ax2.set_title('Packet Loss vs Traffic Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Delay vs Distance
            ax3 = axes[0, 2]
            for rate in sorted(self.df['traffic_rate_num'].unique()):
                rate_data = self.df[self.df['traffic_rate_num'] == rate]
                ax3.plot(rate_data['distance'], rate_data['avg_delay'], 
                        marker='^', label=f'{rate}Mbps')
            ax3.set_xlabel('Distance (m)')
            ax3.set_ylabel('Average Delay (ms)')
            ax3.set_title('Delay vs Distance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Throughput Distribution
            ax4 = axes[1, 0]
            ax4.hist(self.df['throughput'], bins=20, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Throughput (Mbps)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Throughput Distribution')
            ax4.grid(True, alpha=0.3)
            
            # 5. Packet Loss Distribution
            ax5 = axes[1, 1]
            ax5.hist(self.df['packet_loss'], bins=20, alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Packet Loss (%)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Packet Loss Distribution')
            ax5.grid(True, alpha=0.3)
            
            # 6. Performance by Interference
            ax6 = axes[1, 2]
            interference_stats = self.df.groupby('interferers').agg({
                'throughput': 'mean',
                'packet_loss': 'mean',
                'avg_delay': 'mean'
            })
            interference_stats.plot(kind='bar', ax=ax6)
            ax6.set_xlabel('Number of Interferers')
            ax6.set_ylabel('Average Value')
            ax6.set_title('Performance by Interference Level')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'performance_dashboard.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('performance_dashboard.png')
            self.logger.info("Created performance dashboard")
            
        except Exception as e:
            self.logger.error(f"Error creating performance dashboard: {e}")
    
    def _create_throughput_analysis(self) -> None:
        """Create detailed throughput analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Throughput Analysis', fontsize=14, fontweight='bold')
            
            # 1. Throughput vs Traffic Rate (all scenarios)
            ax1 = axes[0, 0]
            for distance in sorted(self.df['distance'].unique()):
                for speed in sorted(self.df['speed'].unique()):
                    scenario_data = self.df[(self.df['distance'] == distance) & 
                                          (self.df['speed'] == speed)]
                    if len(scenario_data) > 0:
                        ax1.plot(scenario_data['traffic_rate_num'], scenario_data['throughput'], 
                               marker='o', label=f'Dist:{distance}m, Speed:{speed}m/s')
            ax1.set_xlabel('Traffic Rate (Mbps)')
            ax1.set_ylabel('Throughput (Mbps)')
            ax1.set_title('Throughput vs Traffic Rate')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. Throughput efficiency (throughput/traffic_rate ratio)
            ax2 = axes[0, 1]
            self.df['throughput_efficiency'] = self.df['throughput'] / self.df['traffic_rate_num']
            efficiency_by_distance = self.df.groupby('distance')['throughput_efficiency'].mean()
            ax2.bar(efficiency_by_distance.index, efficiency_by_distance.values)
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Throughput Efficiency')
            ax2.set_title('Throughput Efficiency by Distance')
            ax2.grid(True, alpha=0.3)
            
            # 3. Throughput by packet size
            ax3 = axes[1, 0]
            packet_size_stats = self.df.groupby('packet_size').agg({
                'throughput': ['mean', 'std']
            })
            packet_size_stats.plot(kind='bar', ax=ax3)
            ax3.set_xlabel('Packet Size (bytes)')
            ax3.set_ylabel('Throughput (Mbps)')
            ax3.set_title('Throughput by Packet Size')
            ax3.legend(['Mean', 'Std Dev'])
            ax3.grid(True, alpha=0.3)
            
            # 4. Throughput heatmap by distance and traffic rate
            ax4 = axes[1, 1]
            throughput_pivot = self.df.pivot_table(
                values='throughput', 
                index='distance', 
                columns='traffic_rate_num', 
                aggfunc='mean'
            )
            sns.heatmap(throughput_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4)
            ax4.set_title('Throughput Heatmap (Distance vs Traffic Rate)')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'throughput_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('throughput_analysis.png')
            self.logger.info("Created throughput analysis plots")
            
        except Exception as e:
            self.logger.error(f"Error creating throughput analysis: {e}")
    
    def _create_packet_loss_analysis(self) -> None:
        """Create detailed packet loss analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Packet Loss Analysis', fontsize=14, fontweight='bold')
            
            # 1. Packet Loss vs Traffic Rate
            ax1 = axes[0, 0]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax1.plot(dist_data['traffic_rate_num'], dist_data['packet_loss'], 
                        marker='o', label=f'Distance: {distance}m')
            ax1.set_xlabel('Traffic Rate (Mbps)')
            ax1.set_ylabel('Packet Loss (%)')
            ax1.set_title('Packet Loss vs Traffic Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Packet Loss vs Distance
            ax2 = axes[0, 1]
            for rate in sorted(self.df['traffic_rate_num'].unique()):
                rate_data = self.df[self.df['traffic_rate_num'] == rate]
                ax2.plot(rate_data['distance'], rate_data['packet_loss'], 
                        marker='s', label=f'Rate: {rate}Mbps')
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Packet Loss (%)')
            ax2.set_title('Packet Loss vs Distance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Packet Loss by interference level
            ax3 = axes[1, 0]
            interference_loss = self.df.groupby('interferers')['packet_loss'].agg(['mean', 'std'])
            interference_loss.plot(kind='bar', ax=ax3)
            ax3.set_xlabel('Number of Interferers')
            ax3.set_ylabel('Packet Loss (%)')
            ax3.set_title('Packet Loss by Interference Level')
            ax3.legend(['Mean', 'Std Dev'])
            ax3.grid(True, alpha=0.3)
            
            # 4. Packet Loss heatmap
            ax4 = axes[1, 1]
            loss_pivot = self.df.pivot_table(
                values='packet_loss', 
                index='distance', 
                columns='traffic_rate_num', 
                aggfunc='mean'
            )
            sns.heatmap(loss_pivot, annot=True, fmt='.2f', cmap='Reds', ax=ax4)
            ax4.set_title('Packet Loss Heatmap (Distance vs Traffic Rate)')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'packet_loss_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('packet_loss_analysis.png')
            self.logger.info("Created packet loss analysis plots")
            
        except Exception as e:
            self.logger.error(f"Error creating packet loss analysis: {e}")
    
    def _create_delay_analysis(self) -> None:
        """Create detailed delay analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Delay Analysis', fontsize=14, fontweight='bold')
            
            # 1. Delay vs Traffic Rate
            ax1 = axes[0, 0]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax1.plot(dist_data['traffic_rate_num'], dist_data['avg_delay'], 
                        marker='o', label=f'Distance: {distance}m')
            ax1.set_xlabel('Traffic Rate (Mbps)')
            ax1.set_ylabel('Average Delay (ms)')
            ax1.set_title('Delay vs Traffic Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Delay vs Distance
            ax2 = axes[0, 1]
            for rate in sorted(self.df['traffic_rate_num'].unique()):
                rate_data = self.df[self.df['traffic_rate_num'] == rate]
                ax2.plot(rate_data['distance'], rate_data['avg_delay'], 
                        marker='s', label=f'Rate: {rate}Mbps')
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Average Delay (ms)')
            ax2.set_title('Delay vs Distance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Delay distribution
            ax3 = axes[1, 0]
            ax3.hist(self.df['avg_delay'], bins=20, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Average Delay (ms)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Delay Distribution')
            ax3.grid(True, alpha=0.3)
            
            # 4. Delay by packet size
            ax4 = axes[1, 1]
            packet_delay = self.df.groupby('packet_size')['avg_delay'].agg(['mean', 'std'])
            packet_delay.plot(kind='bar', ax=ax4)
            ax4.set_xlabel('Packet Size (bytes)')
            ax4.set_ylabel('Average Delay (ms)')
            ax4.set_title('Delay by Packet Size')
            ax4.legend(['Mean', 'Std Dev'])
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'delay_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('delay_analysis.png')
            self.logger.info("Created delay analysis plots")
            
        except Exception as e:
            self.logger.error(f"Error creating delay analysis: {e}")
    
    def _create_distance_impact_analysis(self) -> None:
        """Create analysis of distance impact on performance."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Distance Impact Analysis', fontsize=14, fontweight='bold')
            
            # 1. All metrics vs Distance
            ax1 = axes[0, 0]
            distance_stats = self.df.groupby('distance').agg({
                'throughput': 'mean',
                'packet_loss': 'mean',
                'avg_delay': 'mean'
            })
            distance_stats.plot(kind='line', marker='o', ax=ax1)
            ax1.set_xlabel('Distance (m)')
            ax1.set_ylabel('Average Value')
            ax1.set_title('Performance Metrics vs Distance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Distance impact on throughput by traffic rate
            ax2 = axes[0, 1]
            for rate in sorted(self.df['traffic_rate_num'].unique()):
                rate_data = self.df[self.df['traffic_rate_num'] == rate]
                ax2.plot(rate_data['distance'], rate_data['throughput'], 
                        marker='s', label=f'{rate}Mbps')
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Throughput (Mbps)')
            ax2.set_title('Distance Impact on Throughput')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Distance impact on packet loss
            ax3 = axes[1, 0]
            for rate in sorted(self.df['traffic_rate_num'].unique()):
                rate_data = self.df[self.df['traffic_rate_num'] == rate]
                ax3.plot(rate_data['distance'], rate_data['packet_loss'], 
                        marker='^', label=f'{rate}Mbps')
            ax3.set_xlabel('Distance (m)')
            ax3.set_ylabel('Packet Loss (%)')
            ax3.set_title('Distance Impact on Packet Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Distance performance summary
            ax4 = axes[1, 1]
            distance_summary = self.df.groupby('distance').agg({
                'throughput': ['mean', 'std'],
                'packet_loss': ['mean', 'std']
            })
            distance_summary.plot(kind='bar', ax=ax4)
            ax4.set_xlabel('Distance (m)')
            ax4.set_ylabel('Value')
            ax4.set_title('Distance Performance Summary')
            ax4.legend(['Throughput Mean', 'Throughput Std', 'Loss Mean', 'Loss Std'])
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'distance_impact_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('distance_impact_analysis.png')
            self.logger.info("Created distance impact analysis")
            
        except Exception as e:
            self.logger.error(f"Error creating distance impact analysis: {e}")
    
    def _create_speed_impact_analysis(self) -> None:
        """Create analysis of speed impact on performance."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Speed Impact Analysis', fontsize=14, fontweight='bold')
            
            # 1. Performance by speed
            ax1 = axes[0, 0]
            speed_stats = self.df.groupby('speed').agg({
                'throughput': 'mean',
                'packet_loss': 'mean',
                'avg_delay': 'mean'
            })
            speed_stats.plot(kind='bar', ax=ax1)
            ax1.set_xlabel('Speed (m/s)')
            ax1.set_ylabel('Average Value')
            ax1.set_title('Performance by Speed')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Speed impact on throughput by distance
            ax2 = axes[0, 1]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax2.plot(dist_data['speed'], dist_data['throughput'], 
                        marker='o', label=f'Distance: {distance}m')
            ax2.set_xlabel('Speed (m/s)')
            ax2.set_ylabel('Throughput (Mbps)')
            ax2.set_title('Speed Impact on Throughput')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Speed impact on packet loss
            ax3 = axes[1, 0]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax3.plot(dist_data['speed'], dist_data['packet_loss'], 
                        marker='s', label=f'Distance: {distance}m')
            ax3.set_xlabel('Speed (m/s)')
            ax3.set_ylabel('Packet Loss (%)')
            ax3.set_title('Speed Impact on Packet Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Speed performance heatmap
            ax4 = axes[1, 1]
            speed_pivot = self.df.pivot_table(
                values='throughput', 
                index='distance', 
                columns='speed', 
                aggfunc='mean'
            )
            sns.heatmap(speed_pivot, annot=True, fmt='.2f', cmap='Blues', ax=ax4)
            ax4.set_title('Speed Impact Heatmap (Distance vs Speed)')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'speed_impact_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('speed_impact_analysis.png')
            self.logger.info("Created speed impact analysis")
            
        except Exception as e:
            self.logger.error(f"Error creating speed impact analysis: {e}")
    
    def _create_interference_impact_analysis(self) -> None:
        """Create analysis of interference impact on performance."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Interference Impact Analysis', fontsize=14, fontweight='bold')
            
            # 1. Performance by interference level
            ax1 = axes[0, 0]
            interference_stats = self.df.groupby('interferers').agg({
                'throughput': 'mean',
                'packet_loss': 'mean',
                'avg_delay': 'mean'
            })
            interference_stats.plot(kind='bar', ax=ax1)
            ax1.set_xlabel('Number of Interferers')
            ax1.set_ylabel('Average Value')
            ax1.set_title('Performance by Interference Level')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Interference impact on throughput by distance
            ax2 = axes[0, 1]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax2.plot(dist_data['interferers'], dist_data['throughput'], 
                        marker='o', label=f'Distance: {distance}m')
            ax2.set_xlabel('Number of Interferers')
            ax2.set_ylabel('Throughput (Mbps)')
            ax2.set_title('Interference Impact on Throughput')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Interference impact on packet loss
            ax3 = axes[1, 0]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax3.plot(dist_data['interferers'], dist_data['packet_loss'], 
                        marker='s', label=f'Distance: {distance}m')
            ax3.set_xlabel('Number of Interferers')
            ax3.set_ylabel('Packet Loss (%)')
            ax3.set_title('Interference Impact on Packet Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Interference performance heatmap
            ax4 = axes[1, 1]
            interference_pivot = self.df.pivot_table(
                values='throughput', 
                index='distance', 
                columns='interferers', 
                aggfunc='mean'
            )
            sns.heatmap(interference_pivot, annot=True, fmt='.2f', cmap='Greens', ax=ax4)
            ax4.set_title('Interference Impact Heatmap (Distance vs Interferers)')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'interference_impact_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('interference_impact_analysis.png')
            self.logger.info("Created interference impact analysis")
            
        except Exception as e:
            self.logger.error(f"Error creating interference impact analysis: {e}")
    
    def _create_packet_size_impact_analysis(self) -> None:
        """Create analysis of packet size impact on performance."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Packet Size Impact Analysis', fontsize=14, fontweight='bold')
            
            # 1. Performance by packet size
            ax1 = axes[0, 0]
            packet_stats = self.df.groupby('packet_size').agg({
                'throughput': 'mean',
                'packet_loss': 'mean',
                'avg_delay': 'mean'
            })
            packet_stats.plot(kind='bar', ax=ax1)
            ax1.set_xlabel('Packet Size (bytes)')
            ax1.set_ylabel('Average Value')
            ax1.set_title('Performance by Packet Size')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Packet size impact on throughput by distance
            ax2 = axes[0, 1]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax2.plot(dist_data['packet_size'], dist_data['throughput'], 
                        marker='o', label=f'Distance: {distance}m')
            ax2.set_xlabel('Packet Size (bytes)')
            ax2.set_ylabel('Throughput (Mbps)')
            ax2.set_title('Packet Size Impact on Throughput')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Packet size impact on packet loss
            ax3 = axes[1, 0]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax3.plot(dist_data['packet_size'], dist_data['packet_loss'], 
                        marker='s', label=f'Distance: {distance}m')
            ax3.set_xlabel('Packet Size (bytes)')
            ax3.set_ylabel('Packet Loss (%)')
            ax3.set_title('Packet Size Impact on Packet Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Packet size efficiency analysis
            ax4 = axes[1, 1]
            efficiency_by_size = self.df.groupby('packet_size')['throughput_efficiency'].mean()
            ax4.bar(efficiency_by_size.index, efficiency_by_size.values)
            ax4.set_xlabel('Packet Size (bytes)')
            ax4.set_ylabel('Throughput Efficiency')
            ax4.set_title('Throughput Efficiency by Packet Size')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'packet_size_impact_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('packet_size_impact_analysis.png')
            self.logger.info("Created packet size impact analysis")
            
        except Exception as e:
            self.logger.error(f"Error creating packet size impact analysis: {e}")
    
    def _create_traffic_rate_impact_analysis(self) -> None:
        """Create analysis of traffic rate impact on performance."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Traffic Rate Impact Analysis', fontsize=14, fontweight='bold')
            
            # 1. Performance by traffic rate
            ax1 = axes[0, 0]
            rate_stats = self.df.groupby('traffic_rate_num').agg({
                'throughput': 'mean',
                'packet_loss': 'mean',
                'avg_delay': 'mean'
            })
            rate_stats.plot(kind='line', marker='o', ax=ax1)
            ax1.set_xlabel('Traffic Rate (Mbps)')
            ax1.set_ylabel('Average Value')
            ax1.set_title('Performance by Traffic Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Traffic rate impact on throughput by distance
            ax2 = axes[0, 1]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax2.plot(dist_data['traffic_rate_num'], dist_data['throughput'], 
                        marker='o', label=f'Distance: {distance}m')
            ax2.set_xlabel('Traffic Rate (Mbps)')
            ax2.set_ylabel('Throughput (Mbps)')
            ax2.set_title('Traffic Rate Impact on Throughput')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Traffic rate impact on packet loss
            ax3 = axes[1, 0]
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                ax3.plot(dist_data['traffic_rate_num'], dist_data['packet_loss'], 
                        marker='s', label=f'Distance: {distance}m')
            ax3.set_xlabel('Traffic Rate (Mbps)')
            ax3.set_ylabel('Packet Loss (%)')
            ax3.set_title('Traffic Rate Impact on Packet Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Traffic rate efficiency analysis
            ax4 = axes[1, 1]
            efficiency_by_rate = self.df.groupby('traffic_rate_num')['throughput_efficiency'].mean()
            ax4.plot(efficiency_by_rate.index, efficiency_by_rate.values, marker='o')
            ax4.set_xlabel('Traffic Rate (Mbps)')
            ax4.set_ylabel('Throughput Efficiency')
            ax4.set_title('Throughput Efficiency by Traffic Rate')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = (self.results_dir / 'traffic_rate_impact_analysis.png')
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('traffic_rate_impact_analysis.png')
            self.logger.info("Created traffic rate impact analysis")
            
        except Exception as e:
            self.logger.error(f"Error creating traffic rate impact analysis: {e}")
    
    def _create_scenario_heatmap(self) -> None:
        """Create scenario performance heatmap."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(f'{self.protocol_name} - Scenario Performance Heatmaps', fontsize=14, fontweight='bold')
            
            # Throughput heatmap
            throughput_pivot = self.df.pivot_table(
                values='throughput', 
                index='distance', 
                columns='traffic_rate_num', 
                aggfunc='mean'
            )
            sns.heatmap(throughput_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0])
            axes[0].set_title('Throughput (Mbps)')
            
            # Packet loss heatmap
            loss_pivot = self.df.pivot_table(
                values='packet_loss', 
                index='distance', 
                columns='traffic_rate_num', 
                aggfunc='mean'
            )
            sns.heatmap(loss_pivot, annot=True, fmt='.2f', cmap='Reds', ax=axes[1])
            axes[1].set_title('Packet Loss (%)')
            
            # Delay heatmap
            delay_pivot = self.df.pivot_table(
                values='avg_delay', 
                index='distance', 
                columns='traffic_rate_num', 
                aggfunc='mean'
            )
            sns.heatmap(delay_pivot, annot=True, fmt='.3f', cmap='Blues', ax=axes[2])
            axes[2].set_title('Average Delay (ms)')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'scenario_heatmap.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('scenario_heatmap.png')
            self.logger.info("Created scenario heatmap")
            
        except Exception as e:
            self.logger.error(f"Error creating scenario heatmap: {e}")
    
    def _create_performance_distributions(self) -> None:
        """Create performance distribution analysis."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'{self.protocol_name} - Performance Distributions', fontsize=14, fontweight='bold')
            
            # Throughput distribution
            axes[0, 0].hist(self.df['throughput'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Throughput (Mbps)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Throughput Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Packet loss distribution
            axes[0, 1].hist(self.df['packet_loss'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Packet Loss (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Packet Loss Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Delay distribution
            axes[0, 2].hist(self.df['avg_delay'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 2].set_xlabel('Average Delay (ms)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Delay Distribution')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Throughput boxplot by distance
            self.df.boxplot(column='throughput', by='distance', ax=axes[1, 0])
            axes[1, 0].set_xlabel('Distance (m)')
            axes[1, 0].set_ylabel('Throughput (Mbps)')
            axes[1, 0].set_title('Throughput Distribution by Distance')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Packet loss boxplot by traffic rate
            self.df.boxplot(column='packet_loss', by='traffic_rate_num', ax=axes[1, 1])
            axes[1, 1].set_xlabel('Traffic Rate (Mbps)')
            axes[1, 1].set_ylabel('Packet Loss (%)')
            axes[1, 1].set_title('Packet Loss Distribution by Traffic Rate')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Delay boxplot by packet size
            self.df.boxplot(column='avg_delay', by='packet_size', ax=axes[1, 2])
            axes[1, 2].set_xlabel('Packet Size (bytes)')
            axes[1, 2].set_ylabel('Average Delay (ms)')
            axes[1, 2].set_title('Delay Distribution by Packet Size')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = (self.results_dir / 'performance_distributions.png')
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('performance_distributions.png')
            self.logger.info("Created performance distributions")
            
        except Exception as e:
            self.logger.error(f"Error creating performance distributions: {e}")
    
    def _create_correlation_analysis(self) -> None:
        """Create correlation analysis between metrics."""
        try:
            # Select numeric columns for correlation
            numeric_cols = ['distance', 'speed', 'interferers', 'packet_size', 
                           'traffic_rate_num', 'throughput', 'packet_loss', 'avg_delay']
            correlation_data = self.df[numeric_cols].corr()
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax)
            ax.set_title(f'{self.protocol_name} - Metric Correlations')
            
            plt.tight_layout()
            plot_path = (self.results_dir / 'correlation_analysis.png')
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('correlation_analysis.png')
            self.logger.info("Created correlation analysis")
            
        except Exception as e:
            self.logger.error(f"Error creating correlation analysis: {e}")
    
    def _create_efficiency_analysis(self) -> None:
        """Create efficiency analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Efficiency Analysis', fontsize=14, fontweight='bold')
            
            # Throughput efficiency by scenario
            ax1 = axes[0, 0]
                        # Throughput efficiency by scenario
            ax1 = axes[0, 0]
            efficiency_by_scenario = self.df.groupby('scenario_group')['throughput_efficiency'].mean().sort_values(ascending=False)
            efficiency_by_scenario.plot(kind='bar', ax=ax1)
            ax1.set_xlabel('Scenario Group')
            ax1.set_ylabel('Throughput Efficiency')
            ax1.set_title('Throughput Efficiency by Scenario')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Performance vs efficiency scatter
            ax2 = axes[0, 1]
            ax2.scatter(self.df['throughput'], self.df['throughput_efficiency'], alpha=0.6)
            ax2.set_xlabel('Throughput (Mbps)')
            ax2.set_ylabel('Throughput Efficiency')
            ax2.set_title('Throughput vs Efficiency')
            ax2.grid(True, alpha=0.3)
            
            # Efficiency by distance and traffic rate
            ax3 = axes[1, 0]
            efficiency_pivot = self.df.pivot_table(
                values='throughput_efficiency', 
                index='distance', 
                columns='traffic_rate_num', 
                aggfunc='mean'
            )
            sns.heatmap(efficiency_pivot, annot=True, fmt='.3f', cmap='Greens', ax=ax3)
            ax3.set_title('Efficiency Heatmap (Distance vs Traffic Rate)')
            
            # Best and worst performing scenarios
            ax4 = axes[1, 1]
            scenario_performance = self.df.groupby('scenario_group').agg({
                'throughput': 'mean',
                'packet_loss': 'mean',
                'avg_delay': 'mean'
            })
            scenario_performance['performance_score'] = (
                scenario_performance['throughput'] * 0.5 - 
                scenario_performance['packet_loss'] * 0.3 - 
                scenario_performance['avg_delay'] * 0.2
            )
            top_scenarios = scenario_performance.nlargest(5, 'performance_score')
            top_scenarios['performance_score'].plot(kind='bar', ax=ax4)
            ax4.set_xlabel('Scenario Group')
            ax4.set_ylabel('Performance Score')
            ax4.set_title('Top 5 Performing Scenarios')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'efficiency_analysis.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('efficiency_analysis.png')
            self.logger.info("Created efficiency analysis")
            
        except Exception as e:
            self.logger.error(f"Error creating efficiency analysis: {e}")
    
    def _create_statistical_summary_plots(self) -> None:
        """Create statistical summary plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.protocol_name} - Statistical Summary', fontsize=14, fontweight='bold')
            
            # Summary statistics table
            ax1 = axes[0, 0]
            ax1.axis('tight')
            ax1.axis('off')
            summary_stats = self.df[['throughput', 'packet_loss', 'avg_delay']].describe()
            table = ax1.table(cellText=summary_stats.values,
                             rowLabels=summary_stats.index,
                             colLabels=summary_stats.columns,
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax1.set_title('Statistical Summary')
            
            # Performance by distance summary
            ax2 = axes[0, 1]
            distance_summary = self.df.groupby('distance').agg({
                'throughput': ['mean', 'std'],
                'packet_loss': ['mean', 'std']
            })
            distance_summary.plot(kind='bar', ax=ax2)
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Value')
            ax2.set_title('Performance Summary by Distance')
            ax2.legend(['Throughput Mean', 'Throughput Std', 'Loss Mean', 'Loss Std'])
            ax2.grid(True, alpha=0.3)
            
            # Performance by traffic rate summary
            ax3 = axes[1, 0]
            rate_summary = self.df.groupby('traffic_rate_num').agg({
                'throughput': ['mean', 'std'],
                'packet_loss': ['mean', 'std']
            })
            rate_summary.plot(kind='line', marker='o', ax=ax3)
            ax3.set_xlabel('Traffic Rate (Mbps)')
            ax3.set_ylabel('Value')
            ax3.set_title('Performance Summary by Traffic Rate')
            ax3.legend(['Throughput Mean', 'Throughput Std', 'Loss Mean', 'Loss Std'])
            ax3.grid(True, alpha=0.3)
            
            # Performance by interference summary
            ax4 = axes[1, 1]
            interference_summary = self.df.groupby('interferers').agg({
                'throughput': ['mean', 'std'],
                'packet_loss': ['mean', 'std']
            })
            interference_summary.plot(kind='bar', ax=ax4)
            ax4.set_xlabel('Number of Interferers')
            ax4.set_ylabel('Value')
            ax4.set_title('Performance Summary by Interference')
            ax4.legend(['Throughput Mean', 'Throughput Std', 'Loss Mean', 'Loss Std'])
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'statistical_summary_plots.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('statistical_summary_plots.png')
            self.logger.info("Created statistical summary plots")
            
        except Exception as e:
            self.logger.error(f"Error creating statistical summary plots: {e}")
    
    def calculate_comprehensive_statistics(self) -> None:
        """Calculate comprehensive statistical analysis."""
        try:
            self.logger.info("Calculating comprehensive statistics...")
            
            # Basic statistics
            self.analysis_results['basic_stats'] = self.df[['throughput', 'packet_loss', 'avg_delay']].describe()
            
            # Performance by distance
            self.analysis_results['distance_stats'] = self.df.groupby('distance').agg({
                'throughput': ['mean', 'std', 'min', 'max'],
                'packet_loss': ['mean', 'std', 'min', 'max'],
                'avg_delay': ['mean', 'std', 'min', 'max']
            })
            
            # Performance by traffic rate
            self.analysis_results['traffic_rate_stats'] = self.df.groupby('traffic_rate_num').agg({
                'throughput': ['mean', 'std', 'min', 'max'],
                'packet_loss': ['mean', 'std', 'min', 'max'],
                'avg_delay': ['mean', 'std', 'min', 'max']
            })
            
            # Performance by interference
            self.analysis_results['interference_stats'] = self.df.groupby('interferers').agg({
                'throughput': ['mean', 'std', 'min', 'max'],
                'packet_loss': ['mean', 'std', 'min', 'max'],
                'avg_delay': ['mean', 'std', 'min', 'max']
            })
            
            # Performance by packet size
            self.analysis_results['packet_size_stats'] = self.df.groupby('packet_size').agg({
                'throughput': ['mean', 'std', 'min', 'max'],
                'packet_loss': ['mean', 'std', 'min', 'max'],
                'avg_delay': ['mean', 'std', 'min', 'max']
            })
            
            # Performance by speed
            self.analysis_results['speed_stats'] = self.df.groupby('speed').agg({
                'throughput': ['mean', 'std', 'min', 'max'],
                'packet_loss': ['mean', 'std', 'min', 'max'],
                'avg_delay': ['mean', 'std', 'min', 'max']
            })
            
            # Scenario performance ranking
            scenario_performance = self.df.groupby('scenario_group').agg({
                'throughput': 'mean',
                'packet_loss': 'mean',
                'avg_delay': 'mean'
            })
            scenario_performance['performance_score'] = (
                scenario_performance['throughput'] * 0.5 - 
                scenario_performance['packet_loss'] * 0.3 - 
                scenario_performance['avg_delay'] * 0.2
            )
            self.analysis_results['scenario_ranking'] = scenario_performance.sort_values('performance_score', ascending=False)
            
            # Efficiency analysis
            self.analysis_results['efficiency_stats'] = {
                'mean_efficiency': self.df['throughput_efficiency'].mean(),
                'std_efficiency': self.df['throughput_efficiency'].std(),
                'max_efficiency': self.df['throughput_efficiency'].max(),
                'min_efficiency': self.df['throughput_efficiency'].min(),
                'efficiency_by_distance': self.df.groupby('distance')['throughput_efficiency'].mean(),
                'efficiency_by_traffic_rate': self.df.groupby('traffic_rate_num')['throughput_efficiency'].mean()
            }
            
            # Correlation analysis
            numeric_cols = ['distance', 'speed', 'interferers', 'packet_size', 
                           'traffic_rate_num', 'throughput', 'packet_loss', 'avg_delay']
            self.analysis_results['correlations'] = self.df[numeric_cols].corr()
            
            self.logger.info("Comprehensive statistics calculated successfully")
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
    
    def create_excel_report(self) -> None:
        """Create comprehensive Excel report with all analysis results."""
        try:
            excel_path = self.results_dir / f'{self.protocol_name}_comprehensive_analysis.xlsx'
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Basic statistics
                self.analysis_results['basic_stats'].to_excel(writer, sheet_name='Basic_Statistics')
                
                # Distance analysis
                self.analysis_results['distance_stats'].to_excel(writer, sheet_name='Distance_Analysis')
                
                # Traffic rate analysis
                self.analysis_results['traffic_rate_stats'].to_excel(writer, sheet_name='Traffic_Rate_Analysis')
                
                # Interference analysis
                self.analysis_results['interference_stats'].to_excel(writer, sheet_name='Interference_Analysis')
                
                # Packet size analysis
                self.analysis_results['packet_size_stats'].to_excel(writer, sheet_name='Packet_Size_Analysis')
                
                # Speed analysis
                self.analysis_results['speed_stats'].to_excel(writer, sheet_name='Speed_Analysis')
                
                # Scenario ranking
                self.analysis_results['scenario_ranking'].to_excel(writer, sheet_name='Scenario_Ranking')
                
                # Efficiency analysis
                efficiency_df = pd.DataFrame({
                    'Metric': ['Mean Efficiency', 'Std Efficiency', 'Max Efficiency', 'Min Efficiency'],
                    'Value': [
                        self.analysis_results['efficiency_stats']['mean_efficiency'],
                        self.analysis_results['efficiency_stats']['std_efficiency'],
                        self.analysis_results['efficiency_stats']['max_efficiency'],
                        self.analysis_results['efficiency_stats']['min_efficiency']
                    ]
                })
                efficiency_df.to_excel(writer, sheet_name='Efficiency_Analysis', index=False)
                
                # Efficiency by distance
                self.analysis_results['efficiency_stats']['efficiency_by_distance'].to_excel(
                    writer, sheet_name='Efficiency_by_Distance'
                )
                
                # Efficiency by traffic rate
                self.analysis_results['efficiency_stats']['efficiency_by_traffic_rate'].to_excel(
                    writer, sheet_name='Efficiency_by_Traffic_Rate'
                )
                
                # Correlation matrix
                self.analysis_results['correlations'].to_excel(writer, sheet_name='Correlations')
                
                # Raw data
                self.df.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            self.logger.info(f"Excel report created: {excel_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating Excel report: {e}")
    
    def generate_comprehensive_summary(self) -> str:
        """Generate comprehensive human-readable summary."""
        try:
            summary = []
            summary.append("=" * 80)
            summary.append(f"COMPREHENSIVE PROTOCOL ANALYSIS: {self.protocol_name}")
            summary.append("=" * 80)
            summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary.append(f"Total Scenarios: {len(self.df)}")
            summary.append(f"Unique Scenarios: {self.df['scenario_group'].nunique()}")
            summary.append("")
            
            # Overall performance summary
            summary.append("OVERALL PERFORMANCE SUMMARY:")
            summary.append("-" * 40)
            summary.append(f"Average Throughput: {self.df['throughput'].mean():.3f} Mbps")
            summary.append(f"Average Packet Loss: {self.df['packet_loss'].mean():.3f}%")
            summary.append(f"Average Delay: {self.df['avg_delay'].mean():.3f} ms")
            summary.append(f"Average Efficiency: {self.df['throughput_efficiency'].mean():.3f}")
            summary.append("")
            
            # Best and worst scenarios
            summary.append("TOP 5 PERFORMING SCENARIOS:")
            summary.append("-" * 40)
            top_scenarios = self.analysis_results['scenario_ranking'].head()
            for idx, (scenario, row) in enumerate(top_scenarios.iterrows(), 1):
                summary.append(f"{idx}. {scenario}")
                summary.append(f"   Throughput: {row['throughput']:.3f} Mbps")
                summary.append(f"   Packet Loss: {row['packet_loss']:.3f}%")
                summary.append(f"   Delay: {row['avg_delay']:.3f} ms")
                summary.append(f"   Score: {row['performance_score']:.3f}")
                summary.append("")
            
            # Distance analysis
            summary.append("DISTANCE IMPACT ANALYSIS:")
            summary.append("-" * 40)
            for distance in sorted(self.df['distance'].unique()):
                dist_data = self.df[self.df['distance'] == distance]
                summary.append(f"Distance {distance}m:")
                summary.append(f"  Avg Throughput: {dist_data['throughput'].mean():.3f} Mbps")
                summary.append(f"  Avg Packet Loss: {dist_data['packet_loss'].mean():.3f}%")
                summary.append(f"  Avg Delay: {dist_data['avg_delay'].mean():.3f} ms")
            summary.append("")
            
            # Traffic rate analysis
            summary.append("TRAFFIC RATE IMPACT ANALYSIS:")
            summary.append("-" * 40)
            for rate in sorted(self.df['traffic_rate_num'].unique()):
                rate_data = self.df[self.df['traffic_rate_num'] == rate]
                summary.append(f"Traffic Rate {rate}Mbps:")
                summary.append(f"  Avg Throughput: {rate_data['throughput'].mean():.3f} Mbps")
                summary.append(f"  Avg Packet Loss: {rate_data['packet_loss'].mean():.3f}%")
                summary.append(f"  Avg Delay: {rate_data['avg_delay'].mean():.3f} ms")
            summary.append("")
            
            # Interference analysis
            summary.append("INTERFERENCE IMPACT ANALYSIS:")
            summary.append("-" * 40)
            for interferers in sorted(self.df['interferers'].unique()):
                intf_data = self.df[self.df['interferers'] == interferers]
                summary.append(f"Interferers: {interferers}")
                summary.append(f"  Avg Throughput: {intf_data['throughput'].mean():.3f} Mbps")
                summary.append(f"  Avg Packet Loss: {intf_data['packet_loss'].mean():.3f}%")
                summary.append(f"  Avg Delay: {intf_data['avg_delay'].mean():.3f} ms")
            summary.append("")
            
            # Key insights
            summary.append("KEY INSIGHTS:")
            summary.append("-" * 40)
            
            # Best distance
            best_distance = self.df.groupby('distance')['throughput'].mean().idxmax()
            summary.append(f"• Best performing distance: {best_distance}m")
            
            # Best traffic rate
            best_rate = self.df.groupby('traffic_rate_num')['throughput'].mean().idxmax()
            summary.append(f"• Best performing traffic rate: {best_rate}Mbps")
            
            # Most efficient scenario
            most_efficient = self.df.loc[self.df['throughput_efficiency'].idxmax()]
            summary.append(f"• Most efficient scenario: {most_efficient['scenario_group']}")
            summary.append(f"  (Throughput: {most_efficient['throughput']:.3f} Mbps, Efficiency: {most_efficient['throughput_efficiency']:.3f})")
            
            # Performance variability
            throughput_cv = self.df['throughput'].std() / self.df['throughput'].mean()
            summary.append(f"• Throughput variability (CV): {throughput_cv:.3f}")
            
            summary.append("")
            summary.append("=" * 80)
            summary.append("ANALYSIS COMPLETED SUCCESSFULLY")
            summary.append(f"Results saved to: {self.results_dir}")
            summary.append(f"Total plots generated: {len(self.plots_generated)}")
            summary.append("=" * 80)
            
            return "\n".join(summary)
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"
    
    def run_complete_analysis(self) -> bool:
        """Run complete comprehensive analysis."""
        try:
            self.logger.info("Starting comprehensive single protocol analysis...")
            
            # Load data
            if not self.load_data():
                return False
            
            # Calculate comprehensive statistics
            self.calculate_comprehensive_statistics()
            self.logger.info("Statistical analysis completed")
            
            # Generate comprehensive plots
            self.generate_comprehensive_plots()
            self.logger.info(f"Generated {len(self.plots_generated)} plots")
            
            # Create Excel report
            self.create_excel_report()
            self.logger.info("Excel report generated")
            
            # Generate and save summary
            summary = self.generate_comprehensive_summary()
            summary_path = self.results_dir / 'comprehensive_summary.txt'
            
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            self.logger.info(f"Summary saved to: {summary_path}")
            print(summary)
            
            self.logger.info("=" * 80)
            self.logger.info("COMPREHENSIVE SINGLE PROTOCOL ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info(f"Results directory: {self.results_dir}")
            self.logger.info(f"Total plots generated: {len(self.plots_generated)}")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return False


def main():
    """Main function to run the comprehensive single protocol analysis."""
    
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Comprehensive Single Protocol Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python protocol_analysis.py --protocol aarf
  python protocol_analysis.py --protocol smartv3 --name "SmartV3"
  python protocol_analysis.py -p smartv1 -n "SmartV1"
  python protocol_analysis.py --list-protocols
        """
    )
    
    parser.add_argument(
        '--protocol', '-p',
        type=str,
        default='aarf',
        help='Protocol name to analyze (default: aarf)'
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        default=None,
        help='Custom display name for protocol (auto-detected if not provided)'
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
    
    # Convert protocol name to CSV filename
    protocol_mapping = {
        'aarf': 'aarf-benchmark.csv',
        'smartv1': 'smartv1-benchmark.csv',
        'smartv2': 'smartv2-benchmark.csv', 
        'smartv3': 'smartv3-benchmark.csv',
        'smartrf': 'smartrf-benchmark-oracle.csv',
        'smartrfv3': 'smartrf-benchmark-v3.csv'
    }
    
    protocol_csv = protocol_mapping.get(args.protocol.lower())
    
    if not protocol_csv:
        print(f"❌ ERROR: Unknown protocol '{args.protocol}'. Use --list-protocols to see available options.")
        return False
    
    # Optional: Specify custom name (auto-detected from filename if None)
    protocol_name = args.name
    
    print("=" * 80)
    print("COMPREHENSIVE SINGLE PROTOCOL ANALYZER")
    print("=" * 80)
    print(f"Protocol CSV: {protocol_csv}")
    print()
    
    # Create analyzer instance
    try:
        analyzer = ComprehensiveSingleProtocolAnalyzer(
            protocol_csv=protocol_csv,
            protocol_name=protocol_name
        )
        
        print(f"Analyzing: {analyzer.protocol_name}")
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
            print("• 14 comprehensive analysis plots (.png files)")
            print("• Detailed Excel report with multiple sheets")
            print("• Comprehensive text summary report")
            print("• Analysis log file")
            print("\nKey features:")
            print("• Performance dashboard with 6 subplots")
            print("• Throughput, packet loss, and delay analysis")
            print("• Distance, speed, interference, and packet size impact analysis")
            print("• Traffic rate impact and efficiency analysis")
            print("• Scenario heatmaps and performance distributions")
            print("• Correlation analysis and statistical summaries")
        else:
            print("\n" + "=" * 80)
            print("ANALYSIS FAILED!")
            print("=" * 80)
            print("Check the log file for details about the error.")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPlease check:")
        print("1. CSV file path is correct")
        print("2. CSV file exists in the specified location")
        print("3. CSV file has the required columns")
        return False
    
    return True


if __name__ == "__main__":
    main()