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

class ProtocolComparisonAnalyzer:
    """Complete side-by-side comparison analyzer for two Wi-Fi protocols."""
    
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
        self.results_dir = self.base_dir / 'test_side_by_side' / f'{self.protocol1_name}_vs_{self.protocol2_name}'
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
        log_file = self.results_dir / 'comparison_analysis.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True
        )
        
        logger.info(f"=== Protocol Comparison: {self.protocol1_name} vs {self.protocol2_name} ===")
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
        
        # Convert traffic_rate to numeric
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
    
    def create_comparison_plots(self) -> None:
        """Create comprehensive comparison plots."""
        if self.combined_df is None:
            logger.warning("No data available for plotting")
            return
        
        logger.info("Creating comparison plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall throughput comparison
        self._create_metric_comparison_plot('throughput', 'Throughput (Mbps)')
        
        # 2. Packet loss comparison
        self._create_metric_comparison_plot('packet_loss', 'Packet Loss (%)')
        
        # 3. Delay comparison
        self._create_metric_comparison_plot('avg_delay', 'Average Delay (ms)')
        
        # 4. Throughput vs Distance
        self._create_grouped_comparison_plot('distance', 'throughput', 
                                           'Distance (m)', 'Throughput (Mbps)',
                                           'throughput_vs_distance')
        
        # 5. Throughput vs Traffic Rate
        self._create_grouped_comparison_plot('traffic_rate', 'throughput',
                                           'Traffic Rate', 'Throughput (Mbps)',
                                           'throughput_vs_traffic_rate')
        
        # 6. Throughput vs Speed
        self._create_grouped_comparison_plot('speed', 'throughput',
                                           'Speed (m/s)', 'Throughput (Mbps)',
                                           'throughput_vs_speed')
        
        # 7. Throughput vs Interferers
        self._create_grouped_comparison_plot('interferers', 'throughput',
                                           'Number of Interferers', 'Throughput (Mbps)',
                                           'throughput_vs_interferers')
        
        # 8. Multi-metric comparison heatmap
        self._create_heatmap_comparison()
        
        # 9. Correlation matrix
        self._create_correlation_comparison()
        
        # 10. Performance distribution plots
        self._create_distribution_plots()
    
    def _create_metric_comparison_plot(self, metric: str, ylabel: str) -> None:
        """Create a comparison plot for a specific metric."""
        if metric not in self.combined_df.columns:
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Box plot
            sns.boxplot(data=self.combined_df, x='Protocol', y=metric, ax=ax1)
            ax1.set_title(f'{ylabel} Distribution by Protocol')
            ax1.set_ylabel(ylabel)
            
            # Violin plot
            sns.violinplot(data=self.combined_df, x='Protocol', y=metric, ax=ax2)
            ax2.set_title(f'{ylabel} Distribution (Violin)')
            ax2.set_ylabel(ylabel)
            
            plt.tight_layout()
            plot_path = self.results_dir / f'{metric}_comparison.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append(f'{metric}_comparison.png')
            logger.info(f"Created {metric} comparison plot")
            
        except Exception as e:
            logger.error(f"Error creating {metric} plot: {e}")
    
    def _create_grouped_comparison_plot(self, group_var: str, metric: str, 
                                      xlabel: str, ylabel: str, filename: str) -> None:
        """Create a grouped comparison plot."""
        if group_var not in self.combined_df.columns or metric not in self.combined_df.columns:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sns.boxplot(data=self.combined_df, x=group_var, y=metric, hue='Protocol', ax=ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs {xlabel}: {self.protocol1_name} vs {self.protocol2_name}')
            ax.legend(title='Protocol')
            
            if group_var in ['traffic_rate']:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = self.results_dir / f'{filename}.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append(f'{filename}.png')
            logger.info(f"Created {filename} plot")
            
        except Exception as e:
            logger.error(f"Error creating {filename} plot: {e}")
    
    def _create_heatmap_comparison(self) -> None:
        """Create a heatmap showing mean performance across scenarios."""
        try:
            # Create pivot tables for each protocol
            metrics = ['throughput', 'packet_loss', 'avg_delay']
            
            for metric in metrics:
                if metric not in self.combined_df.columns:
                    continue
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                
                # Protocol 1 heatmap
                pivot1 = self.combined_df[self.combined_df['Protocol'] == self.protocol1_name].pivot_table(
                    values=metric, index='distance', columns='traffic_rate', aggfunc='mean'
                )
                sns.heatmap(pivot1, annot=True, fmt='.2f', cmap='viridis', ax=ax1)
                ax1.set_title(f'{self.protocol1_name} - {metric.replace("_", " ").title()}')
                
                # Protocol 2 heatmap
                pivot2 = self.combined_df[self.combined_df['Protocol'] == self.protocol2_name].pivot_table(
                    values=metric, index='distance', columns='traffic_rate', aggfunc='mean'
                )
                sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='viridis', ax=ax2)
                ax2.set_title(f'{self.protocol2_name} - {metric.replace("_", " ").title()}')
                
                # Difference heatmap
                diff = pivot1 - pivot2
                sns.heatmap(diff, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax3)
                ax3.set_title(f'Difference ({self.protocol1_name} - {self.protocol2_name})')
                
                plt.tight_layout()
                plot_path = self.results_dir / f'{metric}_heatmap_comparison.png'
                fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
                plt.close(fig)
                
                self.plots_generated.append(f'{metric}_heatmap_comparison.png')
            
            logger.info("Created heatmap comparisons")
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
    
    def _create_correlation_comparison(self) -> None:
        """Create correlation matrix comparison."""
        try:
            numeric_cols = ['distance', 'speed', 'interferers', 'packet_size', 
                          'traffic_rate_num', 'throughput', 'packet_loss', 'avg_delay']
            available_cols = [col for col in numeric_cols if col in self.combined_df.columns]
            
            if len(available_cols) < 3:
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Protocol 1 correlation
            corr1 = self.df1[available_cols].corr()
            sns.heatmap(corr1, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax1)
            ax1.set_title(f'{self.protocol1_name} - Variable Correlations')
            
            # Protocol 2 correlation
            corr2 = self.df2[available_cols].corr()
            sns.heatmap(corr2, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax2)
            ax2.set_title(f'{self.protocol2_name} - Variable Correlations')
            
            plt.tight_layout()
            plot_path = self.results_dir / 'correlation_comparison.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('correlation_comparison.png')
            logger.info("Created correlation comparison")
            
        except Exception as e:
            logger.error(f"Error creating correlation plot: {e}")
    
    def _create_distribution_plots(self) -> None:
        """Create distribution comparison plots."""
        try:
            metrics = ['throughput', 'packet_loss', 'avg_delay']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                if metric not in self.combined_df.columns:
                    continue
                
                # Histogram
                ax1 = axes[i]
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    data = self.combined_df[self.combined_df['Protocol'] == protocol][metric]
                    ax1.hist(data, alpha=0.7, label=protocol, bins=30)
                ax1.set_xlabel(metric.replace('_', ' ').title())
                ax1.set_ylabel('Frequency')
                ax1.set_title(f'{metric.replace("_", " ").title()} Distribution')
                ax1.legend()
                
                # CDF
                ax2 = axes[i + 3]
                for protocol in [self.protocol1_name, self.protocol2_name]:
                    data = self.combined_df[self.combined_df['Protocol'] == protocol][metric]
                    sorted_data = np.sort(data)
                    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    ax2.plot(sorted_data, y, label=protocol, linewidth=2)
                ax2.set_xlabel(metric.replace('_', ' ').title())
                ax2.set_ylabel('Cumulative Probability')
                ax2.set_title(f'{metric.replace("_", " ").title()} CDF')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.results_dir / 'distribution_comparison.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            
            self.plots_generated.append('distribution_comparison.png')
            logger.info("Created distribution comparison")
            
        except Exception as e:
            logger.error(f"Error creating distribution plots: {e}")
    
    def generate_human_readable_summary(self) -> str:
        """Generate human-readable comparison summary."""
        if not self.comparison_results:
            return "No comparison results available."
        
        summary = []
        summary.append(f"\n{'='*80}")
        summary.append(f"PROTOCOL COMPARISON SUMMARY")
        summary.append(f"{self.protocol1_name} vs {self.protocol2_name}")
        summary.append(f"{'='*80}")
        summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"User: ahmedjk34")
        summary.append("")
        
        # Overall performance comparison
        if 'overall' in self.comparison_results:
            summary.append("OVERALL PERFORMANCE COMPARISON")
            summary.append("-" * 50)
            
            overall = self.comparison_results['overall']
            
            # Throughput comparison
            if 'throughput' in overall:
                th_stats = overall['throughput']
                p1_mean = th_stats.loc[self.protocol1_name, 'mean']
                p2_mean = th_stats.loc[self.protocol2_name, 'mean']
                
                if p1_mean > p2_mean:
                    winner = self.protocol1_name
                    improvement = ((p1_mean - p2_mean) / p2_mean) * 100
                else:
                    winner = self.protocol2_name
                    improvement = ((p2_mean - p1_mean) / p1_mean) * 100
                
                summary.append(f"THROUGHPUT WINNER: {winner}")
                summary.append(f"   • {self.protocol1_name}: {p1_mean:.3f} Mbps (avg)")
                summary.append(f"   • {self.protocol2_name}: {p2_mean:.3f} Mbps (avg)")
                summary.append(f"   • Improvement: {improvement:.1f}% better")
                summary.append("")
            
            # Packet Loss comparison
            if 'packet_loss' in overall:
                pl_stats = overall['packet_loss']
                p1_mean = pl_stats.loc[self.protocol1_name, 'mean']
                p2_mean = pl_stats.loc[self.protocol2_name, 'mean']
                
                if p1_mean < p2_mean:
                    winner = self.protocol1_name
                    improvement = ((p2_mean - p1_mean) / p2_mean) * 100
                else:
                    winner = self.protocol2_name
                    improvement = ((p1_mean - p2_mean) / p1_mean) * 100
                
                summary.append(f"PACKET LOSS WINNER: {winner}")
                summary.append(f"   • {self.protocol1_name}: {p1_mean:.3f}% loss (avg)")
                summary.append(f"   • {self.protocol2_name}: {p2_mean:.3f}% loss (avg)")
                summary.append(f"   • Improvement: {improvement:.1f}% better")
                summary.append("")
            
            # Delay comparison
            if 'avg_delay' in overall:
                dl_stats = overall['avg_delay']
                p1_mean = dl_stats.loc[self.protocol1_name, 'mean']
                p2_mean = dl_stats.loc[self.protocol2_name, 'mean']
                
                if p1_mean < p2_mean:
                    winner = self.protocol1_name
                    improvement = ((p2_mean - p1_mean) / p2_mean) * 100
                else:
                    winner = self.protocol2_name
                    improvement = ((p1_mean - p2_mean) / p1_mean) * 100
                
                summary.append(f"DELAY WINNER: {winner}")
                summary.append(f"   • {self.protocol1_name}: {p1_mean:.3f} ms (avg)")
                summary.append(f"   • {self.protocol2_name}: {p2_mean:.3f} ms (avg)")
                summary.append(f"   • Improvement: {improvement:.1f}% better")
                summary.append("")
        
        # Scenario-specific analysis
        summary.append("SCENARIO-SPECIFIC ANALYSIS")
        summary.append("-" * 50)
        
        scenarios_analyzed = []
        
        # Analyze by distance
        if 'by_distance' in self.comparison_results:
            distance_data = self.comparison_results['by_distance']
            if 'throughput' in distance_data:
                th_by_dist = distance_data['throughput']
                
                summary.append("Performance by Distance:")
                for distance in sorted(th_by_dist['distance'].unique()):
                    dist_data = th_by_dist[th_by_dist['distance'] == distance]
                    p1_perf = dist_data[dist_data['Protocol'] == self.protocol1_name]['mean'].iloc[0] if len(dist_data[dist_data['Protocol'] == self.protocol1_name]) > 0 else 0
                    p2_perf = dist_data[dist_data['Protocol'] == self.protocol2_name]['mean'].iloc[0] if len(dist_data[dist_data['Protocol'] == self.protocol2_name]) > 0 else 0
                    
                    if p1_perf > p2_perf:
                        summary.append(f"   • {distance}m: {self.protocol1_name} wins ({p1_perf:.2f} vs {p2_perf:.2f} Mbps)")
                    elif p2_perf > p1_perf:
                        summary.append(f"   • {distance}m: {self.protocol2_name} wins ({p2_perf:.2f} vs {p1_perf:.2f} Mbps)")
                    else:
                        summary.append(f"   • {distance}m: Tie ({p1_perf:.2f} Mbps)")
                summary.append("")
        
        # Analyze by traffic rate
        if 'by_traffic_rate' in self.comparison_results:
            traffic_data = self.comparison_results['by_traffic_rate']
            if 'throughput' in traffic_data:
                th_by_traffic = traffic_data['throughput']
                
                summary.append("Performance by Traffic Rate:")
                for rate in sorted(th_by_traffic['traffic_rate'].unique()):
                    rate_data = th_by_traffic[th_by_traffic['traffic_rate'] == rate]
                    p1_perf = rate_data[rate_data['Protocol'] == self.protocol1_name]['mean'].iloc[0] if len(rate_data[rate_data['Protocol'] == self.protocol1_name]) > 0 else 0
                    p2_perf = rate_data[rate_data['Protocol'] == self.protocol2_name]['mean'].iloc[0] if len(rate_data[rate_data['Protocol'] == self.protocol2_name]) > 0 else 0
                    
                    if p1_perf > p2_perf:
                        summary.append(f"   • {rate}: {self.protocol1_name} wins ({p1_perf:.2f} vs {p2_perf:.2f} Mbps)")
                    elif p2_perf > p1_perf:
                        summary.append(f"   • {rate}: {self.protocol2_name} wins ({p2_perf:.2f} vs {p1_perf:.2f} Mbps)")
                    else:
                        summary.append(f"   • {rate}: Tie ({p1_perf:.2f} Mbps)")
                summary.append("")
        
        # Key insights
        summary.append("KEY INSIGHTS")
        summary.append("-" * 50)
        
        # Count wins by scenario
        p1_wins = 0
        p2_wins = 0
        ties = 0
        
        if 'by_scenario' in self.comparison_results:
            for scenario, data in self.comparison_results['by_scenario'].items():
                if 'throughput' in data and len(data['throughput']) >= 2:
                    try:
                        p1_th = data['throughput'].loc[self.protocol1_name, 'mean']
                        p2_th = data['throughput'].loc[self.protocol2_name, 'mean']
                        
                        if p1_th > p2_th:
                            p1_wins += 1
                        elif p2_th > p1_th:
                            p2_wins += 1
                        else:
                            ties += 1
                    except:
                        pass
        
        total_scenarios = p1_wins + p2_wins + ties
        if total_scenarios > 0:
            summary.append(f"Overall Scenario Performance:")
            summary.append(f"   • {self.protocol1_name}: {p1_wins}/{total_scenarios} scenarios won ({(p1_wins/total_scenarios)*100:.1f}%)")
            summary.append(f"   • {self.protocol2_name}: {p2_wins}/{total_scenarios} scenarios won ({(p2_wins/total_scenarios)*100:.1f}%)")
            summary.append(f"   • Ties: {ties}/{total_scenarios} scenarios ({(ties/total_scenarios)*100:.1f}%)")
            summary.append("")
        
        # Recommendations
        summary.append("RECOMMENDATIONS")
        summary.append("-" * 50)
        
        if p1_wins > p2_wins:
            summary.append(f"{self.protocol1_name} shows superior overall performance")
            summary.append(f"   • Consider using {self.protocol1_name} for production deployment")
            summary.append(f"   • {self.protocol2_name} may need further optimization")
        elif p2_wins > p1_wins:
            summary.append(f"{self.protocol2_name} shows superior overall performance")
            summary.append(f"   • Consider using {self.protocol2_name} for production deployment")
            summary.append(f"   • {self.protocol1_name} may need further optimization")
        else:
            summary.append("Both protocols show similar performance")
            summary.append("   • Choice may depend on specific use case requirements")
            summary.append("   • Consider other factors like complexity, power consumption, etc.")
        
        summary.append("")
        summary.append("Detailed results and plots available in:")
        summary.append(f"   {self.results_dir}")
        summary.append(f"{'='*80}")
        
        return "\n".join(summary)
    
    def export_to_excel(self) -> None:
        """Export comprehensive comparison results to Excel."""
        excel_path = self.results_dir / f'{self.protocol1_name}_vs_{self.protocol2_name}_comparison.xlsx'
        
        try:
            logger.info(f"Exporting comparison results to Excel: {excel_path}")
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Combined raw data
                self.combined_df.to_excel(writer, sheet_name='Combined_Data', index=False)
                
                # Protocol 1 data
                self.df1.to_excel(writer, sheet_name=f'{self.protocol1_name}_Data', index=False)
                
                # Protocol 2 data
                self.df2.to_excel(writer, sheet_name=f'{self.protocol2_name}_Data', index=False)
                
                # Overall statistics
                if 'overall' in self.comparison_results:
                    for metric, stats in self.comparison_results['overall'].items():
                        stats.to_excel(writer, sheet_name=f'Overall_{metric.title()}')
                
                # Statistics by grouping variables
                for key, data in self.comparison_results.items():
                    if key.startswith('by_') and key != 'by_scenario':
                        var_name = key[3:]
                        for metric, stats in data.items():
                            sheet_name = f'{var_name.title()}_{metric.title()}'[:31]  # Excel sheet name limit
                            stats.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Analysis summary
                summary_data = pd.DataFrame({
                    'Analysis_Info': [
                        'Protocol 1', 'Protocol 2', 'Total Records P1', 'Total Records P2',
                        'Analysis Date', 'Plots Generated', 'User'
                    ],
                    'Value': [
                        self.protocol1_name, self.protocol2_name,
                        len(self.df1), len(self.df2),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        len(self.plots_generated), 'ahmedjk34'
                    ]
                })
                summary_data.to_excel(writer, sheet_name='Analysis_Summary', index=False)
            
            # Apply formatting
            self._format_excel_file(excel_path)
            logger.info(f"Excel file exported successfully: {excel_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
    
    def _format_excel_file(self, excel_path: Path) -> None:
        """Apply formatting to Excel file."""
        try:
            wb = load_workbook(excel_path)
            
            # Define styles
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                
                if ws.max_row > 0:
                    # Format headers
                    for cell in ws[1]:
                        cell.font = header_font
                        cell.fill = header_fill
                        cell.alignment = Alignment(horizontal='center')
                    
                    # Auto-fit columns
                    for column in ws.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if cell.value:
                                    max_length = max(max_length, len(str(cell.value)))
                            except:
                                pass
                        
                        adjusted_width = min(max_length + 2, 50)
                        ws.column_dimensions[column_letter].width = adjusted_width
            
            wb.save(excel_path)
            logger.info("Excel formatting applied successfully")
            
        except Exception as e:
            logger.error(f"Error formatting Excel file: {e}")
    
    def run_complete_comparison(self) -> None:
        """Run the complete comparison analysis pipeline."""
        logger.info(f"=== Starting Complete Protocol Comparison ===")
        logger.info(f"Comparing: {self.protocol1_name} vs {self.protocol2_name}")
        
        # Load and preprocess data
        if not self.load_data():
            logger.error("Comparison failed - could not load data")
            return
        
        self.preprocess_data()
        
        # Calculate statistics
        self.calculate_statistical_comparison()
        
        # Generate plots
        self.create_comparison_plots()
        
        # Export to Excel
        self.export_to_excel()
        
        # Generate and display human-readable summary
        summary = self.generate_human_readable_summary()
        print(summary)
        
        # Save summary to file - FIX: Use UTF-8 encoding
        summary_path = self.results_dir / 'comparison_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Final summary
        logger.info("=== Comparison Analysis Complete ===")
        logger.info(f"Results saved in: {self.results_dir}")
        logger.info(f"Plots generated: {len(self.plots_generated)}")
        logger.info(f"Files created:")
        logger.info(f"  - Excel: {self.protocol1_name}_vs_{self.protocol2_name}_comparison.xlsx")
        logger.info(f"  - Summary: comparison_summary.txt")
        logger.info(f"  - Plots: {len(self.plots_generated)} PNG files")
        logger.info(f"  - Log: comparison_analysis.log")
        
        print(f"\n{'='*80}")
        print(f"COMPARISON ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Results directory: {self.results_dir}")
        print(f"Excel file: {self.protocol1_name}_vs_{self.protocol2_name}_comparison.xlsx")
        print(f"Plots generated: {len(self.plots_generated)}")
        print(f"Summary file: comparison_summary.txt")
        print(f"Log file: comparison_analysis.log")
        print(f"{'='*80}")


def main():
    """Main function to run the comparison analysis."""
    try:
        # CSV files in parent directory
        protocol1_csv = "aarf-benchmark.csv"
        protocol2_csv = "smartxgb-benchmark.csv"
        
        analyzer = ProtocolComparisonAnalyzer(protocol1_csv, protocol2_csv)
        analyzer.run_complete_comparison()
        
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()