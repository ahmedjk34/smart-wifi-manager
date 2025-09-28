#!/usr/bin/env python3
"""
Debug SmartRF & AARF Performance Issues
Analyzes benchmark results to identify root causes of poor performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_dataset(name, filename, output_prefix):
    try:
        df = pd.read_csv(filename)

        print(f"\n🔍 {name.upper()} PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Distance analysis
        print("\n📏 Distance Performance:")
        for dist in [20, 40, 60]:
            dist_data = df[df['Distance'] == dist]
            if not dist_data.empty:
                avg_throughput = dist_data['Throughput(Mbps)'].mean()
                avg_loss = dist_data['PacketLoss(%)'].mean()
                print(f"  {dist}m: {avg_throughput:.2f} Mbps, {avg_loss:.1f}% loss")

        # Interferer analysis
        print("\n📡 Interferer Impact:")
        for intf in [0, 3]:
            intf_data = df[df['Interferers'] == intf]
            if not intf_data.empty:
                avg_throughput = intf_data['Throughput(Mbps)'].mean()
                avg_loss = intf_data['PacketLoss(%)'].mean()
                print(f"  {intf} interferers: {avg_throughput:.2f} Mbps, {avg_loss:.1f}% loss")

        # Rate adaptation analysis (only if column exists)
        if "RateChanges" in df.columns:
            print("\n🔄 Rate Adaptation:")
            avg_changes = df["RateChanges"].mean()
            print(f"  Average rate changes: {avg_changes:.1f}")
        else:
            print("\n🔄 Rate Adaptation: N/A (not available in this dataset)")

        # # Plot performance
        # fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # # Throughput vs Distance
        # df.groupby("Distance")["Throughput(Mbps)"].mean().plot(kind="bar", ax=axes[0,0])
        # axes[0,0].set_title("Throughput vs Distance")

        # # Loss vs Distance
        # df.groupby("Distance")["PacketLoss(%)"].mean().plot(kind="bar", ax=axes[0,1])
        # axes[0,1].set_title("Packet Loss vs Distance")

        # # Throughput vs Interferers
        # df.groupby("Interferers")["Throughput(Mbps)"].mean().plot(kind="bar", ax=axes[1,0])
        # axes[1,0].set_title("Throughput vs Interferers")

        # # Rate Changes vs Performance (if available)
        # if "RateChanges" in df.columns:
        #     axes[1,1].scatter(df["RateChanges"], df["Throughput(Mbps)"])
        #     axes[1,1].set_title("Rate Changes vs Throughput")
        #     axes[1,1].set_xlabel("Rate Changes")
        #     axes[1,1].set_ylabel("Throughput (Mbps)")
        # else:
        #     axes[1,1].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        #     axes[1,1].set_title("Rate Changes vs Throughput")

        # plt.tight_layout()
        # output_file = f"{output_prefix}_debug_analysis.png"
        # plt.savefig(output_file)
        # plt.close(fig)

        # print(f"\n📊 Analysis plot saved to: {output_file}")

    except Exception as e:
        print(f"❌ {name} analysis failed: {e}")


def analyze_performance_issues():
    analyze_dataset("SmartRF", "smartrf-benchmark-results.csv", "smartrf")
    analyze_dataset("AARF", "aarf-benchmark.csv", "aarf")


if __name__ == "__main__":
    analyze_performance_issues()
