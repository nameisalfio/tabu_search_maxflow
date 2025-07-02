#!/usr/bin/env python3
"""
Script per eseguire esperimenti su tutti i network
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import Config
from main import run_multiple_experiments

def main():
    """Esegue esperimenti su tutti i network nella directory data/networks"""
    
    config = Config()
    
    # Trova tutti i file network
    networks_dir = Path("data/networks")
    network_files = list(networks_dir.glob("network_*.txt"))
    
    if not network_files:
        print("❌ No network files found in data/networks/")
        print("   Please add network files with pattern: network_*.txt")
        return
    
    print(f"🔬 Found {len(network_files)} network files")
    print("🚀 Starting batch experiments...")
    
    all_results = {}
    
    for i, network_file in enumerate(network_files):
        print(f"\n{'='*60}")
        print(f"🎯 Processing network {i+1}/{len(network_files)}: {network_file.name}")
        print(f"{'='*60}")
        
        try:
            results = run_multiple_experiments(str(network_file), config)
            if results:
                all_results[network_file.stem] = results
                print(f"✅ Completed {network_file.name}")
            else:
                print(f"❌ Failed {network_file.name}")
                
        except Exception as e:
            print(f"❌ Error processing {network_file.name}: {e}")
            continue
    
    # Save aggregate results
    if all_results:
        output_dir = Path("data/results")
        output_dir.mkdir(exist_ok=True)
        
        # Save complete results
        results_file = output_dir / "batch_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Batch results saved to: {results_file}")
        
        # Create summary
        print(f"\n📊 BATCH SUMMARY:")
        print(f"{'Network':<15} {'Best':<8} {'Mean':<8} {'Std':<6} {'Time(s)':<8}")
        print("-" * 50)
        
        for network_name, result in all_results.items():
            stats = result['statistics']
            print(f"{network_name:<15} {stats['best']:<8} {stats['mean']:<8.1f} "
                  f"{stats['std']:<6.2f} {stats['mean_execution_time']:<8.1f}")
        
        # Generate comparison plot if multiple networks
        if len(all_results) > 1:
            try:
                from src.visualization.plotter import FlowPlotter
                
                plotter = FlowPlotter(
                    theme=config.get('visualization.theme', 'dark'),
                    figsize=(15, 10)
                )
                
                # Prepare data for comparison plot
                comparison_data = {}
                for network_name, result in all_results.items():
                    stats = result['statistics']
                    comparison_data[network_name] = {
                        'best': stats['best'],
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'execution_time': stats['mean_execution_time']
                    }
                
                plotter.plot_statistics_summary(
                    comparison_data,
                    save_path=str(output_dir / "batch_comparison.png") if config.get('visualization.save_plots') else None,
                    show=config.get('visualization.show_plots', False)
                )
                
                print(f"📊 Comparison plot saved to: {output_dir / 'batch_comparison.png'}")
                
            except Exception as e:
                print(f"⚠️  Could not generate comparison plot: {e}")
    
    else:
        print("\n❌ No successful experiments!")
    
    print(f"\n✅ Batch processing completed!")

if __name__ == "__main__":
    main()
