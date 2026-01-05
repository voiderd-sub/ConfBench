#!/usr/bin/env python3
"""
Run ConfBench Benchmark

Evaluates structure predictions against ground truth.
"""
import argparse
from confbench import ConfBenchmark


def main():
    parser = argparse.ArgumentParser(
        description='Run ConfBench benchmark on predicted structures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--confbench-data', type=str, required=True,
                        help='Path to confbench_data directory (from prepare_confbench_data.py)')
    parser.add_argument('--predictions-dir', type=str, required=True,
                        help='Path to directory containing predicted structures')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Path to save results CSV')
    
    # Filtering options
    parser.add_argument('--lddt-threshold', type=float, default=0.6,
                        help='Minimum lddt value for filtering')
    parser.add_argument('--max-holo-resolution', type=float, default=4.5,
                        help='Maximum holo resolution (Angstroms)')
    parser.add_argument('--max-apo-resolution', type=float, default=4.5,
                        help='Maximum apo resolution (Angstroms)')
    parser.add_argument('--min-rmsd', type=float, default=1.5,
                        help='Minimum RMSD threshold - at least one of 3 RMSDs must exceed this')
    parser.add_argument('--filter-crystal-contacts', action='store_true',
                        help='Filter out systems with crystal contacts (system_num_atoms_with_crystal_contacts >= 1)')
    
    # Other options
    parser.add_argument('--pocket-distance-cutoff', type=float, default=10.0,
                        help='Distance cutoff for pocket definition (Angstroms)')
    parser.add_argument('--prediction-target', type=str, default='holo', choices=['holo', 'apo'],
                        help='What structure the model predicts: holo or apo (default: holo)')
    parser.add_argument('--n-pairs', type=int, default=None,
                        help='Optional: limit number of pairs to evaluate')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of parallel workers')
    
    # No filter options
    parser.add_argument('--no-lddt-filter', action='store_true',
                        help='Disable lddt filtering')
    parser.add_argument('--no-resolution-filter', action='store_true',
                        help='Disable resolution filtering')
    parser.add_argument('--no-rmsd-filter', action='store_true',
                        help='Disable RMSD filtering')
    
    args = parser.parse_args()
    
    # Process filter options
    lddt_threshold = None if args.no_lddt_filter else args.lddt_threshold
    max_holo_resolution = None if args.no_resolution_filter else args.max_holo_resolution
    max_apo_resolution = None if args.no_resolution_filter else args.max_apo_resolution
    min_rmsd = None if args.no_rmsd_filter else args.min_rmsd
    filter_crystal_contacts = args.filter_crystal_contacts
    
    print("=" * 60)
    print("ConfBench Benchmark")
    print("=" * 60)
    print(f"\nConfBench data: {args.confbench_data}")
    print(f"Predictions: {args.predictions_dir}")
    print(f"Output: {args.output_csv}")
    print(f"\nFiltering options:")
    print(f"  lddt threshold: {lddt_threshold}")
    print(f"  max holo resolution: {max_holo_resolution}")
    print(f"  max apo resolution: {max_apo_resolution}")
    print(f"  min RMSD: {min_rmsd}")
    print(f"  filter crystal contacts: {filter_crystal_contacts}")
    print(f"\nPrediction target: {args.prediction_target}")
    print(f"Pocket distance cutoff: {args.pocket_distance_cutoff}")
    print(f"Number of workers: {args.n_workers}")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = ConfBenchmark(
        confbench_data_dir=args.confbench_data,
        predictions_dir=args.predictions_dir,
        pocket_distance_cutoff=args.pocket_distance_cutoff,
        prediction_target=args.prediction_target
    )
    
    # Run benchmark
    results_df = benchmark.run(
        output_csv=args.output_csv,
        lddt_threshold=lddt_threshold,
        max_holo_resolution=max_holo_resolution,
        max_apo_resolution=max_apo_resolution,
        min_rmsd=min_rmsd,
        filter_crystal_contacts=filter_crystal_contacts,
        n_pairs=args.n_pairs,
        n_workers=args.n_workers
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
