"""
ConfBench Benchmark Runner

Main benchmark class for running evaluation on all pairs.
"""
import os
import glob
import pandas as pd
from typing import Optional, List
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from .evaluator import Evaluator


class ConfBenchmark:
    """
    Main benchmark runner for ConfBench.
    
    Evaluates structure predictions against ground truth holo structures,
    computing ConfBench scores for each pair.
    """
    
    def __init__(
        self,
        confbench_data_dir: str,
        predictions_dir: str,
        pocket_distance_cutoff: float = 10.0,
        prediction_target: str = 'holo'
    ):
        """
        Initialize the benchmark.
        
        Args:
            confbench_data_dir: Path to confbench_data directory created by prepare_confbench_data.py
            predictions_dir: Path to directory containing predicted structures
            pocket_distance_cutoff: Distance cutoff for pocket definition (Angstroms)
            prediction_target: 'holo' or 'apo' - what the model is predicting
        """
        self.data_dir = confbench_data_dir
        self.predictions_dir = predictions_dir
        self.pocket_distance_cutoff = pocket_distance_cutoff
        self.prediction_target = prediction_target
        self.evaluator = Evaluator(pocket_distance_cutoff)
        
        # Load metadata
        self.metadata_path = os.path.join(confbench_data_dir, 'metadata.csv')
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        self.metadata = pd.read_csv(self.metadata_path)
    
    def apply_filters(
        self,
        lddt_threshold: Optional[float] = None,
        max_holo_resolution: Optional[float] = None,
        max_apo_resolution: Optional[float] = None,
        min_rmsd: Optional[float] = None,
        filter_crystal_contacts: bool = False
    ) -> pd.DataFrame:
        """
        Apply filtering conditions to metadata.
        
        Args:
            lddt_threshold: Minimum lddt value (default: None, no filter)
            max_holo_resolution: Maximum holo resolution (default: None, no filter)
            max_apo_resolution: Maximum apo resolution (default: None, no filter)
            min_rmsd: Minimum RMSD threshold - at least one of 3 RMSDs must exceed this (default: None)
            filter_crystal_contacts: If True, filter out systems with crystal contacts (default: False)
            
        Returns:
            Filtered DataFrame
        """
        df = self.metadata.copy()
        
        if lddt_threshold is not None and 'lddt' in df.columns:
            df = df[df['lddt'] >= lddt_threshold]
        
        if max_holo_resolution is not None and 'holo_resolution' in df.columns:
            # Keep rows where resolution is NaN or <= threshold
            mask = df['holo_resolution'].isna() | (df['holo_resolution'] <= max_holo_resolution)
            df = df[mask]
        
        if max_apo_resolution is not None and 'apo_resolution' in df.columns:
            mask = df['apo_resolution'].isna() | (df['apo_resolution'] <= max_apo_resolution)
            df = df[mask]
        
        if min_rmsd is not None:
            # At least one RMSD must exceed threshold
            mask = (
                (df['global_rmsd'] > min_rmsd) |
                (df['pocket_ca_rmsd'] > min_rmsd) |
                (df['pocket_all_rmsd'] > min_rmsd)
            )
            df = df[mask]
        
        if filter_crystal_contacts:
            col_name = 'system_num_atoms_with_crystal_contacts'
            if col_name in df.columns:
                before_count = len(df)
                df = df[df[col_name] == 0]
                after_count = len(df)
                print(f"Crystal contact filter: {before_count} -> {after_count} ({before_count - after_count} removed)")
            else:
                print(f"Warning: Column '{col_name}' not found in metadata. Crystal contact filter skipped.")
        
        return df.reset_index(drop=True)
    
    def _get_structure_paths(self, holo_id: str, apo_id: str):
        """Get paths to holo, apo, and ligand files."""
        holo_dir = os.path.join(self.data_dir, 'holo', holo_id)
        
        # Holo structure
        holo_cif = os.path.join(holo_dir, 'receptor.cif')
        holo_pdb = os.path.join(holo_dir, 'receptor.pdb')
        holo_path = holo_cif if os.path.exists(holo_cif) else (holo_pdb if os.path.exists(holo_pdb) else None)
        
        # Apo structure
        apo_path = os.path.join(self.data_dir, 'apo', f'{apo_id}.cif')
        if not os.path.exists(apo_path):
            apo_path = None
        
        # Ligand files
        ligand_dir = os.path.join(holo_dir, 'ligand_files')
        ligand_files = glob.glob(os.path.join(ligand_dir, '*.sdf')) if os.path.exists(ligand_dir) else []
        
        return holo_path, apo_path, ligand_files
    
    def _evaluate_single_pair(self, args):
        """Evaluate a single holo-apo pair (for multiprocessing)."""
        holo_id, apo_id = args
        
        # Get structure paths
        holo_path, apo_path, ligand_files = self._get_structure_paths(holo_id, apo_id)
        
        if holo_path is None:
            return {'holo_id': holo_id, 'apo_id': apo_id, 'status': 'error', 'message': 'Holo not found'}
        
        if apo_path is None:
            return {'holo_id': holo_id, 'apo_id': apo_id, 'status': 'error', 'message': 'Apo not found'}
        
        if not ligand_files:
            return {'holo_id': holo_id, 'apo_id': apo_id, 'status': 'error', 'message': 'No ligands'}
        
        # Find prediction file based on prediction_target
        if self.prediction_target == 'holo':
            # Model predicts holo: prediction files named by holo_id
            pred_path = self.evaluator.find_prediction_file(self.predictions_dir, holo_id)
        else:
            # Model predicts apo: prediction files named by apo_id
            pred_path = self.evaluator.find_prediction_file(self.predictions_dir, apo_id)
        
        if pred_path is None:
            search_id = holo_id if self.prediction_target == 'holo' else apo_id
            return {'holo_id': holo_id, 'apo_id': apo_id, 'status': 'error', 'message': f'Prediction not found ({search_id})'}
        
        # Calculate scores
        try:
            results = self.evaluator.calculate_scores(
                pred_path=pred_path,
                holo_path=holo_path,
                apo_path=apo_path,
                ligand_files=ligand_files,
                holo_id=holo_id,
                prediction_target=self.prediction_target
            )
            results['holo_id'] = holo_id
            results['apo_id'] = apo_id
            results['pred_file'] = os.path.basename(pred_path)
            return results
        except Exception as e:
            return {'holo_id': holo_id, 'apo_id': apo_id, 'status': 'error', 'message': f'Exception: {str(e)[:50]}'}
    
    def run(
        self,
        output_csv: str,
        lddt_threshold: Optional[float] = 0.6,
        max_holo_resolution: Optional[float] = 4.5,
        max_apo_resolution: Optional[float] = 4.5,
        min_rmsd: Optional[float] = 1.5,
        filter_crystal_contacts: bool = False,
        n_pairs: Optional[int] = None,
        n_workers: int = 1
    ) -> pd.DataFrame:
        """
        Run the benchmark.
        
        Args:
            output_csv: Path to save results CSV
            lddt_threshold: Minimum lddt value (default: 0.6)
            max_holo_resolution: Maximum holo resolution (default: 4.5)
            max_apo_resolution: Maximum apo resolution (default: 4.5)
            min_rmsd: Minimum RMSD threshold (default: 1.5)
            filter_crystal_contacts: If True, filter out systems with crystal contacts (default: False)
            n_pairs: Optional limit on number of pairs to evaluate
            n_workers: Number of parallel workers (default: 1)
            
        Returns:
            DataFrame with benchmark results
        """
        # Apply filters
        filtered_df = self.apply_filters(
            lddt_threshold=lddt_threshold,
            max_holo_resolution=max_holo_resolution,
            max_apo_resolution=max_apo_resolution,
            min_rmsd=min_rmsd,
            filter_crystal_contacts=filter_crystal_contacts
        )
        
        print(f"Total pairs in metadata: {len(self.metadata)}")
        print(f"Pairs after filtering: {len(filtered_df)}")
        
        if n_pairs is not None:
            filtered_df = filtered_df.head(n_pairs)
            print(f"Pairs to evaluate (limited): {len(filtered_df)}")
        
        # Prepare arguments
        pair_args = [(row['holo_id'], row['apo_id']) for _, row in filtered_df.iterrows()]
        
        # Run evaluation
        results_list = []
        
        if n_workers == 1:
            # Single-threaded
            for args in tqdm(pair_args, desc="Evaluating"):
                results_list.append(self._evaluate_single_pair(args))
        else:
            # Multi-threaded
            with Pool(n_workers) as pool:
                results_list = list(tqdm(
                    pool.imap(self._evaluate_single_pair, pair_args),
                    total=len(pair_args),
                    desc="Evaluating"
                ))
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Reorder columns
        column_order = [
            'holo_id', 'apo_id', 'pred_file',
            'global_score', 'pocket_ca_score', 'pocket_all_score',
            'rmsd_pred_holo_global', 'rmsd_pred_holo_pocket_ca', 'rmsd_pred_holo_pocket_all',
            'rmsd_pred_apo_global', 'rmsd_pred_apo_pocket_ca', 'rmsd_pred_apo_pocket_all',
            'rmsd_apo_holo_global', 'rmsd_apo_holo_pocket_ca', 'rmsd_apo_holo_pocket_all',
            'status', 'message'
        ]
        existing_cols = [c for c in column_order if c in results_df.columns]
        results_df = results_df[existing_cols]
        
        # Save results
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
        
        # Print summary
        success_df = results_df[results_df['status'] == 'success']
        if len(success_df) > 0:
            print(f"\n=== Summary ===")
            print(f"Total evaluated: {len(results_df)}")
            print(f"Successful: {len(success_df)} ({100*len(success_df)/len(results_df):.1f}%)")
            print(f"\nMean scores:")
            print(f"  Global Score: {success_df['global_score'].mean():.3f}")
            print(f"  Pocket CA Score: {success_df['pocket_ca_score'].mean():.3f}")
            print(f"  Pocket All Score: {success_df['pocket_all_score'].mean():.3f}")
        
        return results_df
