"""
ConfBench Evaluator

Evaluates predicted structures against ground truth.
"""
import os
import glob
from typing import Dict, List, Optional
from .utils import (
    parse_holo_id,
    calculate_pairwise_rmsds,
    calculate_confbench_score,
)


class Evaluator:
    """
    Evaluate predicted structures against ground truth.
    
    Calculates ConfBench scores based on the formula:
    Score = (RMSD(Pred, Apo) - RMSD(Pred, Holo)) / sqrt(0.5 * (RMSD_pa^2 + RMSD_ph^2 + RMSD_ah^2))
    """
    
    def __init__(self, pocket_distance_cutoff: float = 10.0):
        """
        Initialize the evaluator.
        
        Args:
            pocket_distance_cutoff: Distance cutoff for pocket definition (Angstroms)
        """
        self.pocket_distance_cutoff = pocket_distance_cutoff
    
    def calculate_scores(
        self,
        pred_path: str,
        holo_path: str,
        apo_path: str,
        ligand_files: List[str],
        holo_id: Optional[str] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calculate ConfBench scores for a single prediction.
        
        Args:
            pred_path: Path to predicted structure file (.cif, .pdb, .xyz)
            holo_path: Path to ground truth holo structure
            apo_path: Path to apo structure
            ligand_files: List of ligand file paths (.sdf)
            holo_id: Optional holo ID for extracting chain information
            
        Returns:
            Dictionary with scores and intermediate RMSD values:
            {
                'global_score': float,
                'pocket_ca_score': float,
                'pocket_all_score': float,
                'rmsd_pred_holo_global': float,
                'rmsd_pred_holo_pocket_ca': float,
                'rmsd_pred_holo_pocket_all': float,
                'rmsd_pred_apo_global': float,
                'rmsd_pred_apo_pocket_ca': float,
                'rmsd_pred_apo_pocket_all': float,
                'rmsd_apo_holo_global': float,
                'rmsd_apo_holo_pocket_ca': float,
                'rmsd_apo_holo_pocket_all': float,
            }
        """
        results = {
            'global_score': None,
            'pocket_ca_score': None,
            'pocket_all_score': None,
            'rmsd_pred_holo_global': None,
            'rmsd_pred_holo_pocket_ca': None,
            'rmsd_pred_holo_pocket_all': None,
            'rmsd_pred_apo_global': None,
            'rmsd_pred_apo_pocket_ca': None,
            'rmsd_pred_apo_pocket_all': None,
            'rmsd_apo_holo_global': None,
            'rmsd_apo_holo_pocket_ca': None,
            'rmsd_apo_holo_pocket_all': None,
            'status': 'success',
            'message': ''
        }
        
        # Validate input files
        if not os.path.exists(pred_path):
            results['status'] = 'error'
            results['message'] = f'Prediction file not found: {pred_path}'
            return results
        
        if not os.path.exists(holo_path):
            results['status'] = 'error'
            results['message'] = f'Holo file not found: {holo_path}'
            return results
        
        if not os.path.exists(apo_path):
            results['status'] = 'error'
            results['message'] = f'Apo file not found: {apo_path}'
            return results
        
        if not ligand_files:
            results['status'] = 'error'
            results['message'] = 'No ligand files provided'
            return results
        
        # Extract holo chain IDs if available
        holo_chain_ids = None
        if holo_id:
            _, protein_chains, _ = parse_holo_id(holo_id)
            holo_chain_ids = protein_chains if protein_chains else None
        
        # Calculate all pairwise RMSDs
        try:
            rmsds = calculate_pairwise_rmsds(
                holo_path=holo_path,
                apo_path=apo_path,
                pred_path=pred_path,
                ligand_files=ligand_files,
                pocket_distance_cutoff=self.pocket_distance_cutoff,
                holo_chain_ids=holo_chain_ids
            )
        except Exception as e:
            results['status'] = 'error'
            results['message'] = f'RMSD calculation failed: {str(e)[:100]}'
            return results
        
        # Copy RMSD values to results
        for key in rmsds:
            results[key] = rmsds[key]
        
        # Calculate scores
        results['global_score'] = calculate_confbench_score(
            rmsds['rmsd_pred_apo_global'],
            rmsds['rmsd_pred_holo_global'],
            rmsds['rmsd_apo_holo_global']
        )
        
        results['pocket_ca_score'] = calculate_confbench_score(
            rmsds['rmsd_pred_apo_pocket_ca'],
            rmsds['rmsd_pred_holo_pocket_ca'],
            rmsds['rmsd_apo_holo_pocket_ca']
        )
        
        results['pocket_all_score'] = calculate_confbench_score(
            rmsds['rmsd_pred_apo_pocket_all'],
            rmsds['rmsd_pred_holo_pocket_all'],
            rmsds['rmsd_apo_holo_pocket_all']
        )
        
        # Check if scores were calculated successfully
        if all(v is None for v in [results['global_score'], results['pocket_ca_score'], results['pocket_all_score']]):
            results['status'] = 'error'
            results['message'] = 'Failed to calculate scores'
        
        return results
    
    def find_prediction_file(
        self,
        predictions_dir: str,
        holo_id: str,
        extensions: List[str] = ['.cif', '.pdb', '.xyz']
    ) -> Optional[str]:
        """
        Find prediction file for a given holo_id.
        
        Searches for files matching the pattern: {holo_id}.{ext} or {holo_id}-*.{ext}
        
        Args:
            predictions_dir: Directory containing prediction files
            holo_id: Holo system ID
            extensions: List of file extensions to search for
            
        Returns:
            Path to prediction file, or None if not found
        """
        for ext in extensions:
            # Exact match
            path = os.path.join(predictions_dir, f"{holo_id}{ext}")
            if os.path.exists(path):
                return path
            
            # Pattern match (e.g., holo_id-0-rmsd0.45.cif)
            pattern = os.path.join(predictions_dir, f"{holo_id}*{ext}")
            matches = glob.glob(pattern)
            if matches:
                # Return first match (or you could implement sorting logic)
                return matches[0]
        
        return None
