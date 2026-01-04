import pandas as pd
import numpy as np
import os
import glob
import re
import argparse
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pymol import cmd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, MMCIFParser, Selection
from Bio.PDB.Polypeptide import three_to_one
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

# ===== Configuration (will be set via arguments) =====
PLINDER_BASE = None
SYSTEMS_DIR = None
LINKED_STRUCTURES_DIR = None
LINKS_FILE = None
POCKET_DISTANCE_CUTOFF = None
OUTPUT_DIR = None
ANNOTATION_TABLE = None

# ===== Resolution lookup (will be populated at runtime) =====
PDB_RESOLUTION_MAP = {}  # pdb_id -> resolution


def safe_three_to_one(resname):
    """
    Safely convert 3-letter amino acid code to 1-letter code.
    Returns 'X' for non-standard amino acids.
    """
    try:
        return three_to_one(resname)
    except KeyError:
        return 'X'


# ===== ID Parsing Functions =====

def parse_holo_id(holo_id):
    """
    Parse holo_id to extract PDB ID, protein chain IDs, and ligand chain IDs.
    
    Format: {pdb_id}__{number}__{protein_chains}__{ligand_chains}
    Example: 182l__1__1.A__1.E
             -> pdb_id='182l', protein_chains=['1.A'], ligand_chains=['1.E']
    
    Note: CIF files use the full format '1.A' as chain IDs
    """
    parts = holo_id.split('__')
    
    if len(parts) < 4:
        return None, [], []
    
    pdb_id = parts[0]
    
    # Parse protein chains (format: 1.A or 1.A_1.B_2.A)
    protein_part = parts[2]
    protein_chains = []
    for segment in protein_part.split('_'):
        if segment:
            protein_chains.append(segment)
    
    # Parse ligand chains (format: 1.E or 1.E_1.F)
    ligand_part = parts[3]
    ligand_chains = []
    for segment in ligand_part.split('_'):
        if segment:
            ligand_chains.append(segment)
    
    return pdb_id, protein_chains, ligand_chains


def parse_apo_id(apo_id):
    """
    Parse apo_id to extract PDB ID and chain ID.
    
    Format: {pdb_id}_{chain}
    Example: 3hh4_A -> pdb_id='3hh4', chain='A'
    """
    if '_' not in apo_id:
        return None, None
    
    parts = apo_id.rsplit('_', 1)
    if len(parts) != 2:
        return None, None
    
    return parts[0], parts[1]


def get_resolution(pdb_id):
    """Get resolution for a PDB ID from the preloaded map."""
    return PDB_RESOLUTION_MAP.get(pdb_id, None)


# ===== Helper Functions =====

def get_holo_apo_pairs(n_pairs=None):
    """Get holo-apo pairs from the parquet file."""
    df = pd.read_parquet(LINKS_FILE, engine='pyarrow')
    
    # Sort by reference_system_id, then by pocket_fident, lddt, bb_lddt (descending)
    df_sorted = df.sort_values(
        by=['reference_system_id', 'pocket_fident', 'lddt', 'bb_lddt'],
        ascending=[True, False, False, False]
    )
    
    # Get first row per group
    df_top = df_sorted.groupby('reference_system_id', as_index=False).first()
    
    # Select and rename columns
    df_result = df_top[['reference_system_id', 'id']].rename(
        columns={'reference_system_id': 'holo_id', 'id': 'apo_id'}
    )
    
    # Sort by holo_id
    df_result = df_result.sort_values(by='holo_id').reset_index(drop=True)
    
    if n_pairs is not None:
        df_result = df_result.head(n_pairs)
    
    return df_result


def get_structure_paths(holo_id, apo_id):
    """Get file paths for holo and apo structures."""
    _, _, ligand_chains = parse_holo_id(holo_id)
    
    # Holo structure path - prefer CIF (has correct chain IDs) over PDB
    holo_dir = os.path.join(SYSTEMS_DIR, holo_id)
    holo_cif = os.path.join(holo_dir, 'receptor.cif')
    holo_pdb = os.path.join(holo_dir, 'receptor.pdb')
    
    if os.path.exists(holo_cif):
        holo_path = holo_cif
    elif os.path.exists(holo_pdb):
        holo_path = holo_pdb
    else:
        holo_path = None
    
    # Apo structure path
    apo_cif = os.path.join(LINKED_STRUCTURES_DIR, f'{apo_id}.cif')
    apo_path = apo_cif if os.path.exists(apo_cif) else None
    
    # Ligand files - only get files matching ligand chains from holo_id
    ligand_dir = os.path.join(holo_dir, 'ligand_files')
    ligand_files = []
    
    if os.path.exists(ligand_dir) and ligand_chains:
        for ligand_chain in ligand_chains:
            patterns = [
                f"*.{ligand_chain}.sdf",
                f"{ligand_chain}.sdf",
            ]
            for pattern in patterns:
                matches = glob.glob(os.path.join(ligand_dir, pattern))
                ligand_files.extend(matches)
        
        ligand_files = list(set(ligand_files))
        
        if not ligand_files:
            ligand_files = glob.glob(os.path.join(ligand_dir, '*.sdf'))
    elif os.path.exists(ligand_dir):
        ligand_files = glob.glob(os.path.join(ligand_dir, '*.sdf'))
    
    return holo_path, apo_path, ligand_files


def extract_sequence_from_structure(structure, target_chain_ids=None):
    """Extract amino acid sequence from a structure."""
    sequence = ""
    residue_list = []
    
    for model in structure:
        for chain in model:
            if target_chain_ids is not None and chain.id not in target_chain_ids:
                continue
            
            for residue in chain:
                if residue.id[0] == ' ':
                    one_letter = safe_three_to_one(residue.resname)
                    if one_letter != 'X':
                        sequence += one_letter
                        residue_list.append(residue)
    
    return sequence, residue_list


def get_residue_key(residue):
    """Get a unique key for a residue (chain_id, resseq, icode)."""
    return (residue.parent.id, residue.id[1], residue.id[2])


def build_selection_string(residue_keys):
    """Build PyMOL selection string from a list of residue keys."""
    if not residue_keys:
        return "none"
    
    # Group by chain
    chain_resis = {}
    for chain, resnum, icode in residue_keys:
        if chain not in chain_resis:
            chain_resis[chain] = []
        
        # Handle insertion code
        resi_str = str(resnum)
        if icode and icode.strip():
             resi_str += icode
        chain_resis[chain].append(resi_str)
        
    parts = []
    for chain, resis in chain_resis.items():
        # Separate integers (for range optimization) and strings
        ints = []
        strs = []
        for r in resis:
            try:
                ints.append(int(r))
            except ValueError:
                strs.append(r)
        
        # Optimize integer ranges (e.g., 1,2,3 -> 1-3)
        ints = sorted(list(set(ints)))
        ranges = []
        if ints:
            start = ints[0]
            prev = ints[0]
            for x in ints[1:]:
                if x == prev + 1:
                    prev = x
                else:
                    if start == prev:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{prev}")
                    start = x
                    prev = x
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
        
        final_resis = ranges + strs
        resi_part = "+".join(final_resis)
        # Use quotes for chain ID to handle special characters like dot
        parts.append(f"(chain \"{chain}\" and resi {resi_part})")
        
    return "(" + " or ".join(parts) + ")"


def create_residue_mapping(holo_residues, apo_residues):
    """Create 1:1 mapping between holo and apo residues using LOCAL sequence alignment."""
    holo_seq = ""
    for res in holo_residues:
        holo_seq += safe_three_to_one(res.resname)
    
    apo_seq = ""
    for res in apo_residues:
        apo_seq += safe_three_to_one(res.resname)
    
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10 
    aligner.extend_gap_score = -0.5
    
    alignments = aligner.align(holo_seq, apo_seq)
    if len(alignments) == 0:
        return {}
    
    alignment = alignments[0]
    mapping = {}
    aligned_holo, aligned_apo = alignment.aligned
    
    for (h_start, h_end), (a_start, a_end) in zip(aligned_holo, aligned_apo):
        for h_pos, a_pos in zip(range(h_start, h_end), range(a_start, a_end)):
            if h_pos < len(holo_residues) and a_pos < len(apo_residues):
                holo_key = get_residue_key(holo_residues[h_pos])
                apo_key = get_residue_key(apo_residues[a_pos])
                mapping[holo_key] = apo_key
    
    return mapping


def get_ligand_coords(ligand_files):
    """Extract coordinates from SDF ligand files using PyMOL."""
    all_coords = []
    
    for ligand_file in ligand_files:
        cmd.delete('all')
        try:
            cmd.load(ligand_file, 'ligand')
            model = cmd.get_model('ligand')
            for atom in model.atom:
                all_coords.append([atom.coord[0], atom.coord[1], atom.coord[2]])
        except Exception:
            pass
        finally:
            cmd.delete('all')
    
    return np.array(all_coords) if all_coords else None


def find_pocket_residues(structure, ligand_coords, cutoff=10.0, target_chain_ids=None):
    """Find residues within cutoff distance of ligand."""
    if ligand_coords is None or len(ligand_coords) == 0:
        return []
    
    pocket_residues = set()
    
    for model in structure:
        for chain in model:
            if target_chain_ids is not None and chain.id not in target_chain_ids:
                continue
            
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                
                for atom in residue:
                    atom_coord = atom.coord
                    distances = np.linalg.norm(ligand_coords - atom_coord, axis=1)
                    if np.min(distances) <= cutoff:
                        pocket_residues.add(get_residue_key(residue))
                        break
    
    return pocket_residues


def calculate_rmsd_with_pymol(holo_path, apo_path, selection_holo, selection_apo):
    """Calculate RMSD using PyMOL's align command with cycles=0."""
    cmd.delete('all')
    
    try:
        cmd.load(holo_path, 'holo')
        cmd.load(apo_path, 'apo')
        
        result = cmd.align(selection_holo, selection_apo, cycles=0)
        
        rmsd = result[0]
        n_atoms = result[1]
        
        cmd.delete('all')
        
        if n_atoms == 0:
            return None, 0
        
        return rmsd, n_atoms
    
    except Exception:
        cmd.delete('all')
        return None, 0


def calculate_all_rmsds(holo_id, apo_id, holo_path, apo_path, ligand_files):
    """Calculate all three types of RMSD for a holo-apo pair."""
    results = {
        'holo_id': holo_id,
        'apo_id': apo_id,
        'global_rmsd': None,
        'pocket_ca_rmsd': None,
        'pocket_all_rmsd': None,
        'status': 'success',
        'message': ''
    }
    
    _, holo_protein_chains, _ = parse_holo_id(holo_id)
    holo_chains = holo_protein_chains if holo_protein_chains else None
    
    # Load structures
    if holo_path.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    
    try:
        holo_structure = parser.get_structure('holo', holo_path)
    except Exception as e:
        results['status'] = 'error'
        results['message'] = f'Failed to load holo: {str(e)[:50]}'
        return results
    
    try:
        apo_structure = MMCIFParser(QUIET=True).get_structure('apo', apo_path)
    except Exception as e:
        results['status'] = 'error'
        results['message'] = f'Failed to load apo: {str(e)[:50]}'
        return results
    
    # Get ligand coordinates
    ligand_coords = get_ligand_coords(ligand_files)
    if ligand_coords is None or len(ligand_coords) == 0:
        results['status'] = 'error'
        results['message'] = 'No ligand coordinates'
        return results
    
    # Find pocket residues
    holo_pocket_keys = find_pocket_residues(
        holo_structure, ligand_coords, POCKET_DISTANCE_CUTOFF, target_chain_ids=holo_chains
    )
    if len(holo_pocket_keys) == 0:
        results['status'] = 'error'
        results['message'] = 'No pocket residues'
        return results
    
    # Extract residue lists
    _, holo_residues = extract_sequence_from_structure(holo_structure, target_chain_ids=holo_chains)
    _, apo_residues = extract_sequence_from_structure(apo_structure, target_chain_ids=None)  # No filter for apo
    
    # [FIX] Create FULL residue mapping (Full Sequence vs Full Sequence)
    full_mapping = create_residue_mapping(holo_residues, apo_residues)
    
    # [FIX] Extract only pocket residues from the full mapping
    residue_mapping = {}
    for h_key in holo_pocket_keys:
        if h_key in full_mapping:
            residue_mapping[h_key] = full_mapping[h_key]
    
    # Check mapping ratio
    mapped_pocket_keys = [key for key in holo_pocket_keys if key in residue_mapping]
    mapping_ratio = len(mapped_pocket_keys) / len(holo_pocket_keys) if len(holo_pocket_keys) > 0 else 0
    
    if mapping_ratio < 0.5:
        results['status'] = 'skipped'
        results['message'] = f'Low mapping: {len(mapped_pocket_keys)}/{len(holo_pocket_keys)}'
        return results

    # 1. Global RMSD (Calculate using ONLY mapped residues from the full mapping)
    holo_global_keys = list(full_mapping.keys())
    apo_global_keys = list(full_mapping.values())
    
    holo_sele_str = build_selection_string(holo_global_keys)
    apo_sele_str = build_selection_string(apo_global_keys)
    
    holo_global_sele = f"holo and {holo_sele_str} and name CA"
    apo_global_sele = f"apo and {apo_sele_str} and name CA"

    rmsd, n_atoms = calculate_rmsd_with_pymol(holo_path, apo_path, holo_global_sele, apo_global_sele)
    results['global_rmsd'] = rmsd
    
    if rmsd is None or n_atoms == 0:
        results['status'] = 'error'
        results['message'] = 'Global alignment failed'
        return results
    
    # Build pocket selections
    holo_pocket_keys_mapped = mapped_pocket_keys
    apo_pocket_keys_mapped = [residue_mapping[k] for k in mapped_pocket_keys]
    
    holo_pocket_str = build_selection_string(holo_pocket_keys_mapped)
    apo_pocket_str = build_selection_string(apo_pocket_keys_mapped)
    
    if holo_pocket_str == "none" or apo_pocket_str == "none":
        results['status'] = 'error'
        results['message'] = 'Could not create pocket selections'
        return results
    
    holo_pocket_sele = f"holo and {holo_pocket_str}"
    apo_pocket_sele = f"apo and {apo_pocket_str}"
    
    # 2. Pocket CA RMSD
    rmsd, n_atoms = calculate_rmsd_with_pymol(
        holo_path, apo_path,
        f"{holo_pocket_sele} and name CA",
        f"{apo_pocket_sele} and name CA"
    )
    results['pocket_ca_rmsd'] = rmsd
    
    if rmsd is None or n_atoms == 0:
        results['status'] = 'error'
        results['message'] = 'Pocket CA alignment failed'
        return results
    
    # 3. Pocket All-atom RMSD
    rmsd, n_atoms = calculate_rmsd_with_pymol(
        holo_path, apo_path,
        f"{holo_pocket_sele} and not elem H",
        f"{apo_pocket_sele} and not elem H"
    )
    results['pocket_all_rmsd'] = rmsd
    
    if rmsd is None or n_atoms == 0:
        results['status'] = 'error'
        results['message'] = 'Pocket all-atom alignment failed'
        return results
    
    results['message'] = f'{len(mapped_pocket_keys)} pocket residues'
    
    return results


def process_single_pair(args):
    """Worker function to process a single holo-apo pair."""
    holo_id, apo_id = args
    
    # Get resolution for holo and apo
    holo_pdb_id = holo_id.split('__')[0]
    apo_pdb_id = apo_id.rsplit('_', 1)[0]
    holo_resolution = get_resolution(holo_pdb_id)
    apo_resolution = get_resolution(apo_pdb_id)
    
    # Initialize PyMOL for this process (if needed)
    try:
        cmd.reinitialize()
    except Exception:
        pass
    
    # Get structure paths
    holo_path, apo_path, ligand_files = get_structure_paths(holo_id, apo_id)
    
    if holo_path is None:
        return {
            'holo_id': holo_id, 'apo_id': apo_id,
            'holo_resolution': holo_resolution, 'apo_resolution': apo_resolution,
            'global_rmsd': None, 'pocket_ca_rmsd': None, 'pocket_all_rmsd': None,
            'status': 'error', 'message': 'Holo not found'
        }
    
    if apo_path is None:
        return {
            'holo_id': holo_id, 'apo_id': apo_id,
            'holo_resolution': holo_resolution, 'apo_resolution': apo_resolution,
            'global_rmsd': None, 'pocket_ca_rmsd': None, 'pocket_all_rmsd': None,
            'status': 'error', 'message': 'Apo not found'
        }
    
    if not ligand_files:
        return {
            'holo_id': holo_id, 'apo_id': apo_id,
            'holo_resolution': holo_resolution, 'apo_resolution': apo_resolution,
            'global_rmsd': None, 'pocket_ca_rmsd': None, 'pocket_all_rmsd': None,
            'status': 'error', 'message': 'No ligands'
        }
    
    # Calculate RMSDs
    try:
        results = calculate_all_rmsds(holo_id, apo_id, holo_path, apo_path, ligand_files)
        results['holo_resolution'] = holo_resolution
        results['apo_resolution'] = apo_resolution
    except Exception as e:
        results = {
            'holo_id': holo_id, 'apo_id': apo_id,
            'holo_resolution': holo_resolution, 'apo_resolution': apo_resolution,
            'global_rmsd': None, 'pocket_ca_rmsd': None, 'pocket_all_rmsd': None,
            'status': 'error', 'message': f'Exception: {str(e)[:50]}'
        }
    
    return results


def main():
    """Main function to calculate RMSDs for all holo-apo pairs."""
    parser = argparse.ArgumentParser(description='Calculate RMSD between holo and apo structures')
    
    # Configuration arguments (required)
    parser.add_argument('--plinder-base', type=str, required=True, help='Base directory for Plinder')
    parser.add_argument('--systems-dir', type=str, required=True, help='Directory containing system structures')
    parser.add_argument('--linked-structures-dir', type=str, required=True, help='Directory containing linked structures')
    parser.add_argument('--links-file', type=str, required=True, help='Path to links.parquet file')
    parser.add_argument('--annotation-table', type=str, required=True, help='Path to annotation_table.parquet file')
    parser.add_argument('--pocket-distance-cutoff', type=float, required=True, help='Distance cutoff for pocket definition (Angstroms)')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory for output results')

    parser.add_argument('--n-pairs', type=int, default=None, 
                        help='Number of pairs to process (default: all)')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Save results every N pairs (default: 1000)')
    args = parser.parse_args()
    
    # Set global configuration
    global PLINDER_BASE, SYSTEMS_DIR, LINKED_STRUCTURES_DIR, LINKS_FILE, POCKET_DISTANCE_CUTOFF, OUTPUT_DIR, ANNOTATION_TABLE, PDB_RESOLUTION_MAP
    PLINDER_BASE = args.plinder_base
    SYSTEMS_DIR = args.systems_dir
    LINKED_STRUCTURES_DIR = args.linked_structures_dir
    LINKS_FILE = args.links_file
    ANNOTATION_TABLE = args.annotation_table
    POCKET_DISTANCE_CUTOFF = args.pocket_distance_cutoff
    OUTPUT_DIR = args.output_dir
    
    # Load resolution information from annotation table
    print("\nLoading resolution information from annotation table...")
    ann_df = pq.read_table(
        ANNOTATION_TABLE, 
        columns=['system_id', 'entry_resolution', 'entry_determination_method']
    ).to_pandas()
    ann_df['pdb_id'] = ann_df['system_id'].str.split('__').str[0]
    
    # NMR methods get resolution = 0
    nmr_methods = ['SOLUTION NMR', 'SOLID-STATE NMR']
    ann_df['resolution'] = ann_df.apply(
        lambda row: 0.0 if row['entry_determination_method'] in nmr_methods else row['entry_resolution'],
        axis=1
    )
    
    # Create PDB ID -> resolution map
    pdb_resolution = ann_df.groupby('pdb_id')['resolution'].first()
    PDB_RESOLUTION_MAP = pdb_resolution.to_dict()
    print(f"Loaded resolution for {len(PDB_RESOLUTION_MAP)} PDB entries")
    
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output or os.path.join(OUTPUT_DIR, f'rmsd_results_{timestamp}.csv')
    
    n_workers = args.n_workers or max(1, cpu_count() - 2)
    
    print("=" * 80)
    print("Holo-Apo RMSD Calculation (Batch Mode)")
    print("=" * 80)
    
    # Get holo-apo pairs
    print("\nLoading holo-apo pairs...")
    pairs_df = get_holo_apo_pairs(n_pairs=args.n_pairs)
    total_pairs = len(pairs_df)
    print(f"Total pairs to process: {total_pairs}")
    print(f"Using {n_workers} workers")
    print(f"Output file: {output_file}")
    print()
    
    # Prepare arguments for workers
    pair_args = list(zip(pairs_df['holo_id'].tolist(), pairs_df['apo_id'].tolist()))
    
    # Process with multiprocessing
    results_list = []
    
    print("Processing pairs...")
    with Pool(processes=n_workers) as pool:
        # Use imap for progress tracking
        for i, result in enumerate(tqdm(pool.imap(process_single_pair, pair_args), 
                                       total=total_pairs, 
                                       desc="RMSD Calculation")):
            results_list.append(result)
            
            # Periodic save
            if (i + 1) % args.batch_size == 0:
                temp_df = pd.DataFrame(results_list)
                temp_df.to_csv(output_file.replace('.csv', '_partial.csv'), index=False)
                print(f"\n  Saved intermediate results ({i + 1}/{total_pairs})")
    
    # Create final DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Reorder columns
    column_order = ['holo_id', 'apo_id', 'holo_resolution', 'apo_resolution', 'global_rmsd', 'pocket_ca_rmsd', 'pocket_all_rmsd', 'status', 'message']
    results_df = results_df[column_order]
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Remove partial file if exists
    partial_file = output_file.replace('.csv', '_partial.csv')
    if os.path.exists(partial_file):
        os.remove(partial_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    success_count = len(results_df[results_df['status'] == 'success'])
    skipped_count = len(results_df[results_df['status'] == 'skipped'])
    error_count = len(results_df[results_df['status'] == 'error'])
    
    print(f"\nProcessed: {len(results_df)} pairs")
    print(f"  - Success: {success_count} ({100*success_count/len(results_df):.1f}%)")
    print(f"  - Skipped: {skipped_count} ({100*skipped_count/len(results_df):.1f}%)")
    print(f"  - Error:   {error_count} ({100*error_count/len(results_df):.1f}%)")
    
    if success_count > 0:
        success_df = results_df[results_df['status'] == 'success']
        print(f"\nRMSD Statistics (successful pairs only):")
        print(f"  Global RMSD:     {success_df['global_rmsd'].mean():.3f} ± {success_df['global_rmsd'].std():.3f} Å")
        print(f"  Pocket CA RMSD:  {success_df['pocket_ca_rmsd'].mean():.3f} ± {success_df['pocket_ca_rmsd'].std():.3f} Å")
        print(f"  Pocket All RMSD: {success_df['pocket_all_rmsd'].mean():.3f} ± {success_df['pocket_all_rmsd'].std():.3f} Å")
    
    # Error breakdown
    if error_count > 0:
        print(f"\nError breakdown:")
        error_df = results_df[results_df['status'] == 'error']
        error_counts = error_df['message'].value_counts().head(10)
        for msg, count in error_counts.items():
            print(f"  - {msg}: {count}")


if __name__ == '__main__':
    main()