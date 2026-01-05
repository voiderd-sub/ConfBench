#!/usr/bin/env python3
"""
Prepare ConfBench Data

Copies structure files from PLINDER dataset to a standalone confbench_data directory.
No filtering is applied; all success pairs are copied.
Filtering is done at benchmark runtime (run_benchmark.py).
"""
import os
import shutil
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq


def get_structure_paths(holo_id: str, apo_id: str, systems_dir: str, linked_structures_dir: str):
    """Get paths for holo, apo structures and ligand files."""
    # Parse holo_id to get ligand chains
    parts = holo_id.split('__')
    ligand_chains = []
    if len(parts) >= 4:
        ligand_part = parts[3]
        for segment in ligand_part.split('_'):
            if segment:
                ligand_chains.append(segment)
    
    # Holo structure path
    holo_dir = os.path.join(systems_dir, holo_id)
    holo_cif = os.path.join(holo_dir, 'receptor.cif')
    holo_pdb = os.path.join(holo_dir, 'receptor.pdb')
    
    if os.path.exists(holo_cif):
        holo_path = holo_cif
    elif os.path.exists(holo_pdb):
        holo_path = holo_pdb
    else:
        holo_path = None
    
    # Apo structure path
    apo_cif = os.path.join(linked_structures_dir, f'{apo_id}.cif')
    apo_path = apo_cif if os.path.exists(apo_cif) else None
    
    # Ligand files
    ligand_dir = os.path.join(holo_dir, 'ligand_files')
    ligand_files = []
    
    if os.path.exists(ligand_dir):
        import glob
        for ligand_chain in ligand_chains:
            patterns = [f"*.{ligand_chain}.sdf", f"{ligand_chain}.sdf"]
            for pattern in patterns:
                matches = glob.glob(os.path.join(ligand_dir, pattern))
                ligand_files.extend(matches)
        
        ligand_files = list(set(ligand_files))
        
        if not ligand_files:
            ligand_files = glob.glob(os.path.join(ligand_dir, '*.sdf'))
    
    return holo_path, apo_path, ligand_files, holo_dir


def main():
    parser = argparse.ArgumentParser(description='Prepare ConfBench data from PLINDER dataset')
    parser.add_argument('--rmsd-csv', type=str, required=True,
                        help='Path to RMSD results CSV file (from calculate_rmsd.py)')
    parser.add_argument('--links-file', type=str, required=True,
                        help='Path to links.parquet file (for lddt values)')
    parser.add_argument('--systems-dir', type=str, required=True,
                        help='Path to PLINDER systems directory')
    parser.add_argument('--linked-structures-dir', type=str, required=True,
                        help='Path to linked_structures directory')
    parser.add_argument('--annotation-table', type=str, required=True,
                        help='Path to annotation_table.parquet (for crystal contact info)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for confbench_data')
    parser.add_argument('--n-pairs', type=int, default=None,
                        help='Optional: limit number of pairs to copy')
    
    args = parser.parse_args()
    
    # Load RMSD results
    print(f"Loading RMSD results from {args.rmsd_csv}...")
    rmsd_df = pd.read_csv(args.rmsd_csv)
    print(f"Total pairs: {len(rmsd_df)}")
    
    # Filter to success only
    rmsd_df = rmsd_df[rmsd_df['status'] == 'success']
    print(f"Success pairs: {len(rmsd_df)}")
    
    # Load lddt values from links file
    print(f"\nLoading lddt values from {args.links_file}...")
    links_df = pd.read_parquet(
        args.links_file,
        columns=['reference_system_id', 'id', 'lddt', 'bb_lddt', 'pocket_fident']
    )
    
    # Merge lddt values
    rmsd_df = rmsd_df.merge(
        links_df.rename(columns={'reference_system_id': 'holo_id', 'id': 'apo_id'}),
        on=['holo_id', 'apo_id'],
        how='left'
    )
    print(f"Merged with lddt: {rmsd_df['lddt'].notna().sum()} pairs have lddt values")
    
    # Load crystal contact info from annotation table
    print(f"\nLoading crystal contact info from {args.annotation_table}...")
    ann_df = pq.read_table(
        args.annotation_table,
        columns=['system_id', 'system_num_atoms_with_crystal_contacts']
    ).to_pandas()
    
    # Merge crystal contact info
    rmsd_df = rmsd_df.merge(
        ann_df.rename(columns={'system_id': 'holo_id'}),
        on='holo_id',
        how='left'
    )
    crystal_col = 'system_num_atoms_with_crystal_contacts'
    print(f"Merged with crystal contacts: {rmsd_df[crystal_col].notna().sum()} pairs have values")
    
    # Limit if requested
    if args.n_pairs is not None:
        rmsd_df = rmsd_df.head(args.n_pairs)
        print(f"Limited to {len(rmsd_df)} pairs")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    holo_dir = output_dir / 'holo'
    apo_dir = output_dir / 'apo'
    
    holo_dir.mkdir(parents=True, exist_ok=True)
    apo_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print(f"\nCopying files to {args.output_dir}...")
    
    copied_pairs = []
    copied_apos = set()
    
    for _, row in tqdm(rmsd_df.iterrows(), total=len(rmsd_df), desc="Copying"):
        holo_id = row['holo_id']
        apo_id = row['apo_id']
        
        holo_path, apo_path, ligand_files, source_holo_dir = get_structure_paths(
            holo_id, apo_id, args.systems_dir, args.linked_structures_dir
        )
        
        if holo_path is None or apo_path is None or not ligand_files:
            continue
        
        # Copy holo structure
        dest_holo_dir = holo_dir / holo_id
        dest_holo_dir.mkdir(parents=True, exist_ok=True)
        
        dest_holo_path = dest_holo_dir / os.path.basename(holo_path)
        if not dest_holo_path.exists():
            shutil.copy2(holo_path, dest_holo_path)
        
        # Copy ligand files
        dest_ligand_dir = dest_holo_dir / 'ligand_files'
        dest_ligand_dir.mkdir(parents=True, exist_ok=True)
        
        for ligand_file in ligand_files:
            dest_ligand_path = dest_ligand_dir / os.path.basename(ligand_file)
            if not dest_ligand_path.exists():
                shutil.copy2(ligand_file, dest_ligand_path)
        
        # Copy apo structure (only once per apo_id)
        if apo_id not in copied_apos:
            dest_apo_path = apo_dir / f'{apo_id}.cif'
            if not dest_apo_path.exists():
                shutil.copy2(apo_path, dest_apo_path)
            copied_apos.add(apo_id)
        
        # Record copied pair
        copied_pairs.append({
            'holo_id': holo_id,
            'apo_id': apo_id,
            'holo_resolution': row.get('holo_resolution'),
            'apo_resolution': row.get('apo_resolution'),
            'global_rmsd': row.get('global_rmsd'),
            'pocket_ca_rmsd': row.get('pocket_ca_rmsd'),
            'pocket_all_rmsd': row.get('pocket_all_rmsd'),
            'lddt': row.get('lddt'),
            'bb_lddt': row.get('bb_lddt'),
            'pocket_fident': row.get('pocket_fident'),
            'system_num_atoms_with_crystal_contacts': row.get('system_num_atoms_with_crystal_contacts'),
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(copied_pairs)
    metadata_path = output_dir / 'metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"\n=== Summary ===")
    print(f"Total pairs copied: {len(copied_pairs)}")
    print(f"Unique apo structures: {len(copied_apos)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
