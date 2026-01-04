"""
ConfBench Utilities

RMSD calculation and structure alignment utilities extracted from calculate_rmsd.py.
"""
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import three_to_one
from pymol import cmd


def safe_three_to_one(resname: str) -> str:
    """
    Safely convert 3-letter amino acid code to 1-letter code.
    Returns 'X' for non-standard amino acids.
    """
    try:
        return three_to_one(resname)
    except KeyError:
        return 'X'


# ===== ID Parsing Functions =====

def parse_holo_id(holo_id: str) -> Tuple[Optional[str], List[str], List[str]]:
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


def parse_apo_id(apo_id: str) -> Tuple[Optional[str], Optional[str]]:
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


# ===== Structure Loading =====

def load_structure(structure_path: str, structure_name: str = 'structure'):
    """
    Load a structure from file (supports .pdb, .cif).
    
    Args:
        structure_path: Path to structure file
        structure_name: Name for the structure object
        
    Returns:
        BioPython Structure object
    """
    if structure_path.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    
    return parser.get_structure(structure_name, structure_path)


# ===== Sequence and Residue Functions =====

def extract_sequence_from_structure(structure, target_chain_ids: Optional[List[str]] = None) -> Tuple[str, List]:
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


def get_residue_key(residue) -> Tuple[str, int, str]:
    """Get a unique key for a residue (chain_id, resseq, icode)."""
    return (residue.parent.id, residue.id[1], residue.id[2])


def build_selection_string(residue_keys: List[Tuple[str, int, str]]) -> str:
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


def create_residue_mapping(holo_residues: List, apo_residues: List) -> Dict:
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


# ===== Ligand and Pocket Functions =====

def get_ligand_coords(ligand_files: List[str]) -> Optional[np.ndarray]:
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


def find_pocket_residues(
    structure, 
    ligand_coords: np.ndarray, 
    cutoff: float = 10.0, 
    target_chain_ids: Optional[List[str]] = None
) -> Set[Tuple[str, int, str]]:
    """Find residues within cutoff distance of ligand."""
    if ligand_coords is None or len(ligand_coords) == 0:
        return set()
    
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


# ===== RMSD Calculation =====

def calculate_rmsd_with_pymol(
    structure1_path: str, 
    structure2_path: str, 
    selection1: str, 
    selection2: str,
    object1_name: str = 'struct1',
    object2_name: str = 'struct2'
) -> Tuple[Optional[float], int]:
    """
    Calculate RMSD using PyMOL's align command with cycles=0.
    
    Args:
        structure1_path: Path to first structure file
        structure2_path: Path to second structure file
        selection1: PyMOL selection string for first structure
        selection2: PyMOL selection string for second structure
        object1_name: Name for first structure in PyMOL
        object2_name: Name for second structure in PyMOL
        
    Returns:
        Tuple of (RMSD, number of atoms aligned)
    """
    cmd.delete('all')
    
    try:
        cmd.load(structure1_path, object1_name)
        cmd.load(structure2_path, object2_name)
        
        result = cmd.align(selection1, selection2, cycles=0)
        
        rmsd = result[0]
        n_atoms = result[1]
        
        cmd.delete('all')
        
        if n_atoms == 0:
            return None, 0
        
        return rmsd, n_atoms
    
    except Exception:
        cmd.delete('all')
        return None, 0


def calculate_pairwise_rmsds(
    holo_path: str,
    apo_path: str,
    pred_path: str,
    ligand_files: List[str],
    pocket_distance_cutoff: float = 10.0,
    holo_chain_ids: Optional[List[str]] = None
) -> Dict[str, Optional[float]]:
    """
    Calculate all pairwise RMSDs for ConfBench score calculation.
    
    Returns dict with keys:
        - rmsd_pred_holo_global, rmsd_pred_holo_pocket_ca, rmsd_pred_holo_pocket_all
        - rmsd_pred_apo_global, rmsd_pred_apo_pocket_ca, rmsd_pred_apo_pocket_all
        - rmsd_apo_holo_global, rmsd_apo_holo_pocket_ca, rmsd_apo_holo_pocket_all
    """
    results = {
        'rmsd_pred_holo_global': None,
        'rmsd_pred_holo_pocket_ca': None,
        'rmsd_pred_holo_pocket_all': None,
        'rmsd_pred_apo_global': None,
        'rmsd_pred_apo_pocket_ca': None,
        'rmsd_pred_apo_pocket_all': None,
        'rmsd_apo_holo_global': None,
        'rmsd_apo_holo_pocket_ca': None,
        'rmsd_apo_holo_pocket_all': None,
    }
    
    # Load structures
    try:
        holo_structure = load_structure(holo_path, 'holo')
        apo_structure = load_structure(apo_path, 'apo')
        pred_structure = load_structure(pred_path, 'pred')
    except Exception:
        return results
    
    # Get ligand coordinates for pocket definition
    ligand_coords = get_ligand_coords(ligand_files)
    if ligand_coords is None or len(ligand_coords) == 0:
        return results
    
    # Find pocket residues in holo
    holo_pocket_keys = find_pocket_residues(
        holo_structure, ligand_coords, pocket_distance_cutoff, target_chain_ids=holo_chain_ids
    )
    if len(holo_pocket_keys) == 0:
        return results
    
    # Extract residue lists
    _, holo_residues = extract_sequence_from_structure(holo_structure, target_chain_ids=holo_chain_ids)
    _, apo_residues = extract_sequence_from_structure(apo_structure, target_chain_ids=None)
    _, pred_residues = extract_sequence_from_structure(pred_structure, target_chain_ids=None)
    
    # Create residue mappings
    holo_to_apo = create_residue_mapping(holo_residues, apo_residues)
    holo_to_pred = create_residue_mapping(holo_residues, pred_residues)
    
    # Get mapped pocket residues
    holo_pocket_mapped = [k for k in holo_pocket_keys if k in holo_to_apo and k in holo_to_pred]
    if len(holo_pocket_mapped) < len(holo_pocket_keys) * 0.5:
        return results  # Low mapping ratio
    
    apo_pocket_keys = [holo_to_apo[k] for k in holo_pocket_mapped]
    pred_pocket_keys = [holo_to_pred[k] for k in holo_pocket_mapped]
    
    # Build selection strings
    holo_global_keys = list(holo_to_apo.keys())
    apo_global_keys = list(holo_to_apo.values())
    
    holo_global_str = build_selection_string(holo_global_keys)
    apo_global_str = build_selection_string(apo_global_keys)
    holo_pocket_str = build_selection_string(list(holo_pocket_mapped))
    apo_pocket_str = build_selection_string(apo_pocket_keys)
    pred_pocket_str = build_selection_string(pred_pocket_keys)
    
    # Calculate RMSDs between pred and holo
    rmsd, _ = calculate_rmsd_with_pymol(
        pred_path, holo_path,
        f"pred and {build_selection_string([holo_to_pred[k] for k in holo_global_keys if k in holo_to_pred])} and name CA",
        f"holo and {holo_global_str} and name CA",
        'pred', 'holo'
    )
    results['rmsd_pred_holo_global'] = rmsd
    
    rmsd, _ = calculate_rmsd_with_pymol(
        pred_path, holo_path,
        f"pred and {pred_pocket_str} and name CA",
        f"holo and {holo_pocket_str} and name CA",
        'pred', 'holo'
    )
    results['rmsd_pred_holo_pocket_ca'] = rmsd
    
    rmsd, _ = calculate_rmsd_with_pymol(
        pred_path, holo_path,
        f"pred and {pred_pocket_str} and not elem H",
        f"holo and {holo_pocket_str} and not elem H",
        'pred', 'holo'
    )
    results['rmsd_pred_holo_pocket_all'] = rmsd
    
    # Calculate RMSDs between pred and apo
    pred_global_keys = [holo_to_pred[k] for k in holo_global_keys if k in holo_to_pred]
    pred_global_str = build_selection_string(pred_global_keys)
    
    rmsd, _ = calculate_rmsd_with_pymol(
        pred_path, apo_path,
        f"pred and {pred_global_str} and name CA",
        f"apo and {apo_global_str} and name CA",
        'pred', 'apo'
    )
    results['rmsd_pred_apo_global'] = rmsd
    
    rmsd, _ = calculate_rmsd_with_pymol(
        pred_path, apo_path,
        f"pred and {pred_pocket_str} and name CA",
        f"apo and {apo_pocket_str} and name CA",
        'pred', 'apo'
    )
    results['rmsd_pred_apo_pocket_ca'] = rmsd
    
    rmsd, _ = calculate_rmsd_with_pymol(
        pred_path, apo_path,
        f"pred and {pred_pocket_str} and not elem H",
        f"apo and {apo_pocket_str} and not elem H",
        'pred', 'apo'
    )
    results['rmsd_pred_apo_pocket_all'] = rmsd
    
    # Calculate RMSDs between apo and holo (these should already be computed)
    rmsd, _ = calculate_rmsd_with_pymol(
        holo_path, apo_path,
        f"holo and {holo_global_str} and name CA",
        f"apo and {apo_global_str} and name CA",
        'holo', 'apo'
    )
    results['rmsd_apo_holo_global'] = rmsd
    
    rmsd, _ = calculate_rmsd_with_pymol(
        holo_path, apo_path,
        f"holo and {holo_pocket_str} and name CA",
        f"apo and {apo_pocket_str} and name CA",
        'holo', 'apo'
    )
    results['rmsd_apo_holo_pocket_ca'] = rmsd
    
    rmsd, _ = calculate_rmsd_with_pymol(
        holo_path, apo_path,
        f"holo and {holo_pocket_str} and not elem H",
        f"apo and {apo_pocket_str} and not elem H",
        'holo', 'apo'
    )
    results['rmsd_apo_holo_pocket_all'] = rmsd
    
    return results


def calculate_confbench_score(
    rmsd_pred_apo: float,
    rmsd_pred_holo: float,
    rmsd_apo_holo: float
) -> Optional[float]:
    """
    Calculate ConfBench score.
    
    Score = (RMSD(Pred, Apo) - RMSD(Pred, Holo)) / sqrt(0.5 * (RMSD_pa^2 + RMSD_ph^2 + RMSD_ah^2))
    
    Returns:
        Score value, or None if calculation fails
    """
    if rmsd_pred_apo is None or rmsd_pred_holo is None or rmsd_apo_holo is None:
        return None
    
    denominator = np.sqrt(0.5 * (rmsd_pred_apo**2 + rmsd_pred_holo**2 + rmsd_apo_holo**2))
    
    if denominator == 0:
        return None
    
    score = (rmsd_pred_apo - rmsd_pred_holo) / denominator
    return score
