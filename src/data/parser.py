"""Parser for RNA structure files (.st format)"""

from pathlib import Path
from typing import Dict, Optional
import pickle


def parse_st_file(file_path: str) -> Dict[str, str]:
    """
    Parse a .st file to extract RNA sequence and structure information.

    The .st file format contains header lines (starting with #) followed by 4 essential lines:
    1. Sequence: nucleotide sequence (A, U, G, C, N)
    2. Dot-bracket: secondary structure notation
    3. Structural annotation: E/S/H/I/M/B/X characters
    4. Pseudoknot annotation: N/K characters

    Args:
        file_path: Path to the .st file

    Returns:
        Dict with keys: 'sequence', 'dot_bracket', 'structure', 'pseudoknot'

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header lines (those starting with #)
    data_lines = []
    for line in lines:
        if not line.startswith('#'):
            stripped = line.strip()
            if stripped:  # Only add non-empty lines
                data_lines.append(stripped)

    # Validate we have exactly 4 data lines
    if len(data_lines) < 4:
        raise ValueError(
            f"Expected at least 4 data lines in {file_path}, got {len(data_lines)}. "
            f"Lines: {data_lines[:10]}"
        )

    # Extract the 4 essential lines
    sequence = data_lines[0]
    dot_bracket = data_lines[1]
    structure = data_lines[2]
    pseudoknot = data_lines[3]

    # Validate lengths match
    seq_len = len(sequence)
    if not (len(dot_bracket) == len(structure) == len(pseudoknot) == seq_len):
        raise ValueError(
            f"Length mismatch in {file_path}: "
            f"seq={seq_len}, dot={len(dot_bracket)}, "
            f"struct={len(structure)}, pk={len(pseudoknot)}"
        )

    return {
        'sequence': sequence,
        'dot_bracket': dot_bracket,
        'structure': structure,
        'pseudoknot': pseudoknot,
    }


def extract_rfid(reference_name: str) -> str:
    """
    Extract RFAM family ID from reference name.

    Args:
        reference_name: Reference name like "RF00266_AAGV020173148.1_4127-4193"

    Returns:
        RFAM ID like "RF00266"
    """
    return reference_name.split('_')[0]


def load_rfam_types(rfam_types_path: str = 'rfam/rfam_types_full.pkl') -> Dict[str, str]:
    """
    Load RFAM type mappings from pickle file.

    Args:
        rfam_types_path: Path to rfam_types_full.pkl

    Returns:
        Dict mapping RFID to meta-type string
    """
    with open(rfam_types_path, 'rb') as f:
        return pickle.load(f)


def get_meta_type(rfid: str, rfam_types: Dict[str, str]) -> Optional[str]:
    """
    Get meta-type for a given RFID.

    Args:
        rfid: RFAM ID like "RF00266"
        rfam_types: Dict mapping RFID to meta-type

    Returns:
        Meta-type string like "Gene; snRNA; snoRNA; CD-box;" or None if not found
    """
    return rfam_types.get(rfid)
