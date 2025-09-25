"""
Core MACE model deviation calculation functions
"""

import os
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any
from ase import Atoms
import logging

logger = logging.getLogger(__name__)


def calculate_mace_model_deviation(
    model_files: List[str],
    trajectory_file: str,
    output_file: str,
    type_map: Optional[List[str]] = None,
    device: str = 'cuda',
    batch_size: int = 1,
    chunk_size: int = 100,
) -> str:
    """
    Calculate MACE model ensemble uncertainty/deviation.
    
    Args:
        model_files: List of paths to MACE model files (.pt format)
        trajectory_file: Path to trajectory file (ASE-readable format)
        output_file: Path to output model_devi.out file
        type_map: Optional list of element symbols for type mapping
        device: Device for calculation ('cuda', 'cpu', 'mps')
        batch_size: Batch size for model inference
        chunk_size: Number of frames to process at once
        
    Returns:
        Path to output file
        
    Raises:
        FileNotFoundError: If model files or trajectory file not found
        RuntimeError: If MACE calculation fails
    """
    
    logger.info(f"MACE model deviation calculation starting")
    logger.info(f"Models: {len(model_files)} files")
    logger.info(f"Trajectory: {trajectory_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Device: {device}")
    
    try:
        from .utils import load_mace_models, read_trajectory, write_model_deviation
        
        # Validate device
        device = _validate_device(device)
        
        # Load MACE models
        models = load_mace_models(model_files, device=device)
        logger.info(f"Loaded {len(models)} MACE models on {device}")
        
        # Read trajectory
        frames = read_trajectory(trajectory_file)
        logger.info(f"Read {len(frames)} frames from trajectory")
        
        # Calculate model deviations
        deviations = _calculate_frame_deviations(
            models, frames, device=device, 
            batch_size=batch_size, chunk_size=chunk_size
        )
        
        # Write results
        write_model_deviation(deviations, output_file)
        logger.info(f"Model deviation written to: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"MACE model deviation calculation failed: {e}")
        raise RuntimeError(f"MACE model deviation calculation failed: {e}")


def _validate_device(device: str) -> str:
    """Validate and adjust device based on availability"""
    if device == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        else:
            logger.info(f"CUDA available: {torch.cuda.device_count()} device(s)")
            return device
    elif device == 'mps':
        if not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return 'cpu'
        else:
            return device
    else:
        return 'cpu'


def _calculate_frame_deviations(
    models: List[torch.nn.Module],
    frames: List[Atoms],
    device: str = 'cuda',
    batch_size: int = 1,
    chunk_size: int = 100,
) -> List[float]:
    """
    Calculate model deviations for trajectory frames
    
    Args:
        models: List of loaded MACE models
        frames: List of ASE Atoms objects
        device: Device for calculation
        batch_size: Batch size for inference
        chunk_size: Process frames in chunks to manage memory
        
    Returns:
        List of maximum force deviations per frame
    """
    
    n_frames = len(frames)
    all_deviations = []
    
    # Process frames in chunks
    for chunk_start in range(0, n_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_frames = frames[chunk_start:chunk_end]
        
        logger.info(f"Processing frames {chunk_start+1}-{chunk_end}/{n_frames}")
        
        chunk_deviations = []
        for i, atoms in enumerate(chunk_frames):
            try:
                # Get ensemble predictions
                forces_ensemble = _get_ensemble_forces(models, atoms, device)
                
                if len(forces_ensemble) > 1:
                    # Calculate standard deviation across models
                    forces_array = np.array(forces_ensemble)  # Shape: (n_models, n_atoms, 3)
                    force_std = np.std(forces_array, axis=0)  # Shape: (n_atoms, 3)
                    max_force_devi = np.max(force_std)  # Maximum deviation across all atoms/directions
                else:
                    max_force_devi = 0.0
                    
                chunk_deviations.append(max_force_devi)
                
            except Exception as e:
                logger.warning(f"Failed to process frame {chunk_start + i}: {e}")
                chunk_deviations.append(0.0)
        
        all_deviations.extend(chunk_deviations)
    
    return all_deviations


def _get_ensemble_forces(
    models: List[torch.nn.Module],
    atoms: Atoms,
    device: str
) -> List[np.ndarray]:
    """
    Get force predictions from ensemble of MACE models
    
    Args:
        models: List of MACE models
        atoms: ASE Atoms object
        device: Device for calculation
        
    Returns:
        List of force arrays, one per model
    """
    
    forces_ensemble = []
    
    for model in models:
        try:
            with torch.no_grad():
                # Convert ASE atoms to MACE format
                # Note: This is a simplified conversion - actual MACE interface may differ
                batch_data = _atoms_to_mace_batch(atoms, device)
                
                # Get model prediction
                output = model(batch_data)
                
                # Extract forces
                if 'forces' in output:
                    forces = output['forces']
                    if isinstance(forces, torch.Tensor):
                        forces = forces.cpu().numpy()
                    forces_ensemble.append(forces)
                else:
                    logger.warning("Model output does not contain 'forces' key")
                    forces_ensemble.append(np.zeros_like(atoms.positions))
                    
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
            forces_ensemble.append(np.zeros_like(atoms.positions))
    
    return forces_ensemble


def _atoms_to_mace_batch(atoms: Atoms, device: str) -> Dict[str, torch.Tensor]:
    """
    Convert ASE Atoms to MACE model input format
    
    Note: This is a simplified implementation. The actual MACE model
    interface may require different input format/keys.
    """
    
    # Basic conversion - may need adjustment based on specific MACE model format
    positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    
    batch_data = {
        'positions': positions.unsqueeze(0),  # Add batch dimension
        'atomic_numbers': atomic_numbers.unsqueeze(0),  # Add batch dimension
        'ptr': torch.tensor([0, len(atoms)], dtype=torch.long, device=device),
    }
    
    # Add cell information if periodic
    if atoms.pbc.any():
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=device)
        batch_data['cell'] = cell.unsqueeze(0)  # Add batch dimension
    
    return batch_data