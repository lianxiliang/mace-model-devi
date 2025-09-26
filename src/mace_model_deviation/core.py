"""
Core MACE model deviation calculation functions

Comprehensive implementation matching ai2-kit functionality
"""

import os
import numpy as np
import torch
from typing import List, Optional, Dict, Any
import ase.io
from ase import Atoms
import logging

logger = logging.getLogger(__name__)


def calculate_mace_model_deviation(
    model_files: List[str],
    trajectory_file: str,
    output_file: str,
    type_map: Optional[List[str]] = None,
    device: str = 'cuda',
    batch_size: int = 64,
    default_dtype: str = 'float64',
    enable_cueq: bool = False
) -> str:
    """
    Calculate MACE model deviation using DeepMD strategy
    
    Follows DeepMD approach:
    1. Read reference trajectory (from model 1 or LAMMPS/DFT reference)  
    2. Evaluate all models on the same configurations
    3. Calculate RMS deviation from ensemble mean (DeepMD methodology)
    4. Write results in DeepMD model_devi.out format
    
    Args:
        model_files: List of paths to MACE model files (.pt format)
        trajectory_file: Path to trajectory file (ASE-readable format)
        output_file: Path to output model_devi.out file
        type_map: Optional list of element symbols for type mapping
        device: Device for calculation ('cuda', 'cpu', 'mps')
        batch_size: Batch size for DataLoader (controls memory usage)
        default_dtype: Default data type for torch ('float32', 'float64')
        enable_cueq: Enable CuEq acceleration for faster inference
        
    Returns:
        Path to output file
        
    Raises:
        FileNotFoundError: If model files or trajectory file not found
        RuntimeError: If MACE calculation fails
    """
    
    # Input validation
    if not model_files:
        raise ValueError("No model files provided")
    
    if len(model_files) < 2:
        raise ValueError(f"Need at least 2 models for meaningful deviation calculation, got {len(model_files)}")
    
    if not trajectory_file:
        raise ValueError("Trajectory file path is required")
    
    if not output_file:
        raise ValueError("Output file path is required")
    
    logger.info(f"Calculating MACE model deviation with {len(model_files)} models on device: {device}")
    
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
    
    # Validate model files exist
    missing_models = [f for f in model_files if not os.path.exists(f)]
    if missing_models:
        raise FileNotFoundError(f"Model files not found: {missing_models}")
    
    # Check if MACE is available
    try:
        import torch
        from mace import data
        from mace.tools import torch_tools, utils, torch_geometric
        logger.info("MACE package found - using proper MACE evaluation approach")
    except ImportError as e:
        logger.error(f"MACE package not found: {e}")
        logger.error("Cannot calculate model deviation without MACE - this is required for MACE workflows")
        raise ImportError(f"MACE packages required for model deviation calculation: {e}")
    
    try:
        from .utils import load_mace_models, read_trajectory, write_model_deviation
        
        # Set up torch configuration following MACE eval_configs.py style
        torch_tools.set_default_dtype(default_dtype)
        
        # Validate device - use MACE's device initialization
        device_obj = torch_tools.init_device(device)
        device_str = str(device_obj).replace('cuda:', 'cuda')  # Convert to string for compatibility
        
        # Load MACE models with optional CuEq acceleration
        models = load_mace_models(model_files, device=device_str, default_dtype=default_dtype, enable_cueq=enable_cueq)
        logger.info(f"Loaded {len(models)} MACE models on {device_obj}")
        
        # Read trajectory
        frames = read_trajectory(trajectory_file, type_map)
        logger.info(f"Read {len(frames)} frames from trajectory")
        
        # Calculate model deviations using proper MACE pipeline
        deviations = _calculate_frame_deviations_mace(
            models, frames, device=device_obj, batch_size=batch_size
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


def _calculate_frame_deviations_mace(
    models: List[torch.nn.Module],
    frames: List[Atoms],
    device: torch.device,
    batch_size: int = 64,
) -> List[Dict[str, float]]:
    """
    Calculate model deviations for trajectory frames using proper MACE pipeline
    
    Args:
        models: List of loaded MACE models
        frames: List of ASE Atoms objects
        device: Device for calculation
        batch_size: Batch size for DataLoader (controls memory usage)
        
    Returns:
        List of deviation statistics per frame (DeepMD format)
    """
    from mace import data
    from mace.tools import torch_geometric, utils, torch_tools
    
    n_frames = len(frames)
    all_energies = []
    all_forces = []
    
    # Get predictions from each model
    for model_idx, model in enumerate(models):
        logger.info(f"Evaluating model {model_idx + 1}/{len(models)}")
        
        # Extract model properties (exactly like eval_configs.py)
        z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])  # type: ignore[attr-defined]
        cutoff = float(model.r_max)  # type: ignore[attr-defined,arg-type]
        
        try:
            heads = model.heads  # type: ignore[attr-defined] 
        except AttributeError:
            heads = None
            
        logger.info(f"Model properties - atomic numbers: {z_table.zs}, cutoff: {cutoff}, heads: {heads}")
        
        # Convert ASE atoms to MACE configurations (following eval_configs.py exactly)
        configs = [data.config_from_atoms(atoms) for atoms in frames]
        
        # Create MACE DataLoader for all frames at once (no chunking - MACE standard)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[  # type: ignore[arg-type]  # MACE standard: list of AtomicData
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=cutoff, heads=heads  # type: ignore[arg-type]
                )
                for config in configs
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        # Collect model outputs (following eval_configs.py pattern)
        model_energies = []
        model_forces = []
        
        for batch in data_loader:
            batch = batch.to(device)
            
            # Enable gradient computation for force calculation
            batch.positions.requires_grad_(True)
            
            # Get model output - proper MACE interface
            output = model(batch.to_dict())
            
            # Extract energies and forces 
            energies = torch_tools.to_numpy(output["energy"])
            forces = np.split(
                torch_tools.to_numpy(output["forces"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )[:-1]  # drop last as it's empty
            
            model_energies.extend(energies)
            model_forces.extend(forces)
        
        all_energies.append(model_energies)
        all_forces.append(model_forces)
    
    # Calculate deviations using DeepMD methodology
    frame_deviations = []
    all_energies = np.array(all_energies)  # Shape: (n_models, n_frames)
    
    for frame_idx in range(n_frames):
        # Energy deviation for this frame
        frame_energies = all_energies[:, frame_idx]  # (n_models,)
        energy_std = np.std(frame_energies)
        
        # Use energy std as "virial" deviation for DeepMD compatibility
        max_energy_devi = float(energy_std)
        min_energy_devi = float(energy_std)
        avg_energy_devi = float(energy_std)
        
        # Force deviation for this frame
        frame_forces = np.array([all_forces[model_idx][frame_idx] for model_idx in range(len(models))])
        # shape: (n_models, n_atoms, 3)
        
        if len(models) > 1:
            # DeepMD-style force deviation calculation
            mean_forces = np.mean(frame_forces, axis=0)  # (n_atoms, 3)
            deviations = frame_forces - mean_forces[np.newaxis, :, :]  # (n_models, n_atoms, 3)
            squared_norms = np.sum(deviations**2, axis=2)  # (n_models, n_atoms)
            force_deviations_per_atom = np.sqrt(np.mean(squared_norms, axis=0))  # (n_atoms,)
            
            max_force_devi = float(np.max(force_deviations_per_atom))
            min_force_devi = float(np.min(force_deviations_per_atom))
            avg_force_devi = float(np.mean(force_deviations_per_atom))
        else:
            max_force_devi = 0.0
            min_force_devi = 0.0
            avg_force_devi = 0.0
        
        # Frame deviation statistics
        frame_deviation = {
            'max_devi_v': max_energy_devi,
            'min_devi_v': min_energy_devi,
            'avg_devi_v': avg_energy_devi,
            'max_devi_f': max_force_devi,
            'min_devi_f': min_force_devi,
            'avg_devi_f': avg_force_devi,
        }
        
        frame_deviations.append(frame_deviation)
        
        # Progress logging
        if (frame_idx + 1) % 100 == 0:
            logger.info(f"Processed {frame_idx + 1}/{len(frames)} frames")
    
    return frame_deviations