"""
Utility functions for MACE model deviation calculation
"""

import logging
import os
from typing import Dict, List, Optional, Union
import ase.io
import torch
from ase import Atoms

logger = logging.getLogger(__name__)


def load_mace_models(
    model_files: List[str], 
    device: str = 'cuda', 
    default_dtype: str = 'float64',
    enable_cueq: bool = False
) -> List[torch.nn.Module]:
    """
    Load MACE models from file paths following the reference implementation
    
    Args:
        model_files: List of paths to MACE model files
        device: Device to load models on
        default_dtype: Default data type for torch ('float32', 'float64')
        enable_cueq: Enable CuEq acceleration (requires cupy and e3nn-jax)
        
    Returns:
        List of loaded MACE models
        
    Raises:
        FileNotFoundError: If any model file is not found
        RuntimeError: If model loading fails
    """
    
    models = []
    
    # Set up torch device
    torch_device = torch.device(device)
    
    for model_idx, model_file in enumerate(model_files):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"MACE model file not found: {model_file}")
        
        try:
            model = torch.load(f=model_file, map_location=torch_device)
            
            # Apply CuEq acceleration if requested (following eval_configs.py)
            if enable_cueq:
                try:
                    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
                    # Convert torch.device to string for CuEq function
                    device_str = str(torch_device)
                    cueq_model = run_e3nn_to_cueq(model, device=device_str)
                    if cueq_model is not None:
                        model = cueq_model
                except (ImportError, Exception) as e:
                    logger.warning(f"CuEq acceleration unavailable: {e}")
            
            # Set model dtype
            model = model.double() if default_dtype == 'float64' else model.float()
            
            # Set model to evaluation mode and move to device
            # (this line helps with CUDA problems as noted in eval_configs.py)
            model = model.to(torch_device)
            model.eval()
            
            # Disable gradients for inference to save memory
            for param in model.parameters():
                param.requires_grad = False
            
            models.append(model)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MACE model {model_file}: {e}")
    
    return models


def read_trajectory(trajectory_file: str, type_map: Optional[List[str]] = None) -> List[Atoms]:
    """
    Read trajectory file using ASE with proper format detection
    
    Args:
        trajectory_file: Path to trajectory file
        type_map: Optional type mapping for atom ordering (specorder for LAMMPS dumps)
        
    Returns:
        List of ASE Atoms objects
        
    Raises:
        FileNotFoundError: If trajectory file is not found
        RuntimeError: If trajectory reading fails
    """
    
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
    
    try:
        from pathlib import Path
        
        # Determine format based on file extension
        file_path = Path(trajectory_file)
        if file_path.suffix in ['.lammpstrj', '.dump']:
            file_format = 'lammps-dump-text'
        elif file_path.suffix in ['.xyz']:
            file_format = 'xyz'
        else:
            # Auto-detect format
            file_format = None
        
        # Read all frames - no sampling needed since trajectory is pre-sampled
        index = ':'
        
        # Read frames using ASE
        read_kwargs = {'index': index}
        if file_format:
            read_kwargs['format'] = file_format
            if file_format == 'lammps-dump-text' and type_map:
                read_kwargs['specorder'] = type_map
        frames = ase.io.read(trajectory_file, **read_kwargs)
        
        # Ensure frames is a list
        if not isinstance(frames, list):
            frames = [frames]
        
        return frames
        
    except Exception as e:
        logger.error(f"Failed to read trajectory file {trajectory_file}: {e}")
        raise RuntimeError(f"Failed to read trajectory {trajectory_file}: {e}")


def write_model_deviation(frame_deviations: List[Dict[str, float]], output_file: str) -> None:
    """
    Write model deviation results to file in DeepMD format
    
    Args:
        frame_deviations: List of deviation statistics for each frame
        output_file: Path to output file
        
    Raises:
        RuntimeError: If writing fails
    """
    
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            # Write header in DeepMD format - keep 'devi_v' for compatibility (represents energy deviation for MACE)
            f.write('#        step     max_devi_v     min_devi_v     avg_devi_v     max_devi_f     min_devi_f     avg_devi_f\n')
            
            # Write data
            for frame_idx, frame_deviation in enumerate(frame_deviations):
                # Use frame index as timestep since trajectory is pre-sampled
                timestep = frame_idx
                
                # Write results in DeepMD format with energy and force deviations
                f.write(f"{timestep:>12d} {frame_deviation['max_devi_v']:>14.6e} "
                       f"{frame_deviation['min_devi_v']:>14.6e} {frame_deviation['avg_devi_v']:>14.6e} "
                       f"{frame_deviation['max_devi_f']:>14.6e} {frame_deviation['min_devi_f']:>14.6e} "
                       f"{frame_deviation['avg_devi_f']:>14.6e}\n")
        
    except Exception as e:
        raise RuntimeError(f"Failed to write model deviation file {output_file}: {e}")


def setup_logging(level: str = 'INFO') -> None:
    """
    Setup logging for mace-model-deviation package
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )