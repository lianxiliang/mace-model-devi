"""
Utility functions for MACE model deviation calculation
"""

import os
import torch
import ase.io
from ase import Atoms
from typing import List, Union, Dict, Optional
import logging

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
            logger.info(f"Loading model {model_idx + 1}/{len(model_files)}: {os.path.basename(model_file)}")
            
            # Load model using torch.load (following reference implementation)
            model = torch.load(f=model_file, map_location=torch_device)
            
            # Validate model was loaded successfully
            if model is None:
                raise RuntimeError(f"Model file {model_file} loaded as None")
            
            # Apply CuEq acceleration if requested (following eval_configs.py)
            if enable_cueq:
                logger.info("Converting model to CuEq for acceleration")
                try:
                    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
                    # Convert torch.device to string for CuEq function
                    device_str = str(torch_device)
                    cueq_model = run_e3nn_to_cueq(model, device=device_str)
                    if cueq_model is not None:
                        model = cueq_model
                    else:
                        logger.warning("CuEq conversion returned None, using original model")
                except ImportError as e:
                    logger.warning(f"CuEq acceleration not available (missing dependencies): {e}")
                    logger.warning("Continuing without CuEq acceleration")
                except Exception as e:
                    logger.warning(f"Failed to apply CuEq acceleration: {e}")
                    logger.warning("Continuing without CuEq acceleration")
            
            # Ensure model is in the right dtype
            if default_dtype == 'float64':
                model = model.double()
            elif default_dtype == 'float32':
                model = model.float()
            
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
    
    logger.info(f"Successfully loaded {len(models)} MACE models on {device}")
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
        
        logger.info(f"Reading trajectory: {trajectory_file}")
        
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
        logger.info("Reading all frames from pre-sampled trajectory")
        
        # Read frames using ASE with efficient reading
        if file_format == 'lammps-dump-text' and type_map:
            logger.info(f"Reading LAMMPS trajectory with specorder: {type_map}")
            frames = ase.io.read(trajectory_file, index=index, format=file_format, specorder=type_map)
        elif file_format:
            frames = ase.io.read(trajectory_file, index=index, format=file_format)
        else:
            frames = ase.io.read(trajectory_file, index=index)
        
        # Ensure frames is a list
        if not isinstance(frames, list):
            frames = [frames]
        
        logger.info(f"Successfully read {len(frames)} frames from {trajectory_file}")
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
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        logger.info(f"Writing model deviations to: {output_file}")
        
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
        
        logger.info(f"Model deviation results written successfully")
        
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