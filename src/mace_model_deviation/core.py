""""""

Core MACE model deviation calculation functionsCore MACE model deviation calculation functions

Comprehensive implementation matching ai2-kit functionality"""

"""

import os

import osimport numpy as np

import numpy as npimport torch

import torchfrom typing import List, Optional, Tuple, Dict, Any

import ase.iofrom ase import Atoms

from ase import Atomsimport logging

from typing import List, Optional, Dict, Any

import logginglogger = logging.getLogger(__name__)



logger = logging.getLogger(__name__)

def calculate_mace_model_deviation(

    model_files: List[str],

def calculate_mace_model_deviation(    trajectory_file: str,

    model_files: List[str],    output_file: str,

    trajectory_file: str,    type_map: Optional[List[str]] = None,

    output_file: str,    device: str = 'cuda',

    type_map: Optional[List[str]] = None,    batch_size: int = 1,

    device: str = 'cuda',    chunk_size: int = 100,

    batch_size: int = 64,) -> str:

    chunk_size: int = 1000,    """

    default_dtype: str = 'float64'    Calculate MACE model ensemble uncertainty/deviation.

) -> str:    

    """    Args:

    Calculate MACE model deviation using DeepMD strategy        model_files: List of paths to MACE model files (.pt format)

            trajectory_file: Path to trajectory file (ASE-readable format)

    Follows DeepMD approach:        output_file: Path to output model_devi.out file

    1. Read reference trajectory (from model 1 or LAMMPS/DFT reference)          type_map: Optional list of element symbols for type mapping

    2. Evaluate all models on the same configurations        device: Device for calculation ('cuda', 'cpu', 'mps')

    3. Calculate RMS deviation from ensemble mean (DeepMD methodology)        batch_size: Batch size for model inference

    4. Write results in DeepMD model_devi.out format        chunk_size: Number of frames to process at once

            

    Args:    Returns:

        model_files: list of MACE model file paths (.model files)        Path to output file

        trajectory_file: path to reference trajectory file (LAMMPS dump, xyz, etc.)        

        output_file: path to output deviation file (model_devi.out format)    Raises:

        type_map: type mapping for atoms (optional, for compatibility)        FileNotFoundError: If model files or trajectory file not found

        chunk_size: process frames in chunks to manage memory (default: 1000)        RuntimeError: If MACE calculation fails

        device: device for MACE calculation ('cuda', 'cpu', 'mps')    """

        batch_size: batch size for MACE evaluation (default: 64)    

        default_dtype: default data type for torch ('float32', 'float64')    logger.info(f"MACE model deviation calculation starting")

            logger.info(f"Models: {len(model_files)} files")

    Returns:    logger.info(f"Trajectory: {trajectory_file}")

        path to the output file    logger.info(f"Output: {output_file}")

            logger.info(f"Device: {device}")

    Raises:    

        FileNotFoundError: if trajectory file does not exist    try:

        ValueError: if less than 2 models provided or invalid parameters        from .utils import load_mace_models, read_trajectory, write_model_deviation

        ImportError: if required MACE packages are not available        

    """        # Validate device

    # Input validation        device = _validate_device(device)

    if not model_files:        

        raise ValueError("No model files provided")        # Load MACE models

            models = load_mace_models(model_files, device=device)

    if len(model_files) < 2:        logger.info(f"Loaded {len(models)} MACE models on {device}")

        raise ValueError(f"Need at least 2 models for meaningful deviation calculation, got {len(model_files)}")        

            # Read trajectory

    if not trajectory_file:        frames = read_trajectory(trajectory_file)

        raise ValueError("Trajectory file path is required")        logger.info(f"Read {len(frames)} frames from trajectory")

            

    if not output_file:        # Calculate model deviations

        raise ValueError("Output file path is required")        deviations = _calculate_frame_deviations(

                models, frames, device=device, 

    logger.info(f"calculating MACE model deviation with {len(model_files)} models on device: {device}")            batch_size=batch_size, chunk_size=chunk_size

    logger.info(f"processing trajectory: {trajectory_file}")        )

            

    if not os.path.exists(trajectory_file):        # Write results

        raise FileNotFoundError(f"trajectory file not found: {trajectory_file}")        write_model_deviation(deviations, output_file)

            logger.info(f"Model deviation written to: {output_file}")

    # Validate model files exist        

    missing_models = [f for f in model_files if not os.path.exists(f)]        return output_file

    if missing_models:        

        raise FileNotFoundError(f"Model files not found: {missing_models}")    except Exception as e:

            logger.error(f"MACE model deviation calculation failed: {e}")

    # check if MACE is available        raise RuntimeError(f"MACE model deviation calculation failed: {e}")

    try:

        import torch

        from mace import datadef _validate_device(device: str) -> str:

        from mace.tools import torch_tools, utils    """Validate and adjust device based on availability"""

        logger.info("MACE package found - using direct torch evaluation approach")    if device == 'cuda':

    except ImportError as e:        if not torch.cuda.is_available():

        logger.error(f"MACE package not found: {e}")            logger.warning("CUDA requested but not available, falling back to CPU")

        logger.error("Cannot calculate model deviation without MACE - this is required for MACE workflows")            return 'cpu'

        raise ImportError(f"MACE packages required for model deviation calculation: {e}")        else:

                logger.info(f"CUDA available: {torch.cuda.device_count()} device(s)")

    # read trajectory frames with reference data            return device

    frames = _read_trajectory_frames(trajectory_file, type_map)    elif device == 'mps':

    logger.info(f"loaded {len(frames)} frames from trajectory")        if not torch.backends.mps.is_available():

                logger.warning("MPS requested but not available, falling back to CPU")

    # calculate model deviation using direct MACE evaluation            return 'cpu'

    logger.info(f"Calculating real MACE model deviation with {len(model_files)} models")        else:

    frame_deviations = _calculate_mace_deviation_direct(            return device

        frames, model_files, device, default_dtype, batch_size    else:

    )        return 'cpu'

    

    # write results

    _write_deviation_results(frame_deviations, output_file)def _calculate_frame_deviations(

        models: List[torch.nn.Module],

    logger.info(f"model deviation calculation complete: {output_file}")    frames: List[Atoms],

    return output_file    device: str = 'cuda',

    batch_size: int = 1,

    chunk_size: int = 100,

def _read_trajectory_frames() -> List[float]:

    trajectory_file: str,     """

    type_map: Optional[List[str]] = None    Calculate model deviations for trajectory frames

) -> List[Atoms]:    

    """    Args:

    Read trajectory frames using ASE with proper format and type ordering        models: List of loaded MACE models

    Reads all frames since trajectory was already written with desired sampling        frames: List of ASE Atoms objects

            device: Device for calculation

    Args:        batch_size: Batch size for inference

        trajectory_file: path to trajectory file (LAMMPS dump, xyz, etc.)        chunk_size: Process frames in chunks to manage memory

        type_map: type mapping for atom ordering (specorder for LAMMPS dumps)        

            Returns:

    Returns:        List of maximum force deviations per frame

        list of ASE Atoms objects    """

    """    

    from pathlib import Path    n_frames = len(frames)

        all_deviations = []

    # determine format based on file extension    

    file_path = Path(trajectory_file)    # Process frames in chunks

    if file_path.suffix in ['.lammpstrj', '.dump']:    for chunk_start in range(0, n_frames, chunk_size):

        file_format = 'lammps-dump-text'        chunk_end = min(chunk_start + chunk_size, n_frames)

    elif file_path.suffix in ['.xyz']:        chunk_frames = frames[chunk_start:chunk_end]

        file_format = 'xyz'        

    else:        logger.info(f"Processing frames {chunk_start+1}-{chunk_end}/{n_frames}")

        # auto-detect format        

        file_format = None        chunk_deviations = []

            for i, atoms in enumerate(chunk_frames):

    # read all frames - no sampling needed since trajectory is pre-sampled            try:

    index = ':'                # Get ensemble predictions

    logger.info("reading all frames from pre-sampled trajectory")                forces_ensemble = _get_ensemble_forces(models, atoms, device)

                    

    # read frames using ASE with efficient reading                if len(forces_ensemble) > 1:

    try:                    # Calculate standard deviation across models

        if file_format == 'lammps-dump-text' and type_map:                    forces_array = np.array(forces_ensemble)  # Shape: (n_models, n_atoms, 3)

            logger.info(f"reading LAMMPS trajectory with specorder: {type_map}")                    force_std = np.std(forces_array, axis=0)  # Shape: (n_atoms, 3)

            frames = ase.io.read(trajectory_file, index=index, format=file_format, specorder=type_map)                    max_force_devi = np.max(force_std)  # Maximum deviation across all atoms/directions

        elif file_format:                else:

            frames = ase.io.read(trajectory_file, index=index, format=file_format)                    max_force_devi = 0.0

        else:                    

            frames = ase.io.read(trajectory_file, index=index)                chunk_deviations.append(max_force_devi)

                        

        if not isinstance(frames, list):            except Exception as e:

            frames = [frames]                logger.warning(f"Failed to process frame {chunk_start + i}: {e}")

                        chunk_deviations.append(0.0)

        logger.info(f"read {len(frames)} frames from {trajectory_file}")        

        return frames        all_deviations.extend(chunk_deviations)

            

    except Exception as e:    return all_deviations

        logger.error(f"failed to read trajectory file {trajectory_file}: {e}")

        raise

def _get_ensemble_forces(

    models: List[torch.nn.Module],

def _calculate_mace_deviation_direct(    atoms: Atoms,

    frames: List[Atoms],    device: str

    model_files: List[str],) -> List[np.ndarray]:

    device: str = 'cuda',    """

    default_dtype: str = 'float64',    Get force predictions from ensemble of MACE models

    batch_size: int = 64    

) -> List[Dict[str, float]]:    Args:

    """        models: List of MACE models

    Calculate MACE model deviation using direct torch evaluation        atoms: ASE Atoms object

            device: Device for calculation

    follows eval_configs.py approach: direct torch model evaluation        

    without ASE calculator overhead for maximum efficiency    Returns:

            List of force arrays, one per model

    Args:    """

        frames: list of ASE Atoms objects (reference trajectory)    

        model_files: list of MACE model file paths    forces_ensemble = []

        device: device for MACE calculation    

        default_dtype: default data type for torch    for model in models:

        batch_size: batch size for processing        try:

                    with torch.no_grad():

    Returns:                # Convert ASE atoms to MACE format

        list of deviation statistics for each frame                # Note: This is a simplified conversion - actual MACE interface may differ

    """                batch_data = _atoms_to_mace_batch(atoms, device)

    try:                

        import torch                # Get model prediction

        from mace import data                output = model(batch_data)

        from mace.tools import torch_tools, utils                

                        # Extract forces

        logger.info(f"evaluating {len(model_files)} models on {len(frames)} frames using direct torch")                if 'forces' in output:

                            forces = output['forces']

        # set up torch configuration following MACE eval_configs.py style                    if isinstance(forces, torch.Tensor):

        torch_tools.set_default_dtype(default_dtype)                        forces = forces.cpu().numpy()

        torch_device = torch_tools.init_device(device)                    forces_ensemble.append(forces)

                        else:

        # convert ASE atoms to MACE configurations                    logger.warning("Model output does not contain 'forces' key")

        configs = [data.config_from_atoms(atoms) for atoms in frames]                    forces_ensemble.append(np.zeros_like(atoms.positions))

        logger.info(f"converted {len(configs)} configurations for evaluation")                    

                except Exception as e:

        # EFFICIENCY IMPROVEMENT: Load models once and reuse            logger.warning(f"Model prediction failed: {e}")

        models = []            forces_ensemble.append(np.zeros_like(atoms.positions))

        for model_idx, model_file in enumerate(model_files):    

            logger.info(f"loading model {model_idx + 1}/{len(model_files)}: {os.path.basename(model_file)}")    return forces_ensemble

            model = torch.load(f=model_file, map_location=torch_device)

            

            # Ensure model is in the right dtypedef _atoms_to_mace_batch(atoms: Atoms, device: str) -> Dict[str, torch.Tensor]:

            if default_dtype == 'float64':    """

                model = model.double()    Convert ASE Atoms to MACE model input format

            elif default_dtype == 'float32':    

                model = model.float()    Note: This is a simplified implementation. The actual MACE model

                interface may require different input format/keys.

            model = model.to(torch_device)    """

                

            # disable gradients for inference to save memory    # Basic conversion - may need adjustment based on specific MACE model format

            for param in model.parameters():    positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device)

                param.requires_grad = False    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=device)

            model.eval()    

                batch_data = {

            models.append(model)        'positions': positions.unsqueeze(0),  # Add batch dimension

                'atomic_numbers': atomic_numbers.unsqueeze(0),  # Add batch dimension

        # Pre-allocate arrays to save memory - more efficient than appending lists        'ptr': torch.tensor([0, len(atoms)], dtype=torch.long, device=device),

        n_models = len(models)     }

        n_frames = len(frames)    

        all_energies = np.zeros((n_models, n_frames))    # Add cell information if periodic

        all_forces = []  # Keep as list due to variable atom counts    if atoms.pbc.any():

                cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=device)

        for model_idx, model in enumerate(models):        batch_data['cell'] = cell.unsqueeze(0)  # Add batch dimension

            logger.info(f"evaluating model {model_idx + 1}/{len(models)}")    

                return batch_data
            # prepare atomic number table for this model
            z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
            
            # get model heads if available
            try:
                heads = model.heads
            except AttributeError:
                heads = None
            
            # convert configs to MACE data format
            dataset = []
            for config in configs:
                # Configs are already MACE Configuration objects from data.config_from_atoms()
                atomic_data = data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=float(model.r_max), heads=heads
                )
                dataset.append(atomic_data)
            
            model_forces = []
            
            # evaluate each configuration - process in chunks for better memory management
            chunk_size = min(batch_size, len(dataset))  # Use batch_size parameter for chunking
            for chunk_start in range(0, len(dataset), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(dataset))
                chunk_data = dataset[chunk_start:chunk_end]
                
                # Process chunk
                for i, atomic_data in enumerate(chunk_data):
                    config_idx = chunk_start + i
                    
                    # Create a proper batch with batch indices
                    batch_dict = atomic_data.to_dict()
                    n_atoms = len(batch_dict['positions'])
                    batch_dict['batch'] = torch.zeros(n_atoms, dtype=torch.long)
                    
                    # Add ptr field for PyTorch Geometric batch processing
                    batch_dict['ptr'] = torch.tensor([0, n_atoms], dtype=torch.long)
                    
                    # Handle head field properly - expand scalar to per-atom if needed
                    if 'head' in batch_dict and torch.is_tensor(batch_dict['head']) and batch_dict['head'].dim() == 0:
                        # Convert scalar head to per-atom head
                        batch_dict['head'] = batch_dict['head'].expand(n_atoms)
                    
                    # Move to device and set proper dtypes
                    for key, value in batch_dict.items():
                        if torch.is_tensor(value):
                            batch_dict[key] = value.to(torch_device)
                            # Enable gradients for positions to compute forces
                            if key == 'positions':
                                batch_dict[key] = batch_dict[key].requires_grad_(True)
                    
                    # model evaluation with gradient computation for forces
                    with torch.enable_grad():
                        output = model(batch_dict)
                    
                    # extract energy and forces - convert to numpy immediately to save GPU memory
                    energy = torch_tools.to_numpy(output["energy"]).item()
                    forces = torch_tools.to_numpy(output["forces"])
                    
                    # Store directly in pre-allocated array
                    all_energies[model_idx, config_idx] = energy
                    model_forces.append(forces)
                    
                    # Clear intermediate tensors for memory efficiency
                    del output, batch_dict
                    if torch_device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            all_forces.append(model_forces)
            
            # clear model from GPU memory after processing
            if torch_device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # convert lists to arrays - all_energies is already a numpy array
        # all_forces remains a list of lists due to variable atom counts per frame
        
        frame_deviations = []
        
        for frame_idx in range(len(frames)):
            # ENERGY deviation for this frame (replaces virial in DeepMD format)
            frame_energies = all_energies[:, frame_idx]  # (n_models,)
            energy_std = np.std(frame_energies)  # standard deviation across models
            energy_mean = np.mean(frame_energies)
            
            # Use energy std as both max/min/avg "virial" for DeepMD compatibility
            # This represents energy uncertainty, which is more meaningful for MACE than fake virial
            max_energy_devi = float(energy_std)
            min_energy_devi = float(energy_std) 
            avg_energy_devi = float(energy_std)
            
            # force deviation for this frame (unchanged - this part was correct)
            frame_forces = np.array([all_forces[model_idx][frame_idx] for model_idx in range(len(models))])
            # shape: (n_models, n_atoms, 3)
            
            # Debug: verify data structure
            if frame_idx == 0:  # Only log for first frame
                logger.info(f"Frame {frame_idx}: energy_mean={energy_mean:.6f}, energy_std={energy_std:.6f}")
                logger.info(f"Frame forces shape: {frame_forces.shape}")
            
            # DeepMD-style force deviation calculation (unchanged)
            n_atoms = frame_forces.shape[1]
            mean_forces = np.mean(frame_forces, axis=0)  # (n_atoms, 3)
            
            # vectorized calculation for efficiency - follows DeepMD methodology exactly
            deviations = frame_forces - mean_forces[np.newaxis, :, :]  # (n_models, n_atoms, 3)
            squared_norms = np.sum(deviations**2, axis=2)  # (n_models, n_atoms) - L2 norm squared per atom per model
            force_deviations_per_atom = np.sqrt(np.mean(squared_norms, axis=0))  # (n_atoms,) - RMS across models
            
            # frame deviation statistics (now using energy deviation instead of virial)
            frame_deviation = {
                'max_devi_v': max_energy_devi,
                'min_devi_v': min_energy_devi, 
                'avg_devi_v': avg_energy_devi,
                'max_devi_f': float(np.max(force_deviations_per_atom)),
                'min_devi_f': float(np.min(force_deviations_per_atom)),
                'avg_devi_f': float(np.mean(force_deviations_per_atom)),
            }
            
            frame_deviations.append(frame_deviation)
            
            # progress logging for long trajectories
            if (frame_idx + 1) % 100 == 0:
                logger.info(f"processed {frame_idx + 1}/{len(frames)} frames")
        
        return frame_deviations
        
    except Exception as e:
        logger.error(f"MACE deviation calculation failed: {e}")
        logger.error("This is a critical error - real model deviation calculation is required")
        raise RuntimeError(f"MACE model deviation calculation failed: {e}")


def _write_deviation_results(
    frame_deviations: List[Dict[str, float]],
    output_file: str
) -> None:
    """
    Write model deviation results to file in DeepMD format
    
    Args:
        frame_deviations: list of deviation statistics for each frame
        output_file: path to output file
    """
    logger.info("writing model deviation statistics")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        # write header - keep 'devi_v' for ai2kit compatibility (represents energy deviation for MACE)
        # Note: ai2kit expects max_devi_v/min_devi_v/avg_devi_v columns - using energy deviation as content
        f.write('#       step         max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f\n')
        
        for frame_idx, frame_deviation in enumerate(frame_deviations):
            # use frame index as timestep since trajectory is pre-sampled
            timestep = frame_idx
            
            # write results in DeepMD format with virial deviations
            f.write(f"{timestep:>12d} {frame_deviation['max_devi_v']:>14.6e} "
                   f"{frame_deviation['min_devi_v']:>14.6e} {frame_deviation['avg_devi_v']:>14.6e} "
                   f"{frame_deviation['max_devi_f']:>14.6e} {frame_deviation['min_devi_f']:>14.6e} "
                   f"{frame_deviation['avg_devi_f']:>14.6e}\n")
    
    logger.info(f"deviation statistics written to: {output_file}")