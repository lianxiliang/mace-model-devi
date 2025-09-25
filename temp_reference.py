from ai2_kit.core.log import get_logger
from ai2_kit.core.util import ensure_dir

from typing import List, Optional, Dict, Any, Union
import ase.io
from ase import Atoms
import numpy as np
import os

logger = get_logger(__name__)


class MaceModelDeviTool:
    """
    CLI tool for MACE model deviation calculation.
    """
    
    def calculate(
        self,
        models: str,
        traj: str,
        output: str,
        type_map: Optional[Union[str, tuple, list]] = None,
        device: str = 'cuda',
        batch_size: int = 64,
        default_dtype: str = 'float64'
    ) -> str:
        """
        Calculate MACE model deviation from command line.
        
        :param models: space-separated list of MACE model file paths
        :param traj: path to trajectory file
        :param output: path to output model_devi.out file
        :param type_map: element symbols (comma-separated string or tuple/list)
        :param device: device for calculation ('cuda', 'cpu', 'mps')
        :param batch_size: batch size for processing
        :param default_dtype: torch dtype ('float32', 'float64')
        :return: path to output file
        """
        # Parse model files from space-separated string
        model_files = models.strip().split()
        
        # Parse type_map from comma-separated string or tuple
        type_map_list = None
        if type_map:
            if isinstance(type_map, str):
                # Handle string input: "O,H" -> ["O", "H"]
                type_map_list = [t.strip() for t in type_map.split(',')]
            elif isinstance(type_map, (tuple, list)):
                # Handle tuple/list input from Fire CLI: ("O", "H") -> ["O", "H"] 
                type_map_list = [str(t).strip() for t in type_map]
            else:
                raise ValueError(f"Invalid type_map format: {type(type_map)}. Expected string or tuple.")
        
        logger.info(f"MACE model deviation calculation starting")
        logger.info(f"Models: {len(model_files)} files")
        logger.info(f"Trajectory: {traj}")
        logger.info(f"Output: {output}")
        logger.info(f"Device: {device}")
        
        # Call the main calculation function
        result = calculate_mace_model_deviation(
            model_files=model_files,
            traj_file=traj,
            output_file=output,
            type_map=type_map_list,
            device=device,
            batch_size=batch_size,
            default_dtype=default_dtype
        )
        
        logger.info(f"MACE model deviation calculation completed: {result}")
        return result


def calculate_mace_model_deviation(
    model_files: List[str],
    traj_file: str,
    output_file: str,
    type_map: Optional[List[str]] = None,
    chunk_size: int = 1000,
    device: str = 'cuda',
    batch_size: int = 64,
    default_dtype: str = 'float64'
) -> str:
    """
    Calculate MACE model deviation using DeepMD strategy
    
    Follows DeepMD approach:
    1. Read reference trajectory (from model 1 or LAMMPS/DFT reference)  
    2. Evaluate all models on the same configurations
    3. Calculate RMS deviation from ensemble mean (DeepMD methodology)
    4. Write results in DeepMD model_devi.out format
    
    :param model_files: list of MACE model file paths (.model files)
    :param traj_file: path to reference trajectory file (LAMMPS dump, xyz, etc.)
    :param output_file: path to output deviation file (model_devi.out format)
    :param type_map: type mapping for atoms (optional, for compatibility)
    :param chunk_size: process frames in chunks to manage memory (default: 1000)
    :param device: device for MACE calculation ('cuda', 'cpu', 'mps')
    :param batch_size: batch size for MACE evaluation (default: 64)
    :param default_dtype: default data type for torch ('float32', 'float64')
    :return: path to the output file
    :raises FileNotFoundError: if trajectory file does not exist
    :raises ValueError: if less than 2 models provided or invalid parameters
    :raises ImportError: if required MACE packages are not available
    """
    # Input validation
    if not model_files:
        raise ValueError("No model files provided")
    
    if len(model_files) < 2:
        raise ValueError(f"Need at least 2 models for meaningful deviation calculation, got {len(model_files)}")
    
    if not traj_file:
        raise ValueError("Trajectory file path is required")
    
    if not output_file:
        raise ValueError("Output file path is required")
    
    logger.info(f"calculating MACE model deviation with {len(model_files)} models on device: {device}")
    logger.info(f"processing trajectory: {traj_file}")
    
    if not os.path.exists(traj_file):
        raise FileNotFoundError(f"trajectory file not found: {traj_file}")
    
    # Validate model files exist
    missing_models = [f for f in model_files if not os.path.exists(f)]
    if missing_models:
        raise FileNotFoundError(f"Model files not found: {missing_models}")
    
    # check if MACE is available
    try:
        import torch
        from mace import data
        from mace.tools import torch_tools, utils
        use_real_mace = True
        logger.info("MACE package found - using direct torch evaluation approach")
    except ImportError as e:
        logger.error(f"MACE package not found: {e}")
        logger.error("Cannot calculate model deviation without MACE - this is required for MACE workflows")
        raise ImportError(f"MACE packages required for model deviation calculation: {e}")
    
    # read trajectory frames with reference data
    frames = _read_trajectory_frames(traj_file, type_map)
    logger.info(f"loaded {len(frames)} frames from trajectory")
    
    # calculate model deviation using direct MACE evaluation
    logger.info(f"Calculating real MACE model deviation with {len(model_files)} models")
    frame_deviations = _calculate_mace_deviation_direct(
        frames, model_files, device, default_dtype, batch_size
    )
    # write results
    _write_deviation_results(frame_deviations, output_file)
    
    logger.info(f"model deviation calculation complete: {output_file}")
    return output_file


def _read_trajectory_frames(
    trajectory_file: str, 
    type_map: Optional[List[str]] = None
) -> List[Atoms]:
    """
    Read trajectory frames using ASE with proper format and type ordering
    Reads all frames since trajectory was already written with desired sampling
    
    :param trajectory_file: path to trajectory file (LAMMPS dump, xyz, etc.)
    :param type_map: type mapping for atom ordering (specorder for LAMMPS dumps)
    :return: list of ASE Atoms objects
    """
    import ase.io
    from pathlib import Path
    
    # determine format based on file extension
    file_path = Path(trajectory_file)
    if file_path.suffix in ['.lammpstrj', '.dump']:
        file_format = 'lammps-dump-text'
    elif file_path.suffix in ['.xyz']:
        file_format = 'xyz'
    else:
        # auto-detect format
        file_format = None
    
    # read all frames - no sampling needed since trajectory is pre-sampled
    index = ':'
    logger.info("reading all frames from pre-sampled trajectory")
    
    # read frames using ASE with efficient reading
    try:
        if file_format == 'lammps-dump-text' and type_map:
            logger.info(f"reading LAMMPS trajectory with specorder: {type_map}")
            frames = ase.io.read(trajectory_file, index=index, format=file_format, specorder=type_map)
        elif file_format:
            frames = ase.io.read(trajectory_file, index=index, format=file_format)
        else:
            frames = ase.io.read(trajectory_file, index=index)
        
        if not isinstance(frames, list):
            frames = [frames]
        
        logger.info(f"read {len(frames)} frames from {trajectory_file}")
        return frames
        
    except Exception as e:
        logger.error(f"failed to read trajectory file {trajectory_file}: {e}")
        raise


def _calculate_mace_deviation_direct(
    frames: List[Atoms],
    model_files: List[str],
    device: str = 'cuda',
    default_dtype: str = 'float64',
    batch_size: int = 64
) -> List[Dict[str, float]]:
    """
    Calculate MACE model deviation using direct torch evaluation
    
    follows eval_configs.py approach: direct torch model evaluation
    without ASE calculator overhead for maximum efficiency
    
    :param frames: list of ASE Atoms objects (reference trajectory)
    :param model_files: list of MACE model file paths
    :param device: device for MACE calculation
    :param default_dtype: default data type for torch
    :return: list of deviation statistics for each frame
    """
    try:
        import torch
        from mace import data
        from mace.tools import torch_tools, utils
        
        logger.info(f"evaluating {len(model_files)} models on {len(frames)} frames using direct torch")
        
        # set up torch configuration following MACE eval_configs.py style
        torch_tools.set_default_dtype(default_dtype)
        torch_device = torch_tools.init_device(device)
        
        # convert ASE atoms to MACE configurations
        configs = [data.config_from_atoms(atoms) for atoms in frames]
        logger.info(f"converted {len(configs)} configurations for evaluation")
        
        # EFFICIENCY IMPROVEMENT: Load models once and reuse
        models = []
        for model_idx, model_file in enumerate(model_files):
            logger.info(f"loading model {model_idx + 1}/{len(model_files)}: {os.path.basename(model_file)}")
            model = torch.load(f=model_file, map_location=torch_device)
            
            # Ensure model is in the right dtype
            if default_dtype == 'float64':
                model = model.double()
            elif default_dtype == 'float32':
                model = model.float()
            
            model = model.to(torch_device)
            
            # disable gradients for inference to save memory
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            
            models.append(model)
        
        # Pre-allocate arrays to save memory - more efficient than appending lists
        n_models = len(models) 
        n_frames = len(frames)
        all_energies = np.zeros((n_models, n_frames))
        all_forces = []  # Keep as list due to variable atom counts
        
        for model_idx, model in enumerate(models):
            logger.info(f"evaluating model {model_idx + 1}/{len(models)}")
            
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
    
    :param frame_deviations: list of deviation statistics for each frame
    :param output_file: path to output file
    """
    logger.info("writing model deviation statistics")
    
    ensure_dir(output_file)
    
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