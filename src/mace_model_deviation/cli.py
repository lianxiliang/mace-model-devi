"""
Command-line interface for MACE model deviation calculation
"""

import argparse
import logging
import sys
from typing import List, Optional

from .core import calculate_mace_model_deviation
from .utils import setup_logging


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Calculate MACE model ensemble uncertainty/deviation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  mace-model-devi --models model1.pt model2.pt model3.pt --traj trajectory.xyz --output model_devi.out

  # With GPU device specification
  mace-model-devi --models *.pt --traj traj.lammpstrj --output results.out --device cuda

  # With type mapping
  mace-model-devi --models *.pt --traj trajectory.xyz --output model_devi.out --type-map O,H
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='MACE model files (.pt format). Can use wildcards or space-separated list.'
    )
    
    parser.add_argument(
        '--traj',
        required=True,
        help='Trajectory file path (ASE-readable format)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output file path for model deviation results'
    )
    
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device for calculation (default: cuda)'
    )
    
    parser.add_argument(
        '--type-map',
        help='Comma-separated list of element symbols (e.g., "O,H")'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for model inference (default: 1)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Number of frames to process at once (default: 100)'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='mace-model-deviation 0.1.0'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point"""
    
    try:
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("MACE Model Deviation Calculator v0.1.0")
        logger.info(f"Models: {len(args.models)} files")
        logger.info(f"Trajectory: {args.traj}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Device: {args.device}")
        
        # Parse type map
        type_map = None
        if args.type_map:
            type_map = [t.strip() for t in args.type_map.split(',')]
            logger.info(f"Type map: {type_map}")
        
        # Run calculation
        output_file = calculate_mace_model_deviation(
            model_files=args.models,
            trajectory_file=args.traj,
            output_file=args.output,
            type_map=type_map,
            device=args.device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        
        logger.info(f"✅ MACE model deviation calculation completed successfully!")
        logger.info(f"Results written to: {output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Calculation interrupted by user")
        return 1
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        logging.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())