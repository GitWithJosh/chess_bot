import chess
import chess.pgn
import numpy as np
import os
import logging
from typing import List, Tuple, Optional
from tqdm import tqdm
from pathlib import Path

# Input/Output Directories
BATCH_SOURCE_DIR = "01_Data/Batches/Batch_GM"  # Source batch folder (from 01_Data)
TENSOR_OUTPUT_DIR = "02_PreProcessing/Tensors/GM"  # Output tensor folder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessDataProcessor:
    """Processes PGN files into training data for neural networks"""
    
    def __init__(self):
        """Initialize the chess data processor with move mapping"""
        self.move_to_index = {}
        self.index_to_move = {}
        self._build_move_mapping()
        
        # Create output directory if it doesn't exist
        self.output_dir = Path(TENSOR_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChessDataProcessor initialized")
        logger.info(f"Input directory: {BATCH_SOURCE_DIR}")
        logger.info(f"Output directory: {TENSOR_OUTPUT_DIR}")
    
    def _build_move_mapping(self):
        """Build mapping from moves to indices (AlphaZero-style)"""
        index = 0
        
        # Queen moves (56 directions × 7 squares = 392)
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        for direction in directions:
            for distance in range(1, 8):
                for from_rank in range(8):
                    for from_file in range(8):
                        to_rank = from_rank + direction[0] * distance
                        to_file = from_file + direction[1] * distance
                        if 0 <= to_rank < 8 and 0 <= to_file < 8:
                            from_square = chess.square(from_file, from_rank)
                            to_square = chess.square(to_file, to_rank)
                            move_key = f"{from_square}_{to_square}"
                            if move_key not in self.move_to_index:
                                self.move_to_index[move_key] = index
                                self.index_to_move[index] = move_key
                                index += 1
        
        # Knight moves (8 directions)
        knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
        for knight_move in knight_moves:
            for from_rank in range(8):
                for from_file in range(8):
                    to_rank = from_rank + knight_move[0]
                    to_file = from_file + knight_move[1]
                    if 0 <= to_rank < 8 and 0 <= to_file < 8:
                        from_square = chess.square(from_file, from_rank)
                        to_square = chess.square(to_file, to_rank)
                        move_key = f"{from_square}_{to_square}"
                        if move_key not in self.move_to_index:
                            self.move_to_index[move_key] = index
                            self.index_to_move[index] = move_key
                            index += 1
        
        # Promotions (simplified - just queen promotions for now)
        for from_file in range(8):
            # White pawn promotions
            from_square = chess.square(from_file, 6)
            to_square = chess.square(from_file, 7)
            move_key = f"{from_square}_{to_square}_q"
            if move_key not in self.move_to_index:
                self.move_to_index[move_key] = index
                self.index_to_move[index] = move_key
                index += 1
            
            # Black pawn promotions
            from_square = chess.square(from_file, 1)
            to_square = chess.square(from_file, 0)
            move_key = f"{from_square}_{to_square}_q"
            if move_key not in self.move_to_index:
                self.move_to_index[move_key] = index
                self.index_to_move[index] = move_key
                index += 1
        
        logger.info(f"Built move mapping with {len(self.move_to_index)} unique moves")
    
    def board_to_tensor(self, board: chess.Board) -> np.ndarray:
        """Convert chess board to 8x8x18 tensor representation
        
        Args:
            board: Chess board position
            
        Returns:
            numpy array of shape (8, 8, 18) representing the board state
        """
        tensor = np.zeros((8, 8, 18), 
                         dtype=getattr(np, "float16"))
        
        # Piece locations (channels 0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                piece_type = piece.piece_type - 1  # 0-5 for pawn-king
                color_offset = 0 if piece.color == chess.WHITE else 6
                tensor[rank, file, piece_type + color_offset] = 1
        
        # Castling rights (channels 12-15)
        tensor[:, :, 12] = float(board.has_kingside_castling_rights(chess.WHITE))
        tensor[:, :, 13] = float(board.has_queenside_castling_rights(chess.WHITE))
        tensor[:, :, 14] = float(board.has_kingside_castling_rights(chess.BLACK))
        tensor[:, :, 15] = float(board.has_queenside_castling_rights(chess.BLACK))
        
        # En passant target square (channel 16)
        if board.ep_square is not None:
            ep_rank, ep_file = divmod(board.ep_square, 8)
            tensor[ep_rank, ep_file, 16] = 1
        
        # Side to move (channel 17)
        tensor[:, :, 17] = float(board.turn == chess.WHITE)
        
        return tensor
    
    def encode_move(self, move: chess.Move) -> int:
        """Encode chess move to index using the move mapping
        
        Args:
            move: Chess move to encode
            
        Returns:
            Integer index representing the move (0 if move not found)
        """
        from_square = move.from_square
        to_square = move.to_square
        
        if move.promotion:
            promotion_piece = chess.piece_symbol(move.promotion)
            move_key = f"{from_square}_{to_square}_{promotion_piece}"
        else:
            move_key = f"{from_square}_{to_square}"
        
        return self.move_to_index.get(move_key, 0)  # Return 0 if move not found
    
    def process_pgn_batch(self, pgn_files: List[str], batch_size: int = None) -> List[Tuple]:
        """Process PGN files into training samples
        
        Args:
            pgn_files: List of PGN file paths to process
            batch_size: Maximum number of games to process (None for all)
            
        Returns:
            List of tuples (board_tensor, move_index, game_result)
        """
        if batch_size is None:
            batch_size = float('inf')
            
        samples = []
        games_processed = 0
        
        for pgn_file in pgn_files:
            logger.info(f"Processing {pgn_file}")
            
            try:
                with open(pgn_file, 'r', encoding='utf-8') as f:
                    while games_processed < batch_size:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        
                        # Determine game result
                        result = game.headers.get("Result", "*")
                        if result == "1-0":
                            game_result = 1.0
                        elif result == "0-1":
                            game_result = -1.0
                        else:
                            game_result = 0.0
                        
                        # Process moves
                        board = game.board()
                        for move in game.mainline_moves():
                            if move not in board.legal_moves:
                                logger.warning(f"Invalid move found in game, skipping: {move}")
                                break
                                
                                
                            # Create training sample
                            board_tensor = self.board_to_tensor(board)
                            move_index = self.encode_move(move)
                            
                            # Adjust result based on whose turn it is
                            adjusted_result = game_result if board.turn == chess.WHITE else -game_result
                            
                            samples.append((board_tensor, move_index, adjusted_result))
                            board.push(move)
                        
                        games_processed += 1
                        
                        if games_processed % 1000 == 0:
                            logger.info(f"Processed {games_processed} games, {len(samples)} samples")
                
            except Exception as e:
                logger.error(f"Error processing {pgn_file}: {e}")
                continue
        
        logger.info(f"Total samples created: {len(samples)}")
        return samples
    
    def save_batch(self, samples: List[Tuple], filename: str):
        """Save processed batch to compressed file
        
        Args:
            samples: List of training samples
            filename: Output filename (without extension)
        """
        if not samples:
            logger.warning("No samples to save")
            return
        
        # Convert to numpy arrays
        board_tensors = np.array([s[0] for s in samples])
        move_labels = np.array([s[1] for s in samples], dtype=getattr(np, "int32"))
        results = np.array([s[2] for s in samples], dtype=getattr(np, "float32"))
        
        # Create full file path
        output_path = self.output_dir / f"{filename}.npz"
        
        # Save compressed
        np.savez_compressed(
            output_path,
            X=board_tensors,
            y_move=move_labels,
            y_result=results
        )
        logger.info(f"Saved batch to {output_path}")
        logger.info(f"Batch contains {len(samples)} samples")
    
    def process_all_batches(self):
        """Process all PGN files in the configured batch directory"""
        batch_dir = Path(BATCH_SOURCE_DIR)
        
        if not batch_dir.exists():
            raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
        
        # Find all PGN files
        pgn_files = list(batch_dir.glob(f"*.pgn"))
        
        if not pgn_files:
            raise FileNotFoundError(f"No PGN files found in: {batch_dir}")
        
        # Sort files numerically by extracting batch number
        def extract_batch_number(file_path):
            try:
                # Extract number from filename like "Batch123.pgn"
                filename = file_path.stem  # Remove extension
                if filename.startswith("Batch"):
                    return int(filename[5:])  # Remove "Batch" prefix
                else:
                    return float('inf')  # Put non-batch files at the end
            except ValueError:
                return float('inf')  # Put files with invalid numbers at the end
        
        pgn_files.sort(key=extract_batch_number)
        
        logger.info(f"Found {len(pgn_files)} PGN files to process")
        
        # Process each batch file individually
        for pgn_file in pgn_files:
            logger.info(f"Processing batch: {pgn_file.name}")
            
            # Process the single file
            samples = self.process_pgn_batch([str(pgn_file)])
            
            # Save with same name as source file
            output_name = pgn_file.stem  # Filename without extension
            self.save_batch(samples, output_name)
            
            logger.info(f"Completed processing {pgn_file.name}")
        
        logger.info("All batches processed successfully!")
    
def main():
    """Main function to process batches"""
    logger.info("Starting Chess Data Processing")
    
    # Initialize processor
    processor = ChessDataProcessor()

    # Process all batches
    try:
        processor.process_all_batches()
        logger.info("Chess data processing completed successfully!")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
