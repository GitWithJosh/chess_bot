Source: Neural Networks for Chess, The magic of deep and reinforcement learning revealed  
Dominik Klein June 11, 2022

### AlphaZero
- is essentially AlphaGoZero adapted to chess
- 1 deep residual network (CNN with skip connections)
- combines Policy and value network

### Network Input
- basic building block: binary planes of size 8x8 (so either ones or zeros)
- 119 planes are necessary
- 12 planes: 6 planes for white pieces, 6 planes for black pieces
- 2 planes:  to track repetition of a position (first one all ones if occured once, second one all ones if twice)
- 98 planes: The 7 Previous Positions (7*14=98), at start history planes are all zeros (although author says 8 prev.)
- 1 plane: Encodes whose turn it is (all ones if white, all zeros if black)
- 4 planes: for castling right (white-kingside, white-queenside,etc.), set to all 1 if castling rights exist
- 2: Counters (directly as numbers) (one for total move count and one for progress = number of moves w.o. capture or pawn move)
- together 117 planes and 2 numbers (but also planes? to get to 119, author is unclear)
- 119 * (8x8 plane) = 7616 nodes

### Network Output
- Probabilites of every possible chess move (indicating how good the move is)
- a Value for that position, scalar (how likely white wins, black wins or draw)
- outputs for move probabilities is a little more tricky
- for every source square, count all possible moves, if a "superpiec" stood there (a queenhorse)
- each of these moves gets one output -> later illegal moves get set to zero and recalculate to get probabilities sum =1
- Output Move Encoding:
- Source Square | Direction | Number of Squares
- Source squares: Squares that have pieces on them
- Direction: Up, Down, Left, Right, Up Right, Down Right, Up Left, Down Left, two right and up, two up and right,...)
- Number of squares: 1-7
- Underpromotions: Source square | move type | promotes to
- move type: advance, capture left, capture right (how a pawn gets to the last rank)
- underpromotes to: knight, bishop, rook
- Yes, we do this for all sqaures, even tho most of them illegal (set to zero later)
- Where is promotion to queen? This is in the Queen-like moves, assuming the pawn queens there
- Now, calculate how many outputs we get:
- "Queen"-like moves: 64 (souurce sq.) * 8 (queen directions) * 7 (number of sq.) = 3854
- "Knight"-likie moves: 64 (source sq.) * 8 (possible sq.) = 512
- "underpromotions" (not queen promoted): 64 (source sq.) * (3 directions of pawn move) * 3 (pieces, NBR) = 576
- 3584 + 512 + 576 = 4672 outputs
- Note: There is another appraoch, that has less outputs but trains slower (enumerates all possible moves "a1ba1, a1c1,...)

### Network training
- trained usind Monte Carlo Tree Search (self-played games)
- for each move MCT search is conducted  using the current state of the net
- positions of these games, MCT search results and final outcome are used as training data 

### Once trained
- engine works by using Monte Carlo Tree search to find the best move in a given positoin
- guided by the network, because during selection network is queried for move probabilities in a position
- child nodes are selected that have highest probabilities
- second, for the simulation there are no random playouts,
- instead network is queried with position and the value returned is used
