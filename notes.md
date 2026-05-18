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
- training is done by just using the results of MCT-searches
- for each move MCT search is conducted  using the current state of the net
- positions of these games, MCT search results and final outcome are used as training data

- For each possible move in a position, we query the network to get move probabilities (prior p)
- We assign those p to each edge corresponding to each legal move

- Secondly we also get the evaluation (value v) of the position from the network

### Step by step: MCTS and training

- start with randomly initialized weight parameters $\Theta$
- Initialize Q = W = N = 0
- Q is move value
- W is a helper variable that stores the sum of move values
- N is the visit count
- P is the prior probability from the network's policy head
  
#### Selection
- Start at root node and at each step choose the next node that maximizes $Q + u$
- How is $u$ computed given a move $m$: $c \cdot P \cdot \dfrac{\sqrt{\displaystyle \sum^m' N_{m'}}}{1 + N_m}$
- $c$ is a constant
- $P$ is prior probability which led to this node
- enumerator: node count of the parent (all possible moves but we chose m)
- $1 + N_m$ is the counter how often we visit this move during the game
- Impact of $u$: First explore moves with high initial prior $p$, later $u$ gets less important since it reduces (counter in the denominator rises)
- Consequence: move value $Q$ gets more important (behaviour is human-inspired, focus more in promising candidates)
- We continue to select until we find a leaf node (a node with unvisited moves $N_{m_i} = 0$)

#### Expansions
- query network with leaf node position; policy head results are the prior probability $p_m$ for each possible move of the leaf node
- value head results is value $v$ of this position
- expand leaf node by adding edges for each move
- each edge initializes $Q = W = N = 0$ and $p = p_m$

#### Simulation 
- Rollout-simulations are replaced by simpling computing value $v$
  
#### Backpropagation
- Starting from the leaf node the values Q, W and N are updated to the root
- Update formulas: $W = W + v$; $N+=1$ and $Q=V/N$

#### Select move 
- move is selected given policy $\pi_m$ in the root position
- $\pi_m = \dfrac{N^{1/ \tau}_m}{\displaystyle \sum^n N^{1/ \tau}_m}$
- denominator: sum over all possible moves $n$ at the root
- This is almost the same as just taking the move with the highest visit count
- $\tau$ is for exploration


### Once trained
- engine works by using Monte Carlo Tree search to find the best move in a given positoin
- guided by the network, because during selection network is queried for move probabilities in a position
- child nodes are selected that have highest probabilities
- second, for the simulation there are no random playouts,
- instead network is queried with position and the value returned is used
