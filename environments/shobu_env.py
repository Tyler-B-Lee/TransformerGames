import logging
logger = logging.getLogger(__name__)

import random
import gymnasium as gym
import torch
import numpy as np

# docker-compose exec app mpirun -np 2 python3 train.py -e shobu -ef 25600 -ne 50 -tpa 2560 -ob 2560
# docker-compose exec app mpirun -np 2 python3 train.py -e shobu -ef 25600 -ne 50 -tpa 2560 -ob 2560 -t 0.15 -ent 0.01 -oe 8 -os 0.0001
# Generation 6, games shorter on average (~120 turns total)
# docker-compose exec app mpirun -np 2 python3 train.py -e shobu -ef 20480 -ne 70 -tpa 2048 -ob 2048 -ent 0.01 -oe 8 -os 0.0001
# Generation 12, games much shorter again (~45 to 50 turns in total)
# docker-compose exec app mpirun -np 2 python3 train.py -e shobu -ne 120 -ent 0.01 -oe 5 -os 0.0001 -t 0.4
# Generation 16, racing to getting an unbeatable opening (15 to 20 moves total)
# docker-compose exec app mpirun -np 2 python3 train.py -e shobu -ne 150 -ent 0.03 -oe 5 -os 0.0001 -t 0.32
# Generation 21, much longer games again, around 40-60 moves now
# docker-compose exec app mpirun -np 2 python3 train.py -e shobu -ne 120 -ent 0.03 -oe 5 -os 0.0001 -t 0.36

class ShobuEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(ShobuEnv, self).__init__()
        self.name = 'shobu'
        self.n_players = 2
        self.manual = manual

        # all possible moves for a piece (direction, distance)
        # 0 = up, towards the opponent's side, continues clockwise through 7
        self.moves = {(x,y) for x in range(8) for y in (1,2)}
        # translates a movement direction into the (x,y) coordinate change for 1 move in that direction
        self.direction_list = [
            (0,1), # 0
            (1,1), # 1
            (1,0), # 2
            (1,-1), # 3
            (0,-1), # 4
            (-1,-1), # 5
            (-1,0), # 6
            (-1,1) # 7
        ]

        self.action_space = gym.spaces.Discrete( # size = 1024
            1024
        )
        self.observation_space = gym.spaces.Box(0, 1, ( # size = 320 + 1024 = 1,344
            64 * 5 # board info from view of current player
            + self.action_space.n  # legal_actions, which indirectly tells the passive direction chosen
            , )
        )
        self.verbose = verbose

    @property
    def current_player(self):
        return self.players[self.current_player_num]
    
    @property
    def opposing_player(self):
        i = (self.current_player_num + 1) % 2
        return self.players[i]
        
    @property
    def observation(self):
        # ret = self.get_board_array()
        # if self.current_player_num == 1: # flip boards / reverse color for white
        #     ret = ret[::-1] * -1
        #     for i in range(4):
        #         ret[i] = np.flip(ret[i])
        
        # foo = np.zeros(3)
        # if self.passive_move_next:
        #     foo[0] = 1
        # else:
        #     foo[1] = 1
        # foo[2] = self.turns_taken / 500
        # ret = np.append(ret, foo)

        # return np.append(ret, self.legal_actions)

        
        board_array = self.get_board_array()
        if self.current_player_num == 1: # flip boards / reverse color for white
            board_array = board_array[::-1] * -1
            for i in range(4):
                board_array[i] = np.flip(board_array[i])
        board_array = board_array.flatten()

        ret = []
        draw_frac = self.actions_until_draw / 200
        for pos_i in range(64):
            token = [0.0] * 5

            if board_array[pos_i] > 0.5: # own piece
                token[0] = 1.0
            elif board_array[pos_i] < -0.5:
                token[1] = 1.0
            token[2] = draw_frac
            if self.passive_move_next:
                token[3] = 1.0
            else:
                token[4] = 1.0

            ret.extend(token)

        ret = torch.tensor(ret, dtype=torch.float32)
        return torch.cat([ret, self.legal_actions])


    def is_off_board(self, row: int, col: int):
        "Returns True if the given row and column IS OFF of one of the 4x4 game boards, False if it IS ON a board."
        return (row < 0) or (col < 0) or (row > 3) or (col > 3)

    def is_friendly_piece(self,piece_num:int):
        "Returns True if the given piece number is a friendly piece of the current player (black is 1, white is -1)."
        if self.current_player_num == 0:
            return (piece_num == 1)
        return (piece_num == -1)

    def valid_aggressive_helper(self, board_i:int, start_row:int, start_col:int, move: tuple):
        """Finds if the piece at the given start coords can make the given move from its current position
        as a legal AGGRESSIVE move. Rotates moves for the white player.
        Moves are a tuple: ( direction ID (0 to 7), distance (1 or 2) )
        Returns a 3-tuple: a bool for if the move is legal, and the x and y coords of the
        target square if legal."""
        board = self.boards[ board_i ]
        dist = move[1]
        # we must move the opposite direction for the white player
        move_id = move[0] if (self.current_player_num == 0) else (move[0] + 4) % 8
        direction = self.direction_list[ move_id ] # returns tuple for relative movement (x,y)
        # Current location -> Space A -> Space B -> Space C
        A_row, A_col = start_row - direction[1], start_col + direction[0]
        if self.is_off_board(A_row, A_col):
            return False,A_row,A_col
        if dist == 1:
            A = board.get((A_row,A_col),None)
            if A is None:
                return True,A_row,A_col
            if self.is_friendly_piece(A):
                return False,A_row,A_col
            B_row, B_col = A_row - direction[1], A_col + direction[0]
            if self.is_off_board(B_row, B_col):
                return True,A_row,A_col
            return (board.get((B_row,B_col),None) is None),A_row,A_col

        # dist == 2 here
        B_row, B_col = A_row - direction[1], A_col + direction[0]
        if self.is_off_board(B_row, B_col):
            return False,B_row,B_col
        A,B = board.get((A_row,A_col),None), board.get((B_row,B_col),None)
        if (A is None) and (B is None):
            return True,B_row,B_col # both A and B have no stones, so no pushing happens
        # there is a stone on A or B (or both)
        C_row, C_col = B_row - direction[1], B_col + direction[0]
        if A is not None:
            if self.is_friendly_piece(A):
                return False,B_row,B_col
            # enemy stone on A: legal only if B is empty and C is either off board or empty
            return ((B is None) and (self.is_off_board(C_row,C_col) or (board.get((C_row,C_col),None) is None))),B_row,B_col
        else: # must be a stone on B and not on A
            if self.is_friendly_piece(B):
                return False,B_row,B_col
            return (self.is_off_board(C_row,C_col) or (board.get((C_row,C_col),None) is None)),B_row,B_col


    def find_legal_passive_directions(self):
        """Finds and saves two sets in this object: one for aggressive directions legal on the current player's left side
        board and one for ones on their right side board. Note that the aggressive legal on one side = passive legal on other.
        The sets will contain integers tuples, representing each possible move a piece can make (direction ID, distance)."""
        self.left_side_aggressive_moves = set()
        self.right_side_aggressive_moves = set()
        # first, find the passive moves on the right side for this player - need aggressive directions on left
        left_boards = (0,2) if (self.current_player_num == 0) else (3,1) # which boards are on the left
        right_boards = (3,1) if (self.current_player_num == 0) else (0,2) # which boards are on the left
        # find aggresive directions possible on left
        # logger.debug(f"> Finding legal aggressive directions on left side...")
        for board_i in left_boards:
            b = self.boards[board_i]
            # logger.debug(f"\tChecking board {board_i}...")
            for coords,color in b.items():
                if self.is_friendly_piece(color):
                    # logger.debug(f"\tChecking piece at {coords}...")
                    moves_to_check = self.moves - self.left_side_aggressive_moves
                    # logger.debug(f"\tMoves left to check: {moves_to_check}")
                    for move in moves_to_check:
                        valid,x,y = self.valid_aggressive_helper(board_i,coords[0],coords[1],move)
                        if valid:
                            # logger.debug(f"\t\tMove {move} is valid!")
                            self.left_side_aggressive_moves.add(move)
        # next, for the right side
        # logger.debug(f"> Finding legal aggressive directions on right side...")
        for board_i in right_boards:
            b = self.boards[board_i]
            # logger.debug(f"\tChecking board {board_i}...")
            for coords,color in b.items():
                if self.is_friendly_piece(color):
                    # logger.debug(f"\tChecking piece at {coords}...")
                    moves_to_check = self.moves - self.right_side_aggressive_moves
                    # logger.debug(f"\tMoves left to check: {moves_to_check}")
                    for move in moves_to_check:
                        valid,x,y = self.valid_aggressive_helper(board_i,coords[0],coords[1],move)
                        if valid:
                            # logger.debug(f"\t\tMove {move} is valid!")
                            self.right_side_aggressive_moves.add(move)

    def valid_passive_helper(self, board_i:int, start_row:int, start_col:int, move: tuple):
        """Finds if the piece at the given coords can make the given move from its current position as
        a legal PASSIVE move. Rotates moves for the white player. Assumes an aggressive
        mirroring move can be made. Moves are a tuple: ( direction ID (0 to 7), distance (1 or 2) ).
        Returns a tuple: with 3 parts: a bool for if the move is legal, and the x and y coords of the
        target square if legal."""
        board = self.boards[ board_i ]
        dist = move[1]
        # we must move the opposite direction for the white player
        move_id = move[0] if (self.current_player_num == 0) else (move[0] + 4) % 8
        direction = self.direction_list[ move_id ] # returns tuple for relative movement (x,y)
        # Current location -> Space A -> Space B -> Space C
        A_row, A_col = start_row - direction[1], start_col + direction[0]
        if self.is_off_board(A_row, A_col):
            return False,A_row,A_col
        if dist == 1:
            return (board.get((A_row,A_col),None) is None),A_row,A_col
        # dist == 2 here
        B_row, B_col = A_row - direction[1], A_col + direction[0]
        if self.is_off_board(B_row, B_col):
            return False,B_row,B_col
        A,B = board.get((A_row,A_col),None), board.get((B_row,B_col),None)
        return ( (A is None) and (B is None) ),B_row,B_col

    @property
    def legal_actions(self):
        moves = []
        if self.passive_move_next: # do we need to calculate ahead for the player to pick a passive move
            home_board_IDs = (0,1) if (self.current_player_num == 0) else (3,2)
            self.find_legal_passive_directions()
            left_home_pieces = [coords for coords,color in self.boards[home_board_IDs[0]].items() if self.is_friendly_piece(color)]
            for start_row,start_col in left_home_pieces: # passives on LEFT side of home board possible
                for move in self.right_side_aggressive_moves:
                    valid,end_row,end_col = self.valid_passive_helper(home_board_IDs[0],start_row,start_col,move)
                    if valid:
                        if self.current_player_num == 0:
                            moveID = home_board_IDs[0]*256 + (start_row*4+start_col)*16 + (end_row*4+end_col)
                        else:
                            moveID = (3-home_board_IDs[0])*256 + ((3-start_row)*4+(3-start_col))*16 + ((3-end_row)*4+(3-end_col))
                        moves.append(moveID)
            right_home_pieces = [coords for coords,color in self.boards[home_board_IDs[1]].items() if self.is_friendly_piece(color)]
            for start_row,start_col in right_home_pieces: # passives on RIGHT side of home board possible
                for move in self.left_side_aggressive_moves:
                    valid,end_row,end_col = self.valid_passive_helper(home_board_IDs[1],start_row,start_col,move)
                    if valid:
                        if self.current_player_num == 0:
                            moveID = home_board_IDs[1]*256 + (start_row*4+start_col)*16 + (end_row*4+end_col)
                        else:
                            moveID = (3-home_board_IDs[1])*256 + ((3-start_row)*4+(3-start_col))*16 + ((3-end_row)*4+(3-end_col))
                        moves.append(moveID)

        else: # otherwise, we must show which pieces can make an aggressive move
            if (self.passive_side == 'left'):
                opposite_boards = (1,3) if (self.current_player_num == 0) else (2,0)
            else:
                opposite_boards = (0,2) if (self.current_player_num == 0) else (3,1)
            # logger.debug(f"Passive Side: {self.passive_side}")
            # logger.debug(f"Passive Move Made Last: {self.passive_move_made}")

            # find aggressive moves on the appropriate boards, opposite passive side
            opposite_pieces_1 = [coords for coords,color in self.boards[opposite_boards[0]].items() if self.is_friendly_piece(color)]
            # logger.debug(f"opp_pieces 1: {opposite_pieces_1}")
            for start_row,start_col in opposite_pieces_1:
                valid,end_row,end_col = self.valid_aggressive_helper(opposite_boards[0],start_row,start_col,self.passive_move_made)
                if valid:
                    if self.current_player_num == 0:
                        moveID = opposite_boards[0]*256 + (start_row*4+start_col)*16 + (end_row*4+end_col)
                    else:
                        moveID = (3-opposite_boards[0])*256 + ((3-start_row)*4+(3-start_col))*16 + ((3-end_row)*4+(3-end_col))
                    moves.append(moveID)
                    
            opposite_pieces_2 = [coords for coords,color in self.boards[opposite_boards[1]].items() if self.is_friendly_piece(color)]
            # logger.debug(f"opp_pieces 2: {opposite_pieces_2}")
            for start_row,start_col in opposite_pieces_2:
                valid,end_row,end_col = self.valid_aggressive_helper(opposite_boards[1],start_row,start_col,self.passive_move_made)
                if valid:
                    if self.current_player_num == 0:
                        moveID = opposite_boards[1]*256 + (start_row*4+start_col)*16 + (end_row*4+end_col)
                    else:
                        moveID = (3-opposite_boards[1])*256 + ((3-start_row)*4+(3-start_col))*16 + ((3-end_row)*4+(3-end_col))
                    moves.append(moveID)
        
        legal_actions = torch.zeros(self.action_space.n, dtype=torch.float32)
        if len(moves) == 0:
            legal_actions.put_(torch.tensor(0), torch.tensor(1.0))
        else:
            moves = torch.tensor(moves)
            legal_actions.put_(moves, torch.ones_like(moves, dtype=torch.float32))
        return legal_actions

    def move_piece(self,board_i:int,start_row:int,start_col:int,target_row:int,target_col:int):
        """Updates the row/col stored in the piece and the pointers of where the piece ends up, or removes
        it from the game when the target location is off the board."""
        board = self.boards[ board_i ]
        piece = board.pop((start_row,start_col)) # remove piece from board
        sym = '⚫ Black Piece' if piece == 1 else '⚪ White Piece'
        logger.debug(f"Moving {sym} on board {board_i} from ({start_row},{start_col}) to ({target_row},{target_col})")
        if self.is_off_board(target_row, target_col):
            # we must kill this piece, which must have belonged to the opposing player
            enemy_symbol = ' ⚫ Black ' if (self.current_player_num == 1) else ' ⚪ White '
            logger.debug(f"\n     --- Piece eliminated from board {board_i}! ({enemy_symbol})     ---")
            self.actions_until_draw = 201
            return
        # else, move the piece to the target square
        board[(target_row,target_col)] = piece

    def do_aggressive_move(self,board_i:int,start_row:int,start_col:int,target_row:int,target_col:int) -> None:
        """Aggressively moves the 'piece' object according to the given move tuple (direction, distance). 
        Updates the position of both the board object and the piece object itself, along with any stones
        that were pushed around."""
        board = self.boards[ board_i ]
        dist = max(abs(start_row - target_row),abs(start_col - target_col))
        # Current location -> Space A -> Space B -> Space C
        if dist == 1:
            A_row, A_col = target_row, target_col
            A = board.get((A_row,A_col), None)
            # is the spot to move to empty?
            if A is None:
                self.move_piece(board_i,start_row,start_col,target_row,target_col)
                return
            # we can assume that there is an enemy stone at A
            B_row, B_col = 2*target_row - start_row, 2*target_col - start_col
            self.move_piece(board_i,A_row,A_col,B_row,B_col)
            self.move_piece(board_i,start_row,start_col,A_row,A_col)
            return

        # dist == 2 here
        A_row, A_col = target_row - (target_row - start_row)//2, target_col - (target_col - start_col)//2
        B_row, B_col = target_row, target_col
        A,B = board.get((A_row,A_col),None), board.get((B_row,B_col),None)
        if (A is None) and (B is None):
            # both A and B have no stones, so no pushing happens
            self.move_piece(board_i, start_row, start_col, B_row, B_col)
            return
        # there is a stone on A or B (or both)
        C_row, C_col = 2*B_row - A_row, 2*B_col - A_col
        if A is not None:
            # enemy stone on A: push them to C
            self.move_piece(board_i,A_row,A_col,C_row,C_col)
            self.move_piece(board_i,start_row,start_col,B_row,B_col)
            return
        else: # must be a stone on B and not on A: push them to C
            self.move_piece(board_i,B_row,B_col,C_row,C_col)
            self.move_piece(board_i,start_row,start_col,B_row,B_col)
            return
    
    def get_board_array(self):
        "Returns the current state of the board as a 4x4x4 array of 1/0/-1"
        full_board = np.zeros((4,4,4))
        for board_id in range(4):
            b = self.boards[board_id]
            for coords,piece in b.items():
                full_board[board_id][coords[0]][coords[1]] = piece
        return full_board

    def display_board(self):
        horiz_line = ('-' * 29)
        thick_line = '=' * 22
        full_blank_row = 'I   ' + ('   I   ' * 3) + '   I      |   ' + ('   |   ' * 3) + '   |'

        # for i,j in ((2,3),(0,1)):
        #     b1, b2 = self.boards[i], self.boards[j]
        #     logger.debug('\n\t' + horiz_line)
        #     for row in range(4):
        #         logger.debug('\t' + full_blank_row)
        #         rlist1 = []
        #         rlist2 = []
        #         for col in range(4):
        #             piece = b1[row][col]
        #             if piece:
        #                 rlist1.append(piece.symbol)
        #             else:
        #                 rlist1.append('------')
        #             piece = b2[row][col]
        #             if piece:
        #                 rlist2.append(piece.symbol)
        #             else:
        #                 rlist2.append('------')
        #         logger.debug('\tI' + 'I'.join(rlist1) + 'I      |' + '|'.join(rlist2) + '|')
        #         logger.debug('\t' + full_blank_row)
        #         logger.debug('\t' + horiz_line)
        #     if i == 2:
        #         logger.debug('\n' + thick_line)
        full_board = self.get_board_array()
        for i,j in ((2,3),(0,1)):
            b1, b2 = full_board[i], full_board[j]
            # logger.debug('\t' + horiz_line)
            for row in range(4):
                # logger.debug('\t' + full_blank_row)
                rlist1 = []
                rlist2 = []
                for col in range(4):
                    piece = b1[row][col]
                    if piece != 0:
                        rlist1.append('B' if piece == 1 else 'W')
                    else:
                        rlist1.append('-')
                    piece = b2[row][col]
                    if piece:
                        rlist2.append('B' if piece == 1 else 'W')
                    else:
                        rlist2.append('-')
                logger.debug('I  ' + ''.join(rlist1) + '  I  |  ' + ''.join(rlist2) + '  |')
                # logger.debug('\t' + full_blank_row)
                # logger.debug('\t' + horiz_line)
            if i == 2:
                logger.debug(thick_line)
        

    def error_catch(self):
        logger.debug(f"Current Player: {self.current_player.symbol}")
        logger.debug(f"Picking Passive Turn: {self.passive_move_next}")
        if not self.passive_move_next:
            logger.debug(f"Passive move made last: {self.passive_move_made}")
        self.display_board()

    def do_passive_move(self,board_i:int,start_row:int,start_col:int,target_row:int,target_col:int) -> None:
        """Passively moves the 'piece' object according to the given move tuple (direction, distance). 
        Updates the position of both the board object and the piece object itself."""
        self.move_piece(board_i,start_row,start_col,target_row,target_col)
    
    def is_game_over(self):
        """Checks if the current game state is a game-over situation. Always returns a tuple:
        A boolean for the variables 'done' and 'reward' in the step function."""
        # did anyone just push all of their opponent's stones off one of the boards?
        for board in self.boards:
            if (-1 not in board.values()):
                self.winner = '⬛ Black'
                return True, [1, -1]
            if (1 not in board.values()):
                self.winner = '⬜ White'
                return True, [-1, 1]
        # else, game is not over yet
        return False, [0,0]

    def cutoff_game(self):
        """Since the turn limit was reached, end the game and calculate the final rewards for the players."""
        bonuses_list = [0, 3.5, 2.5, 1.25, 1]
        bscore = wscore = 0
        for boardID in range(4):
            bpieces = self.players[0].active_pieces[boardID]
            wpieces = self.players[1].active_pieces[boardID]
            bcount = wcount = 0
            for i in range(4):
                if bpieces[i]:
                    bcount += 1
                if wpieces[i]:
                    wcount += 1
            bonus = bonuses_list[ min(bcount,wcount) ]
            pts = abs(bcount - wcount) * 0.02 * bonus
            if bcount < wcount:
                pts *= -1
            bscore += pts
            wscore -= pts
        if abs(bscore - wscore) < 0.01:
            score = [0,0]
        elif bscore > wscore:
            score = [0.4,-0.4]
        else:
            score = [-0.4,0.4]
        return True, score


    def step(self, action):
        reward = [0, 0]
        terminated = truncated = False

        # player has no legal passive moves
        if action == 0:
            if self.skipped_last_turn:
                logger.debug(">>> Ending game, neither player has any moves...")
                return self.observation, [0,0], False, True, {}
            else:
                logger.debug(">>> Player has no legal actions, skipping their turn...")
                self.current_player_num = (self.current_player_num + 1) % 2
                self.skipped_last_turn = True
                return self.observation, reward, False, False, {}

        self.skipped_last_turn = False
        # moves have 3 parts: board, start position, end position
        boardID, foo = divmod(action, 256)
        start, end = divmod(foo, 16)
        start_row,start_col = divmod(start,4)
        end_row,end_col = divmod(end,4)
        if self.current_player_num == 1: # white (flipping move for orientation with black's POV)
            boardID = 3 - boardID
            start_row,start_col = 3-start_row,3-start_col
            end_row,end_col = 3-end_row,3-end_col
        dist = max(abs(start_row - end_row),abs(start_col - end_col))
        x_dir = (end_col - start_col) // dist
        y_dir = (start_row - end_row) // dist

        if self.passive_move_next:
            direction = self.direction_list.index((x_dir,y_dir))
            if self.current_player_num == 0: # black
                self.passive_side = 'left' if (boardID == 0) else 'right'
                self.passive_move_made = (direction, dist)
            else: # white (flipping move for orientation with black's POV)
                self.passive_side = 'left' if (boardID == 3) else 'right'
                self.passive_move_made = ((direction + 4) % 8,dist)
            
            self.do_passive_move(boardID,start_row,start_col,end_row,end_col)

            self.passive_move_next = False
            self.actions_until_draw -= 1

        else:
            # play the aggressive move given and check for game over
            self.do_aggressive_move(boardID,start_row,start_col,end_row,end_col)

            self.turns_taken += 1
            self.actions_until_draw -= 1

            # check if draw is reached / game is too long
            if (self.actions_until_draw <= 0) or (self.turns_taken >= 500):
                truncated, reward = True, [0,0]

            # else, check if game is over
            else:
                terminated, reward = self.is_game_over()

            self.current_player_num = (self.current_player_num + 1) % 2
            self.passive_move_next = True
            self.passive_move_made = None

        self.done = terminated or truncated

        return self.observation, reward, terminated, truncated, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.boards = []
        
        for _ in range(4):
            board = {}
            for i in range(4):
                board[(3,i)] = 1 # black pieces are 1 by default
                board[(0,i)] = -1 # white is -1 by default
            self.boards.append(board)
        
        self.current_player_num = 0
        self.passive_move_next = True
        self.passive_move_made = None
        self.skipped_last_turn = False
        self.winner = 'Nobody?'

        self.turns_taken = 0
        self.actions_until_draw = 200
        self.done = False

        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation, {}


    def render(self, mode='human', close=False):
        
        if close:
            return

        if not self.done:
            logger.debug(f'\n\n-------TURN {self.turns_taken + 1}-----------')
            logger.debug(f"It is the {'Black' if self.current_player_num == 0 else 'White'} Player's turn")
            if (self.passive_move_next):
                logger.debug(f"☮️  -- Must Choose a PASSIVE Move --  ☮️")
            else:
                logger.debug(f"💢  -- Must Choose an AGGRESSIVE Move --  💢")
                logger.debug(f"- Needed: {'Left' if (self.passive_side == 'right') else 'Right'} Side with move {self.passive_move_made}")
        else:
            logger.debug(f'\n\n-------FINAL POSITION-----------')
            
        self.display_board()
        
        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')
            logger.debug(f'👑 - {self.winner} is victorious! - 👑')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Shobu!')

if __name__ == "__main__":
    env = ShobuEnv()
    env.reset()
    np.set_printoptions(threshold=np.inf)

    # done = False
    # action_count = 0
    # while action_count < 15:
    # while not done:
    #     env.display_board()
    #     legal_actions = [i for i,o in enumerate(env.legal_actions) if o != 0]
    #     logger.debug(f"> Action {action_count} - Player: {'Black' if env.current_player_num == 0 else 'White'}")
    #     logger.debug(f'Legal actions: {legal_actions}')
    #     print(f"> Action {action_count} - Player: {'Black' if env.current_player_num == 0 else 'White'}")
    #     # print(f'\nLegal actions: {legal_actions}')

    #     # action = 1023
    #     # while legal_actions[action] != 1:
    #     #     action = int(input("Choose a valid action: "))
    #     action = random.choice(legal_actions)
    #     print(f"\tAction Chosen: {action}")
    #     logger.info(f"\t> Action Chosen: {action}")
    #     obs,reward,done,_ = env.step(action)
    #     action_count += 1
    # env.display_board()
    # print("Reward:",reward)
    # print("Actions:",action_count)
    
    # obs = env.get_observation().reshape((40,8,8))
    # for i,sq in enumerate(obs):
    #     logger.debug(f"- Observation Square {i}:\n{sq}\n")

    # logger.setLevel(logging.WARNING)
    N = 100
    glens = []
    for _ in range(N):
        terminated = truncated = False
        action_count = 0
        while not (terminated or truncated):
            legal_actions = [i for i,o in enumerate(env.legal_actions) if o != 0]
            # logger.info(f"Player: {ID_TO_PLAYER[env.to_play()]}")
            # logger.info(f"> Action {action_count} - Legal Actions: {legal_actions}")
            # print(f"Player: {ID_TO_PLAYER[env.to_play()]}")
            # print(f"> Action {action_count} - Legal Actions: {legal_actions}")

            # action = -1
            # while action not in legal_actions:
            #     action = int(input("Choose a valid action: "))
            action = random.choice(legal_actions)
            # print(f"\tAction Chosen: {action}")
            # logger.info(f"\t> Action Chosen: {action}")
            obs,reward,terminated,truncated,_ = env.step(action)
            logger.debug(obs)
            action_count += 1
        glens.append(action_count)
        print(reward)
        env.display_board()
        env.reset()

    print(f"\nGames: {glens}")
    print(f"\nLongest Length: {max(glens)}")
    print(f"Shortest Length: {min(glens)}")
    print(f"Average Length: {sum(glens)/N}")