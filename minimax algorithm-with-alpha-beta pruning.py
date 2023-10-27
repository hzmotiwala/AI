#!/usr/bin/env python
from isolation import Board, game_as_text
from random import randint




class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.
        Note:
            1. Be very careful while doing opponent's moves. You might end up
               reducing your own moves.
            2. If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                param1 (Board): The board and game state.
                param2 (bool): True if maximizing player is active.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """

       

        if maximizing_player_turn:
            MyMoves = len(game.get_legal_moves())
            OppMoves = len(game.get_opponent_moves())
   

            #print ("my eval score: ", MyMoves-OppMoves)
            #print ("my moves: ", MyMoves)
            #print ("opp moves: ", OppMoves)
        else:
            MyMoves = len(game.get_opponent_moves())
            OppMoves = len(game.get_legal_moves())
            #print ("my moves: ", MyMoves)
            #print ("opp moves: ", OppMoves)
            #print ("opp eval score: ", MyMoves-OppMoves)
        return MyMoves-OppMoves


class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.

        """

        
        if maximizing_player_turn:
            MyMoves = len(game.get_legal_moves())
            OppMoves = len(game.get_opponent_moves())
        else:
            MyMoves = len(game.get_opponent_moves())
            OppMoves = len(game.get_legal_moves())
        
        return MyMoves-OppMoves*2 


class CustomPlayer:
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        #print ("evals  here?")
        self.eval_fn = eval_fn
        self.search_depth = search_depth

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent

            Note:
                1. Do NOT change the name of this 'move' function. We are going to call
                the this function directly.
                2. Change the name of minimax function to alphabeta function when
                required. Here we are talking about 'minimax' function call,
                NOT 'move' function name.
                Args:
                game (Board): The board and game state.
                legal_moves (list): List of legal moves
                time_left (function): Used to determine time left before timeout

            Returns:
                tuple: best_move
            """
        
        #print ("legal moves: ", legal_moves)
        #print ("time left: ", time_left())

        utility = float('-inf')
        move = game.get_legal_moves()[0]

        while time_left() > 100:
            for depth in range(1,10):
                next_move, next_utility = self.alphabeta(game, time_left, depth=depth)
                #print ("depth: ", depth)
                #print ("time left: ", time_left())
                #print ("neeext  mooooooooooove: ", next_move)
                #print ("neeext  next_utility: ", next_utility)
                if next_utility == float('inf'):
                    #print ("neeext  mooooooooooove: ", next_move)
                    #print ("neeext  next_utility: ", next_utility)
                    return next_move
                #if next_utility >= utility:
                #move = next_move
                #utility = next_utility
                #print("time left now: ", time_left())
                if time_left() >= 50:
                    move = next_move

                #if time_left() < 50:
                    #print ("mooooooooooove: ", move)
                    #return next_move
                    

        #print ("finaaaaaal mooooooooooove: ", move)
        return move


        


        #previous
        #best_move, utility = self.alphabeta(game, time_left, depth=self.search_depth)
        #return best_move

    def utility(self, game, maximizing_player):
        """Can be updated if desired. Not compulsory."""
        #print ("utility score: ", self.eval_fn.score(game, maximizing_player))
        return self.eval_fn.score(game, maximizing_player)


    def minimax(self, game, time_left, depth, maximizing_player=True):
        #print("active queen: ", game.get_active_players_queen())
        #print("Inactive queen: ", game.get_inactive_players_queen())

        #check if game is over or last node or time is up
        if depth == 0 or time_left() < 50 or len(game.get_legal_moves()) == 0:
            return None, self.utility(game,maximizing_player) 

        # capture the results
        actions = game.get_legal_moves()
        values = []
        for action in actions:
            #print("minimax action: ", action)
            next_board, game_over_or_not, who_won = game.forecast_move(action)
            if game_over_or_not:
                return action, float('inf')
            val = self.min_play(next_board, time_left, depth - 1, not maximizing_player)
            #print ("minmininininsmminsindininiminimax value: ", val)
            values.append(val)
        # print("minimax values: ", values, "\n")
        best_val = max(values)
        #print("minimax best val: ", best_val)
        
        best_move_argmax = values.index(best_val)
        #print("minimax best_move_argmax: ", best_move_argmax)
        best_move = actions[best_move_argmax]
        #print ("minimax best_move: ", best_move, "\n")

        return best_move, best_val


    def max_play(self, game, time_left, depth, maximizing_player):
        #check if game is over or last node or time is up
        if depth == 0 or time_left() < 50 or len(game.get_legal_moves()) == 0:
            return self.utility(game,maximizing_player)

        best_val = float('-inf')
        values = []
        actions = game.get_legal_moves()
        for action in actions:
            next_board, game_over_or_not, who_won = game.forecast_move(action)
            if game_over_or_not:
                return float('inf')
            values.append(self.min_play(next_board, time_left, depth - 1, not maximizing_player))
        if len(values) > 0:
            best_val = max(values)
        return best_val

    def min_play(self, game, time_left, depth, maximizing_player):
        #check if game is over or last node or time is up
        if depth == 0 or time_left() < 50 or len(game.get_legal_moves()) == 0:
            return self.utility(game,maximizing_player)

        best_val = float('inf')
        values = []
        actions = game.get_legal_moves()
        for action in actions:
            next_board, game_over_or_not, who_won = game.forecast_move(action)
            if game_over_or_not:
                return float('-inf')
            values.append(self.max_play(next_board, time_left, depth - 1, not maximizing_player))
        if len(values) > 0:
            best_val = min(values)
        return best_val


    def alphabeta(self, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implementation of the alphabeta algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """
  
        #raise NotImplementedError
        
        #check if game is over or last node or time is up
        if depth == 0 or time_left() < 50 or len(game.get_legal_moves()) == 0:
            return None, self.utility(game,maximizing_player) 

        """
        #added jan 21 640pm
        val = float('-inf')
        best_move = None
        actions = game.get_legal_moves()
        for action in actions:
            next_board, game_over_or_not, who_won = game.forecast_move(action)
            if game_over_or_not:
                return action, float('inf')
            max_value=(self.min_play_ab(next_board, time_left, depth - 1, alpha, beta, not maximizing_player))
            if max_value > val:
                val = max_value
                best_move = action

            if val >= beta:
                break
            alpha = max(alpha, val)

        return best_move, val
        """

        
        # capture the results
        actions = game.get_legal_moves()
        values = []
        for action in actions:
            next_board, game_over_or_not, who_won = game.forecast_move(action)
            if game_over_or_not:
                return action, float('inf')
            values.append(self.min_play_ab(next_board, time_left, depth - 1, alpha, beta, not maximizing_player))
        #print("ab values: ", values, "\n")
        val = max(values)
        #print("ab best val: ", val)
        best_move_argmax = values.index(val)
        #print("ab best_move_argmax: ", best_move_argmax)
        best_move = actions[best_move_argmax]
        #print ("ab best_move: ", best_move, "\n")

        return best_move, val
        

    def max_play_ab(self, game, time_left, depth, alpha, beta, maximizing_player=True):
        #check if game is over or last node or time is up

        if depth == 0 or time_left() < 50 or len(game.get_legal_moves()) == 0:
            return self.utility(game,maximizing_player)

        value = float('-inf')
        values = []
        actions = game.get_legal_moves()
        for action in actions:
            next_board, game_over_or_not, who_won = game.forecast_move(action)

            if game_over_or_not:
                return float('inf')
            new_value = self.min_play_ab(next_board, time_left, depth - 1, alpha, beta, not maximizing_player)
            value = max(value, new_value)
            if value >= beta:
                return value
            alpha = max(alpha,value)
        return value

    def min_play_ab(self, game, time_left, depth, alpha, beta, maximizing_player=True):
        #check if game is over or last node or time is up
        if depth == 0 or time_left() < 50 or len(game.get_legal_moves()) == 0:
            return self.utility(game,maximizing_player)

        value = float('inf')
        values = []
        actions = game.get_legal_moves()
        for action in actions:
            next_board, game_over_or_not, who_won = game.forecast_move(action)
            if game_over_or_not:
                return float('-inf')
            new_value = self.max_play_ab(next_board, time_left, depth - 1, alpha, beta, not maximizing_player)
            value = min(value, new_value)
            if value <= alpha:
                return value
            beta = min(beta,value)
        return value