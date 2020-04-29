
from sample_players import DataPlayer
import random
import math
from isolation import Isolation

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """ 
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.alphabeta(state, depth=4))      
        
    def alphabeta(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = random.choice(state.actions())
        scores = []
        for a in state.actions():
            v = self.ab_min_value(state.result(a), alpha, beta, depth)
            alpha = max(alpha, v)
            scores.append(v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move
    
    def ab_min_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.score(state)
        value = float("inf")
        for action in state.actions():
            value = min(value, self.ab_max_value(state.result(action), alpha, beta, depth - 1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def ab_max_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.score(state)
        value = float("-inf")
        for action in state.actions():
            value = max(value, self.ab_min_value(state.result(action), alpha, beta, depth - 1))
            if value >= beta:
               return value
            alpha = max(alpha, value)
        return value
    
    def score_central(self, state):
        '''
        This score either gives back the basic score if we have more or less moves than them. But if we have the same amount
        of moves it will then take into account the advantage in position based on how close it is to the center.
        '''
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        if len(own_liberties) != len(opp_liberties):
            return len(own_liberties) - len(opp_liberties)
        else:
            cen_x, cen_y = float(math.ceil(11 / 2)), float(math.ceil(9 / 2))
            x, y, opp_x, opp_y = own_loc % 13, own_loc // 13, opp_loc % 13, opp_loc // 13
            own_centrality = ((11 - cen_x) ** 2 - (x - cen_x) ** 2) + ((9 - cen_y) ** 2  - (y - cen_y) ** 2)
            opp_centrality = ((11 - cen_x) ** 2 - (opp_x - cen_x) ** 2) + ((9 - cen_y) ** 2  - (opp_y - cen_y) ** 2)
            return float(own_centrality - opp_centrality) / 10
        
    def score(self, state):
        '''
        Basic scoring heuristic.
        '''
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    
    def score_aggressive(self, state):
        '''
        Basic scoring heuristic with an added multiplier to make you more aggressive.
        '''
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return 1.5 * len(own_liberties) - len(opp_liberties)
    
    def score_build(self, state):
        '''
        Basic scoring heuristic that makes your player more defensive as the game goes on.
        '''
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        blocks_left = str(state.board).count('1')
        blocks_left_percent = blocks_left/99
        if blocks_left_percent > 0.9:
            multiplier = 1.50
        elif blocks_left_percent > 0.7:
            multiplier = 1.75
        elif blocks_left_percent > 0.5:
            multiplier = 2
        else:
            multiplier = 1
        return len(own_liberties) - multiplier * len(opp_liberties)
    