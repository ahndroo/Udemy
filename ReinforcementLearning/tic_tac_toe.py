import numpy as np
import matplotlib.pyplot as plt

LENGTH = 3; # length of game board

class Agent:
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps # greedy epsilon
        self.alpha = alpha # learning rate
        self.verbose = False
        self.state_history = []

    def setV(self, V):
        self.V = V

    def set_symbol(self, symbol):
        self.sym = symbol;

    def set_verbose(self, v):
        # if true will print vlaues for each pos on board
        self.verbose = v

    def reset_history(self):
        self.state_history = []

    def take_action(self, env):
        # choose action based on epsilon greedy
        r = np.random.rand()
        best_state = None
        if r < self.eps:
            # take a random action
            if self.verbose:
                print "taking random action"
            possible_moves = []
            for i in xrange(LENGTH):
                for j in xrange(LENGTH):
                    if env.is_empty(i ,j):
                        possible_moves.append((i, j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
            pos2value = {}
            next_move = None
            best_value = -1
            for i in xrange(LENGTH):
                for j in xrange(LENGTH):
                    if env.is_empty(i,j):
                        # what is state if made this move?
                        env.board[i,j] = self.sym
                        state = env.get_state()
                        env.board[i,j] = 0 # change it back!
                        pos2value[(i,j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i,j)
            if self.verbose:
                print("Taking greedy action")
                for i in xrange(LENGTH):
                    print "------------------"
                    for j in xrange(LENGTH):
                        if env.is_empty(i,j):
                            print "%.2f|" % pos2value[(i, j)],
                        else:
                            print " ",
                            if env.board[i,j] == env.x:
                                print "x |",
                            elif env.board[i,j] == env.o:
                                print "o |",
                            else:
                                print " |",
                    print ""
                print "------------------"
        env.board[next_move[0], next_move[1]] = self.sym


    def update_state_history(self, s):
        # needs to be updated every iteration--cannot be put in take_action since it
        # only happens once every other iteration for each player
        self.state_history.append(s)

    def update(self, env):
         # want to backtrack over the states so that:
         # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
         # where V(next_state) = reward if its hte most current state
         # only do this at end of episode
         reward = env.reward(self.sym)
         target = reward
         for prev in reversed(self.state_history):
             value = self.V[prev] + self.alpha*(target - self.V[prev])
             self.V[prev] = value
             target = value
         self.reset_history()

class Environment:
    def __init__(self):
        self.board = np.zeros((LENGTH, LENGTH))
        self.x = -1; # represents x on board, p1
        self.o = 1; # represents o on board, p2
        self.winner = None;
        self.ended = False;
        self.num_states = 3**(LENGTH*LENGTH);

    def is_empty(self, i, j):
        return self.board[i,j]==0; # true if pos on board is 0, false otherwise

    def reward(self, sym):
        # no reward until game is over
        if not self.game_over():
            return 0
        # sym will be self.x or self.o -- agents needs to know its own symbol
        return 1 if self.winner == sym else 0;

    def get_state(self):
        # returns current state, represented as an int from 0....|S|-1, where
        # S=set of all possible states.  |S| 3^(BOARD_SIZE), since each cell can
        # have 3 possible values- empty(0), x(1), o(2).
        k = 0;
        h = 0;
        for i in xrange(LENGTH):
            for j in xrange(LENGTH):
                if self.board[i,j] == 0:
                    v=0;
                elif self.board[i,j] == self.x:
                    v = 1;
                elif self.board[i,j] == self.o:
                    v = 2;
                h += (3**k) * v;
                k += 1;
        return h;

    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended

        # check rows
        for i in xrange(LENGTH):
            for player in (self.x, self.o):
                if self.board[i].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = True;
                    return True
        # check cols
        for j in xrange(LENGTH):
            for player in (self.x, self.o):
                if self.board[j].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = True
                    return True
        # check diags
        for player in (self.x, self.o):
            if self.board.trace() == player*LENGTH:
                self.winner = player;
                self.ended = True;
                return True
        # top-right -> bottom-left diag
        if np.fliplr(self.board).trace() == player*LENGTH:
            self.winner = player
            self.ended = True
            return True
        # check if draw
        if np.all((self.board==0) == False):
            self.winner = None
            self.ended = True
            return True

        self.winner = None
        return False

    def is_draw(self):
        return self.ended and self.winner is None

    def draw_board(self):
        for i in xrange(LENGTH):
            print "-------------"
            for j in xrange(LENGTH):
                print " ",
                if self.board[i, j] == self.x:
                    print "x",
                elif self.board[i,j] == self.o:
                    print "o",
                else:
                    print " ",
            print ""
        print "-------------"

class Human:
    def __init__(self):
        pass

    def set_symbol(self, sym):
        self.sym = sym

    def take_action(self, env):
        while True:
            # break if illegal move
            move = raw_input("Enter coordinates i,j for next move (i,j=0...2): ")
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env.board[i, j] = self.sym
                break

    def update(self, env):
        pass

    def update_state_history(self, s):
        pass

def play_game(p1, p2, env, draw=False):
    # loops until game is over
    current_player = None
    while not env.game_over():
        # alternate between p1, p2--p1 always goes first
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        # draw the board before the user who wants to see it to make a move
        if draw:
            if draw == 1 and current_player==p1:
                env.draw_board()
            if draw==2 and current_player==p2:
                env.draw_board()
        current_player.take_action(env)

        # update state histories
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

    if draw:
        env.draw_board()
    # do the value function update
    p1.update(env)
    p2.update(env)

def get_state_hash_and_winner(env, i=0, j=0):
    results = []

    for v in (0, env.x, env.o):
        env.board[i, j] = v # if empty board-should be 0
        if j==2:
            #j goes back to 0, increase i, unless i==2--we done
            if i==2:
                #board is full, collect results and return
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i+1, 0)
        else:
            results+= get_state_hash_and_winner(env, i, j+1)
    return results

def initialV_x(env, state_winner_triples):
    # initialize state values as follows:
    # if x wins, V(s)=1
    # if x loses or draw, V(s)=0
    # otherwise V(s)=.5
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v=0
        else:
            v=0.5
        V[state] = v
    return V

def initialV_o(env, state_winner_triples):
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v=0
        else:
            v=0.5
        V[state]=v
    return V

if __name__ == "__main__":
    # train agent
    p1 = Agent()
    p2 = Agent()

    # set initial V for p1 and p2
    env = Environment()
    state_winner_triples = get_state_hash_and_winner(env)

    Vx = initialV_x(env, state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env, state_winner_triples)
    p2.setV(Vo)

    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    T = 10000
    for t in xrange(T):
        if t%200==0:
            print t
        play_game(p1, p2, Environment())

    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, Environment(), draw=2)
        answer = raw_input("Play again? [Y/n]:")
        if answer and answer.lower()[0] == 'n':
            break
