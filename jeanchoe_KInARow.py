'''
<yourUWNetID>_KInARow.py
Authors: <your name(s) here, lastname first and partners separated by ";">
  Example:
    Authors: Smith, Jane; Lee, Laura

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 473, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

from agent_base import KAgent
from game_types import State, Game_Type
import openai
import random

AUTHORS = 'Ankit Gowda'

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        super().__init__(twin)
        self.nickname = 'Strategic Sally'
        if twin:
            self.nickname += ' Twin'
        self.long_name = 'Strategic Sally: The K-in-Row Expert'
        self.playing = None
        self.game_info = {}  # Initialize as empty dict
        self.use_llm = True
        try:
            openai.api_key = "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"
        except:
            self.use_llm = False

        # Set up conversation tracking
        self.conversation_history = []
        self.opponent_utterances = []
        self.my_utterances = []

    def introduce(self):
        return (
            f"\nGreetings! I'm {self.nickname}, a strategic K-in-Row player.\n"
            "Created by Ankit Gowda (netid: agowda) to combine wit and wisdom in game play.\n"
            "Let's have an engaging match!\n"
            f"{'(Twin mode activated!)' if self.twin else ''}\n"
        )

    def prepare(self,
                game_type,
                what_side_to_play,
                opponent_nickname,
                expected_time_per_move=0.1,
                utterances_matter=True):
        self.playing = what_side_to_play
        self.game_info = {
            'type': game_type,
            'k': game_type.k,
            'rows': game_type.n,
            'cols': game_type.m,
            'opponent': opponent_nickname,
            'time_limit': expected_time_per_move,
            'utterances_matter': utterances_matter
        }
        return "OK"

    def get_legal_moves(self, state):
        """Returns list of legal moves as (row, col) tuples."""
        moves = []
        for i in range(len(state.board)):
            for j in range(len(state.board[0])):
                if state.board[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def static_eval(self, state):
        """Evaluates board state. Higher values favor X, lower values favor O."""
        def count_sequences(board, player, k):
            score = 0
            rows, cols = len(board), len(board[0])

            # Helper to evaluate a sequence
            def eval_seq(seq):
                player_count = seq.count(player)
                empty_count = seq.count(' ')
                if empty_count + player_count == k:
                    return 2 ** player_count
                return 0

            # Check rows
            for row in board:
                for start in range(cols - k + 1):
                    score += eval_seq(row[start:start + k])

            # Check columns
            for col in range(cols):
                for start in range(rows - k + 1):
                    score += eval_seq([board[start + i][col] for i in range(k)])

            # Check diagonals
            for row in range(rows - k + 1):
                for col in range(cols - k + 1):
                    # Down-right diagonal
                    score += eval_seq([board[row + i][col + i] for i in range(k)])
                    # Down-left diagonal
                    if col >= k - 1:
                        score += eval_seq([board[row + i][col - i] for i in range(k)])

            return score

        k = self.game_info['k']
        x_score = count_sequences(state.board, 'X', k)
        o_score = count_sequences(state.board, 'O', k)
        return x_score - o_score

    def minimax(self, state, depth_remaining, time_limit=None, alpha=float('-inf'), beta=float('inf'), z_hashing=None):
        if depth_remaining == 0 or not self.get_legal_moves(state):
            return self.static_eval(state), None

        is_maximizing = state.whose_move == 'X'
        best_move = None
        best_value = float('-inf') if is_maximizing else float('inf')

        for move in self.get_legal_moves(state):
            new_state = self.apply_move(state, move)
            value, _ = self.minimax(new_state, depth_remaining - 1, time_limit, alpha, beta, z_hashing)

            if is_maximizing:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        return best_value, best_move

    def apply_move(self, state, move):
        """Creates new state with move applied."""
        new_state = State(old=state)
        new_state.board[move[0]][move[1]] = state.whose_move
        new_state.whose_move = 'O' if state.whose_move == 'X' else 'X'
        return new_state

    def get_llm_utterance(self, curr_board, new_move, new_board, score, current_remark):
        """Generate response using OpenAI API with fallback options."""
        if not self.use_llm:
            return self.get_fallback_utterance()

        try:
            try:
                board_str = '\n'.join([' '.join(row) for row in curr_board.board])
                new_board_str = '\n'.join([' '.join(row) for row in new_board.board])
            except AttributeError:
                board_str = str(curr_board)
                new_board_str = str(new_board)

            prompt = f"""You are Strategic Sally, a witty K-in-Row player.
            Game Info: K={self.game_info['k']}, Board={self.game_info['rows']}x{self.game_info['cols']}

            Current board:
            {board_str}

            My move: {new_move}
            Resulting board:
            {new_board_str}

            Playing as: {self.playing}
            Board evaluation: {score}
            Opponent's last remark: {current_remark}

            Generate a short, strategic response (1-2 sentences) that:
            1. Comments on the game situation
            2. Occasionally responds to opponent's remarks
            3. Shows personality but stays focused on the game
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a strategic but friendly game player."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {str(e)}")  # Added error logging
            return self.get_fallback_utterance()

    def get_fallback_utterance(self):
        """Returns a fallback utterance when LLM is unavailable."""
        utterances = [
            "Hmm, this position looks interesting!",
            "Let's see how you handle this move.",
            "A strategic choice, if I do say so myself.",
            "Every move brings us closer to victory!",
            "This game is getting quite exciting.",
        ]
        return random.choice(utterances)

    def makeMove(self, currentState, currentRemark, timeLimit=10000):
        start_time = time.time()

        # Start with depth 2 and increase if time permits
        depth = 2
        best_value = float('-inf') if self.playing == 'X' else float('inf')
        best_move = None

        while (time.time() - start_time) < (timeLimit/2000):  # Use half time for search
            value, move = self.minimax(currentState, depth)
            if move:
                best_value = value
                best_move = move
            depth += 1

        if not best_move:
            legal_moves = self.get_legal_moves(currentState)
            best_move = random.choice(legal_moves) if legal_moves else None

        if not best_move:
            raise RuntimeError("No valid moves available")

        new_state = self.apply_move(currentState, best_move)

        # Generate utterance using remaining time
        utterance = self.get_llm_utterance(
            currentState,
            best_move,
            new_state,
            best_value,
            currentRemark
        )

        # Update conversation history
        self.opponent_utterances.append(currentRemark)
        self.my_utterances.append(utterance)

        return [[best_move, new_state], utterance]