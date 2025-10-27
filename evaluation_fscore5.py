import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import chess
import gym
import jax
import jax.numpy as jnp
from tqdm import tqdm

import os
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from JaxPref.jax_utils import batch_to_jax
import gym
from gym.spaces import *
import chess
import chess.svg
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler

def encode_action(action, num_classes=4672):
    action = action - 1
    one_hot_action = np.eye(num_classes)[action]
    return one_hot_action

def tensor_to_numpy(batch):

    return {key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

WHITE = 1
BLACK = 0

def __deepcopy__(self, memo):
    new_instance = self.__class__.__new__(self.__class__)
    memo[id(self)] = new_instance

    new_instance.board = np.copy(self.board)
    new_instance.state = np.copy(self.state)
    new_instance.action_space = self.action_space

    return new_instance

def is_repetition(self, count: int = 3) -> bool:
    """
    Checks if the current position has repeated 3 (or a given number of)
    times.

    Unlike :func:`~chess.Board.can_claim_threefold_repetition()`,
    this does not consider a repetition that can be played on the next
    move.

    Note that checking this can be slow: In the worst case, the entire
    game has to be replayed because there is no incremental transposition
    table.
    """
    # Fast check, based on occupancy only.
    maybe_repetitions = 1
    for state in reversed(self._stack):
        if state.occupied == self.occupied:
            maybe_repetitions += 1
            if maybe_repetitions >= count:
                break
    if maybe_repetitions < count:
        return False

    # Check full replay.
    transposition_key = self._transposition_key()
    switchyard = []

    try:
        while True:
            if count <= 1:
                return True

            if len(self.move_stack) < count - 1:
                break

            move = self.pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            if self._transposition_key() == transposition_key:
                count -= 1
    finally:
        while switchyard:
            self.push(switchyard.pop())

    return False

chess.Board.is_repetition = is_repetition

class Chess(gym.Env):
    """AlphaGo Chess Environment"""
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self):
        self.board = None

        self.T = 1
        self.M = 3
        self.L = 6

        self.size = (8, 8)

        # self.viewer = None

        # self.knight_move2plane[dCol][dRow]
        """
        [ ][5][ ][3][ ]
        [7][ ][ ][ ][1]
        [ ][ ][K][ ][ ]
        [6][ ][ ][ ][0]
        [ ][4][ ][2][ ]
        """
        self.knight_move2plane = {2: {1: 0, -1: 1}, 1: {2: 2, -2: 3}, -1: {2: 4, -2: 5}, -2: {1: 6, -1: 7}}

        self.observation_space = Dict(
            {"P1 piece": Tuple([MultiBinary((8, 8)) for t in range(self.T) for plane in range(6)]),
             "P2 piece": Tuple([MultiBinary((8, 8)) for t in range(self.T) for plane in range(6)]),
             "Repetitions": Tuple([MultiBinary((8, 8)) for t in range(self.T) for plane in range(2)]),
             "Color": MultiBinary((8, 8)),
             "Total move count": MultiBinary((8, 8)),
             "P1 castling": Tuple([MultiBinary((8, 8)) for rook in range(2)]),
             "P2 castling": Tuple([MultiBinary((8, 8)) for rook in range(2)]),
             "No-progress count": MultiBinary((8, 8))})

        self.action_space = Dict(
            {"Queen moves": Tuple([MultiBinary((8, 8)) for squares in range(7) for direction in range(8)]),
             "Knight moves": Tuple([MultiBinary((8, 8)) for move in range(8)]),
             "Underpromotions": Tuple(MultiBinary((8, 8)) for move in range(9))})

    def repetitions(self):
        count = 0
        for state in reversed(self.board.stack):
            if state.occupied == self.board.occupied:
                count += 1

        return count

    def get_direction(self, fromRow, fromCol, toRow, toCol):
        if fromCol == toCol:
            return 0 if toRow < fromRow else 4
        elif fromRow == toRow:
            return 6 if toCol < fromCol else 2
        else:
            if toCol > fromCol:
                return 1 if toRow < fromRow else 3
            else:
                return 7 if toRow < fromRow else 5

    def get_diagonal(self, fromRow, fromCol, toRow, toCol):
        return int(toRow < fromRow and toCol > fromCol or toRow > fromRow and toCol < fromCol)

    def move_type(self, move):
        return "Knight" if self.board.piece_type_at(move.from_square) == 2 else "Queen"

    def observe(self):
        self.P1_piece_planes = np.zeros((8, 8, 6))
        self.P2_piece_planes = np.zeros((8, 8, 6))

        for pos, piece in self.board.piece_map().items():
            row, col = divmod(pos, 8)

            if piece.color == WHITE:
                self.P1_piece_planes[row, col, piece.piece_type - 1] = 1
            else:
                self.P2_piece_planes[row, col, piece.piece_type - 1] = 1

        self.Repetitions_planes = np.concatenate(
            [np.full((8, 8, 1), int(self.board.is_repetition(repeats))) for repeats in range(1, 3)], axis=-1)
        self.Colour_plane = np.full((8, 8, 1), int(self.board.turn))
        self.Total_move_count_plane = np.full((8, 8, 1), self.board.fullmove_number)
        self.P1_castling_planes = np.concatenate((np.full((8, 8, 1), self.board.has_kingside_castling_rights(WHITE)),
                                                  np.full((8, 8, 1), self.board.has_queenside_castling_rights(WHITE))),
                                                 axis=-1)
        self.P2_castling_planes = np.concatenate((np.full((8, 8, 1), self.board.has_kingside_castling_rights(BLACK)),
                                                  np.full((8, 8, 1), self.board.has_queenside_castling_rights(BLACK))),
                                                 axis=-1)

        # The fifty-move rule in chess states that a player can claim a
        # draw if no capture has been made and no pawn has been moved in
        # the last fifty moves (https://en.wikipedia.org/wiki/Fifty-move_rule)
        self.No_progress_count_plane = np.full((8, 8, 1), self.board.halfmove_clock)

        self.binary_feature_planes = np.concatenate(
            (self.P1_piece_planes, self.P2_piece_planes, self.Repetitions_planes), axis=-1)
        self.constant_value_planes = np.concatenate((self.Colour_plane, self.Total_move_count_plane, \
                                                     self.P1_castling_planes, self.P2_castling_planes, \
                                                     self.No_progress_count_plane), axis=-1)

        self.state_history = self.state_history[:, :, 14:-7]
        self.state_history = np.concatenate(
            (self.state_history, self.binary_feature_planes, self.constant_value_planes), axis=-1)
        return self.state_history

    def reset(self):
        if self.board is None:
            self.board = chess.Board()

        self.board.reset()

        self.turn = WHITE

        self.reward = None
        self.terminal = False

        # Initialize states before timestep 1 to matrices containing all zeros
        self.state_history = np.zeros((8, 8, 14 * self.T + 7))
        return self.observe()

    def legal_move_mask(self):
        mask = np.zeros((8, 8, 73))

        for move in self.board.legal_moves:
            fromRow = 7 - move.from_square // 8
            fromCol = move.from_square % 8

            toRow = 7 - move.to_square // 8
            toCol = move.to_square % 8

            dRow = toRow - fromRow
            dCol = toCol - fromCol

            piece_type = self.board.piece_type_at(move.from_square)

            if piece_type == 2:  # Knight move
                # plane = knight_move2plane[dCol][dRow] + 56 # SH edit
                plane = self.knight_move2plane[dCol][dRow] + 56
            else:  # Queen move
                if move.promotion and move.promotion in [2, 3, 4]:  # Underpromotion move (to knight, biship, or rook)
                    if fromCol == toCol:  # Regular pawn promotion move
                        plane = 64 + move.promotion - 2
                    else:  # Simultaneous diagonal pawn capture from the 7th rank and subsequent promotion
                        diagonal = self.get_diagonal(fromRow, fromCol, toRow, toCol)
                        plane = 64 + (diagonal + 1) * 3 + move.promotion - 2
                else:  # Regular queen move
                    squares = max(abs(toRow - fromRow), abs(toCol - fromCol))
                    direction = self.get_direction(fromRow, fromCol, toRow, toCol)
                    plane = (squares - 1) * 8 + direction

            mask[fromRow, fromCol, plane] = 1

        return mask

    def step(self, p):
        mask = self.legal_move_mask()
        p = p * mask
        pMin, pMax = p.min(), p.max()
        p = (p - pMin) / (pMax - pMin)
        action = np.unravel_index(p.argmax(), p.shape)

        fromRow, fromCol, plane = action

        if plane < 56:  # Queen move
            squares, direction = divmod(plane, 8)
            squares += 1

            """
            7 0 1
            6   2
            5 4 3
            """
            if direction == 0:
                toRow = fromRow - squares
                toCol = fromCol
            elif direction == 1:
                toRow = fromRow - squares
                toCol = fromCol + squares
            elif direction == 2:
                toRow = fromRow
                toCol = fromCol + squares
            elif direction == 3:
                toRow = fromRow + squares
                toCol = fromCol + squares
            elif direction == 4:
                toRow = fromRow + squares
                toCol = fromCol
            elif direction == 5:
                toRow = fromRow + squares
                toCol = fromCol - squares
            elif direction == 6:
                toRow = fromRow
                toCol = fromCol - squares
            else:  # direction == 7
                toRow = fromRow - squares
                toCol = fromCol - squares

            fromSquare = (7 - fromRow) * 8 + fromCol
            toSquare = (7 - toRow) * 8 + toCol
            move = chess.Move(fromSquare, toSquare)
        elif plane < 64:  # Knight move
            """
            [ ][5][ ][3][ ]
            [7][ ][ ][ ][1]
            [ ][ ][K][ ][ ]
            [6][ ][ ][ ][0]
            [ ][4][ ][2][ ]
            """
            if plane == 56:
                toRow = fromRow + 1
                toCol = fromCol + 2
            elif plane == 57:
                toRow = fromRow - 1
                toCol = fromCol + 2
            elif plane == 58:
                toRow = fromRow + 2
                toCol = fromCol + 1
            elif plane == 59:
                toRow = fromRow - 2
                toCol = fromCol + 1
            elif plane == 60:
                toRow = fromRow + 2
                toCol = fromCol - 1
            elif plane == 61:
                toRow = fromRow - 2
                toCol = fromCol - 1
            elif plane == 62:
                toRow = fromRow + 1
                toCol = fromCol - 2
            else:  # plane == 63
                toRow = fromRow - 1
                toCol = fromCol - 2

            fromSquare = (7 - fromRow) * 8 + fromCol
            toSquare = (7 - toRow) * 8 + toCol
            move = chess.Move(fromSquare, toSquare)
        else:  # Underpromotions
            toRow = fromRow - self.board.turn

            if plane <= 66:
                toCol = fromCol
                promotion = plane - 62
            elif plane <= 69:
                diagonal = 0
                promotion = plane - 65
                toCol = fromCol - self.board.turn
            else:  # plane <= 72
                diagonal = 1
                promotion = plane - 68
                toCol = fromCol + self.board.turn

            fromSquare = (7 - fromRow) * 8 + fromCol
            toSquare = (7 - toRow) * 8 + toCol
            move = chess.Move(fromSquare, toSquare, promotion=promotion)

        self.board.push(move)

        # self.board = self.board.mirror()

        result = self.board.result(claim_draw=True)
        self.reward = 0 if result == '*' or result == '1/2-1/2' else 1 if result == '1-0' else -1  # if result == '0-1'
        self.terminal = self.board.is_game_over(claim_draw=True)
        self.info = {'last_move': move, 'turn': self.board.turn}

        return self.observe(), self.reward, self.terminal, self.info

    # def get_image(self):
    #   out = BytesIO()
    #   bytestring = chess.svg.board(self.board, size=1000).encode('utf-8')
    #   cairosvg.svg2png(bytestring=bytestring, write_to=out)
    #   image = Image.open(out).convert("RGB")
    #   return np.asarray(image).astype(np.uint8)
    #
    # def render(self, mode='human'):
    #   img = self.get_image()
    #
    #   if mode == 'rgb_array':
    #     return img
    #   elif mode == 'human':
    #     if self.viewer is None:
    #       from gym.envs.classic_control import rendering
    #       self.viewer = rendering.SimpleImageViewer()
    #
    #     self.viewer.imshow(img)
    #     return self.viewer.isopen
    #   else:
    #     raise NotImplementedError

    # def close(self):
    #   if not self.viewer is None:
    #     self.viewer.close()

def uci_move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    move_index = 64 * from_square + to_square
    return move_index

def collect(df, env, len_query):
    df = df.groupby('game_id', sort=False)

    observations, new_observations, actions, labels, timesteps = [], [], [], [], []

    for _, group in df:
        group = group.reset_index(drop=True)
        labels.append(int(group['label'].values[0]))

        env.reset()

        moves = group['move'].tolist()
        observations_grp, actions_grp, timesteps_grp = [], [], []

        for timestep, move_uci in enumerate(moves):
            move = chess.Move.from_uci(move_uci)
            env.board.push(move)
            obse = env.observe()

            observations_grp.append(obse)
            actions_grp.append(uci_move_to_index(move))
            timesteps_grp.append(timestep)

        if len(observations_grp) < 2:
            continue

        observation_ = np.array(observations_grp[:-1])
        observation_new_ = np.array(observations_grp[1:])
        actions_ = np.array(actions_grp)
        timesteps_ = np.array(timesteps_grp)


        pad_width = len_query - len(observation_)
        if pad_width > 0:
            observation_ = np.pad(observation_, ((0, pad_width), (0, 0), (0, 0), (0, 0)), mode='constant')
            observation_new_ = np.pad(observation_new_, ((0, pad_width), (0, 0), (0, 0), (0, 0)), mode='constant')
            actions_ = np.pad(actions_, (0, pad_width), mode='constant')
            timesteps_ = np.pad(timesteps_, (0, pad_width), mode='constant')


        max_idx = max(0, len(observation_) - len_query)
        idxx = np.random.randint(0, max_idx + 1) if max_idx > 0 else 0

        observations.append(observation_[idxx:idxx + len_query])
        new_observations.append(observation_new_[idxx:idxx + len_query])
        actions.append(actions_[idxx:idxx + len_query])
        timesteps.append(timesteps_[idxx:idxx + len_query])

    return {
        "observations": np.array(observations, dtype=float),
        "next_observations": np.array(new_observations, dtype=float),
        "actions": np.array(actions),
        "labels": np.array(labels),
        "timestep_1": np.array(timesteps)
    }

def combine(prep_batch1, prep_batch2):
    batch = {}

    labels_1 = np.array(prep_batch1['labels'])
    labels_2 = np.array(prep_batch2['labels'])

    combined_labels = np.stack([
        np.where(labels_1 > labels_2, 1, np.where(labels_1 < labels_2, 0, 0.5)),
        np.where(labels_1 > labels_2, 0, np.where(labels_1 < labels_2, 1, 0.5))
    ], axis=1)


    # batch['observations'] = prep_batch1['observations'].reshape(50,7616)
    # batch['next_observations'] = prep_batch1['next_observations'].reshape(50,7616)
    #batch['observations'] = prep_batch1['observations'][:, :, :,:, :12]
    batch['observations'] = prep_batch1['observations'].squeeze(0)

    #batch['observations'] = obs.reshape(50, 64, 12)
    # batch['observations'] = prep_batch1['observations'].reshape(50,8,8,119)
    #batch['next_observations'] = prep_batch1['next_observations'][:, :, :, :,:12]
    batch['next_observations'] = prep_batch1['next_observations'].squeeze(0)

    #batch['next_observations'] = obs_.reshape(50, 64, 12)
    # batch['next_observations'] = prep_batch1['next_observations'].reshape(50,8,8,119)

    batch['actions'] = encode_action(prep_batch1['actions'], 4672)
    batch['actions'] = batch['actions'].reshape(50,4672)

    # batch['observations_2'] = prep_batch2['observations'].reshape(50,7616)
    # batch['next_observations_2'] = prep_batch2['next_observations'].reshape(50,7616)
    #batch['observations_2'] = prep_batch2['observations'][:, :, :,:, :12]
    batch['observations_2'] = prep_batch2['observations'].squeeze(0)

    #batch['observations_2'] = obs2.reshape(50, 64, 12)
    #batch['observations_2'] = prep_batch2['observations'].reshape(50,8,8,119)

    #batch['next_observations_2'] = prep_batch2['next_observations'][:, :, :,:, :12]
    batch['next_observations_2'] = prep_batch2['next_observations'].squeeze(0)

    #batch['next_observations'] = obs2_.reshape(50, 64, 12)
    # batch['next_observations_2'] = prep_batch2['next_observations'].reshape(50,8,8,119)

    batch['actions_2'] = encode_action(prep_batch2['actions'], 4672)
    batch['actions_2'] = batch['actions_2'].reshape(50, 4672)

    batch['timestep_1'] = prep_batch1['timestep_1']
    batch['timestep_1'] = batch['timestep_1'].squeeze(0)
    batch['timestep_2'] = prep_batch2['timestep_1']
    batch['timestep_2'] = batch['timestep_2'].squeeze(0)

    batch['labels'] = combined_labels
    batch['labels'] = batch['labels'].squeeze(0)
    batch['script_labels'] = combined_labels
    batch['script_labels'] = batch['script_labels'].squeeze(0)

    return batch

class ChessDataset(Dataset):
    def __init__(self, data_dir1, data_dir2, env, len_query=50, batch_size=256):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.env = env
        self.len_query = len_query
        self.batch_size = batch_size

        self.file_list1 = sorted([os.path.join(data_dir1, f) for f in os.listdir(data_dir1) if f.endswith(".csv")])
        self.file_list2 = sorted([os.path.join(data_dir2, f) for f in os.listdir(data_dir2) if f.endswith(".csv")])

    def __len__(self):
        return len(self.file_list1)

    def __getitem__(self, idx):
        sigma1 = self.file_list1[idx]
        sigma2 = self.file_list2[idx]

        df1 = pd.read_csv(sigma1)
        df2 = pd.read_csv(sigma2)

        prep_batch1 = collect(df1, self.env, self.len_query)
        prep_batch2 = collect(df2, self.env, self.len_query)

        return combine(prep_batch1, prep_batch2)

def get_chess_dataloader(data_dir1, data_dir2, env, batch_size, len_query, shuffle=True):
    dataset = ChessDataset(data_dir1, data_dir2, env, len_query, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader

def evaluate_model(model):
    all_preds = []
    all_labels = []

    eval_data_dir1 = '/media/hail/HDD/chess_data/f_score/f1_cel'
    eval_data_dir2 = '/media/hail/HDD/chess_data/f_score/f2_cel'
    env = Chess()

    dataset_ = ChessDataset(eval_data_dir1, eval_data_dir2, env, 50, 200)
    eval_start_pt, eval_end_pt = 0, 3000
    batch_idx = range(eval_start_pt, eval_end_pt)
    sampler = SequentialSampler(batch_idx)
    dataloader = DataLoader(dataset_, batch_size=200, sampler=sampler)
    for b in dataloader:
        eval = tensor_to_numpy(b)
    batch_ = batch_to_jax(eval)

    logits = model._eval_pref_step(model.train_states, jax.random.PRNGKey(0), batch_)
    labels = batch_['labels']
    mask = (labels[:, 0] != 0.5)
    logits = logits[mask]
    labels = labels[mask]
    preds = (logits[:, 0] > logits[:, 1]).astype(int)
    true_labels = labels[:,0]
    all_preds.append(np.array(preds))
    all_labels.append(np.array(true_labels))

    # concat
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # score
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    print(f"Evaluation Result Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return acc, f1

# === Main ===
if __name__ == "__main__":
    model_path = "/home/hail/Desktop/chess_pt/JaxPref/reward_model/chess/PrefTransformer/group_expert_beginner/s5/model.pkl"
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    model = data['reward_model']
    evaluate_model(model)
