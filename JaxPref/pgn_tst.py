import os
import numpy as np
import torch
from torch.utils.data import Dataset
from gym.spaces import *
import chess
from torch.utils.data import DataLoader, SubsetRandomSampler
import chess_convert as converter
import io
from jax_utils import batch_to_jax
import pandas as pd

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# # FLAGS_DEF = define_flags_with_default(
#     env='chess',
#     model_type='PrefTransformer',
#     max_traj_length=1000,
#     seed=5,
#     data_seed=13,
#     save_model=True,
#     batch_size=128,
#     early_stop=True,
#     min_delta=5e-3,
#     patience=10,
#     obs_dim=1344,
#     act_dim= 4672,
#     data_size = 30000,
#     reward_scale=1.0,
#     reward_bias=0.0,
#     clip_action=0.999,
#     reward_arch='256-256',
#     orthogonal_init=False,
#     activations='relu',
#     activation_final='none',
#     training=True,
#     n_epochs=1000,
#     eval_period=1,
#     data_dir='/home/hail/PreferenceTransformer/human_label/',
#     num_query=30000,
#     len_query=10,
#     query_len=10,
#     skip_flag=0,
#     balance=False,
#     topk=10,
#     window=2,
#     use_human_label=True,
#     feedback_random=False,
#     feedback_uniform=False,
#     enable_bootstrap=False,

#     comment='group_exinter_blitz',

#     reward=MR.get_default_config(),
#     transformer=PrefTransformer.get_default_config(),
#     logging=WandBLogger.get_default_config(),
# )

def uci_move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    move_index = 64 * from_square + to_square
    return move_index

def collect(pgn, len_query):

    observations, new_observations, actions, labels, timesteps = [], [], [], [], []
    
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    moves = game.mainline_moves()
    
    elo = game.headers.get("WhiteElo")
    elo = int(elo)
    
    if elo>=1775 :
        labels.append(1)
    elif elo <= 1325 :
        labels.append(-1)
    else:
        labels.append(0)
    
    states = []
    observations_grp, actions_grp, timesteps_grp = [], [], []
    
    for move in moves:
        mv=chess.Move.from_uci(move.uci())
        actions_grp.append(uci_move_to_index(mv)) 
    
    for t, move in enumerate(moves):
        states.append(converter.get_board_features(board))
        observations_grp.append(converter.get_lc0_input_planes(
            converter.stack_observations(states[: t + 1], history_size=8)
        ))
        board.push(move)
        timesteps_grp.append(t)
        
    observation_ = np.array([np.transpose(o, (1, 2, 0)) for o in observations_grp[:-1]])
    observation_new_ = np.array([np.transpose(o, (1, 2, 0)) for o in observations_grp[1:]])
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

    batch['observations'] = prep_batch1['observations'].squeeze(0)
    
    batch['next_observations'] = prep_batch1['next_observations'].squeeze(0)

    batch['actions'] = encode_action(prep_batch1['actions'], 4672)
    batch['actions'] = batch['actions'].reshape(8,4672)

    batch['observations_2'] = prep_batch2['observations'].squeeze(0)

    batch['next_observations_2'] = prep_batch2['next_observations'].squeeze(0)

    batch['actions_2'] = encode_action(prep_batch2['actions'], 4672)
    batch['actions_2'] = batch['actions_2'].reshape(8, 4672)

    batch['timestep_1'] = prep_batch1['timestep_1']
    batch['timestep_1'] = batch['timestep_1'].squeeze(0)
    batch['timestep_2'] = prep_batch2['timestep_1']
    batch['timestep_2'] = batch['timestep_2'].squeeze(0)

    batch['labels'] = combined_labels
    batch['labels'] = batch['labels'].squeeze(0)
    batch['script_labels'] = combined_labels
    batch['script_labels'] = batch['script_labels'].squeeze(0)

    return batch

def encode_action(action, num_classes=4672):
    action = action - 1
    one_hot_action = np.eye(num_classes)[action]
    return one_hot_action

def tensor_to_numpy(batch):

    return {key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

class ChessDataset(Dataset):
    def __init__(self, data_dir, env, len_query=50, batch_size=256):
        self.data_dir = data_dir
        self.env = env
        self.len_query = len_query
        self.batch_size = batch_size
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]


    def __getitem__(self, idx):
        sigma1 = self.file_list[idx]
        sigma2 = self.file_list[idx+1]

        df1 = pd.read_csv(sigma1)
        df2 = pd.read_csv(sigma2)

        prep_batch1 = collect(df1, self.env, self.len_query)
        prep_batch2 = collect(df2, self.env, self.len_query)

        return combine(prep_batch1, prep_batch2)

# class ChessDataset(Dataset):
#     def __init__(self, data_dir1, data_dir2, env, len_query=50, batch_size=256):
#         self.data_dir1 = data_dir1
#         self.data_dir2 = data_dir2
#         self.env = env
#         self.len_query = len_query
#         self.batch_size = batch_size
#
#         self.file_list1 = sorted([os.path.join(data_dir1, f) for f in os.listdir(data_dir1) if f.endswith(".csv")])
#         self.file_list2 = sorted([os.path.join(data_dir2, f) for f in os.listdir(data_dir2) if f.endswith(".csv")])
#
#         #assert len(self.file_list1) == len(self.file_list2), "File counts in data_dir1 and data_dir2 must match"
#
#     def __len__(self):
#         return len(self.file_list1)
#
#     def __getitem__(self, idx):
#         sigma1 = self.file_list1[idx]
#         sigma2 = self.file_list2[idx]
#
#         df1 = pd.read_csv(sigma1)
#         df2 = pd.read_csv(sigma2)
#
#         prep_batch1 = collect(df1, self.env, self.len_query)
#         prep_batch2 = collect(df2, self.env, self.len_query)
#
#         return combine(prep_batch1, prep_batch2)

def get_chess_dataloader(data_dir, env, batch_size, len_query, shuffle=True):
    dataset = ChessDataset(data_dir, env, len_query, batch_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader


if __name__ == '__main__':
    #trn_data_dir1 = "/media/hail/HDD/chess_data/tr1"
    #trn_data_dir2 = "/media/hail/HDD/chess_data/tr3"

    eval_data_dir1 = "/media/hail/HDD/chess_data/ev1"
    eval_data_dir2 = "/media/hail/HDD/chess_data/ev2"

    shuffled_idx = [i for i in range(64)]
    dataset = ChessDataset(eval_data_dir1,eval_data_dir1, 8, 64)
    batch_indices = shuffled_idx[:]
    sampler = SubsetRandomSampler(batch_indices)
    dataloader = DataLoader(dataset,64, sampler=sampler)
    for b in  dataloader:
        a = tensor_to_numpy(b)
    batch = batch_to_jax(a)
    print('done')
    #dataset_ = ChessDataset(eval_data_dir1, eval_data_dir2, env, FLAGS.len_query, FLAGS.batch_size )
