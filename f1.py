import pickle
import jax
import jax.numpy as jnp
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from JaxPref.chess_convert import *
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
import torch


def batch_to_jax(batch):
    return {k: jnp.array(v) for k, v in batch.items()}

def collect(df, len_query):
    observations, new_observations, actions, labels, timesteps = [], [], [], [], []
    moves = []
    states = []

    labels.append(int(df['white_elo'].values[0]))

    for _, move in enumerate(df['move']):
        moves.append(chess.Move.from_uci(move))

    board = chess.Board()

    observations_grp, actions_grp, timesteps_grp = [], [], []

    for timestep, move in enumerate(moves):
        states.append(get_board_features(board))
        observations_grp.append(get_lc0_input_planes(
            stack_observations(states[: timestep + 1], history_size=8)
        ))
        board.push(move)
        actions_grp.append(uci_move_to_index(move))
        timesteps_grp.append(timestep)

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

def uci_move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    move_index = 64 * from_square + to_square
    return move_index

def labeler(l, mode):
    if mode == 'expert':
        if not l >= 1739:
            l = 0
        else:
            l = 1
    elif mode == 'intermediate':
        if l <= 1367 or l >= 1739:
            l = 0
        else:
            l = 1
    elif mode == 'beginner':
        if l <= 1367:
            l = 1
        else:
            l = 0
    return l

def combine(prep_batch1, prep_batch2, mode):
    batch = {}

    l1 = prep_batch1['labels']
    l2 = prep_batch2['labels']

    labels_1 = np.array([labeler(l1, mode)])
    labels_2 = np.array([labeler(l2, mode)])

    combined_labels = np.stack([
        np.where(labels_1 > labels_2, 1, np.where(labels_1 < labels_2, 0, 0.5)),
        np.where(labels_1 > labels_2, 0, np.where(labels_1 < labels_2, 1, 0.5))
    ], axis=1)

    batch['observations'] = prep_batch1['observations'].squeeze(0)
    batch['observations'] = np.transpose(batch['observations'], (0, 2, 3, 1))
    batch['next_observations'] = prep_batch1['next_observations'].squeeze(0)
    batch['next_observations'] = np.transpose(batch['next_observations'], (0, 2, 3, 1))
    batch['actions'] = encode_action(prep_batch1['actions'], 4672)
    batch['actions'] = batch['actions'].reshape(5, 4672)
    batch['observations_2'] = prep_batch2['observations'].squeeze(0)
    batch['observations_2'] = np.transpose(batch['observations_2'], (0, 2, 3, 1))
    batch['next_observations_2'] = prep_batch2['next_observations'].squeeze(0)
    batch['next_observations_2'] = np.transpose(batch['next_observations_2'], (0, 2, 3, 1))
    batch['actions_2'] = encode_action(prep_batch2['actions'], 4672)
    batch['actions_2'] = batch['actions_2'].reshape(5, 4672)
    batch['timestep_1'] = prep_batch1['timestep_1']
    batch['timestep_1'] = batch['timestep_1'].squeeze(0)
    batch['timestep_2'] = prep_batch2['timestep_1']
    batch['timestep_2'] = batch['timestep_2'].squeeze(0)
    batch['labels'] = combined_labels
    batch['labels'] = batch['labels'].squeeze(0)
    batch['script_labels'] = combined_labels
    batch['script_labels'] = batch['script_labels'].squeeze(0)

    return batch

def get_chess_dataloader(data_dir, env, batch_size, len_query, shuffle=True):
    dataset = ChessDataset(data_dir, env, len_query, batch_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader

def encode_action(action, num_classes=4672):
    action = action - 1
    one_hot_action = np.eye(num_classes)[action]
    return one_hot_action

def tensor_to_numpy(batch):
    return {key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

class ChessDataset(Dataset):
    def __init__(self, data_dir1, data_dir2, len_query=5, batch_size=64):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
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

        prep_batch1 = collect(df1, self.len_query)
        prep_batch2 = collect(df2, self.len_query)

        return combine(prep_batch1, prep_batch2, 'expert')

def evaluate_f1_score(reward_model, batch):
    logits = reward_model._eval_pref_step(reward_model.train_states, jax.random.PRNGKey(0), batch)
    prob = jax.nn.sigmoid(logits[:, 0])
    threshold = 0.5
    pred_labels = (prob >= threshold).astype(int)
    true_labels = batch['labels'][0].astype(int)
    true_label= np.full(pred_labels.shape, true_labels[0])
    f1 = f1_score(true_label, pred_labels)
    return f1


if __name__ == "__main__":

    f1_scores = []
    num_eval_queries = 1000
    query_len = 5

    data_dir1 = "/media/hail/HDD/chess_data_/new1"
    data_dir2 = "/media/hail/HDD/chess_data_/new2"

    shuffled_idx = [i for i in range(1000)]
    dataset = ChessDataset(data_dir1, data_dir2, 5, 1000)
    batch_indices = shuffled_idx[:]
    sampler = SubsetRandomSampler(batch_indices)
    dataloader = DataLoader(dataset, 1000, sampler=sampler)
    for b in dataloader:
        a = tensor_to_numpy(b)
    batch = batch_to_jax(a)

    model_path = "/home/hail/Desktop/chess_pt/JaxPref/reward_model/chess/PrefTransformer/expert/s5/model.pkl"
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    reward_model = data['reward_model']

    f1_score_value = evaluate_f1_score(reward_model, batch)
    f1_scores.append(f1_score_value)

    print('done')