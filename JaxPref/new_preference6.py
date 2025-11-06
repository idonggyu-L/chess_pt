from collections import defaultdict
import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset
import absl.app
import absl.flags
from flax.training.early_stopping import EarlyStopping
from flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel
from JaxPref.jax_utils import batch_to_jax
from viskit.logging import logger, setup_logger
from JaxPref.MR import MR
from JaxPref.PrefTransformer import PrefTransformer
from JaxPref.utils import Timer, define_flags_with_default, set_random_seed, get_user_flags, prefix_metrics, \
    WandBLogger, save_pickle
import chess.svg
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from chess_convert import *

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

FLAGS_DEF = define_flags_with_default(
    env='chess',
    model_type='PrefTransformer',
    max_traj_length=1000,
    seed=5,
    data_seed=5,
    save_model=True,
    batch_size=256,
    early_stop=True,
    min_delta=1e-2,
    patience=10,
    obs_dim=7618,
    act_dim= 4672,
    data_size =100000,
    eval_data_size = 10000,
    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,
    reward_arch='256-256',
    orthogonal_init=False,
    activations='relu',
    activation_final='none',
    training=True,
    n_epochs=1000,
    eval_period=1,
    data_dir='/home/hail/PreferenceTransformer/human_label/',
    num_query=100000,
    len_query=8,
    query_len=8,
    skip_flag=0,
    balance=False,
    topk=10,
    window=2,
    use_human_label=True,
    feedback_random=False,
    feedback_uniform=False,
    enable_bootstrap=False,
    comment='all_soft',
    reward=MR.get_default_config(),
    transformer=PrefTransformer.get_default_config(),
    logging=WandBLogger.get_default_config(),
)

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

def get_soft_label(r1, r2):
    prob = 1 / (1 + 10 ** ((r2 - r1) / 400))
    return [prob, 1 - prob]

def combine(prep_batch1, prep_batch2, mode):
    batch = {}

    l1 = prep_batch1['labels']
    l2 = prep_batch2['labels']

    labels_1 = np.array(l1)
    labels_2 = np.array(l2)

    # combined_labels = np.stack([
    #     np.where(labels_1 > labels_2, 1, np.where(labels_1 < labels_2, 0, 0.5)),
    #     np.where(labels_1 > labels_2, 0, np.where(labels_1 < labels_2, 1, 0.5))
    # ], axis=1)

    combined_labels = np.array([get_soft_label(r1, r2) for r1, r2 in zip(labels_1, labels_2)])

    batch['observations'] = prep_batch1['observations'].squeeze(0)
    batch['observations'] = np.transpose(batch['observations'], (0, 2, 3, 1))
    batch['next_observations'] = prep_batch1['next_observations'].squeeze(0)
    batch['next_observations'] = np.transpose(batch['next_observations'], (0, 2, 3, 1))
    batch['actions'] = encode_action(prep_batch1['actions'], 4672)
    batch['actions'] = batch['actions'].reshape(8, 4672)
    batch['observations_2'] = prep_batch2['observations'].squeeze(0)
    batch['observations_2'] = np.transpose(batch['observations_2'], (0, 2, 3, 1))
    batch['next_observations_2'] = prep_batch2['next_observations'].squeeze(0)
    batch['next_observations_2'] = np.transpose(batch['next_observations_2'], (0, 2, 3, 1))
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
    def __init__(self, data_dir, len_query=8, batch_size=256):
        self.data_dir = data_dir
        self.len_query = len_query
        self.batch_size = batch_size
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]


    def __getitem__(self, idx):
        sigma1 = self.file_list[idx]
        sigma2 = self.file_list[idx+1]

        df1 = pd.read_csv(sigma1)
        df2 = pd.read_csv(sigma2)

        prep_batch1 = collect(df1, self.len_query)
        prep_batch2 = collect(df2, self.len_query)

        return combine(prep_batch1, prep_batch2, 'none')


def main(_):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    save_dir = FLAGS.logging.output_dir + '/' + FLAGS.env
    save_dir += '/' + str(FLAGS.model_type) + '/'

    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    save_dir += f"{FLAGS.comment}" + "/"
    save_dir += 's' + str(FLAGS.seed)

    setup_logger(
        variant=variant,
        seed=FLAGS.seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False
    )

    FLAGS.logging.output_dir = save_dir
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)

    set_random_seed(FLAGS.seed)

    observation_dim = FLAGS.obs_dim
    action_dim = FLAGS.act_dim
    criteria_key = None

    data_size = FLAGS.data_size
    interval = int(data_size / FLAGS.batch_size) + 1
    early_stop = EarlyStopping(min_delta=FLAGS.min_delta, patience=FLAGS.patience)

    total_epochs = FLAGS.n_epochs
    config = transformers.GPT2Config(
        **FLAGS.transformer
    )

    config.warmup_steps = int(total_epochs * 0.1 * interval)
    config.total_steps = total_epochs * interval

    trans = TransRewardModel(config=config, observation_dim=observation_dim, action_dim=action_dim,
                             activation=FLAGS.activations, activation_final=FLAGS.activation_final)

    trn_data_dir = "/media/hail/HDD/all_trn"

    eval_data_dir = "/media/hail/HDD/all_eval"

    eval_data_size =  FLAGS.eval_data_size
    eval_interval = int(eval_data_size / FLAGS.batch_size) + 1

    reward_model = PrefTransformer(config, trans)
    train_loss = "reward/trans_loss"

    dataset = ChessDataset(trn_data_dir, 8)
    dataset_ = ChessDataset(eval_data_dir, 8)

    for epoch in range(FLAGS.n_epochs + 1):
        metrics = defaultdict(list)
        metrics['epoch'] = epoch
        if epoch:
            shuffled_idx = np.random.permutation(FLAGS.data_size-1)
            for i in range(interval):
                start_pt = i * FLAGS.batch_size
                end_pt = min((i + 1) * FLAGS.batch_size,FLAGS.data_size-1)
                with Timer() as train_timer:
                    batch_indices = shuffled_idx[start_pt:end_pt]
                    sampler = SubsetRandomSampler(batch_indices)
                    dataloader = DataLoader(dataset,batch_size=FLAGS.batch_size, sampler=sampler)
                    for b in dataloader:
                        pref = tensor_to_numpy(b)
                    batch = batch_to_jax(pref)
                    for key, val in prefix_metrics(reward_model.train(batch), 'reward').items():
                        metrics[key].append(val)
            metrics['train_time'] = train_timer()
        else:
            metrics[train_loss] = [float(FLAGS.query_len)]

        if epoch % FLAGS.eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt, eval_end_pt = j * FLAGS.batch_size, min((j + 1) * FLAGS.batch_size, 2000)
                batch_idx = range(eval_start_pt, eval_end_pt)
                sampler = SequentialSampler(batch_idx)
                dataloader = DataLoader(dataset_, batch_size=FLAGS.batch_size, sampler=sampler)
                for b in dataloader:
                    eval = tensor_to_numpy(b)
                batch_ = batch_to_jax(eval)

                for key, val in prefix_metrics(reward_model.evaluation(batch_), 'reward').items():
                    metrics[key].append(val)
            if not criteria_key:
                    criteria_key = key
            criteria = np.mean(metrics[criteria_key])
            early_stop = early_stop.update(criteria)
            has_improved = early_stop.has_improved

            if early_stop.should_stop and FLAGS.early_stop:
                for key, val in metrics.items():
                    if isinstance(val, list):
                        metrics[key] = np.mean(val)
                logger.record_dict(metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                wb_logger.log(metrics)
                print('Met early stopping criteria, breaking...')
                break
            elif epoch > 0 and has_improved:
                metrics["best_epoch"] = epoch
                metrics[f"{key}_best"] = criteria
                save_data = {"reward_model": reward_model, "variant": variant, "epoch": epoch}
                save_pickle(save_data, "best_model.pkl", save_dir)

        for key, val in metrics.items():
            if isinstance(val, list):
                metrics[key] = np.mean(val)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wb_logger.log(metrics)

    if FLAGS.save_model:
        save_data = {'reward_model': reward_model, 'variant': variant, 'epoch': epoch}
        save_pickle(save_data, 'model.pkl', save_dir)

if __name__ == '__main__':

    absl.app.run(main)


