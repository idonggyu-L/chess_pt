import jax.numpy as jnp
from JaxPref.jax_utils import batch_to_jax
from JaxPref.chess_convert import *
import pickle

def flip_uci(uci: str) -> str:

    def flip_square(sq):
        file = sq[0]
        rank = sq[1]
        flipped_file = chr(ord('h') - (ord(file) - ord('a')))
        flipped_rank = str(9 - int(rank))
        return flipped_file + flipped_rank

    from_sq, to_sq = uci[:2], uci[2:4]
    flipped = flip_square(from_sq) + flip_square(to_sq)
    if len(uci) > 4:
        flipped += uci[4]
    return flipped


def uci_move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    move_index = 64 * from_square + to_square
    return move_index

def encode_action(action, num_classes=4672):
    action = action - 1
    one_hot_action = np.eye(num_classes)[action]
    return one_hot_action

def get_rewards_from_moves(moves, reward_model, len_query=8):

    board = chess.Board()
    states, obs_list, action_list = [], [], []


    for t, move_str in enumerate(moves):
        is_black_move = (t % 2 == 1)
        if is_black_move:
            move_str = flip_uci(move_str)
        move = chess.Move.from_uci(move_str)
        states.append(get_board_features(board))
        obs = get_lc0_input_planes(
            stack_observations(states[: t + 1], history_size=8)
        )
        obs_list.append(obs)
        action_list.append(uci_move_to_index(move))
        board.push(move)

    obs_array = np.array(obs_list)
    act_array = np.array(action_list)
    rewards = []

    for i in range(len(moves)):
        start_idx = max(0, i - len_query + 1)
        end_idx = i + 1

        obs_seq = obs_array[start_idx:end_idx]
        act_seq = act_array[start_idx:end_idx]
        timesteps = np.arange(len(obs_seq))
        attn_mask = np.ones(len(obs_seq), dtype=np.int32)

        pad_len = len_query - len(obs_seq)
        if pad_len > 0:
            obs_seq = np.pad(obs_seq, ((pad_len, 0), (0, 0), (0, 0), (0, 0)), mode="constant")
            act_seq = np.pad(act_seq, (pad_len, 0), mode="constant")
            timesteps = np.pad(timesteps, (pad_len, 0), mode="constant")
            attn_mask = np.pad(attn_mask, (pad_len, 0), mode="constant")

        input = {
            # === observations: (1, len_query, 8, 8, 112)
            "observations": obs_seq.reshape(1, len_query, 8, 8, 112),

            # === actions: (1, len_query, 4672)
            "actions": encode_action(act_seq[None, :], 4672),

            # === timestep: (1, len_query)
            "timestep": timesteps[None, :],

            # === attention mask: (1, len_query)
            "attn_mask": attn_mask[None, :]
        }

        jax_input = batch_to_jax(input)
        new_reward, _ = reward_model.get_reward(jax_input)
        new_reward = new_reward.reshape(1, len_query) * input["attn_mask"]
        new_reward = jnp.sum(new_reward, axis=1) / jnp.sum(input["attn_mask"], axis=1)
        reward_value = float(np.asarray(new_reward).squeeze())
        rewards.append(reward_value)

    return rewards

if __name__ == "__main__":
    uci_moves = [
        "g1f3", "c2c3", "d2d4", "g1f3", "e2e3", "d2d4", "h2h3", "d1c2", "b2b3", "c1f4",
        "f1d3", "f4d6", "c2d3", "c2e4", "f3e5", "b1a3", "d1c2", "e2e3", "h3h4", "f3g1",
        "e1e2", "h2h3", "b3b4", "a1b1", "c2a4", "f1d3", "e2f1", "a3c2", "a4d1", "g1e2",
        "d1h5", "c2a3", "h5f7", "e1d1", "f7e6", "d1e1", "e6d6", "b1c1", "h1h3", "a3c4",
        "d4c5", "b2b3", "g2g4", "h1h2", "f1e2", "c1a1", "b1a3", "b3c4", "d6e6", "h3h4",
        "h3f3", "a1d1", "f3f8", "e1f1", "e6f7"
    ]
    model_path = "/home/hail/Desktop/chess_pt/JaxPref/reward_model/chess/PrefTransformer/expert/s5/model.pkl"
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    rm = data['reward_model']
    get_rewards_from_moves(uci_moves,rm,8)