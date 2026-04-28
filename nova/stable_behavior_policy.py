import torch
import torch.nn as nn
import numpy as np
import os
from nova.behavior_net import EncoderRNN, Behavior_Latent_Decoder
from nova.mvp_utils import reparameterize, kl_standard_normal, AdaptiveEtaMLP
from utils.mappo_utils.util import get_grad_norm
import time
import pickle
import copy

EPS = 1e-10

class Behavior_policy:
    def __init__(self, args, logger):
        if args.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.args = args
        self.n_actions = args.n_actions

        # Number of all controllable agents
        self.n_agents = args.n_agents

        # Number of all observed vehicles
        self.max_vehicle_num = args.max_vehicle_num

        # History length stored in the observation wrapper
        self.max_history_len = args.max_history_len

        # Dimension of latent representation
        self.latent_dim = args.latent_dim

        self.optim_eps = args.optim_eps
        self.weight_decay = args.weight_decay

        # A single obs of the kinematic
        self.obs_shape = args.obs_shape

        self.init_behavior_net()

        self.logger = logger
        self.log_prefix = args.log_prefix
        self.log_stats_t = -self.args.learner_log_interval - 1

        self._use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm

        # Stabilization parameters
        self.soft_update_coef = args.soft_update_coef
        self.behavior_variation_penalty = args.behavior_variation_penalty
        self.thres_small_variation = args.thres_small_variation

        # MVP (Direction A+B): Gaussian encoder + adaptive Kalman gain + IB KL.
        self.mvp_enable = getattr(args, "mvp_enable", False)
        if self.mvp_enable:
            self.eta_mlp = [
                AdaptiveEtaMLP(self.latent_dim, args.mvp_eta_mlp_hidden,
                               init_eta=args.soft_update_coef).to(self.device)
                for _ in range(self.n_agents)
            ] if args.mvp_adaptive_eta else None
            if self.eta_mlp is not None:
                for i in range(self.n_agents):
                    self.behavior_optimizer[i].add_param_group(
                        {"params": self.eta_mlp[i].parameters(), "lr": args.lr_behavior}
                    )
            self._mvp_train_step = 0

    # Initialize the behavior encoder / decoder with their optimizers
    def init_behavior_net(self):
        self.behavior_encoder = []
        self.behavior_decoder = []
        self.behavior_optimizer = []

        for i in range(self.n_agents):
            self.behavior_encoder.append(EncoderRNN(input_size=self.args.obs_shape_single,
                                                    hidden_size=self.args.encoder_rnn_dim,
                                                    output_size=self.args.latent_dim,
                                                    num_layers=self.args.num_encoder_layer,
                                                    gaussian=getattr(self.args, "mvp_enable", False)).to(self.device))

            self.behavior_decoder.append(Behavior_Latent_Decoder(
                                                    input_size=self.args.obs_shape_single + self.args.latent_dim,
                                                    hidden_size=self.args.decoder_rnn_dim,
                                                    output_size=self.args.obs_shape_single,
                                                    num_layers=self.args.num_decoder_layer,
                                                    dropout=self.args.decoder_dropout).to(self.device))

            self.encoder_parameters = list(self.behavior_encoder[i].parameters())
            self.decoder_parameters = list(self.behavior_decoder[i].parameters())

            self.behavior_optimizer.append(
                torch.optim.Adam(self.encoder_parameters + self.decoder_parameters, lr=self.args.lr_behavior,
                                                    eps=self.optim_eps,
                                                    weight_decay=self.weight_decay))

    # Update the behavior latent in the rollout
    def latent_update(self, history, encoder_hidden, prev_latent):
        # Similar to choose action
        # (Batch First)
        # history: [n_thread, n_agent, max_vehicle_num, max_history_len, obs_dim]
        # For agent_i, choose history_i [n_thread, max_vehicle_num, max_history_len, obs_dim]
        # RNN (initial): [n_thread, num_layers, n_agent, max_vehicle_num, encoder_rnn_dim]
        # RNN (per agent): [num_layers, n_thread * max_vehicle_num, encoder_rnn_dim]

        # Prev latent / Updated latent: [n_thread, n_agent, max_vehicle_num, latent_dim] (Output)

        # Batch size: n_thread * max_vehicle_num
        # Seq len: max_history_len
        # H_in: obs_dim
        new_latent = []
        per_agent_logvar = []  # MVP only
        encoder_hidden_new = []
        n_thread, n_agent, max_vehicle_num, max_history_len, obs_dim = history.shape
        _, num_layers, _, _, encoder_rnn_dim = encoder_hidden.shape

        for i in range(self.n_agents):
            encoder_hidden_per = torch.Tensor(encoder_hidden[:, :, i]).permute((1, 0, 2, 3)).to(self.device)
            encoder_hidden_per = encoder_hidden_per.reshape(num_layers, n_thread * max_vehicle_num, encoder_rnn_dim)

            history_per = history[:, i, :, :, :].reshape(-1, max_history_len, obs_dim)
            history_per = torch.Tensor(history_per).to(self.device)

            out, encoder_hidden_per, latent = self.behavior_encoder[i](history_per, encoder_hidden_per)
            if self.mvp_enable and self.args.mvp_use_sample_rollout:
                latent = reparameterize(latent, self.behavior_encoder[i].last_logvar)
            new_latent.append(latent.reshape(n_thread, max_vehicle_num, -1).unsqueeze(1))
            if self.mvp_enable:
                per_agent_logvar.append(
                    self.behavior_encoder[i].last_logvar.reshape(n_thread, max_vehicle_num, -1).unsqueeze(1)
                )

            encoder_hidden_per = encoder_hidden_per.reshape(num_layers, n_thread, max_vehicle_num, encoder_rnn_dim)
            encoder_hidden_new.append(encoder_hidden_per.permute((1, 0, 2, 3)).unsqueeze(2))

        new_latent = torch.cat(new_latent, dim=1)  # [n_thread, n_agent, max_vehicle_num, latent_dim]

        if self.mvp_enable and self.args.mvp_adaptive_eta:
            # Data-dependent Kalman gain per agent: eta = sigmoid(MLP(logvar)).
            eta_list = []
            for i in range(self.n_agents):
                with torch.no_grad():
                    eta_i = self.eta_mlp[i](per_agent_logvar[i].squeeze(1))  # [n_thread, mv, 1]
                eta_list.append(eta_i.unsqueeze(1))
            eta = torch.cat(eta_list, dim=1).cpu().numpy()  # [n_thread, n_agent, mv, 1]
            new_latent = new_latent.cpu().detach().numpy()
            new_latent = (1 - eta) * prev_latent + eta * new_latent
        else:
            new_latent = new_latent.cpu().detach().numpy()
            new_latent = (1 - self.soft_update_coef) * prev_latent + new_latent * self.soft_update_coef

        encoder_hidden = torch.cat(encoder_hidden_new, dim=2)
        encoder_hidden.cpu().detach().numpy()

        return new_latent, encoder_hidden

    # Use this as the wrapper for behavior batch generation
    # Sample over existing batches
    # Load: History, mask (If agent terminates), attention latent
    def behavior_traj_wrapper(self, history, step, mask):
        # Input here is the recorded history and masks for a single agent
        # Sampled to reduce the load
        # history: [n_thread, max_episode_len, max_vehicle_num, obs_dim]
        # behavior_latent: [n_thread, max_episode_len, max_vehicle_num, latent_dim]
        # Attention rnn (From episode batch): [n_thread, max_episode_len, max_vehicle_num, attention_dim]
        # Mask: [n_thread, max_episode_len]
        n_thread, max_episode_len, max_vehicle_num, obs_dim = history.shape

        curr_traj = torch.zeros((n_thread, max_vehicle_num, self.max_history_len, obs_dim))
        next_traj = torch.zeros((n_thread, max_vehicle_num, self.max_history_len, obs_dim))

        mask_over_curr_traj = torch.ones_like(curr_traj)
        mask_over_next_traj = torch.ones_like(next_traj)

        start_idx = max(0, step - self.max_history_len + 1)
        plug_in_idx = max(0, self.max_history_len - step - 1)
        end_idx = step + self.max_history_len + 1

        curr_traj[:, :, plug_in_idx:, :] = history[:, start_idx:step + 1, :, :].permute((0, 2, 1, 3))
        next_traj = history[:, step + 1:end_idx, :, :].permute((0, 2, 1, 3))

        for i in range(n_thread):
            for j in range(start_idx, step + 1):
                mask_over_curr_traj[i, :, plug_in_idx + j - start_idx, :] = torch.ones((max_vehicle_num, obs_dim)).to(self.device) * mask[i, j]

            for j in range(self.max_history_len):
                mask_over_next_traj[i, :, j, :] = torch.ones((max_vehicle_num, obs_dim)).to(self.device) * mask[i, step + 1 + j]

        return curr_traj, next_traj, mask_over_curr_traj, mask_over_next_traj

    # Compare the current history with max_history_len and next history with max_history_len
    # Compute the MSE loss between the two as the input of VAE
    def learn(self, batch, t_env):
        train_info = {
            'behavior_loss': 0.,
            "stability_loss": 0.,
            "behavior_total": 0.,
            "behavior_encoder_grad_norm": 0.,
            "behavior_decoder_grad_norm": 0.,
        }
        if self.mvp_enable:
            train_info['mvp_kl_ib'] = 0.
            train_info['mvp_kl_warmup'] = 0.
            train_info['mvp_sigma_mean'] = 0.
            train_info['mvp_sigma_p10'] = 0.
            train_info['mvp_sigma_p90'] = 0.
            if self.args.mvp_adaptive_eta:
                train_info['mvp_eta_mean'] = 0.
                train_info['mvp_eta_std'] = 0.

        behavior_loss = []
        stability_loss = []
        total_loss = []

        # history: [n_thread, max_episode_len, n_agent, max_vehicle_num, obs_dim]
        # behavior_latent: [n_thread, max_episode_len, n_agent, max_vehicle_num, latent_dim]
        # agent_terminate: [n_thread, max_episode_len, n_agent, 1]
        history = batch["history"][:, :-1]
        behavior_latent = batch["behavior_latent"][:, :-1]
        agent_terminate = batch["terminated"][:, :-1]

        n_thread, max_episode_len, n_agent, max_vehicle_num, obs_dim = history.shape
        latent_dim = behavior_latent.shape[-1]

        for i in range(self.n_agents):
            # Agent history: [n_thread, max_episode_len, max_vehicle_num, obs_dim]
            # => [n_thread, num_history, max_history_len, max_vehicle_num, obs_dim]
            agent_history = history[:, :, i]
            agent_behavior_latent = behavior_latent[:, :, i]
            
            if self.args.env == "MPE":
                mask = 1 - agent_terminate[:, :, i, 0]
            else:
                mask = agent_terminate[:, :, i, 0]

            # Prev latent / Updated latent: [n_thread, max_vehicle_num, latent_dim]
            latent = torch.zeros((n_thread, max_vehicle_num, self.latent_dim)).to(self.device)

            # Initialize the hidden state
            encoder_hidden = torch.zeros((self.args.num_encoder_layer, n_thread * self.max_vehicle_num,
                                          self.args.encoder_rnn_dim)).to(self.device)
            decoder_hidden = torch.zeros((self.args.num_decoder_layer, n_thread * self.max_vehicle_num,
                                          self.args.decoder_rnn_dim)).to(self.device)

            behavior_error = 0.
            stability_error = 0.
            kl_error = 0.  # MVP: IB KL accumulator
            sigma_mean_acc = 0.
            sigma_p10_acc = 0.
            sigma_p90_acc = 0.
            eta_mean_acc = 0.
            eta_std_acc = 0.
            for j in range(max_episode_len - 1 - self.max_history_len):
                # Wrap the current and next trajectories with their mask for training
                curr_traj, next_traj, mask_over_curr_traj, mask_over_next_traj = \
                    self.behavior_traj_wrapper(agent_history, j, mask)

                # curr history: [n_thread, max_vehicle_num, max_history_len, obs_dim]
                curr_history = torch.Tensor(curr_traj).to(self.device)
                next_history = torch.Tensor(next_traj).to(self.device)

                mask_over_next_traj = torch.Tensor(mask_over_next_traj).to(self.device)

                # Decoder
                pred_history, decoder_hidden = self.behavior_decoder[i](curr_history, latent, decoder_hidden)
                pred_history = pred_history.reshape(n_thread, max_vehicle_num, self.max_history_len, obs_dim)

                # Encoder
                curr_history_en = curr_history.reshape(n_thread * max_vehicle_num, self.max_history_len, obs_dim)
                _, encoder_hidden, new_latent = self.behavior_encoder[i](curr_history_en, encoder_hidden)

                # Stability Error: Compute the variance between current history and predicted next history
                stability = torch.linalg.norm(curr_history - pred_history, dim=-1).reshape(-1, )

                if self.mvp_enable:
                    # (B) VIB KL(q || N(0,I)) per step; aggregated over latent dim.
                    mu_q = new_latent
                    logvar_q = self.behavior_encoder[i].last_logvar
                    kl_step = kl_standard_normal(mu_q, logvar_q, free_bits=self.args.mvp_free_bits)
                    kl_error = kl_error + kl_step.sum(dim=-1).mean()

                    with torch.no_grad():
                        sigma = (0.5 * logvar_q).exp()
                        sigma_flat = sigma.reshape(-1)
                        sigma_mean_acc += sigma_flat.mean().item()
                        sigma_p10_acc += torch.quantile(sigma_flat, 0.10).item()
                        sigma_p90_acc += torch.quantile(sigma_flat, 0.90).item()

                    # Reparameterized sample for belief update (enables gradient to logvar via L1).
                    z_sample = reparameterize(mu_q, logvar_q)
                    z_sample = z_sample.reshape(n_thread, max_vehicle_num, self.latent_dim)

                    # (A) Adaptive Kalman gain: eta = sigmoid(MLP(logvar_q)).
                    if self.args.mvp_adaptive_eta:
                        logvar_batch = logvar_q.reshape(n_thread, max_vehicle_num, self.latent_dim)
                        eta = self.eta_mlp[i](logvar_batch)  # [n_thread, mv, 1]
                        with torch.no_grad():
                            eta_mean_acc += eta.mean().item()
                            eta_std_acc += eta.std().item()
                    else:
                        eta = self.soft_update_coef
                    latent = (1 - eta) * latent + eta * z_sample
                else:
                    # Original iPLAN path (constant-gain EMA).
                    new_latent = new_latent.reshape(n_thread, max_vehicle_num, self.latent_dim)
                    latent = (1 - self.soft_update_coef) * latent + new_latent * self.soft_update_coef

                # Behavior Error: [n_thread, max_vehicle_num, max_history_len, obs_dim]
                if self.mvp_enable and getattr(self.args, "mvp_laplace_nll", False):
                    logvar_scalar = logvar_q.reshape(n_thread, max_vehicle_num, self.latent_dim).mean(dim=-1, keepdim=True).unsqueeze(-1)
                    inv_sigma = (-0.5 * logvar_scalar).exp()
                    raw_err = torch.abs(next_history - pred_history) * inv_sigma + 0.5 * logvar_scalar
                    error = torch.mul(raw_err, mask_over_next_traj)
                else:
                    error = torch.mul(torch.abs(next_history - pred_history), mask_over_next_traj)
                # Average over history number and threads
                behavior_error += error.sum() / (mask_over_next_traj.sum() + EPS) * obs_dim * max_vehicle_num

                # Stability Error: Only consider the stability error larger than a pre-defined threshold
                stability = torch.clamp(stability - self.thres_small_variation, min=0)
                # Average over history number and threads
                stability_error += stability.sum() / n_thread / self.max_history_len

            # Combine the behavior and stability error together with trade-off coefficients
            # However, in iPLAN, we are not using stability error
            n_steps = max_episode_len - 1 - self.max_history_len
            behavior_error = behavior_error / n_steps
            stability_error = stability_error / n_steps
            loss = behavior_error + self.behavior_variation_penalty * stability_error

            if self.mvp_enable:
                # Linear KL warmup to stabilize early training.
                warmup = min(1.0, self._mvp_train_step / max(1, self.args.mvp_ib_kl_warmup_steps))
                kl_error = kl_error / n_steps
                loss = loss + warmup * self.args.mvp_ib_kl_weight * kl_error
                kl_val = kl_error.item() if torch.is_tensor(kl_error) else float(kl_error)
                train_info['mvp_kl_ib'] += kl_val / self.n_agents
                train_info['mvp_kl_warmup'] += warmup / self.n_agents
                train_info['mvp_sigma_mean'] += (sigma_mean_acc / n_steps) / self.n_agents
                train_info['mvp_sigma_p10'] += (sigma_p10_acc / n_steps) / self.n_agents
                train_info['mvp_sigma_p90'] += (sigma_p90_acc / n_steps) / self.n_agents
                if self.args.mvp_adaptive_eta:
                    train_info['mvp_eta_mean'] += (eta_mean_acc / n_steps) / self.n_agents
                    train_info['mvp_eta_std'] += (eta_std_acc / n_steps) / self.n_agents

            self.behavior_optimizer[i].zero_grad()
            loss.backward()

            # Apply max_grad_norm
            if self._use_max_grad_norm:
                behavior_encoder_grad_norm = nn.utils.clip_grad_norm_(self.behavior_encoder[i].parameters(), self.max_grad_norm)
            else:
                behavior_encoder_grad_norm = get_grad_norm(self.behavior_encoder[i].parameters())

            if self._use_max_grad_norm:
                behavior_decoder_grad_norm = nn.utils.clip_grad_norm_(self.behavior_decoder[i].parameters(), self.max_grad_norm)
            else:
                behavior_decoder_grad_norm = get_grad_norm(self.behavior_decoder[i].parameters())

            self.behavior_optimizer[i].step()

            behavior_loss.append(behavior_error.cpu().detach().numpy())
            stability_loss.append(stability_error.cpu().detach().numpy())
            total_loss.append(loss.cpu().detach().numpy())

            # Update the training metrics for tensorboard
            train_info['behavior_loss'] += behavior_error.item()
            train_info['stability_loss'] += stability_error.item()
            train_info['behavior_total'] += loss.item()
            train_info['behavior_encoder_grad_norm'] += behavior_encoder_grad_norm.item()
            train_info['behavior_decoder_grad_norm'] += behavior_decoder_grad_norm.item()

        if self.mvp_enable:
            self._mvp_train_step += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k, v in train_info.items():
                self.logger.log_stat(self.log_prefix + k, v, t_env)

        return behavior_loss, stability_loss, total_loss

    # Save models
    def save_models(self, path):
        for i, behavior_encoder in enumerate(self.behavior_encoder):
            torch.save(behavior_encoder.state_dict(), f"{path}/behavior_encoder_{i}.th")
        for i, behavior_decoder in enumerate(self.behavior_decoder):
            torch.save(behavior_decoder.state_dict(), f"{path}/behavior_decoder_{i}.th")
        for i in range(self.n_agents):
            torch.save(self.behavior_optimizer[i].state_dict(), "{}/behavior_optimizer_{}_opt.th".format(path, i))
        if self.mvp_enable and self.eta_mlp is not None:
            for i in range(self.n_agents):
                torch.save(self.eta_mlp[i].state_dict(), f"{path}/eta_mlp_{i}.th")

    # Load models
    def load_models(self, paths:list, load_optimisers=False):
        if len(paths) == 1:
            path = copy.copy(paths[0])
            paths = [path for i in range(self.n_agents)]

        for i, behavior_encoder in enumerate(self.behavior_encoder):
            behavior_encoder.load_state_dict(
                torch.load("{}/behavior_encoder_{}.th".format(paths[i], i),
                            map_location=lambda storage, loc: storage))
        for i, behavior_decoder in enumerate(self.behavior_decoder):
            behavior_decoder.load_state_dict(
                torch.load("{}/behavior_decoder_{}.th".format(paths[i], i),
                        map_location=lambda storage, loc: storage))

        if self.mvp_enable and self.eta_mlp is not None:
            for i in range(self.n_agents):
                ckpt = "{}/eta_mlp_{}.th".format(paths[i], i)
                if os.path.exists(ckpt):
                    self.eta_mlp[i].load_state_dict(
                        torch.load(ckpt, map_location=lambda storage, loc: storage))

        if load_optimisers:
            if len(paths) == 1:
                path = copy.copy(paths[0])
                paths = [path for i in range(self.n_agents)]

            for i in range(self.n_agents):
                self.behavior_optimizer[i].load_state_dict(torch.load("{}/behavior_optimizer_{}_opt.th".format(paths[i], i),
                                                                      map_location=lambda storage, loc: storage))
