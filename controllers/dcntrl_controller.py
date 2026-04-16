import copy
import torch as th
from modules.agents.ippo_actor import R_Actor
from modules.critics.ippo_critic import R_Critic
from modules.critics import REGISTRY as critic_REGISTRY
from components.action_selectors import EpsilonGreedyActionSelector


class DcntrlMAC:
    '''This multi-agent controller does not share parameters between agents'''
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        actor_input_shape = self._get_actor_input_shape(scheme)
        critic_input_shape = self._get_critic_input_shape(scheme)

        self._build_agents(actor_input_shape)
        self._build_critics(critic_input_shape)

        self.agent_output_type = args.agent_output_type

        self.action_selector = EpsilonGreedyActionSelector(args)

        self.hidden_states = None
        self.input_scheme = scheme

### IPPO ###
    def select_actions_ippo(self, ep_batch, t_ep, test_mode=False):
        actors_inputs = self._build_actor_inputs(ep_batch, t_ep)
        critic_inputs = self._build_critic_inputs(ep_batch, t_ep)
        avail_actions_actors = ep_batch["avail_actions"][:, t_ep]
        rnn_states_actors = ep_batch["rnn_states_actors"][:, t_ep]
        rnn_states_critics = ep_batch["rnn_states_critics"][:, t_ep]

        values, actions, action_log_probs, rnn_states_actors_new, rnn_states_critics_new = [], [], [], [], []
        for agent_id in range(self.args.n_agents):
            obs = actors_inputs[:, agent_id, :]
            critic_obs = critic_inputs[:, agent_id, :]
            avail_actions = avail_actions_actors[:, agent_id, :]
            rnn_states_actor = rnn_states_actors[:, agent_id, :]
            rnn_states_critic = rnn_states_critics[:, agent_id, :]
            avail_actions_actor = avail_actions.unsqueeze(1) if obs.shape[0] > 1 else avail_actions

            action, action_log_prob, rnn_states_actor = self.agents[agent_id](obs.unsqueeze(1),
                                                                              rnn_states_actor.unsqueeze(0),
                                                                              avail_actions_actor,
                                                                              deterministic=True if test_mode else False)
            if action.dim() == 2:
                action = action.unsqueeze(1)
            if action_log_prob.dim() == 2:
                action_log_prob = action_log_prob.unsqueeze(1)
            actions.append(action)
            action_log_probs.append(action_log_prob)
            rnn_states_actors_new.append(rnn_states_actor.unsqueeze(2))

            value, rnn_states_critic = self.critics[agent_id](critic_obs.unsqueeze(1),
                                                              rnn_states_critic.unsqueeze(0))
            if value.dim() == 2:
                value = value.unsqueeze(1)
            values.append(value)
            rnn_states_critics_new.append(rnn_states_critic.unsqueeze(2))

        values = th.cat(values, dim=2).squeeze(1).cpu().detach().numpy()
        actions = th.cat(actions, dim=2).squeeze(1).cpu().detach().numpy()
        rnn_states_actors_new = th.cat(rnn_states_actors_new, dim=2).permute((1, 0, 2, 3)).cpu().detach().numpy()
        rnn_states_critics_new = th.cat(rnn_states_critics_new, dim=2).permute((1, 0, 2, 3)).cpu().detach().numpy()

        return values, actions, action_log_probs, rnn_states_actors_new, rnn_states_critics_new


    def get_value_ippo(self, agent_id, obs, rnn_states_critic):
        """Inputs have shape (batch_size, feat_size) or (batch_size, ts, feat_size)"""
        obs_feats = obs.shape[-1]
        obs_in = obs.reshape(-1, 1, obs_feats)
        hidden_in = rnn_states_critic.reshape(self.args.recurrent_N, -1, self.args.rnn_hidden_dim)
        value, _ = self.critics[agent_id](obs_in, hidden_in)
        value_out = value.reshape(*obs.shape[:-1], 1)
        return value_out

    def eval_action_ippo(self, agent_id, obs, action, available_actions, rnn_states_actor):
        """Inputs have shape (batch_size, feat_size) or (batch_size, ts, feat_size)"""
        obs_feats = obs.shape[-1]
        obs_in = obs.reshape(-1, 1, obs_feats)
        hidden_in = rnn_states_actor.reshape(self.args.recurrent_N, -1, self.args.rnn_hidden_dim)
        action_in = action.reshape(-1, 1, 1)
        num_actions = available_actions.shape[-1]
        avail_actions_in = available_actions.reshape(-1, 1, num_actions)

        action_log_probs, dist_entropy = self.agents[agent_id].evaluate_actions(obs_in,
                                                                                hidden_in,
                                                                                action_in,
                                                                                avail_actions_in)

        action_log_probs_out = action_log_probs.reshape(*obs.shape[:-1], 1)  # cast to original batch shape, as inferred from obs
        return action_log_probs_out, dist_entropy

    def _build_inputs_ippo(self, agent_id, batch, action_onehot, discr_signal=None):
        return self._build_actor_inputs_ippo(agent_id, batch, action_onehot, discr_signal)

    def _build_actor_inputs_ippo(self, agent_id, batch, action_onehot, discr_signal=None):
        # Assumes homogenous agents with flat observations.
        # builds inputs for one agent, all timesteps
        bs, num_ts, _, _ = batch["history"].shape

        states = []

        states.append(batch["history"])
        states = th.cat(states, dim=-1)

        inputs = []
        inputs.append(states.reshape((bs, num_ts, -1)))  # b1av

        if self.args.obs_last_action:
            last_action_onehot = th.cat([action_onehot[:, 0].unsqueeze(1), action_onehot[:, :-1]], axis=1)
            inputs.append(last_action_onehot)
        if self.args.obs_agent_id:
            agent_id_onehot = th.zeros((bs, num_ts, self.n_agents), device=self.agents[agent_id].device)
            agent_id_onehot[:, :, agent_id] = 1
            inputs.append(agent_id_onehot)

        inputs = th.cat(inputs, dim=-1)
        return inputs

    def _build_critic_inputs_ippo(self, agent_id, batch, action_onehot=None, discr_signal=None):
        state = batch["state"]
        if not getattr(self.args, "critic_use_instant_belief", True):
            return state

        instant_belief = batch["instant_belief"]
        bs, num_ts = state.shape[0], state.shape[1]
        instant_belief = instant_belief.reshape(bs, num_ts, -1)
        critic_inputs = th.cat([state, instant_belief], dim=-1)
        return critic_inputs
### IPPO ###

    def init_hidden(self, batch_size):
        self.hidden_states = []
        for i, agent in enumerate(self.agents):
            self.hidden_states.append(agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1))  # bav

    def parameters(self):
        agents_params = []
        for agent in self.agents:
            agents_params.append(list(agent.parameters()))
        return agents_params

    def critic_parameters(self):
        critics_params = []
        for critic in self.critics:
            critics_params.append(list(critic.parameters()))
        return critics_params

    def load_state(self, other_mac):
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(other_mac.agents[i].state_dict())

    def cuda(self):
        for agent in self.agents:
            agent.cuda()
        for critic in self.critics:
            critic.cuda()

    def set_train_mode(self):
        for agent in self.agents:
            agent.train()
        for critic in self.critics:
            critic.train()

    def set_eval_mode(self):
        for agent in self.agents:
            agent.eval()
        for critic in self.critics:
            critic.eval()

    def save_models(self, path):
        for i, agent in enumerate(self.agents):
            th.save(agent.state_dict(), f"{path}/agent_{i}.th")
        for i, critic in enumerate(self.critics):
            th.save(critic.state_dict(), f"{path}/critic_{i}.th")

    def load_models(self, paths):
        if len(paths) == 1:
            path = copy.copy(paths[0])
            paths = [path for i in range(self.n_agents)]
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(
                th.load("{}/agent_{}.th".format(paths[i], i), 
                        map_location=lambda storage, loc: storage))
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(
                th.load("{}/critic_{}.th".format(paths[i], i), 
                        map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(R_Actor(input_shape, self.args))

    def _build_critics(self, input_shape):
        self.critics = []
        if self.args.critic is not None:
            for i in range(self.n_agents):
                self.critics.append(R_Critic(input_shape, self.args))

    def _build_inputs(self, batch, t):
        return self._build_actor_inputs(batch, t)

    def _build_actor_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        states = []

        states.append(batch["history"][:, t])
        states = th.cat(states, dim=-1)

        inputs = []
        inputs.append(states)  # b1av

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        return inputs

    def _build_critic_inputs(self, batch, t):
        bs = batch.batch_size
        state = batch["state"][:, t]
        if getattr(self.args, "critic_use_instant_belief", True):
            instant_belief = batch["instant_belief"][:, t].reshape(bs, -1)
            critic_inputs = th.cat([state, instant_belief], dim=-1)
        else:
            critic_inputs = state
        critic_inputs = critic_inputs.unsqueeze(1).expand(-1, self.n_agents, -1)
        return critic_inputs

    def _get_actor_input_shape(self, scheme):
        history_shape = scheme["history"]["vshape"]
        input_shape = history_shape[0] * history_shape[1]

        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _get_critic_input_shape(self, scheme):
        state_shape = scheme["state"]["vshape"]
        if isinstance(state_shape, tuple):
            input_shape = 1
            for dim in state_shape:
                input_shape *= dim
        else:
            input_shape = state_shape

        if getattr(self.args, "critic_use_instant_belief", True):
            instant_belief_shape = scheme["instant_belief"]["vshape"]
            if isinstance(instant_belief_shape, tuple):
                instant_dim = 1
                for dim in instant_belief_shape:
                    instant_dim *= dim
            else:
                instant_dim = instant_belief_shape

            input_shape += self.n_agents * instant_dim
        return input_shape

