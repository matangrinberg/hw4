import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem': print(f"CEM params: alpha={self.cem_alpha}, " + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")
    
    
    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' or (self.sample_strategy == 'cem' and obs is None):
            # uniformly sample trajectories and return an array of dimensions (num_sequences, horizon, self.ac_dim) in the range [self.low, self.high]
            return np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
        
        elif self.sample_strategy == 'cem':
            # Implement action selection using CEM.
            actions = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
            
            # Begin with randomly selected actions, then refine the sampling distribution iteratively
            elite_mean = np.zeros(actions.shape[1:])
            elite_std = np.zeros(actions.shape[1:])
            
            for i in range(self.cem_iterations):
                # sample candidate sequences from a Gaussian with the current elite mean and variance
                if i > 0:
                    actions = np.random.normal(elite_mean, elite_std, size=(num_sequences, *elite_mean.shape))
                    
                rewards = self.evaluate_candidate_sequences(actions, obs)
                sorted_indices = sorted(range(len(actions)), key=lambda i: rewards[i])
                elites = actions[sorted_indices][-self.cem_num_elites:] # get the top `self.cem_num_elites` elites
                
                if i == 0:  # for the first iteration, we instead sample uniformly at random just like we do for random-shooting
                    elite_mean = np.mean(elites, axis=0)
                    elite_std = np.std(elites, axis=0)
                else: # update the elite mean and variance
                    elite_mean = self.cem_alpha * np.mean(elites, axis=0) + (1 - self.cem_alpha) * elite_mean
                    elite_std = self.cem_alpha * np.std(elites, axis=0) + (1 - self.cem_alpha) * elite_std

            # set `cem_action` to the appropriate action chosen by CEM
            cem_action = elite_mean
            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # for each model in ensemble, compute the predicted sum of rewards for each candidate action sequence
        predicted_rewards = []
        for model in self.dyn_models:
            sum_of_rewards = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            predicted_rewards.append(sum_of_rewards)

        # calculate the mean predictions across all ensembles
        return np.mean(predicted_rewards, axis=0)

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon, obs=obs)
        
        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)]
            action_to_take = best_action_sequence[0]
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """

        sum_of_rewards = np.zeros((self.N))        
        predicted_obs = np.tile(obs, (self.N, 1))
        
        for t in range(self.horizon):
            # getting N actions (the model can process observations and actions in batch, which can be much faster than looping through each action sequence)
            action = candidate_action_sequences[:, t, :]
            
            # calculating sum of rewards for sequence
            r, _ = self.env.get_reward(predicted_obs, action)
            sum_of_rewards += r
            
            # predicting result of performing actions
            predicted_next_obs = model.get_prediction(predicted_obs, action, self.data_statistics)
            predicted_obs = predicted_next_obs
        
        return sum_of_rewards
