import numpy as np
import tensorflow as tf

class FakeEnv:
    def __init__(self, model, args,
                 is_use_reward=True,
                 is_use_oracle_reward=False,
                 is_fake_deterministic=False):
        self.model = model
        self.args = args
        self.is_use_reward = is_use_reward
        self.is_use_oracle_reward = is_use_oracle_reward
        self.is_fake_deterministic = is_fake_deterministic

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''

    def step(self, obs, act):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)


        ensemble_samples = ensemble_model_means
        samples = np.mean(ensemble_samples, axis=0)
        rewards, next_obs = samples[:,:1], samples[:,1:]

        penalized_rewards = rewards



        return next_obs, penalized_rewards, None
