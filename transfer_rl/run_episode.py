import torch
import numpy as np
import pickle
import transfer_rl.my_env


class EpisodeRunner:
    def __init__(self, env, model, mem, save_name, device='cpu', render=False):
        self.env = env
        self.model = model
        self.mem = mem
        self.device = device
        self.render = render
        self.save_name = save_name

        self.total_steps = 0
        self.last_n_rewards = []
        self.last_n_steps = []
        self.best_batch_reward = None
        self.reset()

    def reset(self):
        """
        Reset state of episode runner

        :return:
        """

        self.total_steps = 0
        self.last_n_rewards = []
        self.last_n_steps = []
        self.best_batch_reward = None

    def run(self, max_frames):
        """
        Run episodes until the total number of frames (steps) is reached
        :return:
        """

        # Run episodes until total number of steps (frames) has been met
        while self.total_steps < max_frames:
            self.run_episode()
            self.post_episode()

        # Save results
        self.save_results()

    def run_episode(self):
        """
        Runs one episode of the current environment.

        The model will also be trained while the episode is running. The frequency
        of the training depends on the conditions set in the model.check_train() function.
        """

        state = self.env.reset()
        cur_reward = 0
        cur_steps = 0
        done = False
        while not done:
            cur_steps += 1
            if self.render:
                self.env.render()

            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action, logp = self.model.sample_action(state)
            prev_state = state
            state, reward, done, _ = self.env.step(action.numpy().squeeze())
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            self.mem.push(action, prev_state, logp, reward, done, state)
            cur_reward += reward

            # Train once memory is larger enough. Don't train if rendering.
            if not self.render:
                self.model.check_train(self.mem)

        # Record results from episode
        self.total_steps += cur_steps
        self.last_n_rewards.append(cur_reward)
        self.last_n_steps.append(cur_steps)

    def post_episode(self):

        n = 200
        n_start = max(0, len(self.last_n_rewards) - n)
        cur_average_reward = np.mean(self.last_n_rewards[n_start:])
        print("Episode: {}, Frames elapsed: {}, Last reward: {}, Average reward: {}".format(
            len(self.last_n_rewards), self.total_steps,
            self.last_n_rewards[-1], cur_average_reward))

        if not self.render:
            # Don't save models if we're rendering
            if self.best_batch_reward is None or cur_average_reward > self.best_batch_reward:
                # If we have the best running reward we've seen so far, save the model
                self.best_batch_reward = cur_average_reward
                self.model.save_model("./trained_models/{}.pt".format(self.save_name))

    def save_results(self):
        culm_steps = [c for c in self.last_n_steps]
        for i in range(1, len(culm_steps)):
            culm_steps[i] += culm_steps[i-1]

        results = [[culm_steps[i], self.last_n_steps[i], self.last_n_rewards[i]]
                   for i in range(len(culm_steps))]
        pickle.dump(results, open('./results/' + self.save_name + '.p', 'wb'))
