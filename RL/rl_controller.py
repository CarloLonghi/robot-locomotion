from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
import torch
from torch.optim import Adam

from revolve2.actor_controller import ActorController
from revolve2.serialization import SerializeError, StaticData

from RL.interaction_buffer import Buffer
from .actor_critic_network import Actor, ActorCritic, Critic, ObservationEncoder
from .config import LR_ACTOR, LR_CRITIC, PPO_CLIP_EPS, LR_ACTOR, LR_CRITIC


class RLcontroller(ActorController):
    _num_input_neurons: int
    _num_output_neurons: int
    _dof_ranges: npt.NDArray[np.float_]

    def __init__(
        self,
        actor_critic: Actor,
        dof_ranges: npt.NDArray[np.float_],
        from_checkpoint: bool = False,
    ):
        """
        The controller for an agent
        args:
            actor_critic: Neural Network controlling the agents
            dof_ranges: value range for the agent motors
            from_checkpoint: if True, resumes training from the last checkpoint
        """
        self._iteration_num = 0
        self._actor_critic = actor_critic
        actor_params = [p for p in self._actor_critic.actor.parameters() if p.requires_grad]
        critic_params = [p for p in self._actor_critic.critic.parameters() if p.requires_grad]
        self.actor_optimizer = Adam(actor_params, lr=LR_ACTOR, eps=1e-5)
        self.critic_optimizer = Adam(critic_params, lr=LR_CRITIC, eps=1e-5)
        #self.optimizer = Adam([p for p in self._actor_critic.parameters() if p.requires_grad])
        if from_checkpoint:
            checkpoint = torch.load("RL/model_states/last_checkpoint")
            self._iteration_num = checkpoint['iteration']
            self._actor_critic.load_state_dict(checkpoint['model_state'])
            #self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
            #self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
        self._dof_ranges = dof_ranges
        #self.device = torch.device("cuda:0")
        #self._actor_critic.to(self.device)

    def get_dof_targets(self, observation) -> List[float]:
        """
        Get the target position for the motors of the body
        """
        action_prob, value, _, _ = self._actor_critic(observation)
        action = action_prob.sample()
        a, b, logp, _ = self._actor_critic(observation, action)
        return action, value, logp
    
    def train(self, buffer: Buffer):
        """
        Train the neural network used as controller
        args:
            buffer: replay buffer containing the data for the last timesteps
        """
        eps = PPO_CLIP_EPS # ratio clipping parameter

        print(f"\nITERATION NUM: {self._iteration_num + 1}")

        # learning rate decreases linearly
        lr_linear_decay(self.actor_optimizer, self._iteration_num, 100, LR_ACTOR)
        lr_linear_decay(self.critic_optimizer, self._iteration_num, 100, LR_CRITIC)
        #lr_linear_decay(self.optimizer, self._iteration_num, 100, LR_CRITIC)

        self._iteration_num += 1
        #buffer._normalize_rewards()
        buffer._compute_advantages()
        for epoch in range(4):
            batch_sampler = buffer.get_sampler()

            ppo_losses = []
            val_losses = []
            losses = []
            approx_kl = 0
            
            for batch in batch_sampler:
                
                # normalize advantages and compute returns
                batch['adv'] = (batch['adv'] - batch['adv'].mean()) / (batch['adv'].std() + 1e-8)
                batch['ret'] = batch['adv'] + batch['val']

                # get value and new log probability for the observation
                _, value, logp, entropy = self._actor_critic(batch['obs'], batch['act'])

                # compute approximate kl distance between the new and old policy
                with torch.no_grad():
                    approx_kl = (batch['logp_old'] - logp).mean().item()
                    if approx_kl > PPO_CLIP_EPS:
                        print(approx_kl)
                        continue

                # compute ratio between new and old policy
                ratio = torch.exp(logp - batch['logp_old'])
                obj1 = ratio * batch['adv']
                obj2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * batch['adv'] # ratio clipping
                print(f"Percentage of clipped ratios: {(abs(ratio) - 1 > eps).sum() / ratio.shape[0]}")
                ppo_loss = -torch.min(obj1, obj2).mean() # policy loss
                val_loss = (batch['ret'] - value).pow(2).mean() # value loss
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                #self.optimizer.zero_grad()
                loss = val_loss + 0.5 * ppo_loss - 0.01 * entropy
                loss.backward()
                ppo_losses.append(ppo_loss.item())
                val_losses.append(val_loss.item())
                losses.append(loss.item())

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self._actor_critic.parameters(), 0.5
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                #self.optimizer.step()

            print(f"EPOCH {epoch + 1} loss ppo:  {np.mean(ppo_losses)}, loss val: {np.mean(val_losses)}, final loss: {np.mean(losses)}")
        state = {
            'iteration': self._iteration_num,
            'model_state': self._actor_critic.state_dict(),
            #'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(state, "RL/model_states/last_checkpoint")



    # TODO
    def step(self, dt: float):
        return

    # TODO
    def serialize(self) -> StaticData:
        return {
            "num_input_neurons": self._num_input_neurons,
            "num_output_neurons": self._num_output_neurons,
            "dof_ranges": self._dof_ranges.tolist(),
        }

    # TODO
    @classmethod
    def deserialize(cls, data: StaticData) -> RLcontroller:
        if (
            not type(data) == dict
            or not "actor_state" in data
            or not "critic_state" in data
            or not "encoder_state" in data
            or not "num_input_neurons" in data
            or not "num_output_neurons" in data
            or not "dof_ranges" in data
            or not all(type(r) == float for r in data["dof_ranges"])
        ):
            raise SerializeError()

        in_dim = data["num_input_neurons"]
        out_dim = data["num_output_neurons"]
        actor = Actor(in_dim, out_dim)
        actor.load_state_dict(data["actor_state"])
        critic = Critic(in_dim, out_dim)
        critic.load_state_dict(data["critic_state"])
        encoder = ObservationEncoder(in_dim)
        encoder.load_state_dict(data["encoder_state"])
        network = ActorCritic(in_dim, out_dim)
        network.actor = actor
        network.critic = critic
        network.encoder = encoder
        return RLcontroller(
            network,
            np.array(data["dof_ranges"]),
        )

def lr_linear_decay(optimizer, iter, total_iters, initial_lr):
    """
    Decrease the learning rate linearly
    """
    lr = initial_lr - (initial_lr * (iter / float(total_iters)))
    print(f"new learning rate: {lr}")
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr