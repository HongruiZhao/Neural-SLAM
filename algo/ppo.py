# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():

    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
            actor_critic.parameters()), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        # Compute normalization stats over valid (non-after-done) samples
        # only, mirroring the filter in `feed_forward_generator`. The env
        # has no per-env auto-reset, so post-termination rewards (e.g.
        # finish_bonus refiring) would otherwise bias the running mean/std
        # and dilute the gradient on valid samples.
        masks_slice = rollouts.masks[:-1]
        after_done = torch.zeros_like(masks_slice, dtype=torch.bool)
        if masks_slice.size(0) > 1:
            after_done[1:] = (masks_slice[1:] == 0) & (masks_slice[:-1] == 0)
        valid = ~after_done
        valid_adv = advantages[valid]
        advantages = (advantages - valid_adv.mean()) / (valid_adv.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        approx_kl_epoch = 0
        clip_fraction_epoch = 0

        for e in range(self.ppo_epoch):

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                        advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                        advantages, self.num_mini_batch)

            for sample in data_generator:

                value_preds = sample['value_preds'] 
                returns = sample['returns'] 
                adv_targ = sample['adv_targ'] 

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = \
                        self.actor_critic.evaluate_actions(
                            sample['obs'], sample['rec_states'],
                            sample['masks'], sample['actions'],
                            extras=sample['extras']
                        )

                ratio = torch.exp(action_log_probs -
                                    sample['old_action_log_probs'])
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                        1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds + \
                                        (values - value_preds).clamp(
                                            -self.clip_param, self.clip_param)
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (value_pred_clipped
                                            - returns).pow(2)
                    value_loss = .5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                        dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item() * self.value_loss_coef
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += - dist_entropy.item() * self.entropy_coef

                with torch.no_grad():
                    approx_kl = (sample['old_action_log_probs'] - action_log_probs).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
                    approx_kl_epoch += approx_kl.item()
                    clip_fraction_epoch += clip_frac.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        approx_kl_epoch /= num_updates
        clip_fraction_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, approx_kl_epoch, clip_fraction_epoch
