"""Independent target, actor-gradient and parameter/buffer update oracles."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
import ddpg_agent

class LinearCritic(nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)
        with torch.no_grad(): self.linear.weight.copy_(torch.tensor([coefficients]))
    def forward(self, state, action):
        return self.linear(torch.cat((state, action), dim=1))

class DDPGOracle(unittest.TestCase):
    def test_target_mask_and_actor_gradient(self):
        agent = ddpg_agent.Agent(2, 1, 1, 7)
        device = ddpg_agent.device
        agent.actor_local = nn.Linear(2, 1, bias=False).to(device)
        agent.actor_target = nn.Linear(2, 1, bias=False).to(device)
        with torch.no_grad():
            agent.actor_local.weight.copy_(torch.tensor([[.2, .1]], device=device))
            agent.actor_target.weight.copy_(torch.tensor([[.5, -.5]], device=device))
        agent.critic_local = LinearCritic([.1, .2, .3]).to(device)
        agent.critic_target = LinearCritic([1., 2., 4.]).to(device)
        agent.actor_optimizer = torch.optim.SGD(agent.actor_local.parameters(), lr=0.)
        agent.critic_optimizer = torch.optim.SGD(agent.critic_local.parameters(), lr=0.)
        states = torch.eye(2, device=device)
        actions = torch.zeros((2, 1), device=device)
        rewards = torch.tensor([[1.], [7.]], device=device)
        terminal = torch.tensor([[0.], [1.]], device=device)
        captured = []; original_loss = ddpg_agent.F.mse_loss
        def capture(prediction, target):
            captured.append(target.detach().clone())
            self.assertFalse(target.requires_grad)
            return original_loss(prediction, target)
        with patch.object(ddpg_agent.F, 'mse_loss', side_effect=capture):
            agent._learn((states, actions, rewards, states, terminal), gamma=.5)
        # First target:1+.5*(1+4*.5)=2.5. Terminal target:reward7.
        torch.testing.assert_close(captured[0].cpu(), torch.tensor([[2.5], [7.]]))
        # d[-mean Q(s, actor(s))]/d actor_weight = -.3*[.5,.5].
        torch.testing.assert_close(agent.actor_local.weight.grad.cpu(), torch.tensor([[-.15, -.15]]))
        self.assertTrue(all(p.grad is None for p in agent.actor_target.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.critic_target.parameters()))

    def test_polyak_parameters_and_batchnorm_buffers(self):
        agent = ddpg_agent.Agent(2, 1, 1, 7)
        local, target = nn.BatchNorm1d(3), nn.BatchNorm1d(3)
        with torch.no_grad():
            local.weight.fill_(6); target.weight.fill_(2)
            local.running_mean.fill_(2); target.running_mean.zero_()
            local.running_var.fill_(4); target.running_var.fill_(1)
            local.num_batches_tracked.fill_(9); target.num_batches_tracked.fill_(1)
        agent._soft_update(local, target, .25)
        torch.testing.assert_close(target.weight, torch.full_like(target.weight, 3.))
        torch.testing.assert_close(target.running_mean, torch.full_like(target.running_mean, .5))
        torch.testing.assert_close(target.running_var, torch.full_like(target.running_var, 1.75))
        self.assertEqual(int(target.num_batches_tracked), 9)
        agent._hard_update(local, target)
        for name, value in local.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], value)

if __name__ == '__main__':
    unittest.main()
