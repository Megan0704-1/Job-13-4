# Personal leraning note

## Off-policy algorithm
DQN: uses neural network to approximate the q_values (like those implemented in value_based method)

But instead of using the max of q values (greedy epsilon in value based method), our network directly picked the action.

This nn is called `Policy network` or `Policy function`. This function accepts a state returns the probablility distribution over the actions.

**Goal:** Train a nn that can generate high probablity for best action.

# MLP Policy Agent
- update flow
    1. convert trajectory data to tensor
    2. compute discounted rewards
    3. calculate the action probability by individual observation if all chosen by the policy
    4. call `torch.distribution.log_prob` to make an action distribution per observation
    5. compute log probability of actual action taken with the predicted action
    6. weight the discounted returns with action log probability.

- reference
    1. torch.distribution
    2. torch.gather
