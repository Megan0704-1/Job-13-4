# Folder structure

|- agents/
|   |-- __init__.py         # public interfaces
|   |-- base_agent.py       # `RLAgent` ABC class
|   |-- value_based/
|   |   |-- __init__.py
|   |   |-- base_agent.py   # `ValueBasedAgent` ABC class
|   |-- policy_based/
|   |   |-- __init__.py
|   |   |-- base_agent.py   # `PolicyBasedAgent` ABC class

# Supported Agent
- MLPPolicyAgent
    - MLPPolicy
- A2CPolicyAgent
    - A2CPolicy

# Supported IO shape
- MLPPolicy
    - 3D Input: (batch, sequence dim, features)
    - 2D Input: (batch, features)
    - OUTPUT: (batch, num of actions)
- A2CPolicy
    - 3D Input: (batch, sequence dim, features)
    - 2D Input: (batch, features)
    - OUTPUT: (batch, num of actions)
