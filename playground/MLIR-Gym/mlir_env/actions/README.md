## Actions
- action.py
- action_space.py

### action.py
This file defines the action object
e.g., Action("linalg-tile", "func", Requirement())
means, I am instantiating an Action object that has name "linalg-tile". This pass is expected to be run on function level

### action_space.py
This file defines the "masking", the pre-condition for an action, if a user wants to apply an action to a state
e.g., requirement: all_of=Phase.TENSOR, none_of=Phase.MEMREF
indicates this pass is expecting to be execute on an IR that has tensor type, but not memref type.
