# Note. Each state class is defined under: observations/classes/<state_class>/*

- config.py: This file loads a config from specified yml path. Is designed as a variable for StateBuilder
- builder.py: This file implements the StateBuilder class, which exposes `state_class` to user. The builder instantiates different state classes according to the config. For `get_observation`, `reset` it internally calls the method defined in state_class.
- interface.py: This file declares the contract for state class. For each state class, it has to register in the `STATE_REGISTRY` via decorating with `register_state(<name>)`. With the consideration that different state may have different space types, `build_space` method is designed to accept different gym.Spaces type. `observe` method is expected to be called by `get_observation` method defined in the builder.py
