# mlir_env/observations/__init__.py
# side effect import, making sure SVEngine is loaded when the package is imported

from .interface import STATE_REGISTRY

# import state classes so they register themselves
from .classes.SV.engine import SVEngine # noqa: F401
from .classes.Graph.engine import GraphEngine # noqa: F401
