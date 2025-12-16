# utils/common_utils.py
from mlir_env.core.immutable import ErrorBits, Requirement

# -----
# status ok:
# - with changed
# - no changed
#
# status error:
# (no changed)
# - masked
# - timedout
# - failed
# -----

ir_has_changed = Requirement(none_of=ErrorBits.W_NO_CHANGE)
ir_not_changed = Requirement(any_of=ErrorBits.W_NO_CHANGE)
action_masked_out = Requirement(any_of=ErrorBits.E_MASKED)
action_timed_out = Requirement(any_of=ErrorBits.E_TIMEDOUT)
action_run_failed = Requirement(any_of=ErrorBits.E_FAILED | ErrorBits.E_EVAL_FAILED)
action_run_success = Requirement()
