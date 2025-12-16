# analysis/view_resolver.py
from dataclasses import dataclass
import mlir.ir as mlir_ir


class ModuleView:
    """Module parser"""
    def __init__(self, ir_text: str):
        self.ctx = mlir_ir.Context()
        self.ctx.allow_unregistered_dialects = True
        with self.ctx:
            self.module = mlir_ir.Module.parse(ir_text)

    def functions(self):
        # Yield (func_name, func_op)
        for op in self.module.body.operations:
            if op.operation.name in {"func.func", "llvm.func"}:
                name = op.attributes["sym_name"].value
                yield name, op


@dataclass(frozen=True)
class AnalysisIntent:
    need_polyhedral_access: bool = True  # access matrix, affine maps
    need_loops: bool = True
    inspect_vector_ops: bool = False  # when true, prefer VECTOR view


def pick_view(intent: AnalysisIntent) -> str:
    from .spec import ViewKind

    if intent.inspect_vector_ops:
        return ViewKind.VECTOR
    if intent.need_polyhedral_access and intent.need_loops:
        return ViewKind.AFFINE
    if intent.need_loops and not intent.need_polyhedral_access:
        return ViewKind.SCF
    if not intent.need_loops:
        return ViewKind.MEMREF

    return ViewKind.AFFINE
