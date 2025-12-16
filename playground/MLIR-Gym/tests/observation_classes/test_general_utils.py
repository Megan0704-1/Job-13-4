# tests/observation_class/analysis/utils/test_general_utils.py

import pytest
from mlir import ir as mlir_ir
import mlir.dialects.func as func_d
import mlir.dialects.scf as scf_d
import mlir.dialects.arith as arith_d
import mlir.dialects.affine as affine_d  # just to ensure registration

from mlir_env.observations.classes.utils.general_utils import walk_ops_recursive


def names_preorder(op):
    """Helper: return operation.name list produced by walk_ops(op)."""
    return [o.operation.name for o in walk_ops_recursive(op)]


def test_preorder_simple_for_loop():
    txt = r"""
    module {
      func.func @main() {
        %c0  = arith.constant 0  : index
        %c10 = arith.constant 10 : index
        %c1  = arith.constant 1  : index
        scf.for %i = %c0 to %c10 step %c1 {
          %t = arith.addi %c0, %c1 : index
          scf.yield
        }
        func.return
      }
    }
    """
    with mlir_ir.Context():
        # importing the dialect modules above registers them in the context
        module = mlir_ir.Module.parse(txt)
        got = names_preorder(module.operation)

    # Pre-order: parent before children. Expect to see module, func, constants, for, body ops, return.
    assert got == [
        "builtin.module",
        "func.func",
        "arith.constant",
        "arith.constant",
        "arith.constant",
        "scf.for",
        "arith.addi",
        "scf.yield",
        "func.return",
    ]


def test_if_two_regions_two_blocks():
    txt = r"""
    module {
      func.func @f(%x: i32, %y: i32) -> i32 {
        %c0 = arith.constant 0 : i32
        %cmp = arith.cmpi eq, %x, %y : i32
        %res = scf.if %cmp -> i32 {
          %s = arith.addi %x, %y : i32
          scf.yield %s : i32
        } else {
          scf.yield %c0 : i32
        }
        func.return %res : i32
      }
    }
    """
    with mlir_ir.Context():
        module = mlir_ir.Module.parse(txt)
        got = names_preorder(module.operation)

    # We expect to see both yield ops from the two regions of scf.if, in pre-order.
    assert got == [
        "builtin.module",
        "func.func",
        "arith.constant",
        "arith.cmpi",
        "scf.if",
        "arith.addi",
        "scf.yield",
        "scf.yield",
        "func.return",
    ]


def test_empty_function_body_is_handled():
    txt = r"""
    module {
      func.func @empty() {
        func.return
      }
    }
    """
    with mlir_ir.Context():
        module = mlir_ir.Module.parse(txt)
        got = names_preorder(module.operation)

    # Only module, func, return should be yielded.
    assert got == ["builtin.module", "func.func", "func.return"]


def test_nested_loops_preorder_and_count():
    txt = r"""
    module {
      func.func @g() {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %c8 step %c1 {
          scf.for %j = %c0 to %c8 step %c1 {
            scf.yield
          }
          scf.yield
        }
        func.return
      }
    }
    """
    with mlir_ir.Context():
        module = mlir_ir.Module.parse(txt)
        ops = list(walk_ops_recursive(module.operation))
        names = [o.operation.name for o in ops]

    # Order: module, func, constants, outer for, inner for, yields, return
    assert names == [
        "builtin.module",
        "func.func",
        "arith.constant",
        "arith.constant",
        "arith.constant",
        "scf.for",
        "scf.for",
        "scf.yield",
        "scf.yield",
        "func.return",
    ]

    # Sanity: exactly 2 scf.for and 2 scf.yield present
    assert names.count("scf.for") == 2
    assert names.count("scf.yield") == 2

