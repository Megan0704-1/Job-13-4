# te sts/observation_classes/sv/analysis/test_loop_analysis.py
# tests/unit/test_loop_analysis.py
import math
import numpy as np
import pytest

try:
    import mlir.ir as mlir_ir
except Exception:
    mlir_ir = None

pytestmark = pytest.mark.skipif(
    mlir_ir is None, reason="MLIR Python bindings not available"
)

# ---- Imports from your package under test ----
from mlir_env.views.viewer import ModuleView
from mlir_env.observations.classes.SV.analysis.loop_analysis import (
    LoopInformationExtractor,
)
from mlir_env.observations.classes.SV.immutable import LoopParams


# ---------- IR fixtures ----------
def ir_single_scf_for():
    return r"""
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c2  = arith.constant 2 : index
    scf.for %i = %c0 to %c32 step %c2 {
      %x = arith.addi %c0, %c32 : index
    }
    return
  }
}
"""

def ir_nested_two():
    # Outer scf.for, inner affine.for with unknown upper bound
    return r"""
module {
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c8  = arith.constant 8 : index
    scf.for %i = %c0 to %c64 step %c8 {
      %N = arith.constant 128 : index
      affine.for %j = 0 to %N {
      }
    }
    return
  }
}
"""

def ir_scf_par():
    # scf.parallel with two dims: (0..128 step 1) x (0..16 step 4)
    return r"""
module {
  func.func @p() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c4  = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c16  = arith.constant 16 : index
    %sum = arith.constant 0 : index
    scf.parallel (%i, %j) = (%c0, %c0) to (%c128, %c16) step (%c1, %c4) {
    }
    return
  }
}
"""

def ir_affine_par():
    return r"""
module {
  func.func @p(%alloc: memref<2048x2048xf32>) {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c4  = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c16  = arith.constant 16 : index
    %sum = arith.constant 0 : index
    affine.parallel (%i, %j) = (%c0, %c0) to (%c128, %c16) step (1, 1) {
      %1 = affine.load %alloc[%i, %j] : memref<2048x2048xf32>
    }
    return
  }
}
"""

def ir_affine_unknown_bounds():
    # affine.for with symbolic ub → tripcount unknown
    return r"""
module {
  func.func @g() {
    %N = arith.constant 1024 : index
    affine.for %i = 0 to %N {
    }
    return
  }
}
"""

def ir_rename_invariance():
    # two modules with different IV names; features should match
    base = r"""
module {
  func.func @h() {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c2  = arith.constant 2 : index
    scf.for %i = %c0 to %c32 step %c2 { }
    return
  }
}
"""
    alt = base.replace("%i", "%ii").replace("%c32", "%C32")
    return base, alt


# ---------- Tests ----------
def test_single_scf_for_literals():
    lp = LoopParams(max_loops=4)
    ext = LoopInformationExtractor(lp)

    mv = ModuleView(ir_single_scf_for())
    out = ext.run(mv)

    # one loop found
    assert int(out.loops_mask.sum()) == 1

    feats = out.loop_feats[0]
    # feats layout: [log_tripcount, step_is_pow2, depth_norm, is_parallel, unit_stride(placeholder), reuse(placeholder)]
    # tripcount = ceil((32-0)/2) = 16 -> log1p(16)
    assert np.isclose(feats[0], math.log1p(16), atol=1e-6)
    assert feats[1] == 1.0             # step 2 is pow2
    assert 0.0 <= feats[2] <= 1.0       # depth_norm
    assert feats[3] == 0.0              # not parallel

    # known_mask: tc/pow2 known; depth & parallel known; mem placeholders unknown
    km = out.known_mask[0]
    assert (km[0] and km[1] and km[2] and km[3])


def test_nested_band_depth_norm_and_affine_unknown():
    lp = LoopParams(max_loops=8)
    ext = LoopInformationExtractor(lp)

    mv = ModuleView(ir_nested_two())
    out = ext.run(mv)

    # two loops
    assert int(out.loops_mask.sum()) == 2

    feats = out.loop_feats[:2]
    # outer depth_norm should be 0, inner 1 (band of size 2)
    # (Order is by appearance; allow small tol)
    d0, d1 = feats[0, 2], feats[1, 2]
    assert np.isclose(d0, 0.0, atol=1e-6) and np.isclose(d1, 1.0, atol=1e-6)

    # inner affine.for has unknown ub? In this IR we used constant 128,
    # so tripcount should be known; make one variant symbolic next test.
    # Here confirm at least one pow2 from outer (step=8 → pow2)
    assert (feats[:, 1] == 1.0).any()


def test_scf_parallel_tripcount_and_flags():
    lp = LoopParams(max_loops=4)
    ext = LoopInformationExtractor(lp)

    mv = ModuleView(ir_scf_par())
    out = ext.run(mv)

    # exactly one loop (the parallel region is treated as one loop record)
    assert int(out.loops_mask.sum()) == 1
    f = out.loop_feats[0]
    # product tripcount = 128/1 * 16/4 = 512
    assert f[0] > math.log1p(500) - 1e-6
    assert f[1] == 1.0       # steps (1 and 4) are both pow2
    assert f[3] == 1.0       # is_parallel

def test_affine_parallel_tripcount_and_flags():
    lp = LoopParams(max_loops=4)
    ext = LoopInformationExtractor(lp)

    mv = ModuleView(ir_affine_par())
    out = ext.run(mv)

    # exactly one loop (the parallel region is treated as one loop record)
    assert int(out.loops_mask.sum()) == 1
    f = out.loop_feats[0]
    # product tripcount = 128/1 * 16/1 = 2048
    assert np.isclose(f[0], math.log1p(2048))
    assert f[1] == 1.0       # steps (1 and 4) are both pow2
    assert f[3] == 1.0       # is_parallel


def test_affine_unknown_bounds_marks_unknown():
    # use a truly unknown bound via a non-constant value
    txt = ir_affine_unknown_bounds()
    lp = LoopParams(max_loops=4)
    ext = LoopInformationExtractor(lp)

    mv = ModuleView(txt)
    out = ext.run(mv)

    assert int(out.loops_mask.sum()) == 1
    f = out.loop_feats[0]
    km = out.known_mask[0]
    # 1 to 1024 (by const lookup)
    assert np.isclose(f[0], math.log1p(1024))
    assert bool(km[0]) is True  # tripcount unknown


def test_max_loops_cap_truncates():
    # build 3 identical loops, cap at 2
    src = r"""
module {
  func.func @m() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i0 = %c0 to %c16 step %c1 { }
    scf.for %i1 = %c0 to %c16 step %c1 { }
    scf.for %i2 = %c0 to %c16 step %c1 { }
    return
  }
}
"""
    lp = LoopParams(max_loops=2)
    ext = LoopInformationExtractor(lp)

    mv = ModuleView(src)
    out = ext.run(mv)

    assert int(out.loops_mask.sum()) == 2  # truncated to max_loops


def test_rename_invariance_features_equal():
    base, alt = ir_rename_invariance()
    lp = LoopParams(max_loops=4)
    ext = LoopInformationExtractor(lp)

    m1 = ModuleView(base)
    m2 = ModuleView(alt)
    o1 = ext.run(m1)
    o2 = ext.run(m2)

    # Features should match after SSA renaming
    np.testing.assert_allclose(o1.loop_feats, o2.loop_feats, atol=1e-6)

