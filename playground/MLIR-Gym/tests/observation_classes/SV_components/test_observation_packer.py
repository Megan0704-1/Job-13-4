import numpy as np
from gymnasium import spaces

from mlir_env.observations.classes.SV.immutable import LoopParams, MemParams, PackerParams
from mlir_env.observations.classes.SV.packer import Packer
from mlir_env.observations.classes.SV.analysis.utils.loop_utils import LoopInfoOut
from mlir_env.observations.classes.SV.analysis.utils.memory_utils import MemoryFeatures, FeatureMask, MemInfoOut

def _make_packer(include_mem=True):
    lp = LoopParams(max_loops=4)
    mp = MemParams() if include_mem else None
    pp = None
    hp = None
    hw = None #HardwareParams()
    pk = PackerParams(include_channels=["loop","mems"] if include_mem else ["loop"])
    return Packer(loop_params=lp, mem_params=mp, probes_params=pp,
                  history_params=hp, hw_params=hw, packer_params=pk)

def _make_dummy_params(N: int, L: int):
    attributes = tuple(np.ndarray(L) for _ in range(N))
    return attributes

def test_build_space_dict_keys_not_setattr():
    pk = _make_packer(include_mem=True)
    sp = pk.build_space()
    assert isinstance(sp, spaces.Dict)
    for k in ["loop_feats","loop_mask","known_mask","mem_unit_stride","mem_stride_known"]:
        assert k in sp.spaces, f"missing key in Dict space: {k}"

def test_pack_uses_mems_keyword_and_no_NameError():
    pk = _make_packer(include_mem=True)
    L = 4
    loopN = 4
    # feats, loop mask, feat mask, records
    loops = LoopInfoOut(*_make_dummy_params(loopN, L))
    # feats, mask
    memN = 5; maskN = 3
    feats = MemoryFeatures(*_make_dummy_params(memN, L))
    masks = FeatureMask(*_make_dummy_params(maskN, L))
    mems = MemInfoOut(feats, masks)

    obs = pk.pack(loops=loops, mems=mems)
    for k in ["mem_unit_stride","mem_reuse_any","mem_contig_score",
              "mem_reuse_cnt","mem_strided_hist",
              "mem_unit_known","mem_contig_known","mem_stride_known"]:
        assert k in obs, f"missing packed tensor: {k}"
    assert obs["mem_unit_stride"].shape == (pk.lp.max_loops,)
