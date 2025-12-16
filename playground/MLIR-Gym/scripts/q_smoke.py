# q_obs_smoke.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import random, math
import numpy as np

from .configs import QConfig
from .utils.general_utils import bin012, bin3f, array, bin3i, get_col

class QTableFromObs:
    """
    Q-table smoke test using observation dict (loop + memory + masks).
    State :
      (n_loops_bin, n_pow2_bin, n_parallel_bin,
       max_depth_bin, unit_cnt_bin, reuse_cnt_bin,
       contig_bin, stride_dom_bin,
       legal_TILE_bin, legal_FUSE_bin, legal_BUFF_bin,
       prev_cat)
    Q[s][a] = value, only set key for action applied.
    """

    _CATS = ("FUSE","BUFFERIZE","TILE","CONVERT","EVAL","OTHER")

    def __init__(self, env, cfg: QConfig = QConfig()):
        self.env = env
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.Q: Dict[Tuple[int,...], Dict[int, float]] = {}
        self.groups = self._build_action_groups()
        self.prev_cat = "OTHER"
        self.STATE_ENCODER = {
            "structured_vector": self.encode_sv,
            "gnn_dataflow": self.encode_graph
        }

    # actino_groups {"fuse", "bufferize", "tile", "convert", "eval", "other"}
    def _build_action_groups(self) -> Dict[str, List[int]]:
        groups = {k: [] for k in self._CATS}
        nA = self.env.action_space.n
        for i in range(nA):
            groups[self._classify_action(i)].append(i)
        if self.cfg.verbose:
            print("[Q] action groups:", {k: len(v) for k,v in groups.items()})
        return groups

    def _classify_action(self, idx: int) -> str:
        act = self.env.action_layer.idx_to_action(idx)
        print(f"Apply action: {act}")

        name = (getattr(act,"name",None) or str(act)).lower()
        if "eval" in name: return "EVAL"
        if "tile" in name: return "TILE"
        if "fuse" in name or "fold" in name: return "FUSE"
        if "buffer" in name: return "BUFFERIZE"
        if "convert" in name or "lower" in name: return "CONVERT"
        return "OTHER"


    # state tuple generator
    def encode_sv(self, obs: dict, legal_mask: np.ndarray) -> Tuple[int,...]:
        """
        An encoder for observation when state is represent as structured_vector
        """
        # 1) loops
        loop_mask = array(obs.get("loop_mask", []), dtype=bool)
        loop_feat = array(obs.get("loop_feats", []), dtype=np.float32)
        lknown = array(obs.get("known_mask", []), dtype=bool)
        if loop_feat.size and lknown.size:
            loop_feat = loop_feat * lknown  # apply mask to loop vals

        n_loops = int(loop_mask.sum())
        n_pow2 = int((loop_feat[:,1][loop_mask] > 0.5).sum()) if loop_feat.size and n_loops>0 else 0
        n_par  = int((loop_feat[:,3][loop_mask] > 0.5).sum()) if loop_feat.size and n_loops>0 else 0
        max_depth = float(loop_feat[:,2][loop_mask].max()) if loop_feat.size and n_loops>0 else 0.0

        # 2) access pattern
        unit = array(obs.get("mem_unit_stride", []), dtype=np.float32)
        ukn  = array(obs.get("mem_unit_known",  []), dtype=bool)
        reuse= array(obs.get("mem_reuse_any",   []), dtype=np.float32)
        unit_cnt = int(((unit==1.0) & ukn).sum()) if unit.size else 0
        reuse_cnt= int((reuse>0.5).sum()) if reuse.size else 0

        contig = array(obs.get("mem_contig_score", []), dtype=np.float32)
        contig_known = array(obs.get("mem_contig_known", []), dtype=bool)
        contig_mean = float((contig[contig_known].mean() if (contig.size and contig_known.any()) else 0.0))

        sh = array(obs.get("mem_strided_hist", []), dtype=np.float32)
        shk= array(obs.get("mem_stride_known",[]), dtype=bool)
        stride_dom = 3  # unknown bucket
        if sh.size and shk.any():
            agg = sh[shk].sum(axis=0)  # [4]
            if np.isfinite(agg).all() and agg.sum()>0:
                stride_dom = int(agg.argmax())  # 0:=1,1:2..W,2:>W,3:unk

        # 3) legal mask
        m = legal_mask.astype(bool)
        n_fuse = int(m[self.groups["FUSE"]].sum()) if self.groups["FUSE"] else 0
        n_buff = int(m[self.groups["BUFFERIZE"]].sum()) if self.groups["BUFFERIZE"] else 0
        n_tile = int(m[self.groups["TILE"]].sum()) if self.groups["TILE"] else 0
        n_convert = int(m[self.groups["CONVERT"]].sum()) if self.groups["CONVERT"] else 0
        prev_idx = {"FUSE":0,"BUFFERIZE":1,"TILE":2,"CONVERT":3,"EVAL":4,"OTHER":5}[self.prev_cat]

        # 4) pack
        s = (
            bin012(n_loops),
            bin012(n_pow2),
            bin012(n_par),
            bin3f(max_depth),
            bin012(unit_cnt),
            bin012(reuse_cnt),
            bin3f(contig_mean),
            int(stride_dom), # 0..3
            bin012(n_tile),
            bin012(n_fuse),
            bin012(n_buff),
            prev_idx,
        )
        return s

    def encode_graph(self, obs: dict, legal_mask: np.ndarray) -> Tuple[int,...]:
        """
        An encoder for observation when state is represent as gnn_dataflow graph.
        """

        # 1) get node_feats, node_mask and adj matrix
        node_feats = array(obs.get("node_features", []), np.float32)
        node_mask = array(obs.get("node_mask", []), bool)
        N = int(node_mask.sum())

        edges = array(obs.get("adj", []), np.int8)
        indices = np.where(node_mask)[0]
        live_edges = edges[np.ix_(indices, indices)].astype(np.int32, copy=False)
        indeg = live_edges.sum(axis=0)
        outdeg = live_edges.sum(axis=1)
        E = int(live_edges.sum())
        density = float(E) / float(max(N * (N - 1), 1))

        assert indeg.shape[0] != 0 and outdeg.shape[0] != 0, "Should never have empty indeg list and out deg list"

        # bucket data type and edge density
        size_bin = bin3i(N)
        density_bin = bin3f(density)

        # kahn layer, approx longeest path
        def _layers_from_live_edges(adj: np.ndarray) -> int:
            if adj.size == 0: return 0
            length = adj.shape[0]
            cur_in = adj.sum(axis=0).astype(np.int32, copy=False)
            depth = np.zeros((length,), dtype=np.int32)
            q = [int(i) for i in np.where(cur_in == 0)[0]]
            seen = 0
            while q:
                u = q.pop(0)
                seen += 1
                neighbors = np.nonzero(adj[u])[0]
                depth_u = int(depth[u])
                for v in neighbors:
                    if depth[v] < depth_u + 1:
                        depth[v] = depth_u + 1
                    cur_in[v] -= 1
                    if cur_in[v] == 0:
                        q.append(int(v))
            layers = int(depth.max()) + 1 if length > 0 else 0
            return layers

        layers = _layers_from_live_edges(live_edges)
        layers_bin = layers # bin3f(min(layers / 8.0, 1.0))

        src_ratio       = float((indeg == 0).mean())
        sink_ratio      = float((outdeg == 0).mean())
        fanin_ge2_ratio = float((indeg >= 2).mean())
        fanout_ge2_ratio= float((outdeg >= 2).mean())

        src_bin    = bin3f(src_ratio)
        sink_bin   = bin3f(sink_ratio)
        fanin_bin  = bin3f(fanin_ge2_ratio)
        fanout_bin = bin3f(fanout_ge2_ratio)

        # node feature
        ranks = get_col(node_feats, node_mask, 0)  # reduction generic
        rank_bin = ranks.mean() # bin3f(float((ranks > 0).mean()))

        in_loop = get_col(node_feats, node_mask, 1) # in loop
        all_loops = in_loop.sum(axis=0)
        loop_bin  = bin3f(min(all_loops / N, 1.0))

        par = get_col(node_feats, node_mask, 5)  # parallel generic
        red = get_col(node_feats, node_mask, 6)  # reduction generic
        par_bin = bin3f(float((par > 0).mean()))
        red_bin = bin3f(float((red > 0).mean()))

        ops = get_col(node_feats, node_mask, 2)  # operands
        res = get_col(node_feats, node_mask, 3)  # results
        ops_ge2_bin = bin3f(float((ops >= 2.0).mean()) if ops.size else 0.0)
        res_ge2_bin = bin3f(float((res >= 2.0).mean()) if res.size else 0.0)

        # legal action mask
        lm = array(legal_mask, dtype=bool)
        n_tile = int(lm[self.groups["TILE"]].sum())      if self.groups["TILE"]      else 0
        n_fuse = int(lm[self.groups["FUSE"]].sum())      if self.groups["FUSE"]      else 0
        n_convert = int(lm[self.groups["CONVERT"]].sum()) if self.groups["CONVERT"] else 0
        tile_bin = bin012(n_tile)
        fuse_bin = bin012(n_fuse)
        conv_bin = bin012(n_convert)

        prev_idx = {"FUSE":0,"BUFFERIZE":1,"TILE":2,"CONVERT":3,"EVAL":4,"OTHER":5}[self.prev_cat]

        s = (
            # structures, relationships, and layers
            size_bin, layers_bin, density_bin,
            src_bin, sink_bin, fanin_bin, fanout_bin,
            # loops, properties
            rank_bin, loop_bin, par_bin, red_bin,
            # arity summaries
            ops_ge2_bin, res_ge2_bin,
            # actions
            tile_bin, fuse_bin, conv_bin,
            prev_idx,
        )
        return s

    def encode(self, obs: dict, legal_mask: np.ndarray) -> Tuple[int,...]:
        '''
        A router for different state representation encoder
        '''
        return self.STATE_ENCODER.get(self.env.state_cfg.name)(obs, legal_mask)


    def _epsilon(self, step: int) -> float:
        frac = min(1.0, step / max(1,self.cfg.epsilon_decay))
        return self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end)*math.exp(-3.0*frac)

    def _pick_action(self, s: Tuple[int,...], legal_ids: np.ndarray, eps: float) -> int:
        if len(legal_ids)==0: raise RuntimeError("legal set empty; ensure at least one always-legal action (e.g., EVAL)")
        if random.random() < eps: return int(np.random.choice(legal_ids))
        # exploit
        best_a, best_q = int(legal_ids[0]), -1e18
        row = self.Q.get(s, {})
        for a in legal_ids:
            q = row.get(int(a), 0.0)
            if q > best_q: best_q, best_a = q, int(a)
        return best_a

    # main train api
    def train(self) -> List[float]:
        rewards: List[float] = []
        eps_counter = 0
        self.prev_cat = "OTHER"

        set_default_state = lambda s: self.Q.setdefault(s, {})

        for ep in range(self.cfg.episodes):
            obs, info = self.env.reset()
            mask = array(info.get("mask", np.ones(self.env.action_space.n, dtype=np.int8)))
            assert mask.sum() > 0, "action mask all zeros."
            s = self.encode(obs, mask)
            set_default_state(s)
            term = trunc = False; ep_ret = 0.0

            while not (term or trunc):
                eps = self._epsilon(eps_counter); eps_counter += 1
                legal_ids = np.where(mask.astype(bool))[0]
                action = self._pick_action(s, legal_ids, eps)

                # record class of previous action
                self.prev_cat = self._classify_action(action)

                next_obs, r, term, trunc, next_info = self.env.step(int(action))
                print(f"Get return: {r}")

                next_mask = array(next_info.get("mask", np.ones(self.env.action_space.n, dtype=np.int8)))

                s2 = self.encode(next_obs, next_mask)
                set_default_state(s2)
                next_legal = np.where(next_mask.astype(bool))[0]
                best_next = max((self.Q[s2].get(int(b), 0.0) for b in next_legal), default=0.0)

                old_q = self.Q[s].get(action, 0.0)
                self.Q[s][action] = old_q + self.cfg.alpha * (float(r) + self.cfg.gamma * best_next - old_q)

                s, mask, ep_ret = s2, next_mask, ep_ret + float(r)

            rewards.append(ep_ret)
            if self.cfg.verbose and ((ep+1) % 10 == 0):
                print("-----------------------")
                print(f"[Q-OBS] ep={ep+1:03d}  return={ep_ret:+.3f}")
                print("-----------------------")
            # episode end, reset prev_cat
            self.prev_cat = "OTHER"
        return rewards

def train_qtable_from_obs(env, cfg: Optional[QConfig]=None, out_path: str = "./trial_x") -> List[float]:
    curve = QTableFromObs(env, cfg or QConfig()).train()
    with open(out_path, "wt") as fh:
        fh.write(curve)

