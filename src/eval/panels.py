import sim.chain as chain
import sim.collider as collider
import sim.confounder as confounder

SMALL = {"n_cells": 5000, "n_conf": 10, "n_direct": 10, "n_null": 10}
SMALL_COLL = {"n_cells": 5000, "n_coll": 10, "n_direct": 10, "n_null": 10}
SMALL_CHAIN = {"n_cells": 5000, "n_chain": 10, "n_direct": 10, "n_null": 10}


def get(name, seed, size="small", counts_kw=None, link="linear"):
    """Повертає (expr, idx, pos, neg, підпис)."""
    if name == "confounder":
        cfg = SMALL if size == "small" else confounder.DEFAULT
        e, i = confounder.matched(seed, cfg, counts_kw=counts_kw, link=link)[:2]
        return e, i, "D", "C", cfg
    if name == "confounder_hidden":
        cfg = SMALL if size == "small" else confounder.DEFAULT
        e, i = confounder.matched(seed, cfg, hide_root=True, counts_kw=counts_kw, link=link)[:2]
        return e, i, "D", "C", cfg
    if name == "collider":
        cfg = SMALL_COLL if size == "small" else collider.CFG
        e, i = collider.build(seed, cfg, counts_kw=counts_kw, link=link)
        return e, i, "K", "N", cfg
    if name == "chain":
        cfg = SMALL_CHAIN if size == "small" else chain.CFG
        e, i = chain.matched(seed, False, cfg, counts_kw=counts_kw, link=link)[:2]
        return e, i, "D", "T", cfg
    if name == "chain_hidden":
        cfg = SMALL_CHAIN if size == "small" else chain.CFG
        e, i = chain.matched(seed, True, cfg, counts_kw=counts_kw, link=link)[:2]
        return e, i, "D", "T", cfg
    raise ValueError(name)


NAMES = ["confounder", "confounder_hidden", "collider", "chain", "chain_hidden"]
