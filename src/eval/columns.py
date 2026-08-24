import sim.chain as chain
import sim.collider as collider
import sim.confounder as confounder

LINK = "linear"
MATCH = "corr"


def set_link(link, match="corr"):
    global LINK, MATCH
    LINK, MATCH = link, match


def confounder_visible(seed):
    expr, idx = confounder.matched(seed, link=LINK, match=MATCH)[:2]
    return expr, idx, "D", "C"


def confounder_hidden(seed):
    expr, idx = confounder.matched(seed, hide_root=True, link=LINK, match=MATCH)[:2]
    return expr, idx, "D", "C"


def collider_column(seed):
    expr, idx = collider.build(seed, link=LINK)
    return expr, idx, "K", "N"


def chain_visible(seed):
    expr, idx = chain.matched(seed, False, link=LINK, match=MATCH)[:2]
    return expr, idx, "D", "T"


def chain_hidden(seed):
    expr, idx = chain.matched(seed, True, link=LINK, match=MATCH)[:2]
    return expr, idx, "D", "T"


COLUMNS = [
    ("1 спільна причина",      confounder_visible, "якнайвище"),
    ("2 прихований регулятор", confounder_hidden,  "якнайвище"),
    ("3 колайдер",             collider_column,    "рівно 0.5"),
    ("4a ланцюг, B видно",     chain_visible,      "якнайвище"),
    ("4b ланцюг, B приховано", chain_hidden,       "якнайвище"),
]
