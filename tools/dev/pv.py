"""Short proxy + val7 for registered production methods: python pv.py name [name...]"""
import sys, json, fe
for n in sys.argv[1:]:
    o = fe.proxy(fe.M(n), mix=0.5, tag=n); v = fe.val7(fe.M(n))
    with open("exp_log.txt", "a") as f: f.write(json.dumps(dict(round="prod", name=n, proxy=round(o["mean"], 2), per=o["per"].tolist(), val7=round(v["mean"], 2), vdom=v["dom"].tolist())) + "\n")
