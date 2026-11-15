"""python exp.py <round> [names...]: runs configs on proxy0.5 ('p') and/or val7 ('v'). Appends to exp_log.txt."""
import sys, json
import fe
from mx import MX
SD = dict(shift_detect=True, shift_thr=10.0)
CUM = dict(alpha_cum=True, alpha_min=0.1)
R = {
 "r3p": [
   ("sd_cum01_gp2", dict(**SD, **CUM, glob_prior=2.0), "p"),
   ("sd_cum01_gp1", dict(**SD, **CUM, glob_prior=1.0), "p"),
   ("sd_cum01_gp3", dict(**SD, **CUM, glob_prior=3.0), "p"),
   ("sd_cum005_gp2", dict(**SD, alpha_cum=True, alpha_min=0.05, glob_prior=2.0), "p"),
 ],
 "r3v": [
   ("sd_cum01", dict(**SD, **CUM), "v"),
   ("sd_cum01_gp2", dict(**SD, **CUM, glob_prior=2.0), "v"),
 ],
}
if __name__ == "__main__":
    rnd = sys.argv[1]; only = sys.argv[2:]
    for name, kw, flags in R[rnd]:
        if only and name not in only: continue
        fac = lambda model, kw=kw: MX(model, **kw)
        res = {}
        if "p" in flags:
            o = fe.proxy(fac, mix=0.5, tag=name); res["proxy"] = round(o["mean"], 2); res["per"] = o["per"].tolist()
        if "v" in flags:
            o = fe.val7(fac); res["val7"] = round(o["mean"], 2); res["vdom"] = o["dom"].tolist()
        with open("exp_log.txt", "a") as f: f.write(json.dumps(dict(round=rnd, name=name, kw=kw, **res)) + "\n")
