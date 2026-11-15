#!/bin/bash
# usage: seeds.sh METHOD  -> runs seeds 1..3 on test into scratchpad/seeds/METHOD_s{seed}
SP=${HW2_DEV_CACHE:-$(dirname $(readlink -f $0))/cache}
cd "$(dirname "$(readlink -f "$0")")/../../starter"
for s in 1 2 3; do for i in 1 2 3 4 5 6; do
  python run.py --method $1 --split test --data ../data --ckpt ../source_resnet50_160.pth --out $SP/seeds/$1_s$s --seed $s > /dev/null 2>&1 && break; sleep 3; done
  echo "$1 seed $s done"; done
