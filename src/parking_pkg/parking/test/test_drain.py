import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collections import deque
from parking.slot_core import drain_pending

class Msg:  # fake cloud
    def __init__(s, t): s.t = t

def run(scenario):
    tf_latest = [0.0]                 # time of newest TF we have
    out = []
    def lookup(m):
        if m.t < tf_latest[0] - 10: return ("fail", None)       # too old
        return ("ok", m.t) if m.t <= tf_latest[0] else ("future", None)
    q = deque(); done = drop = 0
    for step in range(100):
        now = step * 0.1
        q.append((now, Msg(now)))                                # a cloud every 0.1 s
        tf_latest[0] = scenario(now)
        d, x = drain_pending(q, lookup, lambda m, tf: out.append(m.t), now)
        done += d; drop += x
    return done, drop, len(q)

# 1) TF slightly behind the cloud stamp (33 ms): nothing should be lost
print("TF 33 ms behind :", run(lambda now: now - 0.033))
# 2) TF lags by 2 s (the failure seen on your machine): clouds wait max 0.5 s, then are dropped, never blocking
print("TF 2 s behind   :", run(lambda now: now - 2.0))
# 3) TF in the future (normal Gazebo): everything processed
print("TF ahead        :", run(lambda now: now + 0.02))
assert run(lambda now: now - 0.033)[1] == 0
assert run(lambda now: now + 0.02)[0] == 100
print("OK")
