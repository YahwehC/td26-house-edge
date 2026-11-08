"""Check auxiliary forward recursion against exhaustive hidden-bit summation."""
import itertools
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from he.joint import SizeEvidence
from he.model import SIZE_IDX

class ChainTest(unittest.TestCase):
    def test_enumerated_missing_bits(self):
        model = SizeEvidence.__new__(SizeEvidence)
        model.Vn = [[.1, .2, .3, .4], [.7, .1, .1, .1]]
        for lag, sign in itertools.product((1, 2, 3), (0, 1)):
            seq, cutoff = {0: 1, 2: 0, 4: 5}, 6
            wr = (sum(v > 0 for v in seq.values()) + 1) / (len(seq) + 2)
            a, b = [wr * (1 - model.Vn[r ^ sign][0]) for r in (0, 1)]
            stationary = a / (1 - b + a)
            total = numerator = 0.0
            for bits in itertools.product((0, 1), repeat=cutoff + lag):
                states = dict(zip(range(-lag, cutoff), bits))
                weight = math.prod(stationary if states[t] else 1-stationary for t in range(-lag, 0))
                for t in range(cutoff):
                    prev, now = states[t-lag], states[t]
                    if t in seq:
                        value = seq[t]
                        if now != int(value >= 2):
                            weight = 0.0
                            break
                        if value:
                            weight *= model.Vn[prev ^ sign][SIZE_IDX[value]]
                    else:
                        prob = b if prev else a
                        weight *= prob if now else 1-prob
                total += weight
                numerator += weight * (states[cutoff-lag] ^ sign)
            ll, r1 = model._chain(seq, cutoff, lag, sign)
            self.assertAlmostEqual(math.exp(ll), total, places=12)
            self.assertAlmostEqual(r1, numerator / total, places=12)

if __name__ == '__main__':
    unittest.main()
