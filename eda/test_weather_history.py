"""Independent exhaustive check of every smoothed marginal."""
import itertools
import math
import sys
import unittest
sys.path.insert(0, '.')
from he.weather_history import smooth_weather
from he.weather import filter_episode


class SmoothingTest(unittest.TestCase):
    def test_all_marginals(self):
        cutoff = 6
        times = {0: [(.2, 1)], 2: [(-1.1, 0), (.6, 1)], 5: [(.8, 0)]}
        for scale, hazard in ((.4, .02), (.7, .5), (0., .1)):
            total = 0.
            mass = [0.] * (cutoff+1)
            for states in itertools.product((0, 1), repeat=cutoff+1):
                probability = .5
                for tick in range(1, cutoff+1):
                    probability *= hazard if states[tick] != states[tick-1] else 1-hazard
                for tick, rows in times.items():
                    for eta, y in rows:
                        win = 1/(1+math.exp(-(eta+(2*states[tick]-1)*scale)))
                        probability *= win if y else 1-win
                total += probability
                for tick, state in enumerate(states):
                    mass[tick] += probability*state
            calculated = smooth_weather(cutoff, times, scale, hazard)
            for actual, expected in zip(calculated, mass):
                self.assertAlmostEqual(actual, expected/total, places=12)
            self.assertAlmostEqual(calculated[-1], filter_episode((cutoff, times), scale, hazard)[1], places=12)
            contaminated = dict(times, **{})
            contaminated[cutoff] = [(10., 0)]
            self.assertEqual(calculated, smooth_weather(cutoff, contaminated, scale, hazard))

    def test_empty(self):
        self.assertEqual(smooth_weather(4, {}, .4, .02), [.5]*5)


if __name__ == '__main__':
    unittest.main()
