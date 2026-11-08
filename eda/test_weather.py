"""Check shared-state forward recursion by independent path enumeration."""
import sys,itertools,math,unittest,copy,json
sys.path.insert(0,'.')
from he.weather import filter_episode
from he.model import load_split
from he.api import load_params,predict_queries
class WeatherTest(unittest.TestCase):
    def test_exhaustive(self):
        trace=(7,{1:[(.2,1),(-1.1,0)],3:[(.7,0)],5:[(-.3,1)]})
        for scale,hazard in ((.4,.02),(.7,.5),(0.,.08)):
            total=hot=0.
            for states in itertools.product((0,1),repeat=7):
                probability=.5
                for tick in range(2,8):probability*=1-hazard if states[tick-1]==states[tick-2] else hazard
                for tick,rows in trace[1].items():
                    for eta,y in rows:
                        p=1/(1+math.exp(-(eta+(2*states[tick-1]-1)*scale)))
                        probability*=p if y else 1-p
                total+=probability;hot+=probability*states[-1]
            ll,p=filter_episode(trace,scale,hazard)
            self.assertAlmostEqual(math.exp(ll),total,places=12)
            self.assertAlmostEqual(p,hot/total,places=12)
    def test_empty(self):
        self.assertEqual(filter_episode((100,{}),.4,.02),(0.,.5))
if __name__=='__main__':unittest.main()
