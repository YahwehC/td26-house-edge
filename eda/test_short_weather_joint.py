"""Independent enumeration check of the joint forecaster's HMM and quadrature."""
import sys,itertools,math,unittest
sys.path.insert(0,'.')
from he.short_weather_joint import forward,NODES,WEIGHTS
class JointWeatherTest(unittest.TestCase):
    def test_paths(self):
        events=[(1,.2,.4),(3,.8,.7),(5,.13,.23)];cut=7
        for hazard in (.02,.5,0.):
            mass=hot=0.
            for states in itertools.product((0,1),repeat=cut):
                p=.5
                for a,b in zip(states,states[1:]):p*=hazard if a!=b else 1-hazard
                for t,c,w in events:p*=w if states[t-1] else c
                mass+=p;hot+=p*states[-1]
            ll,p=forward(events,cut,hazard)
            self.assertAlmostEqual(math.exp(ll),mass,places=13)
            self.assertAlmostEqual(p,hot/mass,places=13)
    def test_normal_quadrature(self):
        for power,expected in [(0,1),(1,0),(2,1),(3,0),(4,3),(6,15),(8,105)]:
            self.assertAlmostEqual(sum(w*x**power for x,w in zip(NODES,WEIGHTS)),expected,places=10)
    def test_empty(self):self.assertEqual(forward([],10,.02),(0.,.5))
if __name__=='__main__':unittest.main()
