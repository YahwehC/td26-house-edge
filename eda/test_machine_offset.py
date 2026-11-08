"""Check regularized offset fit and heterogeneous weather emissions."""
import sys,math,unittest
sys.path.insert(0,'.')
from he.machine_offset import estimate_offsets
from he.weather import filter_episode
class OffsetTest(unittest.TestCase):
    def test_zero_variance(self):
        self.assertEqual(estimate_offsets([],{'machine_variance':0}),{})
    def test_constant_rate_map(self):
        config={'types':['x'],'coef':[0.]*11,'scale':0.,'hazard':.02,'machine_variance':.04}
        rows=[{'episode':'e','machine':'m','type':'x','high':0,'signal':0,'regime':0.,'peer':3.5,'tick':t,'y':int(t<80)} for t in range(100)]
        value=estimate_offsets(rows,config)['m'];prob=1/(1+math.exp(-value))
        self.assertAlmostEqual(80-100*prob-value/.04,0.,places=9)
        self.assertGreater(value,0.)
        self.assertLess(value,math.log(4))
    def test_emission_formats(self):
        scale=.4;hazard=.03;rows={0:[(.2,1),(-.8,0)],3:[(.4,1)]}
        converted={t:[([1/(1+math.exp(-(eta-scale))),1/(1+math.exp(-(eta+scale)))],y) for eta,y in rs] for t,rs in rows.items()}
        a=filter_episode((7,rows),scale,hazard);b=filter_episode((7,converted),scale,hazard)
        for x,y in zip(a,b):self.assertAlmostEqual(x,y,places=12)
if __name__=='__main__':unittest.main()
