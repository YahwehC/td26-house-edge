"""Check regularized offset fit and heterogeneous weather emissions."""
import sys,math,unittest,itertools
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
    def test_smoothed_peer_only_score(self):
        config={'types':['x'],'coef':[0.]*11,'scale':.4,'hazard':.02,
                'machine_variance':.04,'offset_smoothing':True}
        outcomes={'m':{0:1,2:1,5:0},'peer':{0:1,1:1,2:0,3:1,4:1,5:1}}
        rows=[{'episode':'e','machine':mid,'type':'x','high':0,'signal':0,
               'regime':0.,'peer':3.5,'tick':t,'y':y}
              for mid,seq in outcomes.items() for t,y in seq.items()]
        mass=0.;hot=[0.]*6
        for states in itertools.product((0,1),repeat=6):
            probability=.5
            for a,b in zip(states,states[1:]):probability*=.02 if a!=b else .98
            for t,y in outcomes['peer'].items():
                p=1/(1+math.exp(-(2*states[t]-1)*.4))
                probability*=p if y else 1-p
            mass+=probability
            for t,state in enumerate(states):hot[t]+=probability*state
        value=estimate_offsets(rows,config)['m'];score=-value/.04
        p0=1/(1+math.exp(-(value-.4)));p1=1/(1+math.exp(-(value+.4)))
        for t,y in outcomes['m'].items():
            h=hot[t]/mass;p=(1-h)*p0+h*p1;derivative=(1-h)*p0*(1-p0)+h*p1*(1-p1)
            score+=(y-p)*derivative/(p*(1-p))
        self.assertAlmostEqual(score,0.,places=9)
if __name__=='__main__':unittest.main()
