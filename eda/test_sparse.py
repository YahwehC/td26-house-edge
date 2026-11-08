"""Independent exhaustive check of sparse forward marginalization."""
import sys,math,itertools,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from he.sparse import forward
from he.corr import _solve
class ForwardTest(unittest.TestCase):
    def test_iid(self):
        seq={0:0,2:5,5:1}; p=.3
        ll,pr=forward(seq,8,(1,3),[p,p])
        self.assertAlmostEqual(ll,2*math.log(p)+math.log1p(-p),places=10)
        self.assertAlmostEqual(pr,p,places=12)
    def test_exhaustive(self):
        rates=[.17,.71];pair=(1,2);n=4
        prob=[rates[int((s&1)==((s>>1)&1))] for s in range(n)]
        mat=[[0.0]*n for _ in range(n)]
        for s,p in enumerate(prob):
            mat[(s<<1)&3][s]+=1-p;mat[((s<<1)&3)|1][s]+=p
        eq=[[mat[i][j]-(i==j) for j in range(n)] for i in range(n-1)]+[[1.0]*n]
        stationary=_solve(eq,[0,0,0,1])
        for seq in ({0:1,3:0},{1:0,2:5,4:1},{}):
            likelihood=numerator=0.0
            for initial in range(n):
                for outcomes in itertools.product((0,1),repeat=5):
                    state=initial;weight=stationary[initial]
                    for t,y in enumerate(outcomes):
                        if t in seq and y!=int(seq[t]>0):weight=0;break
                        p=prob[state];weight*=p if y else 1-p;state=((state<<1)&3)|y
                    likelihood+=weight;numerator+=weight*prob[state]
            ll,pr=forward(seq,5,pair,rates)
            self.assertAlmostEqual(math.exp(ll),likelihood,places=9)
            self.assertAlmostEqual(pr,numerator/likelihood,places=9)
if __name__=='__main__':unittest.main()
