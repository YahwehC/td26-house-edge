"""NumPy pilot vs stdlib production equivalence on representative real queries."""
import sys,math
sys.path.insert(0,'.')
from he.api import load_params
from he.model import load_split
from he.sparse_joint import predict,forward
from hx.joint_sparse import predict as reference
p=load_params('runs/main_v6/params.json');qs,ex=load_split('kit/public','train')
short=[q for q in qs if len(q['seq'])<.6*q['cutoff']]
for q in short[:8]:
    a=predict(q,p['size_vectors'],p['cfg']);b=reference(q,p['size_vectors'],p['cfg'])
    d=max(abs(x-y) for x,y in zip(a,b));assert d<1e-10,d
    assert abs(sum(a)-1)<1e-10 and min(a)>=0
    print(q['query_id'],d,flush=True)
# Constant win rate + equal size vectors gives independent categorical observations.
v=[.4,.3,.2,.1];wr=.3;seq={0:0,2:1,3:20,6:2}
ll,pr=forward(seq,8,(1,3),1,[wr,wr],[v,v],0)
expected=math.log(1-wr)+sum(math.log(wr*v[k]) for k in (0,3,1))
assert abs(ll-expected)<1e-9
assert max(abs(a-b) for a,b in zip(pr,[1-wr]+[wr*x for x in v]))<1e-10
print('stdlib equivalence and independent categorical check passed',flush=True)
# Independent path enumeration with a nontrivial coupled size regime.
import itertools
from he.corr import _solve
from he.sparse_joint import topology
rates=[.2,.75];vectors=[[.2,.3,.4,.1],[.6,.1,.1,.2]]
cells,reg,dest=topology(1,2,1);n=len(cells)
for sign in (0,1):
    weights=[]
    for z,r in zip(cells,reg):
        w=rates[z];size=vectors[r^sign];weights.append([1-w,w*size[0],w*(1-size[0])])
    matrix=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for category in range(3):matrix[dest[category][i]][i]+=weights[i][category]
    equations=[[matrix[i][j]-(i==j) for j in range(n)] for i in range(n-1)]+[[1.0]*n]
    stationary=_solve(equations,[0]*(n-1)+[1])
    seq={0:1,2:20};total=0;num=[0.0]*5
    for initial in range(n):
        for path in itertools.product(range(3),repeat=4):
            state=initial;mass=stationary[initial]
            for t,category in enumerate(path):
                if t in seq:
                    val=seq[t];expected=0 if val==0 else (1 if val==1 else 2)
                    if category!=expected:mass=0;break
                    if val==0:mass*=1-rates[cells[state]]
                    else:
                        from he.model import SIZE_IDX
                        mass*=rates[cells[state]]*vectors[reg[state]^sign][SIZE_IDX[val]]
                else:mass*=weights[state][category]
                state=dest[category][state]
            total+=mass;w=rates[cells[state]];v=vectors[reg[state]^sign]
            for k,probability in enumerate([1-w]+[w*x for x in v]):num[k]+=mass*probability
    ll,pr=forward(seq,4,(1,2),1,rates,vectors,sign)
    assert abs(math.exp(ll)-total)<1e-10
    assert max(abs(a-b/total) for a,b in zip(pr,num))<1e-9
print('coupled hidden-path enumeration passed',flush=True)
