"""Joint sparse/weather filter: independent path enumeration and zero-scale limit."""
import sys,itertools,math
import numpy as np
sys.path.insert(0,'.')
from he.sparse_joint import topology
from he.api import load_params
from he.model import load_split
from he.peer import peer_winners
from sparse_weather_numpy import forward,machine_predict
from sparse_glm_numpy import predict as baseline

# Independently solve the stationary linear system then enumerate observed paths.
pair=(1,2);lag=1;cells,reg,dest=topology(*pair,lag);n=len(cells);h=.1;scale=.4
V=np.array([[.4,.3,.2,.1],[.2,.4,.3,.1]]);levels=[-1.,1.];size_beta=-.5
win=lambda weather,j:1/(1+math.exp(-(levels[cells[j]]+size_beta*(reg[j]-.5)+(2*weather-1)*scale)))
matrix=np.zeros((2*n,2*n))
for weather in (0,1):
    for j in range(n):
        w=win(weather,j);prob=[1-w,w*V[reg[j],0],w*(1-V[reg[j],0])]
        for category in range(3):
            for next_weather in (0,1):
                matrix[next_weather*n+dest[category][j],weather*n+j]+=prob[category]*(1-h if weather==next_weather else h)
A=matrix-np.eye(2*n);A[-1]=1.;b=np.zeros(2*n);b[-1]=1.;stationary=np.linalg.solve(A,b)
seq={0:1,2:20};emissions={1:(.8,.3),2:(.6,.9)};total=0.;mass=np.zeros(5)
for weather in (0,1):
    for j in range(n):
        paths=[(weather,j,float(stationary[weather*n+j]))]
        for tick in range(3):
            nxt=[]
            for state,node,probability in paths:
                w=win(state,node);v=V[reg[node]]
                options=[(1,w*v[0])] if tick==0 else [(2,w*v[3])] if tick==2 else [(0,1-w),(1,w*v[0]),(2,w*(1-v[0]))]
                for category,p in options:
                    for next_weather in (0,1):
                        nxt.append((next_weather,dest[category][node],probability*p*emissions.get(tick,(1.,1.))[state]*(1-h if next_weather==state else h)))
            paths=nxt
        for state,node,probability in paths:
            total+=probability;w=win(state,node);v=V[reg[node]];mass+=probability*np.array([1-w]+[w*x for x in v])
ll,pred=forward(seq,3,pair,lag,0,0,V,0.,levels,size_beta,0.,{t:3.5 for t in range(4)},scale,h,emissions)
assert abs(math.exp(ll)-total)<1e-10
assert max(abs(a-b) for a,b in zip(pred,mass/total))<1e-9
print('joint weather/category paths agree with exhaustive enumeration')
p=load_params('runs/main_v13/params.json');qs,_=load_split('kit/public','train');q=next(q for q in qs if len(q['seq'])<.6*q['cutoff']);mates=[r for r in qs if r['episode_id']==q['episode_id']]
peers={t:peer_winners(mates,q,t-3,True) for t in range(q['cutoff']+1)};cfg=dict(p['global_state'],scale=0.)
actual=machine_predict(q,peers,p,cfg,.13,{});expected=baseline(q,peers,p,cfg,.5,.13)
assert max(abs(a-b) for a,b in zip(actual,expected))<1e-10
print('zero weather amplitude recovers existing sparse predictor')
