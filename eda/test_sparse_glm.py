"""An independent IID categorical special case including signal/peer/offset logits."""
import sys
import math
sys.path.insert(0, '.')
from he.sparse_glm import predict
q = {'machine_type':'x','signal':1,'cutoff':12,'seq':{0:0,2:1,6:20,8:2,9:0,11:5}}
vector = [.4,.3,.2,.1]
params = {'cfg':{'pairs':{'1,3':1},'size_lag_given_pair':{'1,3':{'1':1}}},
          'size_vectors':{'x':[vector,vector]}}
coef = [-.4,-.4,0.,0.,0.,0.,.3,0.,0.,0.,.07]
config = {'types':['x'],'coef':coef,'scale':.4}
peers = {t:t%4 for t in range(13)}
peers[12] = 2
hot = .75
for offset in (0.,.2,-.3):
    actual = predict(q,peers,params,config,hot,offset)
    eta = -.4+.3+.07*(2-3.5)+offset
    win = (1-hot)/(1+math.exp(-(eta-.4)))+hot/(1+math.exp(-(eta+.4)))
    expected = [1-win]+[win*v for v in vector]
    assert max(abs(a-b) for a,b in zip(actual,expected)) < 1e-11
print('IID categorical forecast matches independent analytic formula with nonzero offsets')
