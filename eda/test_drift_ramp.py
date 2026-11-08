"""The experimental gradual path must affect drift but not shared change."""
import sys,types,unittest
sys.path.insert(0,'.')
from he.joint import episode_posterior_joint
from he.model import DEFAULT_CFG
class RampTest(unittest.TestCase):
    def test_separate_likelihoods(self):
        key=(1,3);m=types.SimpleNamespace(q={'cutoff':2,'seq':{0:0,1:1}},taus=[1],pairs=[key],pair_logprior={key:0.},lml_stable={key:0.},pred_stable={key:.2},lml_tau={key:[0.]},pred_tau={key:[.8]},drift_lml_tau={key:[0.]},drift_pred_tau={key:[.4]})
        cfg=dict(DEFAULT_CFG,model_prior={'drift':1.},drift_rho=.9)
        p,_=episode_posterior_joint([m],cfg);self.assertAlmostEqual(p[0],.38)
        cfg['model_prior']={'shared':1.};cfg['shared_rho']=.9
        p,_=episode_posterior_joint([m],cfg);self.assertAlmostEqual(p[0],.74)
if __name__=='__main__':unittest.main()
