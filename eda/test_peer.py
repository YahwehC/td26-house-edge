"""Missing peers must be distinguished from observed zero payouts."""
import sys,unittest
sys.path.insert(0,'.')
from he.peer import peer_winners
class PeerTest(unittest.TestCase):
    def test_missing_vs_zero(self):
        own={'machine_id':'a','seq':{2:20}}
        loser={'machine_id':'b','seq':{2:0,1:1}}
        absent={'machine_id':'c','seq':{1:1,3:2}}
        machines=[own,loser,absent]
        self.assertEqual(peer_winners(machines,own,2),0)
        self.assertEqual(peer_winners(machines,own,2,True),.75)
        self.assertEqual(peer_winners(machines,dict(own),2,True),.75)
        self.assertEqual(peer_winners(machines,own,-1,True),0)
        self.assertEqual(peer_winners(machines,own,1,True),2)
if __name__=='__main__':unittest.main()
