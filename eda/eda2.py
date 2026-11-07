import csv, collections, statistics as st
P='F:/CYH/projects/datathon/challenges/house-edge/kit/public/'
def rd(f): return list(csv.DictReader(open(P+f,encoding='utf-8')))
q=rd('validation/queries.csv'); h=rd('validation/history.csv'); t={r['query_id']:r for r in rd('validation/targets.csv')}
c={r['query_id']:r['condition'] for r in rd('validation/conditions.csv')}
seq=collections.defaultdict(dict)
for r in h: seq[(r['episode_id'],r['machine_id'])][int(r['tick'])]=int(r['payout'])
# histogram of target p0
p0=[float(t[r['query_id']]['p_0']) for r in q]
print('p0 hist',collections.Counter(min(9,int(x*10)) for x in p0))
# target p0 vs lag-L win, pooled
for L in range(1,9):
    g=collections.defaultdict(list)
    for r in q:
        s=seq[(r['episode_id'],r['machine_id'])]; v=s.get(int(r['cutoff'])-L)
        if v is None: continue
        g[(r['machine_type'], v>0)].append(float(t[r['query_id']]['p_0']))
    print('L',L,{k:(len(v),round(st.mean(v),3)) for k,v in sorted(g.items())})
# within-machine: per machine, compute P(win_t | win_{t-L}) - P(win_t | loss_{t-L}) from history, show distribution of max |diff| lag
lagcount=collections.Counter(); diffs=[]
for r in q:
    s=seq[(r['episode_id'],r['machine_id'])]
    if len(s)<150: continue
    best=None
    for L in range(1,9):
        a=[];b=[]
        for tk,v in s.items():
            u=s.get(tk-L)
            if u is None: continue
            (a if u>0 else b).append(v>0)
        if len(a)<10 or len(b)<10: continue
        d=st.mean(a)-st.mean(b)
        if best is None or abs(d)>abs(best[1]): best=(L,d)
    lagcount[best[0]]+=1; diffs.append(round(best[1],1))
print('best lag',sorted(lagcount.items())); print('best diff', sorted(collections.Counter(diffs).items()))
