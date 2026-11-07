import csv, collections, statistics as st, sys
P='F:/CYH/projects/datathon/challenges/house-edge/kit/public/'
def rd(f): return list(csv.DictReader(open(P+f,encoding='utf-8')))
for split in ['train','validation','test']:
    q=rd(split+'/queries.csv'); h=rd(split+'/history.csv')
    cnt=collections.Counter((r['episode_id'],r['machine_id']) for r in h)
    print(split,'queries',len(q),'episodes',len({r['episode_id'] for r in q}),'hist rows',len(h))
    print(' types',collections.Counter(r['machine_type'] for r in q),'signal',collections.Counter(r['signal'] for r in q))
    print(' rows/machine', collections.Counter(sorted(cnt.values())).most_common(8))
    print(' cutoffs', min(int(r['cutoff']) for r in q), max(int(r['cutoff']) for r in q))
    print(' payout freq', collections.Counter(r['payout'] for r in h))
    if split!='test':
        c={r['query_id']:r['condition'] for r in rd(split+'/conditions.csv')}
        qq={r['query_id']:r for r in q}
        # rows per machine by condition
        bc=collections.defaultdict(list)
        for r in q: bc[c[r['query_id']]].append(cnt[(r['episode_id'],r['machine_id'])])
        print(' rows by cond',{k:(min(v),max(v)) for k,v in bc.items()})
        # is condition constant per episode?
        ec=collections.defaultdict(set)
        for r in q: ec[r['episode_id']].add(c[r['query_id']])
        print(' conds per episode',collections.Counter(len(v) for v in ec.values()))
    if split=='validation':
        t=rd(split+'/targets.csv')
        g=collections.defaultdict(list)
        for r in t:
            qr=qq[r['query_id']]; g[(qr['machine_type'],qr['signal'])].append([float(r[k]) for k in ['p_0','p_1','p_2','p_5','p_20']])
        for k in sorted(g):
            v=g[k]; print(' target',k,len(v),'mean',[round(st.mean(x[i] for x in v),3) for i in range(5)],'sd',[round(st.pstdev(x[i] for x in v),3) for i in range(5)])
