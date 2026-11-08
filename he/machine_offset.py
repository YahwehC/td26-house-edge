"""Persistent machine logit offsets with a zero-centered Gaussian prior.
Default weather covariates use other machines and strictly earlier ticks.
Optional offset_smoothing uses all supplied peer history, still excluding self.
"""
import math
from collections import defaultdict


def estimate_offsets(rows, config):
    variance = config.get('machine_variance', 0.)
    if not variance:
        return {}
    cf=config['coef'];types=config['types'];scale=config['scale'];hazard=config['hazard']
    by=defaultdict(lambda:defaultdict(list))
    for r in rows:
        i=types.index(r['type'])
        eta=cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)
        by[r['episode']][r['tick']].append((r['machine'],eta,r['y']))
    sigmoid=lambda x:1/(1+math.exp(-max(-700,min(700,x))))
    records=defaultdict(list)
    for times in by.values():
        machines={mid for observations in times.values() for mid,_,_ in observations}
        for mid in sorted(machines):
            if config.get('offset_smoothing'):
                from .weather_history import smooth_weather
                peers = {tick: [(eta,y) for machine,eta,y in observations if machine != mid]
                         for tick,observations in times.items()}
                history = smooth_weather(max(times)+1, peers, scale, hazard)
                for tick,observations in sorted(times.items()):
                    for machine,eta,y in observations:
                        if machine == mid:
                            records[mid].append((eta,history[tick],y))
                continue
            hot=.5;last=None
            for tick,observations in sorted(times.items()):
                if last is not None:
                    hot=.5+(hot-.5)*(1-2*hazard)**(tick-last)
                ll=[0.,0.]
                for machine,eta,y in observations:
                    if machine==mid:
                        records[mid].append((eta,hot,y))
                        continue
                    for j,shift in enumerate((-scale,scale)):
                        z=eta+shift
                        ll[j]+=y*z-max(z,0)-math.log1p(math.exp(-abs(z)))
                z=math.log(max(hot,1e-15)/max(1-hot,1e-15))+ll[1]-ll[0]
                hot=sigmoid(z);last=tick
    offsets={}
    for mid,observations in records.items():
        offset=0.
        for _ in range(8):
            score=-offset/variance;information=1/variance
            for eta,hot,y in observations:
                p0,p1=sigmoid(eta+offset-scale),sigmoid(eta+offset+scale)
                probability=(1-hot)*p0+hot*p1
                derivative=(1-hot)*p0*(1-p0)+hot*p1*(1-p1)
                denom=max(1e-15,probability*(1-probability))
                score+=(y-probability)*derivative/denom
                information+=derivative*derivative/denom
            offset+=score/information
        offsets[mid]=offset
    return offsets
