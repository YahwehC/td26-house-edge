"""Binary-state smoothing, used only with other machines' history emissions."""
import math


def smooth_weather(cutoff, times, scale, hazard):
    """P(hot at t | supplied observations strictly before cutoff), t=0..cutoff.

    Normalized forward/backward messages avoid underflow on long histories.
    The caller is responsible for excluding the machine being evaluated.
    """
    emissions = []
    forward = []
    hot = .5
    for tick in range(cutoff + 1):
        if tick:
            hot = hazard + (1 - 2 * hazard) * hot
        ll = [0., 0.]
        for eta, y in times.get(tick, ()) if tick < cutoff else ():
            for state, shift in enumerate((-scale, scale)):
                if isinstance(eta, (list, tuple)):
                    probability = eta[state]
                    ll[state] += math.log(max(1e-300, probability if y else 1-probability))
                else:
                    z = eta + shift
                    ll[state] += y*z - max(z, 0) - math.log1p(math.exp(-abs(z)))
        maximum = max(ll)
        e = [math.exp(value-maximum) for value in ll]
        emissions.append(e)
        cold, warm = (1-hot)*e[0], hot*e[1]
        hot = warm/(cold+warm)
        forward.append(hot)
    result = [0.] * (cutoff+1)
    backward = [1., 1.]
    for tick in range(cutoff, -1, -1):
        cold, warm = (1-forward[tick])*backward[0], forward[tick]*backward[1]
        result[tick] = warm/(cold+warm)
        e = emissions[tick]
        backward = [(1-hazard)*e[0]*backward[0]+hazard*e[1]*backward[1],
                    hazard*e[0]*backward[0]+(1-hazard)*e[1]*backward[1]]
        norm = max(backward)
        backward = [value/norm for value in backward]
    return result
