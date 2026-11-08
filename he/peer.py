"""Observed peer winners plus empirical expectations for unobserved ticks."""


def peer_winners(machines, own, tick, marginalize=False):
    total = 0.0
    for machine in machines:
        if machine is own or machine['machine_id'] == own['machine_id']:
            continue
        seq = machine['seq']
        value = seq.get(tick)
        if value is not None:
            total += value > 0
        elif marginalize and tick >= 0:
            total += (sum(value > 0 for value in seq.values()) + 1) / (len(seq) + 2)
    return total
