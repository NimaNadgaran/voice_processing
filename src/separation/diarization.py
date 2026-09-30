"""Clip/merge diarization turns and measure overlaps on the original timeline."""
def merge_turns(turns, duration):
    result = []
    for start, end in sorted(turns):
        start, end = max(0., float(start)), min(float(end), duration)
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def overlap_ratio(turns_by_speaker, duration):
    events = []
    for turns in turns_by_speaker:
        for start, end in merge_turns(turns, duration):
            events.extend(((start, 1), (end, -1)))
    previous, active, overlapping = 0., 0, 0.
    for point, delta in sorted(events):
        if active >= 2:
            overlapping += point - previous
        active += delta
        previous = point
    return overlapping / max(duration, 1e-9)
