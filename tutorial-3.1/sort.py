import logging
import operator
import numpy as np

log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

def _sort_walkers_identity(we_driver, ibin, status, **kwargs):
    '''A function that, given  sorts the walkers based on a given criteria. Status indicate which method it's from. The int
    arguments mean the following:

    status = 0    _run_we() - not doing any sorting
    status = 1    _split_by_weight() - check upper ideal weight threshold
    status = 2    _merge_by_weight() - check lower ideal weight threshold
    status = 3    _adjust_count()
    status = 4    _split_by_threshold() - check upper weight threshold
    status = 5    _merge_by_threshold() - check lower weight threshold
    status = 6    _run_we() - merging all segs in one group
    '''
    log.debug('using sort._sort_walkers_identity')
    segments = np.array(sorted(ibin, key=operator.attrgetter('weight')), dtype=np.object_)
    weights = np.array(list(map(operator.attrgetter('weight'), segments)))
    cumul_weight = 0

    if status == 0:  # run_we - not doing any sorting
        ordered_array = []
    elif status == 1:  # _split_by_weight() - check upper ideal weight threshold
        ideal_weight = kwargs['ideal_weight']
        ordered_array = segments[weights > we_driver.weight_split_threshold * ideal_weight]
    elif status == 2:  # _merge_by_weight() - check lower ideal weight threshold
        cumul_weight = np.add.accumulate(weights)
        ideal_weight = kwargs['ideal_weight']
        ordered_array = segments[cumul_weight <= ideal_weight * we_driver.weight_merge_cutoff]
    elif status == 3:  # _adjust_count()
        ordered_array = segments
    elif status == 4:  # _split_by_threshold() - check upper weight threshold
        ordered_array = segments[weights > we_driver.largest_allowed_weight]
    elif status == 5:  # _merge_by_threshold() - check lower weight threshold
        ordered_array = segments[weights < we_driver.smallest_allowed_weight]
    elif status == 6:  # _run_we - merging all segs in one group
        ordered_array = np.add.accumulate(weights)
    else:
        print(status)
        print("Not sure why this is triggered")

    return segments, weights, ordered_array, cumul_weight
