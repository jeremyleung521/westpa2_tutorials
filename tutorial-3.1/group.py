import westpa
import logging
from sklearn import cluster
import numpy

log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

def walkers_by_history(we_driver, ibin, hist_length=25, **kwargs):
    '''Groups walkers inside of a bin according to their history.
    Creates a group, which takes the same data format as a bin, and then passes into the
    normal split/merge functions.'''
    # Pass in the bin object instead of the index
    log.debug('using group.walkers_by_history')
    log.debug('history length: {!r}'.format(hist_length))
    bin = we_driver.next_iter_binning[ibin]
    groups = dict()
    z = 0
    for segment in bin:
        if segment.n_iter > 1:
            ##par_iter = we_driver._find_parent_n(segment, hist_length)
            par_iter = _find_parent_n(segment, hist_length)
        else:
            par_iter = (0, 1)
        try:
            groups[par_iter].add(segment)
        except KeyError:
            groups[par_iter] = set([segment])
    return groups.values()

def _find_parent_n(segment, n):
    iteration = (segment.n_iter - 1)
    parent_id = segment.parent_id
    try:
        if (len(segment.id_hist) < n):
            segment.id_hist = [(parent_id, iteration)] + segment.id_hist
        else:
            segment.id_hist.reverse()
            segment.id_hist.append((segment.parent_id, (segment.n_iter - 1)))
            segment.id_hist.reverse()
            del segment.id_hist[n:]
        parent_id = segment.id_hist[(len(segment.id_hist) - 1)][0]
        iteration = segment.id_hist[(len(segment.id_hist) - 1)][1]
    except AttributeError:
        data_manager = westpa.rc.get_data_manager()
        i = 0
        while (i < n) and parent_id >= 0:
            seg_id = parent_id
            iter_group = data_manager.get_iter_group(iteration)
            seg_index = iter_group['seg_index']
            parent_id = seg_index[seg_id]['parent_id']
            try:
                segment.id_hist.append((parent_id, iteration))
            except AttributeError:
                segment.id_hist = list()
                segment.id_hist.append((parent_id, iteration))
            iteration -= 1
            i += 1
        iteration += 1
    return (parent_id, iteration)


def kmeans(coords, n_clusters, splitting, **kwargs):
    X = numpy.array(coords)
    if X.shape[0] == 1:
        X = X.reshape(-1,1)
    km = cluster.KMeans(n_clusters=n_clusters).fit(X)   
    cluster_centers_indices = km.cluster_centers_
    labels = km.labels_
    if splitting:
        print("cluster centers:", numpy.sort(cluster_centers_indices))
    return labels
