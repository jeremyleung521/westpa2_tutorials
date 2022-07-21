import logging
import operator
import numpy
import mdtraj
import os.path
import pickle

import westpa
from westpa.core.propagators.executable import restart_writer
# Just placeholder until I can clean up the code
from sklearn import cluster

log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

def _sort_walkers_identity(we_driver, ibin, status, **kwargs):
    '''A function that, given  sorts the walkers based on a given criteria. Status indicate which method it's from. The int
    arguments mean the following:

    status = 0    _run_we() - not doing any sorting
    status = 1    _split_by_weight() - check upper ideal weight threshold
    status = 2    _merge_by_weight() - check lower ideal weight threshold
    status = 3    _adjust_count() merge
    status = 4    _split_by_threshold() - check upper weight threshold
    status = 5    _merge_by_threshold() - check lower weight threshold
    status = 6    _run_we() - merging all segs in one group
    '''
    #with open('bin.pickle','wb') as fo:
    #    pickle.dump(ibin,fo)

    log.debug('using sort._sort_walkers_identity')
    segments = numpy.array(sorted(ibin, key=operator.attrgetter('weight')), dtype=numpy.object_)
    weights = numpy.array(list(map(operator.attrgetter('weight'), segments)))
    cumul_weight = 0

    if status == 0:  # run_we - not doing any sorting
        ordered_array = segments
    elif status == 1:  # _split_by_weight() - check upper ideal weight threshold
        ideal_weight = kwargs['ideal_weight']
        ordered_array = segments[weights > we_driver.weight_split_threshold * ideal_weight]
    elif status == 2:  # _merge_by_weight() - check lower ideal weight threshold
        cumul_weight = numpy.add.accumulate(weights)
        ideal_weight = kwargs['ideal_weight']
        ordered_array = segments[cumul_weight <= ideal_weight * we_driver.weight_merge_cutoff]
    elif status == 3:  # _adjust_count()
        ordered_array = segments
    elif status == 4:  # _split_by_threshold() - check upper weight threshold
        ordered_array = segments[weights > we_driver.largest_allowed_weight]
    elif status == 5:  # _merge_by_threshold() - check lower weight threshold
        cumul_weight = numpy.add.accumulate(weights)
        ordered_array = segments[weights < we_driver.smallest_allowed_weight]
    elif status == 6:  # _run_we - merging all segs in one group
        ordered_array = numpy.add.accumulate(weights)
    else:
        print(status)
        print("Not sure why this is triggered")

    return segments, weights, ordered_array, cumul_weight


def _sort_walkers_distmatrix(we_driver, ibin, status, **kwargs):
    '''A function that sorts through a distance matrix, and maps those segments into an ordered list of segments
    for split/merge.
    '''
    # Temporarily sorting them by weight. This is done because there are times where seg_ids are not assigned yet. Also
    # makes the distance matrix order much more predictable...
    we_driver.sorting_function_kwargs['scheme'] = 'paired'
    log.debug('using sort._sort_walkers_distmatrix')
 
    segments = numpy.array(sorted(ibin, key=operator.attrgetter('weight')), dtype=numpy.object_)
    weights = numpy.array(list(map(operator.attrgetter('weight'), segments)))
    cumul_weight = 0

    if status == 0: # run_we - not doing any sorting
        ordered_array = segments
    elif status == 1:  # _split_by_weight() - check upper ideal weight threshold
        ideal_weight = kwargs['ideal_weight']
        ordered_array = segments[weights > we_driver.weight_split_threshold * ideal_weight]
        we_driver.sorting_function_kwargs['scheme'] = 'list'
    elif status == 2:  # _merge_by_weight() - check lower ideal weight threshold
        cumul_weight = numpy.add.accumulate(weights)
        ideal_weight = kwargs['ideal_weight']
        ordered_array = segments[cumul_weight <= ideal_weight * we_driver.weight_merge_cutoff]
        we_driver.sorting_function_kwargs['scheme'] = 'list'
    elif status == 4:  # _split_by_threshold() - check upper weight threshold
        ordered_array = segments[weights > we_driver.largest_allowed_weight]
        we_driver.sorting_function_kwargs['scheme'] = 'list'
    elif status == 5:  # _merge_by_threshold() - check lower weight threshold
        cumul_weight = numpy.add.accumulate(weights)
        we_driver.sorting_function_kwargs['scheme'] = 'list'
        ordered_array = segments[weights < we_driver.smallest_allowed_weight]
    elif status == 6:  # _run_we - merging all segs in one group
        ordered_array = numpy.add.accumulate(weights)
        we_driver.sorting_function_kwargs['scheme'] = 'list'
    elif status == 3:
        log.debug('abcde trying status 3')
        # TODO: rewrite the collect_coordinates
        #if segments[0].data:
        log.debug(f'bin length: {len(ibin)}')

        try:
            all_coords = _collect_aux_coordinates(we_driver, ibin, **kwargs)
            print('using _collect_aux_coordinates')
        except KeyError:
            print('KeyError')
            all_coords = None
            #all_coords = _collect_coordinates(**kwargs)
        print(f'foo {all_coords}') 
        #Probably don't need this backup...    
        if all_coords is None:
            we_driver.sorting_function_kwargs['scheme'] = 'list'
            return segments, weights, segments, cumul_weight
        #    log.warning('running kmeans as a backup')
        #    k_labels = kmeans(coords=coords, n_clusters=n_clusters, splitting=splitting, **kwargs)
        #    return k_labels
    
        #Tranform the coordinates
        distance_matrix = _featurize(all_coords)
        print(distance_matrix)
        # Sort based on the featurized distance matrix
        sorted_flatten = numpy.argsort(distance_matrix, axis=None)
        num_segs = len(distance_matrix)
        sorted_indices = [] 
        sorted_array = []

        # The mapping process, turing indices into pointers to the segment objects
        for jdx in sorted_flatten:
            temp_tuple = tuple([int(numpy.floor(jdx / num_segs)), jdx % num_segs])
            if temp_tuple[1] > temp_tuple[0]: # Prevent double count...
                sorted_indices.append(temp_tuple)
        for paired_index in sorted_indices:
            sorted_array.append(tuple([segments[paired_index[0]],segments[paired_index[1]]]))
        
        print(sorted_array)
        ordered_array = numpy.asarray(sorted_array)
        we_driver.sorting_function_kwargs['scheme'] = 'paired'

        # Passing different things depending on the status number...
        #if status == 1:  # _split_by_weight() - check upper ideal weight threshold
        #    ideal_weight = kwargs['ideal_weight']
        #    ineligible_array = sorted_array[weights <= we_driver.weight_split_threshold * ideal_weight]
        #    ordered_array = _sort_remove_ineligible(sorted_array, sorted_indices, ineligible_array)
        #elif status == 2:  # _merge_by_weight() - check lower ideal weight threshold
        #    cumul_weight = numpy.add.accumulate(weights)
        #    ideal_weight = kwargs['ideal_weight']
        #    ordered_array = segments[cumul_weight <= ideal_weight * we_driver.weight_merge_cutoff]
        #    #ineligible_array = segments[cumul_weight > ideal_weight * we_driver.weight_merge_cutoff]
        #elif status == 3:  # _adjust_count()
        #    ordered_array = segments
        #elif status == 4:  # _split_by_threshold() - check upper weight threshold
        #    ordered_array = segments[weights > we_driver.largest_allowed_weight]
        #elif status == 5:  # _merge_by_threshold() - check lower weight threshold
        #    cumul_weight = numpy.add.accumulate(weights)
        #    ordered_array = segments[weights < we_driver.smallest_allowed_weight]
        #elif status == 6:  # _run_we - merging all segs in one group
        #    ordered_array = numpy.add.accumulate(weights)
        #else:
        #    print(status)
        #    print("Not sure why this is triggered")
    
        #ordered_array = numpy.asarray(ordered_array)

    return segments, weights, ordered_array, cumul_weight


def _sort_remove_ineligible(sorted_array, sorted_indices, ineligible_array):
    """Function that loops through the sorted_array (which are paired segments) and removes them
    in one fell swoop.
    """
    del_list = []
    for idx, paired_idx in enumerate(sorted_indices):
        for jdx in paired_idx: # This is a paired list, so should be 2, but generalizing it...
             if jdx in ineligible_array:
                 del_list.append(idx)
                 break
    del_list = sorted(del_list, reverse=True)

    print(del_list)
    print(f'before: {sorted_array}')
    for val in del_list:
        del sorted_array[val]

    print(f'after: {sorted_array}')

    return sorted_array
    

def _featurize(coordinates):
    """User-defined function that featurizes based on the coordinates generated from _collect_coordinates.
    This version takes the coordinate between RMSD each frame's Cl- (It's superposed on Na+)
    
    Parameters
    ----------
    coordinates : array-like
        An array-like object with the shape (n_frames, n_atoms, n_dimen). Generated from _collect_coordinates().

    Returns
    -------
    matrix : numpy.ndarray
        A square matrix where the element (i,j) is the pair-wise "distance" between structure i and structure j.

    """
    if coordinates is None or 0:
        return None
    else:
        matrix = numpy.zeros((len(coordinates),len(coordinates)))
        num_atoms = len(coordinates[0])
        for i in range(0, len(coordinates)): # picking the frames
            for j in range(i+1, len(coordinates)):
                val = 0
                for atom in range(1, num_atoms): # picking the atom, only Cl-
                    for dimen in range(0, 3): # range x,y,z
                        val += (coordinates[i, atom,dimen] - coordinates[j, atom, dimen])**2
                    matrix[i,j] = numpy.sqrt(val / num_atoms)
                    matrix[j,i] = numpy.sqrt(val / num_atoms)
        return matrix


def _collect_aux_coordinates(we_driver, segments, aux_name='coord', **kwargs):
    """Function that collects all the xyz coordinates from the auxiliary dataset, which defaults to name 'coord'.
    This assumes that it is in the format expected by the haMSM plugin, of the dimensions (segments, frame, atoms, 3).

    Parameters
    ----------
    we_driver : obj
        The we_driver object

    segments : list or array-like
        Sorted list/array of segments from bin.

    aux_name : str
        Name of the auxiliary dataset. Default : 'coord'

    Returns
    -------
    all_coords : array-like object
        An array-like objects containing all the xyz data.
    """
    psegments = numpy.array(sorted(we_driver.current_iter_segments, key=operator.attrgetter('weight')), dtype=numpy.object_)
    if aux_name not in psegments[0].data:
        return None
    all_coords = []
    parent_id_list = []
    for segment in segments:
        #print(segment.data[aux_name])
        element = segment.wtg_parent_ids.pop()
        segment.wtg_parent_ids.add(element)
        parent_id_list.append(element)
        #print(segment.wtg_parent_ids)
    print(parent_id_list)
    
    parent_seg_list = []
    #print(f'segments: {segments}')
    #print(psegments)
    #with open('pickle.pickle','wb') as fo:
    #    pickle.dump(psegments, fo)

    for pid in parent_id_list:
        seg = [i for i in psegments if i.seg_id == pid]
        all_coords.append(seg[0].data[aux_name][-1])
    
    all_coords = numpy.asarray(all_coords)

    print(all_coords)
    return all_coords


def _collect_coordinates(topology_path=None, traj_name=None, ref_path=None, parent_path=None, atom_slice=None, ref_slice=None, **sort_arguments):
    """Function that collect all the xyz coordinates from location listed in `west.cfg` and group_arguments.

    Parameters
    ----------
    topology_path : str
        Path to the topology_path file.

    traj_name : str
        Name of the restart file. 

    ref_file : str
        Path to the reference file, which is used for 
        alignment (MDTraj.superpose).
 
    atom_slice : array-like object or None, optional
        An array-like object with indices of atoms to look at.
        If not provided, all atoms are read by MDTraj.

    ref_slice : array-like object or None, optional
        An array-like object with indices of atoms to superpose.
        If not provided, all atoms are used to superpose.


    Returns
    -------
    all_coords : array-like object
        An array-like objects containing all the xyz data.


    Warns
    ------
    log.warning : str
        If no segment data are found in westpa.rc.sim_manager.segments
    
    """
    # Prep segment data
    segments = westpa.rc.sim_manager.segments

    # Prep all paths/expand environmental variables
    topology_path = os.path.expandvars(topology_path)
    ref_path = os.path.expandvars(ref_path) 
    parent_path = os.path.expandvars(parent_path)
    restart_path = westpa.rc.config['west', 'data', 'data_refs', 'segment']
    restart_path = os.path.expandvars(restart_path)

    # Start loading and stuff
    ref_file = mdtraj.load(ref_path, top=topology_path, atom_indices=atom_slice)
    if segments is None or len(segments) == 0:
        log.warning('No segments found. Unable to collect coordinates for segments. Skipping, assuming is during initialization')
        return []
    elif len(segments) == 1: 
        pathy = restart_path.format(segments[0]) + f'/{traj_name}'
        full_traj = mdtraj.load(pathy, top=topology_path, atom_indices=atom_slice)
    else:
        for n_idx, n_seg in segments.items():
            if n_seg.status==1:
                # For case when you need to load the parent. (i.e. start of each iteration)
                n_seg.n_iter = n_seg.n_iter - 1
                pathy = parent_path.format(segment=n_seg) + f'/{traj_name}'
                n_seg.n_iter = n_seg.n_iter + 1
            else:
                pathy = restart_path.format(segment=n_seg) + f'/{traj_name}'

            try:
                #restart_writer(pathy, nseg)
                try:
                    seg_traj = mdtraj.load(pathy, top=topology_path, atom_indices=atom_slice)
                except OSError as e:
                    return -1
                except TypeError as e:
                    seg_traj = mdtraj.load(pathy, top=topology_path)
                    seg_traj.atom_slice(atom_slice)
                full_traj = full_traj.join(seg_traj)
            except NameError:
                try:
                    full_traj = mdtraj.load(pathy, top=topology_path, atom_indices=atom_slice)
                except TypeError:
                    full_traj = mdtraj.load(pathy, top=topology_path)
                    full_traj.atom_slice(atom_slice)
    full_traj = full_traj.superpose(ref_file, atom_indices=ref_slice)
    all_coords = full_traj._xyz 

    return all_coords 
    #return full_traj 


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


def dist_matrix(coords, n_clusters, splitting, **kwargs):
    """Main function which executes the coordinate collection, featurization and clustering for the "binless" scheme.

    This main function calls _collect_coordinates() to load in all the x,y,z coordinates of the restart, featurizes them
    with _featurize() into a N x N "distance" matrix, then pass it to sklearn's agglomerative clustering function to
    generate a label for each segment. N should be the number of segments in this iteration.

    This is written as general as possible so users can simply modify subfunctions _featurize() and 
    _collect_coordinates() without actually having to modify this dist_matrix() function.

    Parameters
    ----------
    coords : array-like
        An array with all the pcoordinates of the final frame. It's a default inumpy.t.

    n_clusters : int
        A predefined number indicating how many final clusters.


    splitting : bool
        A predefined variabel 

    **kwargs : dict
        A dictionary of many other variables defined in the west.cfg YAML file under the group_arguments. It's made as
        general as possible here since they are not able to be pre-defined in the binless scheme.
        
        For this specific dist_matrix() function, it includes:
            topology_path : str
                Path to the topology file.
            traj_name : str
                Name of the restart file. e.g. seg.xml, seg.ncrst. Necessary for MDTraj loading, so use the full 
                file extension, if possible (i.e. ncrst, not rst; prmtop, not top).
            ref_path : str
                Path to the refernce file for superposing/alignment.
            atom_slice : array-like, 1D, optional, default : None
                An array of atom indices to load from the restart files. If not provided, all atoms are loaded.
            ref_slice : array-like, 1D, optional; default : None
                An array of atom indices to align with the reference file. If not provided, all atoms are used.
            link_type : str, optional, default : 'average' 
                The linkage type used in aggolomerative clustering. 'average' is used as default but can be 
                'complete', 'single'. See sklearn documentations for more information.

    Returns
    -------
    labels : array-like
        An array-like with a group label assigned for each segment. 

    """
    #if westpa.rc.sim_manager.n_iter is 0 or westpa.rc.sim_manager.n_iter is 1:
    #   k_labels = kmeans(coords=coords, n_clusters=n_clusters, splitting=splitting, **kwargs) 
    #   print('cluster_centers'k_labels)

    # Obtain the coordinates
    ## TODO: Write a version which uses the segment.data{} (i.e. HDF5 Framework)
    all_coords = _collect_coordinates(**kwargs)
    
    if all_coords == -1:
        log.warning('running kmeans as a backup')
        k_labels = kmeans(coords=coords, n_clusters=n_clusters, splitting=splitting, **kwargs)
        return k_labels

    #Tranform the coordinates
    transformed_coords = _featurize(all_coords)
   
    #Then pass to normal agglomerative clustering
    if 'link_type' not in locals():
        link_type = 'average'
    try: 
        agc = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage=link_type)
        agc.fit(transformed_coords)
        labels = agc.labels_
        print(f'labels:', labels)
        return labels
    except ValueError as e:
        print(e)
        # Backup - just run k-means on the final pcoords
        log.warning('running kmeans as a backup')
        k_labels = kmeans(coords=coords, n_clusters=n_clusters, splitting=splitting, **kwargs)
        return k_labels


def coords_matrix(coords, n_clusters, splitting, **kwargs):
    """Main function which executes the coordinate collection, featurization and clustering for the "binless" scheme.

    This function assumes that the coordinates are already the pcoord. So ODLD and such.

    This main function calls _collect_coordinates() to load in all the x,y,z coordinates of the restart, featurizes them
    with _featurize() into a N x N "distance" matrix, then pass it to sklearn's agglomerative clustering function to
    generate a label for each segment. N should be the number of segments in this iteration.

    This is written as general as possible so users can simply modify subfunctions _featurize() and 
    _collect_coordinates() without actually having to modify this dist_matrix() function.

    Parameters
    ----------
    coords : array-like
        An array with all the pcoordinates of the final frame. It's a default inumpy.t.

    n_clusters : int
        A predefined number indicating how many final clusters.

    splitting : bool
        A predefined variable 

    **kwargs : dict
        A dictionary of many other variables defined in the west.cfg YAML file under the group_arguments. It's made as
        general as possible here since they are not able to be pre-defined in the binless scheme.
        
        For this specific dist_matrix() function, it includes:
            link_type : str, optional, default : 'average' 
                The linkage type used in aggolomerative clustering. 'average' is used as default but can be 
                'complete', 'single'. See sklearn documentations for more information.

    Returns
    -------
    labels : array-like
        An array-like with a group label assigned for each segment. 

    """
    #if westpa.rc.sim_manager.n_iter is 0 or westpa.rc.sim_manager.n_iter is 1:
    #   k_labels = kmeans(coords=coords, n_clusters=n_clusters, splitting=splitting, **kwargs) 
    #   print('cluster_centers'k_labels)

    # Obtain the coordinates
    ## TODO: Write a version which uses the segment.data{} (i.e. HDF5 Framework)
    #all_coords = _collect_coordinates(**kwargs)
 
    #if all_coords is -1:
    #    log.warning('running kmeans as a backup')
    #    k_labels = kmeans(coords=coords, n_clusters=n_clusters, splitting=splitting, **kwargs)
    #    return k_labels

    #Tranform the coordinates
    #transformed_coords = _featurize(all_coords)
    #with open('coords.pickle','wb') as fo:
    #    pickle.dump(coords,fo)
    

    #Transform the coordinates
    transformed_coords = _featurize(coords)
    
    #Then pass to normal agglomerative clustering
    if 'link_type' not in locals():
        link_type = 'average'
    try:
        agc = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage=link_type)
        agc.fit(transformed_coords)
        labels = agc.labels_
        print(f'labels:', labels)
        return labels
    except ValueError as e:
        print(e)
        # Backup - just run k-means on the final pcoords
        log.warning('running kmeans as a backup')
        k_labels = kmeans(coords=coords, n_clusters=n_clusters, splitting=splitting, **kwargs)
        return k_labels
