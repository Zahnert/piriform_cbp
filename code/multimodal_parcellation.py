### These functions were used to 
### a) create random parcellations for statistical comparison with our results and
### b) to create bimodal parcellations

import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist
import itertools
import numpy as np
import pandas as pd

from typing import Union, List
from sys import float_info
import nibabel as nib
from nibabel.spatialimages import SpatialImage

from scipy.cluster import hierarchy
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import kernel_metrics

from scipy import stats

##### SOME FUNCTIONS FROM CBPTOOLS; see Reuter et al. 2020 and https://github.com/inm7/cbptools #####

def relabel(reference: np.ndarray, x: np.ndarray) -> (np.ndarray, list):
    """Relabel cluster labels to best match a reference"""

    permutations = itertools.permutations(np.unique(x))
    accuracy = 0.
    relabeled = None

    for permutation in permutations:
        d = dict(zip(np.unique(x), permutation))
        y = np.zeros(x.shape).astype(int)

        for k, v in d.items():
            y[x == k] = v

        _accuracy = np.sum(y == reference) / len(reference)

        if _accuracy > accuracy:
            accuracy = _accuracy
            relabeled = y.copy()

    return relabeled, accuracy

def map_labels(img: SpatialImage, labels: np.ndarray,
               indices: np.ndarray) -> SpatialImage:
    """Map cluster labels onto the seed mask

    Parameters
    ----------
    img : SpatialImage
        Mask NIfTI image to which the labels will be mapped
    labels : np.ndarray
        1D array of cluster labels (sklearn.cluster.KMeans._labels)
    indices : np.ndarray
        Indices of all mask image voxels of which the order coincides
        with the order of the voxels in the labels array.

    Returns
    -------
    SpatialImage
        Mask image with the labels mapped onto it.
    """
    if len(indices) != len(labels):
        raise ValueError('Indices and labels do not match')

    mapped_img = np.zeros(img.shape)
    mapped_img[indices[0:, 0], indices[0:, 1], indices[0:, 2]] = labels
    return nib.Nifti1Image(np.float32(mapped_img), img.affine, img.header)


def sort_files(participants: str, files: list, pos: int = -1, sep: str = '_',
               index_col: str = 'participant_id') -> list:
    """
    Parameters
    ----------
    participants : str
        Path to a participants tsv file. The contents of the index
        column (default: 'participant_id') should match
        part of the filename. The ordering of the participants in
        this file will determine the ordering of the listed
        files.
    files : list
        List of filenames to be sorted based on the order of values
        of index_col in the participants file.
    pos : int
        Position at which the participant id found in the filename
        when splitting with the defined separator (sep)
    sep : str
        Separator used to split the filename into multiple parts.
    index_col : str
        The column in participants that defines the participant_id
        which is to be found in the list of filenames.

    Returns
    -------
    list
        Sorted input file names
    """

    df = pd.read_csv(participants, sep='\t').set_index(index_col)
    participant_id = df.index.values.astype(str).tolist()
    sorted_files = sorted(
        files,
        key=lambda x: participant_id.index(x.split(sep)[pos].split('.')[0])
    )
    return sorted_files

def get_mask_indices(img: SpatialImage, order: str = 'C') -> np.ndarray:
    """Get voxel space coordinates (indices) of seed voxels

    Parameters
    ----------
    img : SpatialImage
        Mask NIfTI image
    order : str, optional
        Order that the seed-mask voxels will be extracted in.
        The resulting indices will be listed in this way

    Returns
    -------
    np.ndarray
        2D array of shape (n_voxels, 3) containing the 3D coordinates of
        all mask image voxels
    """

    if order not in ('C', 'F', 'c', 'f'):
        raise ValueError('Order has unexpected value: expected %s, got \'%s\''
                         % ("'C' or 'F'", order))

    data = img.get_fdata()
    indices = np.asarray(tuple(zip(*np.where(data == 1))))

    if order.upper() == 'F':
        # indices are C order and must become F order
        reorder = get_c2f_order(img)
        indices = indices[reorder]

    return indices


##### THE FOLLOWING FUNCTION IS ORIGINALLY FROM REUTER ET AL 2020 AND HAS BEEN MODIFIED BY US #####
## see also https://github.com/inm7/cbptools

def group_level_clustering(input: dict, output: dict, params: dict) -> None:
    """ Perform group-level analysis on all individual participant
    clustering results.

    Parameters
    ----------
    input : dict
        Input files, allowed: {seed_img, participants, seed_coordinates,
        labels}
    output : dict
        Output file, allowed {group_labels, group_img}
    params : dict
        Parameters, allowed {linkage, method}. The options parameter is
        equivalent to grouping in the CBPtools documentation on readthedocs.io
        under the parameters for 'clustering'.

    """

    # Input, output, params
    participants = input.get('participants')
    labels = input.get('labels')
    out_labels = output.get('group_labels')
    out_img = output.get('group_img')
    method = params.get('method')
    linkage = params.get('linkage')
    seed_img = input.get('seed_img')
    seed_coordinates = input.get('seed_coordinates')

    if method not in ('agglomerative', 'mode'):
        raise ValueError('Unknown group cluster method: %s' % method)

    # Aggregate subject-level cluster labels into one matrix
    # Resulting shape is (participants, voxels)
    
    # the following steps have been conducted by our load_individual_labels() function and are therefore commented out
    #labels = sort_files(participants, labels, sep='/', pos=1)
    #labels = np.asarray([np.load(f) for f in labels])

    if len(labels.shape) != 2:
        raise ValueError('Cluster label length mismatch between included '
                         'label files')

    # Hierarchical clustering on all labels
    x = labels.T
    y = pdist(x, metric='hamming')
    z = hierarchy.linkage(y, method=linkage, metric='hamming')
    cophenetic_correlation, *_ = hierarchy.cophenet(z, y)
    group_labels = hierarchy.cut_tree(z, n_clusters=len(np.unique(x)))
    group_labels = np.squeeze(group_labels)  # (N, 1) to (N,)

    # Use the hierarchical clustering as a reference to relabel individual
    # participant clustering results
    relabeled = np.empty((0, labels.shape[1]), int)
    accuracy = []

    # iterate over individual participant labels (rows)
    for label in labels:
        x, acc = relabel(reference=group_labels, x=label)
        relabeled = np.vstack([relabeled, x])
        accuracy.append(acc)

    labels = relabeled

    if method == 'agglomerative':
        np.savez(
            out_labels,
            individual_labels=labels,
            relabel_accuracy=accuracy,
            group_labels=group_labels,
            cophenetic_correlation=cophenetic_correlation,
            method='agglomerative'
        )

    elif method == 'mode':
        mode, count = stats.mode(labels, axis=0)
        np.savez(
            out_labels,
            individual_labels=labels,
            relabel_accuracy=accuracy,
            hierarchical_group_labels=group_labels,
            cophenetic_correlation=cophenetic_correlation,
            group_labels=np.squeeze(mode),
            mode_count=np.squeeze(count),
            method='mode'
        )

        # Set group labels to mode for mapping
        group_labels = np.squeeze(mode)

    # Map labels to seed-mask image based on indices
    seed_img = nib.load(seed_img)
    seed_indices = np.load(seed_coordinates)
    group_labels += 1  # avoid 0-labeling
    group_img = map_labels(
        img=seed_img,
        labels=group_labels,
        indices=seed_indices
    )
    nib.save(group_img, out_img)


def merge_individual_labels(input: dict, output: dict) -> None:
    """ Merge individual label results when no group analysis will be
    performed.

    Parameters
    ----------
    input : dict
        Input files, allowed: {labels}
    output : dict
        Output file, allowed {merged_labels}
    """

    # Input, output, params
    label_files = input.get('labels')
    merged_labels = output.get('merged_labels')

    all_labels = dict()

    for label_file in label_files:
        basename = os.path.basename(label_file)
        basename, _ = os.path.splitext(basename)
        labels = np.load(label_file)
        all_labels[basename] = labels

    np.savez(merged_labels, **all_labels)

#### from https://github.com/inm7/cbptools
    
def group_similarity(input: dict, output: dict, params: dict, solution: int) -> None:
    """Pairwise similarity matrix between all participant clustering results

    Parameters
    ----------
    input : dict
        Input files, allowed: {participants, labels}
    output : dict
        Output file, allowed {group_similarity, cophenetic_correlation}
    params : dict
        Parameters, allowed {metric}. The options parameter is equivalent to
        validity:similarity in the CBPtools documentation on readthedocs.io
        under the parameters for 'clustering'.
    """
    # input, output, params
    participants_file = input.get('participants')
    labels_files = input.get('labels')
    similarity_file = output.get('group_similarity')
    cophenet_file = output.get('cophenetic_correlation')
    metric = params.get('metric').lower()

    participants = pd.read_csv(participants_file, sep='\t')
    participants = participants['participant_id']
    df = pd.DataFrame(columns=['participant_id', 'clusters', 'similarity',
                               'relabel accuracy'])
    df_reference = pd.DataFrame(columns=['clusters', 'cophenetic correlation'])

    if hasattr(metrics, metric):
        f = getattr(metrics, metric)
    else:
        raise ValueError('Metric %s not recognized' % metric)

    for file in labels_files:
        data = np.load(file)
        ilabels = data.get('individual_labels')
        glabels = data.get('group_labels')
        accuracy = data.get('relabel_accuracy')

        # Obtain cluster number from file name | This has been modified, cluster number is now an input
        n_clusters = solution

        # Group similarity & cophenetic correlation
        df_reference = df_reference.append({
            'clusters': n_clusters,
            'cophenetic correlation': data.get('cophenetic_correlation')
        }, ignore_index=True)

        for ppid, labels, acc in zip(participants, ilabels, accuracy):
            df = df.append({
                'participant_id': str(ppid),
                'clusters': n_clusters,
                'similarity': f(glabels, labels),
                'relabel accuracy': acc
            }, ignore_index=True)

    df_reference.clusters = df_reference.clusters.astype(int)
    df.clusters = df.clusters.astype(int)
    df.to_csv(similarity_file, sep='\t', index=False)
    df_reference.to_csv(cophenet_file, sep='\t', index=False)

    
### modified using functions from Reuter et al 2020, see above.
def spectral_clustering_bimodal(input: dict, output: dict, params: dict) -> None:
    """ Perform spectral clustering on the input connectivity matrix.
    Parameters
    ----------
    input : dict
        Input files, allowed: {connectivity}
    output : dict
        Output file, allowed {labels}
    params : dict
        The dict is equivalent to cluster_options in the CBPtools
        documentation on readthedocs.io under the parameters for 'clustering'.
    """

    # Input, output, params
    connectivity_file = input.get('connectivity')
    labels_file = output.get('labels')
    n_init = params.get('n_init')
    kernel = params.get('kernel')
    assign_labels = params.get('assign_labels')
    eigen_solver = params.get('eigen_solver')
    n_clusters = params.get('n_clusters')
    gamma = params.get('gamma', None)
    n_neighbors = params.get('n_neighbors', None)
    degree = params.get('degree', None)
    coef0 = params.get('coef0', None)
    eigen_tol = params.get('eigen_tol', None)

    _, ext = os.path.splitext(connectivity_file)
    connectivity = np.load(connectivity_file)

    if ext == '.npz':
        connectivity = connectivity.get('connectivity')

    # If the connectivity file is empty (connectivity could not be computed),
    # create an empty labels file
    if connectivity.size == 0:
        print('%s is empty, aborting clustering' % connectivity_file)
        np.save(labels_file, np.array([]))
        return

    if isinstance(eigen_tol, str):
        eigen_tol = float(eigen_tol)

    kernels = list(kernel_metrics().keys())
    kernels.extend(['nearest_neighbors', 'precomputed',
                    'precomputed_nearest_neighbors'])
    if kernel not in kernels:
        msg = 'Unknown kernel (affinity): %s' % kernel
        print(msg)

    gamma_kernels = ('rbf', 'polynomial', 'sigmoid', 'laplacian', 'chi2')
    if gamma is None and kernel in gamma_kernels:
        msg = 'Setting gamma to 1./%s (1./n_features)' % connectivity.shape[1]
        print(msg)
        gamma = 1./connectivity.shape[1]

    kwargs = {'n_clusters': n_clusters, 'n_init': n_init, 'affinity': kernel,
              'assign_labels': assign_labels, 'eigen_solver': eigen_solver,
              'gamma': gamma, 'n_neighbors': n_neighbors, 'degree': degree,
              'coef0': coef0}

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    debug_msg = str(['%s=%s' % (k, v) for k, v in kwargs.items()])
    debug_msg = debug_msg.strip('[]').replace('\'', '')
    print('clustering %s with options: %s'
                 % (connectivity_file, debug_msg))

    # Perform spectral clustering on the available tolerances
    try:
        kwargs['eigen_tol'] = eigen_tol
        clustering = SpectralClustering(**kwargs)
        clustering.fit(connectivity)
        labels = clustering.labels_

        if np.unique(labels).size != n_clusters:
            print('%s: %s clusters requested, only %s found'
                          % (labels_file, n_clusters, np.unique(labels).size))
            np.save(labels_file, np.array([]))

        # cluster labels are 0-indexed
        np.save(labels_file, labels)

    except np.linalg.LinAlgError as exc:
        logger.error('%s: %s (try increasing the eigen_tol with arpack '
                     'as eigen_solver)' % (labels_file, exc))
        np.save(labels_file, np.array([]))
    

def load_labels_bimodal(subjects_dir, participants, hemi, solution):
    
    """
    generates a list of paths to each individual bimodal label file.
    ---------
    
    subjects_dir = directory in which overall bimodal clustering results have been saved. contains subject-specific subfolders.
    participants = participants.tsv
    hemi = rh or lh
    solution = clustering solution; range 2-10
    
    returns array of dimensions participants x labeled voxels
    
    """
    
    with open(participants, 'r') as ps:
            lines = [i.rstrip() for i in ps.readlines()]
            subjects = [i for i in lines if len(i) == 6]
            
    l = []
            
    for subject in subjects:
        
        l.append(f"{subjects_dir}/individual/{subject}/{hemi}_{solution}cluster_labels.npy")
        
    seed = np.load(l[0])
                 
    for i in l[1:]:
        
        m = np.load(i)
        seed = np.vstack((seed,m))
        
    return seed


### In this example we show code for spectral clustering using shuffled and real data.
### Here, bimodal parcellations were created. The same code can be used with slight modification
### to create just dmri or fmri parcellations.
##### MERGE DMRI AND FMRI MATRICES #####
# note that here, for both modalities the same seeds and targets were used
# therefore, both matrices have the same dimensions

# analysis dir being the directory in which dmri and fmri results are to be found

randperm = False # switch for creation of shuffled connectivity matrices
mode = 'bimodal'

analysis_dir = '/path/to/connectivity/matrices/'
if not os.path.isdir(f"{analysis_dir}/bimodal/individual"):
    os.makedirs(f"{analysis_dir}/bimodal/individual")
if not os.path.isdir(f"{analysis_dir}/bimodal/group"):
    os.mkdir(f"{analysis_dir}/bimodal/group")

participants = '/path/to/participants.tsv'

# make subjects list
with open(participants, 'r') as ps:
    lines = [i.rstrip() for i in ps.readlines()]
    subjects = [i for i in lines if len(i) == 6] # unelegant hack to extract subject id's
    
# create bimodal parcellations for each subject
for subject in subjects:
    
    for hemi in 'rh', 'lh':

        if not os.path.isdir(f"{analysis_dir}/bimodal/{hemi}/individual/{subject}"):
            
            os.mkdir(f"{analysis_dir}/bimodal/{hemi}/individual/{subject}")
    
        d = np.load(f"{analysis_dir}/dmri/{hemi}/{subject}/connectivity.npz")['connectivity']# load dmri connectivity
        f = np.load(f"{fmri_dir}/fmri/{hemi}/{subject}/connectivity.npz")['connectivity']
        
        if randperm:
            
            d = np.array([np.random.permutation(row) for row in d])
            f = np.array([np.random.permutation(row) for row in f])

        if mode == 'bimodal':
            
            m = np.hstack((f,d)) # stack matrices
        
            np.savez(f"{analysis_dir}/bimodal/{hemi}/individual/{subject}/connectivity.npz", connectivity=m)
            
### Now perform individual level clustering - in this example using the bimodal parcellations

for subject in subjects:
    
    for hemi in ['rh', 'lh']
    
        for solution in range(2,11):

            i = {'connectivity':f"{analysis_dir}/bimodal/{hemi}/individual/{subject}/connectivity.npz"}
            o = {'labels':f"{analysis_dir}/bimodal/{hemi}/individual/{subject}/{solution}cluster_labels.npy"}
            p = {'n_init':256, 'kernel':'nearest_neighbors', 'assign_labels':'kmeans', 
                 'eigen_solver':None, 'n_clusters':solution, 'n_neigbours':10, 'eigen_tol':1e-10}

            spectral_clustering_bimodal(input=i, output=o, params=p)

### PERFORM GROUP LEVEL CLUSTERING AND COMPUTE MEAN ARI ###

bimodal_dir = '/your/path/bimodal/' # above = f"{analysis_dir}/bimodal/ ...
participants = '/your/path/to/participants.tsv'

for hemi in 'rh', 'lh':
    
    seed = f"/your/path/to/{hemi}_seed.nii.gz"
    s = nib.load(f"/your/path/to/{hemi}_seed.nii.gz")
    c = get_mask_indices(s) # voxel coordinates of our seed
    np.save(f'/your/path/to/store/seed_coordinates/{hemi}_2mm_coords.npy', c)
    coords = f'/your/path/to/store/seed_coordinates/{hemi}_2mm_coords.npy'
    
    for solution in range(2,11):
        
        l = load_labels_bimodal(subjects_dir=bimodal_dir, participants=participants, hemi=hemi, solution=solution)
        
        i = {'seed_img':seed, 'participants':participants, 'seed_coordinates':coords, 'labels':l}
        
        imgname = f"{bimodal_dir}/group/{hemi}_bimodal_{solution}labeled_roi.nii.gz" #hier definieren wir wie die outputs heißen sollen
        lblname = f"{bimodal_dir}/group/{hemi}_{solution}_cluster_labels.npz"
        
        o = {'group_labels':lblname, 'group_img':imgname}
        
        group_level_clustering(input=i, output=o, params=parameters)
        
        i_sim = {'participants':participants, 'labels':lblname}
        o_sim = {'group_similarity':f"{bimodal_dir}/group/{hemi}_{solution}_group_similarity.csv", 
                 'cophenetic_correlation':f"{bimodal_dir}/group/{hemi}_{solution}_cophenetic_correlation.csv"}
        p_sim = {'metric':'adjusted_rand_score'}
        
        group_similarity(input=i_sim, output=o_sim, params=p_sim, n_clusters=solution)