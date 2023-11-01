import os
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
from nilearn.image import math_img
from nibabel.spatialimages import SpatialImage

def dice(a, b):
    """ return dice coefficient for two binary arrays"""
    
    return (2.0*np.sum(a*b))/(np.sum(a)+np.sum(b))


# the following is a function from the cbptools toolbox by Reuter et al.;
# to be found at https://github.com/inm7/cbptools
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


# the following is a function from the cbptools toolbox by Reuter et al.;
# to be found at https://github.com/inm7/cbptools
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



def map_roi(seed_in, individual_solution):
    """
    Maps individual labels onto seed roi and returns a nifti image of the labeled roi.
    --------
    
    seed_in = nifti seed image of the hemisphere to be analyzed
    individual_solution = npy file of individual cluster labels for the solution and hemisphere to be analyzed.
    """
    
    
    s = nib.load(seed_in) # load the entire seed
    c = get_mask_indices(s) # get voxel space coordinates of the mask image

    if type(individual_solution) is np.ndarray:
        
        label = individual_solution + 1
    else:
        
        label = np.load(individual_solution)
        label += 1 # because values in label files start with zero (range(0,n)

    labeled_roi = map_labels(img=s, labels=label, indices=c)
    
    return labeled_roi



def separator(inparc, hemi, location_of_subrois, location_of_seed, amygdala='full'):
    """
    Computes dice coefficients of single clusters within a parcellation with amygdala and PC.
    Expects inparc to be a specific clustering solution mapped to seed space of a single subject (or one group parcellation). 
    Uses the dice() function.
    
    inparc = {solution}_cluster_labels.npz['individual_labels'] of shape (participants * voxels)
    
    """
    
    clusternumber = np.max(inparc) + 1 # +1 because 0 is also a cluster label. This is taken care of in the map_roi function below
    print(f"{clusternumber} clusters detected")
    
    if amygdala == 'full':
        amygdala = f"{location_of_subrois}/{hemi}_amygdala_for_sep.nii.gz"
        piriform = f"{location_of_subrois}/{hemi}_pir_for_sep.nii.gz"
    elif amygdala == 'nocortical':
        amygdala = f"{location_of_subrois}/{hemi}_amygdala_nocortical_for_sep.nii.gz"
        piriform = f"{location_of_subrois}/{hemi}_pir_yescortical_for_sep.nii.gz"
    
    insula = f"{location_of_subrois}/{hemi}_insula_for_sep.nii.gz"
    
    amy = nilearn.image.get_data(amygdala)
    pc = nilearn.image.get_data(piriform)
    ins = nilearn.image.get_data(insula)
    
    # all individual separation indices in one list, which will be returned and can be used to create a dataframe
    amy_pir_sep_list = []
    all_sep_list = []
    
    amy_pir_overlap_list = []
    # list of individual mean dice coefficients (over all clusters within a solution) between pir and amy clusters
    mean_majority_pir_amy_dices_added = []    
    
    seedimg = nib.load(location_of_seed)
    overlap_heatmap = np.zeros(seedimg.shape)
    
    combined_pir_amy_dices = []
    combined_amy_pir_dices = []
    mean_combined_pir_amy_dices = []
    
    for individual_label in inparc: #iterate over rows of this array, i.e. over each individual participants' labels.
        
        mapped_parc = map_roi(location_of_seed, individual_label)
        
        majority_pir_amy_dices = []
        majority_pir_ins_dices = []
        majority_amy_pir_dices = []
        majority_amy_ins_dices = []
        majority_ins_pir_dices = []
        majority_ins_amy_dices = []
        
        majority_pir_amy_overlap = []
        majority_pir_ins_overlap = []
        majority_amy_pir_overlap = []
        majority_amy_ins_overlap = []
        majority_ins_pir_overlap = []
        majority_ins_amy_overlap = []
        
        # to obtain heatmap of ectopic voxels
        individual_overlaps = np.zeros(seedimg.shape)
        
        # to get combined majority piriform region
        pir_s = np.zeros(seedimg.shape)
        
        # to get combined majority amygdala region
        amy_s = np.zeros(seedimg.shape)
        
        for i in range(1, int(clusternumber)+1):

            #print(f"dice scores for cluster #{i}")

            c = math_img(f"np.where(img == {i}, 1, 0)", img=mapped_parc) # extract the current cluster as binary image
            d = nilearn.image.get_data(c) # turn image of binarized cluster into an array
            
            dice_pir = dice(d, pc)
            overlap_pir = np.sum(d * pc)
            #print(f"pc dice = {dice_pir}")
            dice_amy = dice(d, amy)
            overlap_amy = np.sum(d * amy)

            #print(f"amy dice = {dice_amy}")
            dice_ins = dice(d, ins)
            overlap_ins = np.sum(d * ins)
            
        
            if dice_pir > dice_amy and dice_pir > dice_ins:

                majority_pir_amy_dices.append(dice_amy)
                majority_pir_ins_dices.append(dice_ins)
                
                # add count of 'ectopically' overlapping voxels - in this case with amy and ins
                majority_pir_amy_overlap.append(overlap_amy)
                majority_pir_ins_overlap.append(overlap_ins)
                
                # accumulate 'wrongly assigned' voxels for later depiction as heatmap on seed across individuals
                
                individual_overlaps = individual_overlaps + (d*amy) + (d*ins)
                
                # create combined majority piriform cluster array
                pir_s += d

            elif dice_amy > dice_pir and dice_amy > dice_ins:

                majority_amy_pir_dices.append(dice_pir)
                majority_amy_ins_dices.append(dice_ins)
                
                majority_amy_pir_overlap.append(overlap_pir)
                majority_amy_ins_overlap.append(overlap_ins)
                
                individual_overlaps = individual_overlaps + (d*pc) + (d*ins)
                
                # create combined majority amygdala cluster array
                amy_s += d

            elif dice_ins > dice_pir and dice_ins > dice_amy:

                majority_ins_pir_dices.append(dice_pir)
                majority_ins_amy_dices.append(dice_amy)
                
                majority_ins_pir_overlap.append(overlap_pir)
                majority_ins_amy_overlap.append(overlap_amy)
                
                individual_overlaps = individual_overlaps + (d*pc) + (d*amy)


            # now the rare case of equal dice in pir and amy
            elif dice_pir > dice_ins and dice_pir == dice_amy:

                majority_pir_amy_dices.append(dice_amy)
                majority_pir_amy_overlap.append(overlap_amy)
                
                individual_overlaps = individual_overlaps + (d*amy)

            # now the rare case of equal dice in pir and ins
            elif dice_pir > dice_amy and dice_pir == dice_ins:

                majority_pir_ins_dices.append(dice_ins)
                majority_pir_ins_overlap.append(overlap_ins)
                
                individual_overlaps = individual_overlaps + (d*ins)
                
        overlap_heatmap += individual_overlaps
        
        dice_overall_majority_pir_w_amy = dice(pir_s, amy)
        dice_overall_majority_amy_w_pir = dice(amy_s, pc)
        combined_pir_amy_dices.append(dice_overall_majority_pir_w_amy)
        combined_amy_pir_dices.append(dice_overall_majority_amy_w_pir)
        mean_combined_pir_amy_dices.append((dice_overall_majority_pir_w_amy + dice_overall_majority_amy_w_pir)/2)
        
        # the larger, the worse
        amy_pir_overlap = np.sum(majority_pir_amy_overlap) + np.sum(majority_amy_pir_overlap)
 
        amy_pir_overlap_list.append(amy_pir_overlap)
    
        
        outimg = nib.Nifti1Image(np.float32(overlap_heatmap), seedimg.affine, seedimg.header)
    
    return combined_pir_amy_dices, combined_amy_pir_dices, mean_combined_pir_amy_dices, amy_pir_overlap_list, outimg


####### now an exemplary function call #######

modalities = ['bimodal','dmri', 'fmri']

df = pd.DataFrame()

for hemi in ['lh','rh']:
    
    location_of_subrois = f'/home/felix/PC_CBP/final_seeds/roi_for_sep/{hemi}/'
    
    seed = f'/home/felix/PC_CBP/final_seeds/{hemi}_full_seed_2mm.nii.gz'
            
    for solution in range(2,11):

        for modality in modalities:
            
            for truth in ['measured', 'shuffled']:
                
                if truth == 'measured':

                    if modality == 'bimodal':

                        i = np.load(f'/home/felix/PC_CBP/06_23_group_results/real/bimodal/group_{hemi}/{solution}_cluster_labels.npz')
                        i = i['individual_labels']

                    else:

                        i = np.load(f'/home/felix/PC_CBP/06_23_group_results/real/{modality}/group_{hemi}/{solution}clusters/labels.npz')
                        i = i['individual_labels']
                    
                elif truth == 'shuffled':
                    
                    i = np.load(f'/home/felix/PC_CBP/06_23_group_results/fake/{modality}/group_{hemi}/{solution}_cluster_labels.npz')
                    i = i['individual_labels']
                    
                for definition in ['full', 'nocortical']:

                    #this function call creates 3 lists of shape(100,) containing individual subjects' values, which are used to create a df below for later statistics
                    combined_pir_amy_dices, combined_amy_pir_dices, mean_combined_pir_amy_dices, amy_pir_overlap, overlap_img = separator(inparc=i, 
                                                                                                                      hemi=hemi, 
                                                                                                                      location_of_subrois=location_of_subrois, 
                                                                                                                      location_of_seed=seed, amygdala=definition)
                    current_df = pd.DataFrame()
                    current_df['combined_pir_amy_dices'] = combined_pir_amy_dices
                    current_df['combined_amy_pir_dices'] = combined_amy_pir_dices
                    current_df['mean_combined_pir_amy_dices'] = mean_combined_pir_amy_dices
                    current_df['amy_pir_overlap'] = amy_pir_overlap

                    current_df['modality'] = [modality for i in range(100)]
                    current_df['hemi'] = [hemi for i in range(100)]
                    current_df['solution'] = [solution for i in range(100)]
                    current_df['truth'] = [truth for i in range(100)]
                    current_df['definition'] = [definition for i in range(100)]

                    df = df.append(current_df)
            
                    if truth == 'measured':
                        nib.save(overlap_img, f'/home/felix/PC_CBP/img/overlap_imgs/{modality}_{hemi}_{solution}_overlap_heatmap_{definition}.nii.gz')

