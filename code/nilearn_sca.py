import os
from nilearn import input_data
from sys import argv, exit
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler

if not argv[0]:
    print("""Missing arguments, 3 arguments required. Usage:
    python nilearn_sca.py <SUB_DIR> <path/to/directory/of/binarized/cluster/masks> <target_mask_dir>
    Expects binarized parcels of the desired cluster solution in the cluster mask dir.
    Expects participants.tsv in mask dir.
    Target_mask_dir = directory where target_MNI4fmri.nii.gz can be found - same as used in cbptools""")
    exit()
    
elif not argv[1]:
    print("path to binarized cluster masks of the cluster solution to be mapped needs to be parsed")
    print("Usage: python nilearn_sca.py <SUB_DIR> <path/to/directory/of/binarized/cluster/masks> <target_mask_dir>")
    exit()
        
elif not argv[2]:
    print("Directory of target masks needs to be set. target_MNI4fmri.nii.gz needs to be in this directory")
    exit()
    
else:
    print(f"Running seed based correlation analysis for all subcluster masks found in {argv[1]}, for each subject individually")

SUB_DIR = argv[1]
mask_dir = argv[2]
target_mask_dir = argv[3]

sub_list = pd.read_csv(mask_dir + '/participants.tsv', delimiter=',')
mask_list =  [i for i in os.listdir(mask_dir)]
mask_list.remove('participants.tsv')

for sub in sub_list:
    print(sub)
    
    os.chdir(f"{SUB_DIR}/{sub}/Resting/REST1") #directory with ica cleaned, zscored adn concatenated (across 4 runs) data
    if not os.path.exists('ts_dir'):
        os.mkdir('ts_dir') # directory for individual timeseries
    if not os.path.exists('correlation_maps'):
        os.mkdir('correlation_maps') # output directory
    
    for mask in mask_list:
        print(mask)
        # extract seed mask timeseries
        os.system(f'fslmeants -i rfMRI_REST1_hp2000_clean_zvals_concat.nii.gz -o ts_dir/{mask[:-7]}_ts.txt -m {mask_dir}/{mask}')
        
        seed_ts = np.loadtxt(f"ts_dir/{mask[:-7]}_ts.txt").reshape(-1,1)
        
       
        brain_masker = input_data.NiftiMasker(mask_img=f"{target_mask_dir}/target_MNI4fmri.nii.gz", 
                                              standardize=False, 
                                              memory_level=1, 
                                              verbose=0)
        
        brain_ts = brain_masker.fit_transform('rfMRI_REST1_hp2000_clean_zvals_concat.nii.gz')
        
        seed_to_voxel_correlations = (np.dot(brain_ts.T, seed_ts) /
                              seed_ts.shape[0]
                              )
        seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
        seed_to_voxel_correlations_img.to_filename((f'correlation_maps/{mask}_correlation_r.nii.gz'))
        
        seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
        seed_to_voxel_correlations_fisher_z_img = brain_masker.inverse_transform(seed_to_voxel_correlations_fisher_z.T)
        seed_to_voxel_correlations_fisher_z_img.to_filename(f'correlation_maps/{mask}_correlation_z.nii.gz')
