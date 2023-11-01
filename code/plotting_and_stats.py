### examplary code for plotting mean ari vs chance
## a paired sample t-test with bonferroni correction is conducted at the same time using the statannot tool

from statannot import add_stat_annotation
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(20,10)})
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
sns.set_palette('Set2')

full_df_lh = full_df.loc[full_df['hemi'] == 'lh']
full_df_rh = full_df.loc[full_df['hemi'] == 'rh']


f, axs = plt.subplots(1,2,figsize=(20,10),sharey=True)

sns.violinplot(x="clusters", 
               y="similarity", 
               ax = axs[1],
               data=full_df_lh, 
               inner='box', 
               hue='truth',
               hue_order=['measured', 'shuffled'],
               cut=0, scale='count')

sns.violinplot(x="clusters", 
               y="similarity", 
               ax = axs[0],
               data=full_df_rh, 
               inner='box', 
               hue='truth', 
               hue_order=['measured', 'shuffled'],
               cut=0, scale='count') 

axs[0].set_xlabel('Clustering solution')
axs[0].set_ylabel('Individual to group similarity (ARI)')
axs[1].set_title('Diffusion MRI, left hemisphere')
axs[0].legend(loc='best', title=None, fontsize=10)

axs[1].set_xlabel('Clustering solution')
axs[1].set_ylabel('Individual to group similarity (ARI)')
axs[0].set_title('Diffusion MRI, right hemisphere')
axs[1].legend(loc='best', title=None, fontsize=10)


order = [i for i in range(2,11)]
test_results = add_stat_annotation(axs[1], data=full_df_lh, x='clusters', y='similarity', hue='truth',
                                   box_pairs=[
                                       ((2, 'shuffled'), (2, 'measured')),
                                        ((3, 'shuffled'), (3, 'measured')),
                                        ((4, 'shuffled'), (4, 'measured')),
                                        ((5, 'shuffled'), (5, 'measured')),
                                        ((6, 'shuffled'), (6, 'measured')),
                                        ((7, 'shuffled'), (7, 'measured')),
                                        ((8, 'shuffled'), (8, 'measured')),
                                       ((9, 'shuffled'), (9, 'measured')),
                                       ((10, 'shuffled'), (10, 'measured'))],
                                   test='t-test_paired', text_format='star', loc='inside', verbose=2)

test_results = add_stat_annotation(axs[0], data=full_df_rh, x='clusters', y='similarity', hue='truth',
                                   box_pairs=[
                                       ((2, 'shuffled'), (2, 'measured')),
                                        ((3, 'shuffled'), (3, 'measured')),
                                        ((4, 'shuffled'), (4, 'measured')),
                                        ((5, 'shuffled'), (5, 'measured')),
                                        ((6, 'shuffled'), (6, 'measured')),
                                        ((7, 'shuffled'), (7, 'measured')),
                                        ((8, 'shuffled'), (8, 'measured')),
                                       ((9, 'shuffled'), (9, 'measured')),
                                       ((10, 'shuffled'), (10, 'measured'))],
                                   test='t-test_paired', text_format='star', loc='inside', verbose=2)


plt.tight_layout()
plt.savefig('/home/felix/PC_CBP/06_23_group_results/results/dmri_ari_vs_shuffled.tif', bbox_inches='tight', dpi=600, format='tif')
plt.show()



# plot PC-amygdala separation with contrast cortical amygdala as PC vs not; modality = dmri
from statannot import add_stat_annotation
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/home/felix/PC_CBP/separation_df_real_and_shuffled_melted_corticaldefinition_dmri_125.csv')
df['definition'] = df['definition'].map({'nocortical': 'cortical_amygdala_as_pc', 'full': 'cortical_amygdala_as_amygdala'})
df = df.loc[df['truth'] == 'measured']
df = df.loc[df['modality'] == 'dmri_125']

df_lh = df.loc[df['hemi'] == 'lh']
df_rh = df.loc[df['hemi'] == 'rh']

sns.set(rc={'figure.figsize':(20,10)})
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
sns.set_palette('pastel')

f, axs = plt.subplots(1,2,figsize=(20,10),sharey=True)

sns.violinplot(x="solution", 
               y="mean_combined_pir_amy_dices",
               ax=axs[1],
               data=df_lh, 
               inner='box', 
               hue='definition', 
               cut=0, 
               scale='count')

sns.violinplot(x="solution", 
               y="mean_combined_pir_amy_dices",
               ax=axs[0],
               data=df_rh, 
               inner='box', 
               hue='definition', 
               cut=0, 
               scale='count')

axs[0].set_xlabel('Clustering solution')
axs[0].set_ylabel('Mean dice between amygdala and piriform')
axs[1].set_title('Diffusion MRI: left hemisphere')
axs[0].get_legend().remove()


axs[1].set_xlabel('Clustering solution')
axs[1].set_ylabel('Mean dice between amygdala and piriform')
axs[0].set_title('Diffusion MRI: right hemisphere')
axs[1].legend(loc='best', title=None, fontsize=10)

test_results = add_stat_annotation(axs[1], data=df_lh, x='solution', y='mean_combined_pir_amy_dices', hue='definition',
                                   box_pairs=[
                                       ((2, 'cortical_amygdala_as_amygdala'), (2, 'cortical_amygdala_as_pc')),
                                        ((3, 'cortical_amygdala_as_amygdala'), (3, 'cortical_amygdala_as_pc')),
                                        ((4, 'cortical_amygdala_as_amygdala'), (4, 'cortical_amygdala_as_pc')),
                                        ((5, 'cortical_amygdala_as_amygdala'), (5, 'cortical_amygdala_as_pc')),
                                        ((6, 'cortical_amygdala_as_amygdala'), (6, 'cortical_amygdala_as_pc')),
                                        ((7, 'cortical_amygdala_as_amygdala'), (7, 'cortical_amygdala_as_pc')),
                                        ((8, 'cortical_amygdala_as_amygdala'), (8, 'cortical_amygdala_as_pc')),
                                       ((9, 'cortical_amygdala_as_amygdala'), (9, 'cortical_amygdala_as_pc')),
                                       ((10, 'cortical_amygdala_as_amygdala'), (10, 'cortical_amygdala_as_pc'))],
                                   test='t-test_paired', text_format='star', loc='inside', verbose=2)

test_results = add_stat_annotation(axs[0], data=df_rh, x='solution', y='mean_combined_pir_amy_dices', hue='definition',
                                   box_pairs=[
                                       ((2, 'cortical_amygdala_as_amygdala'), (2, 'cortical_amygdala_as_pc')),
                                        ((3, 'cortical_amygdala_as_amygdala'), (3, 'cortical_amygdala_as_pc')),
                                        ((4, 'cortical_amygdala_as_amygdala'), (4, 'cortical_amygdala_as_pc')),
                                        ((5, 'cortical_amygdala_as_amygdala'), (5, 'cortical_amygdala_as_pc')),
                                        ((6, 'cortical_amygdala_as_amygdala'), (6, 'cortical_amygdala_as_pc')),
                                        ((7, 'cortical_amygdala_as_amygdala'), (7, 'cortical_amygdala_as_pc')),
                                        ((8, 'cortical_amygdala_as_amygdala'), (8, 'cortical_amygdala_as_pc')),
                                       ((9, 'cortical_amygdala_as_amygdala'), (9, 'cortical_amygdala_as_pc')),
                                       ((10, 'cortical_amygdala_as_amygdala'), (10, 'cortical_amygdala_as_pc'))],
                                   test='t-test_paired', text_format='star', loc='inside', verbose=2)


plt.tight_layout()
plt.show()
