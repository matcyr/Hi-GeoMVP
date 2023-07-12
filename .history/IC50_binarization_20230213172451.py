# %%
from prepare_data.create_drp_dict import read_dr_dict
from prepare_data.create_drug_feat import load_drug_feat
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

# %%
drug_response_dict, drug_name, cell_name = read_dr_dict()

# %%
path_ic50 =  './Data/GDSC_data/IC50reduced.csv'
IC50table = pd.read_csv(path_ic50)
IC50table = IC50table.rename(columns={'0': 'cell_id'})
IC50table = IC50table.set_index('cell_id')

# %%
drug_id_name = pd.read_csv('./Data/GDSC_data/drugid_name.txt',sep='\t',header=None)

# %%
drug_id_name_dict = {item[1]:item[0] for item in drug_id_name.values}
drug_name_id_dict = {item[0]:item[1] for item in drug_id_name.values}

# %%
target_drug = pd.read_csv('/home/yurui/Atten_Geom_DRP/Data/GDSC_data/Table_S12_GDSC_Drug_group.csv')
target_drug = target_drug.set_index('TARGET_PATHWAY').DRUG_NAME.str.split(';', expand=True).stack().reset_index(level=1, drop=True).reset_index()
target_drug = target_drug.pivot(index= 0 , columns='TARGET_PATHWAY', values=0).fillna(0)
target_drug.rename(index = drug_id_name_dict, inplace = True)
target_drug = target_drug[target_drug.index.isin(drug_id_name_dict.values())]
for column in target_drug.columns:
    target_drug[column] = pd.to_numeric(target_drug[column], errors='coerce').fillna(1).astype(int)

# %%
### Fit the distribution of ic50 for each drug
drug_ic50_dict = {drug:[] for drug in drug_name}
def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]
for idx,(cell,drug,ic50,norm_ic50) in enumerate(drug_response_dict):
###upsampling the data
## For each cell line, fit a norm distribution with N(ic_50,0.2)
    normal_dist_sampling = np.random.normal(ic50,0.2,100)
    drug_ic50_dict[drug].append(normal_dist_sampling)
drug_ic50_dict = {drug:flatten_list(drug_ic50_dict[drug]) for drug in drug_name}

# %%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# %%
import numpy as np
tmp = np.array(drug_ic50_dict[10390396])

# %%
cell_drug_resist = pd.read_excel('/home/yurui/Atten_Geom_DRP/Data/GDSC_data/TableS5C.xlsx',index_col=1, header=5)
cell_drug_resist.drop(['Unnamed: 0'], axis=1, inplace=True)
cell_drug_resist.rename(index={'Discretisation \nThreshold (log IC50)/\nCell-Lines': 'Threshold'},inplace=True)

# %%
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')

# %%
def plot_drug_ic50(drug_id, save = True):
    drug_name = drug_name_id_dict[drug_id]
    min_ic50 = np.min(drug_ic50_dict[drug_id])
    max_ic50 = np.max(drug_ic50_dict[drug_id])
    tmp = np.array(drug_ic50_dict[drug_id])
    min_int = int(min_ic50)
    max_int = int(max_ic50)
    step = 0.5
    interval = np.arange(min_int, max_int + step, step)
    s = pd.cut(tmp, bins = [x for x in interval])
    values = s.value_counts().values
    # labels = [str(i+0.5) + '-' + str(i+1) for i in np.arange(-3, 4.5,0.5)]
    labels = [str(x) + '-' + str(x+step) for x in interval[:-1]]
    df = pd.DataFrame(values, index = labels)
    df.plot(kind = 'bar', legend= False)
    # plt.hist(df, bins=interval)
    # plt.xticks(rotation=45)
    plt.xlabel('IC50')
    plt.ylabel('Frequency')
    title_font = fm.FontProperties(family='DejaVu Sans', style='normal', size=14, weight='bold', stretch='normal')
    plt.title(f'Distribution of IC50 for {drug_name}', fontproperties=title_font)
    plt.style.use('ggplot')
    # plt.figure(dpi=100, figsize=(200, 1))
    plt.autoscale(enable=True, axis='x', tight=True)
    # plt.show()
    if save :
        plt.savefig(f"./Figures/{drug_name}.svg", bbox_inches='tight')
# [plot_drug_ic50(drug_id, save = True) for drug_id in drug_name_id_dict.keys()]

# %%
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
t = 0.05
norm_t = norm.ppf(t)
def Get_threshold_drug(drug_id):
    tmp = np.array(drug_ic50_dict[drug_id])
    kde.fit(tmp[:, None])
    min_ic50 = np.min(tmp)
    max_ic50 = np.max(tmp)
    x_d = np.linspace(min_ic50, max_ic50, 1000)
    logprob = kde.score_samples(x_d[:, None])
    score = np.exp(logprob)
    max_pos = np.argmax(score.ravel())
    mu = x_d.ravel()[max_pos]
    ### Check the first derivative
    diff = np.gradient(score)
    sdiff = np.sign(diff)
    zc = np.where(sdiff[:-1] != sdiff[1:])
    if zc is not None:
        theta_pos = zc[0][0]
        theta = x_d[theta_pos]    
        cond_1 = theta < mu
        cond_2 = score[:theta_pos].sum()/score.sum() > 0.05
        cond = cond_1 & cond_2
        if cond:
            print('Theta generate by 1st derivative')
        else :
            print('Theta generate failed by 1st derivative, used f_min instead')
            theta = min_ic50
    else: 
        print('Theta generate failed by 1st derivative, used f_min instead')
        theta = min_ic50
    sigma = np.absolute(np.mean([theta,mu]))
    b = norm_t * sigma + mu ##https://www.nature.com/articles/srep36812  ic50<b means sensitive, others resistant
    return b

# %%
drug_threshold = {drug_id : Get_threshold_drug(drug_id) for drug_id in drug_name_id_dict.keys()}
np.save(os.path.join('./Data/DRP_dataset/', 'drug_threshold.npy'), drug_threshold)

# %%
drug_threshold = np.load('./Data/DRP_dataset/drug_threshold.npy',allow_pickle='TRUE').item()


# %%
drug_response_binary = []
num = 0
for line in drug_response_dict:
    cell_id, drug_id, ic50, norm_ic50 = line
    threshold = drug_threshold[drug_id]
    if ic50 < threshold:
        label = 1
        num += 1
    else: label = 0
    drug_response_binary.append([drug_id , cell_id, label])


# %%
import seaborn as sns
drug_id = 10390396.0
drug_name = drug_name_id_dict[drug_id]
min_ic50 = np.min(drug_ic50_dict[drug_id])
max_ic50 = np.max(drug_ic50_dict[drug_id])
tmp = np.array(drug_ic50_dict[drug_id])
min_int = int(min_ic50)
max_int = int(max_ic50)
step = 0.5
interval = np.arange(min_int, max_int + step, step)
s = pd.cut(tmp, bins = [x for x in interval])
values = s.value_counts().values
# labels = [str(i+0.5) + '-' + str(i+1) for i in np.arange(-3, 4.5,0.5)]
labels = [str(x) + '-' + str(x+step) for x in interval[:-1]]
# df = pd.DataFrame(values, index = labels)
sns.histplot(tmp,bins=50,kde= True, kde_kws = {'bw_adjust' : 0.5}, line_kws={'label': 'kernel density'})
# sns.histplot(tmp,bins=30)
plt.axvline(x = drug_threshold[drug_id] , linestyle='dashed', label = 'Threshold')
# plt.text(-2.2,4000, "%.4f"%drug_threshold[drug_id], fontsize=10)
plt.text(-2.7,3000, "SENSITIVE", fontsize=10)
plt.text(- 1.0,3000, "RESISTANCE", fontsize=10)
# plt.hist(df, bins=interval)
# plt.xticks(rotation=45)
plt.xlabel('LogIC50(μM)')
plt.ylabel('Frequency')
title_font = fm.FontProperties(family='DejaVu Sans', style='normal', size=14, weight='bold', stretch='normal')
plt.title(f'Distribution of IC50 for {drug_name}', fontproperties=title_font)
plt.style.use('ggplot')
# plt.figure(dpi=100, figsize=(200, 1))
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(fontsize=10, loc='upper right')
plt.show()

# %%
"Threshold = %.4f"%drug_threshold[drug_id]



