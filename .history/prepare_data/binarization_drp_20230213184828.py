import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from prepare_data.create_drp_dict import read_dr_dict
from prepare_data.create_drug_feat import load_drug_feat
if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')
    

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]
###upsampling the data
## For each cell line, fit a norm distribution with N(ic_50,0.2)
def drug_ic5


def plot_drug_ic50(drug_id, drug_name_id_dict,drug_ic50_dict  save = True):
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