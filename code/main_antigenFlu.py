
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle

from models import rf_regr, rf_regr_avg


# In[2]:


def create_labels(in_file):
    out_file = in_file + '_theta4.csv'
    
    df = pd.read_csv(in_file + '.csv')
    
    df['Label'] = df['Distance'] > 4
    
    df.to_csv(out_file)
    
    return True


# In[32]:


def dist_file_subtype(subtype):
    df = pd.read_csv('../data/' + subtype + '_dist_theta4.csv')
    
    viral_pair_subtype = []
    for index, row in df.iterrows():
        vnames = row['ViralPair'].split(',')
        row['ViralPair'] = vnames[0] + '/' + subtype + ',' + vnames[1] + '/' + subtype

        viral_pair_subtype.append(row['ViralPair'])
        
    df['ViralPair'] = pd.DataFrame(data = viral_pair_subtype, index = df.index)
    
    return df
    


# In[33]:


# dist_file_subtype('H1N1')


# In[38]:


def merge_dist_files():
      
    df_h1n1 = dist_file_subtype('H1N1')
    df_h3n2 = dist_file_subtype('H3N2')
    df_h5n1 = dist_file_subtype('H5N1')
    df_h9n2 = dist_file_subtype('H9N2')
    
    frames = [df_h1n1, df_h3n2, df_h5n1, df_h9n2]
    
    df = pd.concat(frames).drop(['Unnamed: 0'], axis = 1)
    
    df.to_csv('../data/mixed_dist_theta4.csv')
    
    return True


# In[39]:


# merge_dist_files()


# In[40]:


# create_labels('../data/H1N1_dist')
# create_labels('../data/H5N1_dist')
# create_labels('../data/H9N2_dist')
# create_labels('../data/H3N2_dist')


# In[41]:


def remove_inconsistency(dist, seqs):
        
    dist_viral_pairs = dist['ViralPair']
    dist_viral_pairs_list = list(dist_viral_pairs)
    
    seqs_viral_pairs = seqs['ViralPair']
    # print(len(seqs_viral_pairs))
    # seqs_viral_pairs_list = list(seqs_viral_pairs)
    
    distances = []
    labels = []
    
    for i in range(len(seqs_viral_pairs)):
        # print(i)
        vpname = seqs_viral_pairs[i]
        vnames = vpname.split(',')
        alias = vnames[1] + ',' + vnames[0]
        
        a1 = vpname in dist_viral_pairs_list
        a2 = alias in dist_viral_pairs_list
        
        if a1:
            d = dist.loc[dist['ViralPair'] == vpname]['Distance']
            distances.append(d.values[0])
            l = dist.loc[dist['ViralPair'] == vpname]['Label']
            if l.values[0] == True:
                labels.append(1)
            else:
                labels.append(0)
        elif a2:
            d = dist.loc[dist['ViralPair'] == alias]['Distance']
            distances.append(d.values[0])
            l = dist.loc[dist['ViralPair'] == alias]['Label']
            if l.values[0] == True:
                labels.append(1)
            else:
                labels.append(0)
            # labels.append(l.values[0])
        else:
            distances.append(pd.np.nan)
            labels.append(pd.np.nan)
            
    seqs['Distance'] = pd.DataFrame(data = distances, index = seqs.index)
    seqs['Label'] = pd.DataFrame(data = labels, index = seqs.index)
    
    return seqs


# In[42]:


def get_inputs(dist_file, seqs_file):
    
    df_dist = pd.read_csv(dist_file).drop(['Unnamed: 0'], axis = 1)
    df_seqs = pd.read_csv(seqs_file).rename(index = str, columns = {'Unnamed: 0': 'ViralPair'})
    
    df = remove_inconsistency(df_dist, df_seqs)
    df = df.dropna()
    
    x = df.iloc[:, 1:-2]
    
    corr = x.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(np.bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    # print("len(to_drop): %s" % str(len(to_drop)))

    x = x.drop(to_drop, axis = 1)    
    
    d = df.iloc[:, -2]
    y = df.iloc[:, -1]
    
    return x, d, y


# In[43]:


# dist_file = '../data/H9N2_dist_theta4.csv'
# seqs_file = '../data/H9N2_seqs_WEIL970102.csv'

# x, d, y = get_inputs(dist_file, seqs_file)
# rf_regr(x, d, '../result/H9N2_WEIL970102.svg')
# avg_accu, avg_prec, avg_rec, avg_f1 = rf_regr_avg(x, d, 5, '../result/H9N2_WEIL970102')


# In[44]:


def idxList_exp(subtype):
    with open('./aaindex/indexKeyList_2_pickle', 'rb') as f:
        idxList = pickle.load(f)
    
    out_file = '../result/result_'+subtype+'.csv'
    handle = open(out_file, 'w')
    handle.write('IndexId,Accuracy,Precision,Recall,F-score\n')
    
    dist_file = '../data/'+subtype+'_dist_theta4.csv'
    for idx in idxList:
        seqs_file = '../data/'+subtype+'_seqs_'+idx+'.csv'
        x, d, y = get_inputs(dist_file, seqs_file)
        accu, prec, rec, f1 = rf_regr_avg(x, d, 5, subtype + '_' + idx)
        
        handle.write(idx + ',' + str(accu) + ',' + str(prec) + ',' + str(rec) + ',' + str(f1) + '\n')
        
    handle.close()
    return True


# In[45]:


def exp_subtypes(subtype_list):
    for s in subtype_list:
        idxList_exp(s)
    return True

