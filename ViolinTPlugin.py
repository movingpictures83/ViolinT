#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sympy.interactive import printing
printing.init_printing(use_latex=True)
sns.set(rc={'figure.figsize':(8,6)})
from IPython.display import Image
import os
import numpy as np
from scipy import stats
import csv
all_colors=["orange", "blue","red","black","#2ecc71", "#2e0071",  "#2efdaa"]
subset_colors =  ["red","black","#2ecc71", "#2e0071",  "#2efdaa", "#200daa","#2ffd00"]
include_plots = ['arc','alecar3']
import os.path as path
import xlwt 
from xlwt import Workbook 

#np.random.seed(12345678)


# In[16]:


def writeHeader(sheet1, our_algo, other_algo):
    row = 0
    col= 1
    sheet1.write(row, col, "dataset") 
    col= col+ 1
    sheet1.write(row, col,  "algorithm")
    col= col+ 1
    sheet1.write(row, col,   "cache_size") 
    col= col+ 1
    sheet1.write(row, col, "our_algo_mean") 
    col= col+ 1
    sheet1.write(row, col, "other_algo_mean") 
    col= col+ 1
    sheet1.write(row, col, "other_algo_std") 
    col= col+ 1 
    sheet1.write(row,  col,  "other_algo_std") 
    col= col+ 1 
    sheet1.write(row,  col, "p-value")
    col= col+ 1 
    sheet1.write(row, col,  "color") 
    
def writeInCsv(sheet1,row, df_cache, our_algo, other_algo, datas, cache_size):
    df_our_algo=df_cache[(df_cache['algo']==our_algo)].hit_rate.to_numpy()
    
    df_other_algo=df_cache[(df_cache['algo']==other_algo)].hit_rate.to_numpy()
    print(len(df_our_algo))
    print(len(df_other_algo))
    
    
    t2, p2  = stats.ttest_rel(df_our_algo, df_other_algo)
    
    our_algo_mean = np.mean(np.array(df_our_algo))
    other_algo_mean = np.mean(np.array(df_other_algo))
    # Calculate the standard deviation
    our_algo_std = np.std(np.array(df_our_algo), ddof=1)
    other_algo_std = np.std(np.array(df_other_algo), ddof=1)

    our_algo_sem =  stats.sem(np.array(df_our_algo))
    other_algo_sem =  stats.sem(np.array(df_other_algo))
    print( "*****Dataset:" , datas , "*****Cache Size:" , cache_size , "*******")
    print(our_algo ," Average = " , our_algo_mean, "Standard deviation = " , our_algo_std)
    print("Standard error estimated =", our_algo_sem)
    
    print(other_algo, " Average = " , other_algo_mean, "Standard deviation = " , other_algo_std)
    print("Standard error estimated = ", other_algo_sem)

    print("t-test with respect to", other_algo)
    print("t = " + str(t2))
    print("p-value = " + str(p2))

    color = 0 if p2>0.05  else (1 if our_algo_mean> other_algo_mean  else -1)

    col= 1
    sheet1.write(row, col, datas) 
    col= col+ 1
    sheet1.write(row, col,  other_algo)
    col= col+ 1
    sheet1.write(row, col,   cache_size) 
    col= col+ 1
    sheet1.write(row, col, our_algo_mean) 
    col= col+ 1
    sheet1.write(row, col, other_algo_mean) 
    col= col+ 1
    sheet1.write(row, col, other_algo_std) 
    col= col+ 1 
    sheet1.write(row,  col,  other_algo_std) 
    col= col+ 1 
    sheet1.write(row,col , round(p2,3))
    col= col+ 1 
    sheet1.write(row, col,  color) 
    
    


# In[95]:

class ViolinTPlugin:
 def input(self, inputfile):
  self.infile = inputfile
 def run(self):
     pass
 def output(self, outputfile):
  df = pd.read_csv(self.infile, header=None)
  df.columns = ['traces', 'trace_name', 'algo', 'hits', 'misses', 'writes', 'filters', 
                   'size', 'cache_size', 'requestcount', 'hit_rate', 'time', 'dataset']

  df = df.sort_values(['dataset', 'cache_size', 
                    'traces', 'hit_rate'], ascending=[True, True, True, False])

  our_algo = "cacheus"
  other_algos = ["dlirs"]

# df = pd.read_excel('data/final_results.xlsx')
# our_algo = "ScanALeCaR"
# other_algos = [ 'ARC', "LIRS", "DLIRS", "LeCaR", "ALeCaR2N", "ALeCaRN"]


# our_algo = "ALeCaRN"
# other_algos = [ 'ARC', "LIRS", "DLIRS", "LeCaR", "ALeCaR2N", "ScanALeCaR"]
# #print(df_all)
            
  wb = Workbook() 

  data_list=[]
  other_algo_list=[]
  our_algo_list=[]
  cache_size_list=[]
  dataset_list=[]

  # add_sheet is used to create sheet. 
  filename = our_algo + ' t-test results'
  sheet1 = wb.add_sheet(filename) 
  datasets = df["dataset"].unique()
  row=1
  writeHeader(sheet1, our_algo, "Other Algorithm")
  for other_algo in other_algos:
#     sheet1.write(row, 0, other_algo) 
#     row= row+1
    for datas in datasets:
        df_data= df[ df["dataset"] == datas]
        cache_sizes = df_data["cache_size"].unique()
        for cache_size in cache_sizes:
            df_cache = df_data[(df_data["cache_size"] == cache_size) ]
            
            print(cache_size, datas, other_algo)
            
#             t_test_results.append(l)
            diff = writeInCsv(sheet1,row, df_cache, our_algo, other_algo, datas, cache_size)
            row= row+1
            
            df_our_algo=df_cache[(df_cache['algo']==our_algo)].hit_rate.to_numpy()
            df_other_algo=df_cache[(df_cache['algo']==other_algo)].hit_rate.to_numpy()
            
            data = [a_i - b_i for a_i, b_i in zip(list(df_our_algo), list(df_other_algo))] 
            data_list.append(data)
            
            for item in data:
                other_algo_list.append(other_algo)
                our_algo_list.append(our_algo)
                cache_size_list.append(cache_size)
                dataset_list.append(datas)
        
            
  diff = pd.DataFrame(columns = ['hit_rate_diff']) 
  flat_list = []
  for sublist in data_list:
    for item in sublist:
        flat_list.append(item)
        
  diff['hit_rate_diff'] = flat_list
  diff['other_algo'] = other_algo_list
  diff['our_algo'] = our_algo_list
  diff['dataset'] = dataset_list
  diff['cache_size'] = cache_size_list
  diff['algo_dataset_cache-size'] = list(zip(diff['other_algo'], 
                                           diff['dataset'], diff['cache_size']))
  diff['algo_cache-size'] = list(zip(list(diff['other_algo']), list(diff['cache_size'])))
  diff['dataset_cache-size'] = list(zip(list(diff['dataset']), list(diff['cache_size'])))


  #sheet1.write(row, datas, cache_size, alecar_mean, alecar_std,scanalecar_mean, scanalecar_std,t2, p2) 
  
  wb.save(our_algo+' t-test results.xls')


  # In[96]:


  #diff_MSR = diff[diff['dataset'] == 'MSR']
  #print(diff_MSR)  
  #diff_MSR.to_csv('diff_MSR.csv', index=False)

  fig, axes = plt.subplots()
  fig.set_size_inches(30, 6)
  fontsize = 20

  #fig.set_size_inches(11.7, 8.27)
  print(diff.isnull().values.any())
  diff = diff.dropna()

  print(diff.isnull().values.any())

  # diff.to_csv('out.csv')
  diff['hit_rate_diff'] =  diff['hit_rate_diff'][(np.isnan(diff['hit_rate_diff']) == False) 
                                               & (np.isinf(diff['hit_rate_diff']) == False)]

  datas = diff['dataset'].unique()
  label = ["0.05", "0.1", "0.5", "1", "5", "10"]
  plt.subplots_adjust(wspace=0.01, hspace=0)

  for i, trace in enumerate(datas):
    plot_no = 0
    index = i
    axes = plt.subplot2grid((1, 5), (plot_no, index))
    if i == 0:
        axes.set_ylim(-25, 25)
    elif i <= 3:
        axes.set_ylim(-25, 25)
        axes.set_yticklabels([])
    else:
        axes.set_ylim(-25, 25)
        axes.set_yticklabels([])
        
    diff_trace = diff[diff['dataset'] == trace]
    #diff_trace = diff[diff['other_algo'] == 'lirs']
    #print(trace, diff_trace['algo_cache-size'])
    #print(diff_trace['cache_size'])
    #print(diff_trace['algo_dataset_cache-size'])
    #print(diff_trace)

    print(len(diff['hit_rate_diff']))
    sns.violinplot('dataset_cache-size', 'hit_rate_diff', data = diff_trace, ax = axes, fontsize=20)
    
    #axes.set_title(trace, fontsize=30)
    axes.yaxis.grid(True)
    axes.set_xlabel('')
    axes.set_ylabel('')
    axes.set_xticklabels([])
    axes.tick_params(axis='y', labelsize=20) 
    #axes.set_xticklabels(label, fontsize=fontsize)

  # diff['hit_rate_diff'] =  diff['hit_rate_diff'][(np.isnan(diff['hit_rate_diff']) == False) 
  #                                                & (np.isinf(diff['hit_rate_diff']) == False)]
  # diff['hit_rate_diff'].plot(kind='kde')

  plt.savefig(outputfile, bbox_inches="tight", dpi=300)
  plt.show()


# In[ ]:





# In[ ]:




