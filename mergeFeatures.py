"""
This script implements the following:
-import all three features: band power, ERPs, and peak frequency & power
-merges columns of these features horizontally
-adds the label column
"""


#%%
import sys
sys.path.insert(0, '/Users/joshuaighalo/Documents/BrainNet/Projects/workspace/code_base/laurel_place')
import params as dg 
import sys
sys.path.insert(0, '/Users/joshuaighalo/Documents/BrainNet/Projects/workspace/code_base')
from fn_cfg import*
from df_lib import*
c = list(np.arange(16,64,16))
#%% import features csv files
bandPower = pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/trainingData/bandPower.csv')
erps = pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/trainingData/erps.csv')
#f = np.mean(erps.values, axis=1)
peakFreqPwr = pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/trainingData/peakFreqPwr.csv')
mmse_dmn =  (pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/trainingData/mmse.csv')).dementia.to_list()
mmse_dmn =  [item for item in (list(filter(lambda x: str(x) != 'nan', mmse_dmn))) for i in range(308)]


def classifyMCI(x):
    # Crum RM, Anthony JC, Bassett SS, Folstein MF; Anthony; Bassett; Folstein (May 1993). 
    # "Population-based norms for the Mini-Mental Status Examination by age and educational level". 
    # JAMA. 269 (18): 2386–91. doi:10.1001/jama.1993.03500180078038. PMID 8479064.
    # SCI = severe cognitive impairment; MOCI = moderate cognitive impairment; MICI = mild cognitive impairment; NOCI = no cognitive impairment
    if x <= 9:
        status = 'SCI'
    elif x >= 10 & x <= 18:
        status = 'MOCI' 
    elif x >= 19 & x <= 23:
        status = 'MICI'   
    elif x >= 24 & x <= 30:
        status = 'NOCI'   
    return status
    
    




mmse_ndm =  (pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/trainingData/mmse.csv')).non_dementia.to_list()
mmse_ndm = [item for item in (list(filter(lambda x: str(x) != 'nan', mmse_ndm))) for i in range(308)]
targets = [mmse_dmn,mmse_ndm]
targets = list(itertools.chain(*targets))
#targets = [['dementia']*19712,['nondementia']*10164]
#targets = list(itertools.chain(*targets))
targets = pd.DataFrame(targets,columns = ['target'])
#%% merge features csv 
pd.concat([bandPower,erps,peakFreqPwr,targets],axis=1).to_csv(r"/Volumes/Backup Plus/eegDatasets/laurel_place/trainingData/trainData.csv",index=False, mode='w')
