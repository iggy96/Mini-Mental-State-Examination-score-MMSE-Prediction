"""
Generate ERPs features for dementia prediction
script is used to generate the LPC data for the N200, P300, and N400 trials
these erp results are added to the training data
"""
#%%
import sys
sys.path.insert(0, '/Users/joshuaighalo/Documents/BrainNet/Projects/workspace/code_base/laurel_place')
import params as dg 
import sys
sys.path.insert(0, '/Users/joshuaighalo/Documents/BrainNet/Projects/workspace/code_base')
from fn_cfg import*
from df_lib import*

#%%
# import participant scans from backup
t1_dm = (pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/sqfResults/subjs_grp_timepoint.csv')).t1_subjsDM.to_list()
t1_dm = list(filter(lambda x: str(x) != 'nan', t1_dm))
t1_ndm = (pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/sqfResults/subjs_grp_timepoint.csv')).t1_subjsNDM.to_list()
t1_ndm = list(filter(lambda x: str(x) != 'nan', t1_ndm))
t2_dm = (pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/sqfResults/subjs_grp_timepoint.csv')).t2_subjsDM.to_list()
t2_dm = list(filter(lambda x: str(x) != 'nan', t2_dm))
t2_ndm = (pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/sqfResults/subjs_grp_timepoint.csv')).t2_subjsNDM.to_list()
t2_ndm = list(filter(lambda x: str(x) != 'nan', t2_ndm))
t3_dm = (pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/sqfResults/subjs_grp_timepoint.csv')).t3_subjsDM.to_list()
t3_dm = list(filter(lambda x: str(x) != 'nan', t3_dm))
t3_ndm = (pd.read_csv(r'/Volumes/Backup Plus/eegDatasets/laurel_place/sqfResults/subjs_grp_timepoint.csv')).t3_subjsNDM.to_list()
t3_ndm = list(filter(lambda x: str(x) != 'nan', t3_ndm))

# merge dementia timepoints into one group
dementiaGroup = list(itertools.chain(t1_dm,t2_dm,t3_dm))
non_dementiaGroup = list(itertools.chain(t1_ndm,t2_ndm,t3_ndm))
#dementiaGroup = dementiaGroup[0:3]
#non_dementiaGroup = non_dementiaGroup[0:3]

#%% function for extracting the amplitude and latency of ERPs

def lpcN200P300N400Extractor(version,filename,localPath,line,fs,Q,stimTrig,lowcut,highcut,order,clip):
    ncPipeline(version,filename,localPath,line,fs,Q,stimTrig,lowcut,highcut,order,clip,dispIMG=None,
                y_lim=None,figsize=None,pltclr=None,titles=None)
    tones_fz = (ncPipeline.fz_tones)[1]
    tones_cz = (ncPipeline.cz_tones)[1]
    tones_pz = (ncPipeline.pz_tones)[1]
    words_fz = (ncPipeline.fz_words)[1]
    words_cz = (ncPipeline.cz_words)[1]
    words_pz = (ncPipeline.pz_words)[1]
    latency = ncPipeline.erp_latency

    def meanAmplitude(temporal_range,latency_array,amplitude_array):
        t1 = temporal_range[0]
        absolute_val_array = np.abs(latency_array - t1)
        smallest_difference_index = absolute_val_array.argmin()
        closest_element = latency_array[smallest_difference_index]
        idx_1 = (np.where(latency_array==closest_element))[0]
        idx_1 = idx_1.item()

        t2 = temporal_range[1]
        absolute_val_array = np.abs(latency_array - t2)
        smallest_difference_index = absolute_val_array.argmin()
        closest_element = latency_array[smallest_difference_index]
        idx_2 = (np.where(latency_array==closest_element))[0]
        idx_2 = idx_2.item()
        newLatencyArray = latency_array[idx_1:idx_2]
        newAmplitudeArray = amplitude_array[idx_1:idx_2]
        meanAmp = np.mean(newAmplitudeArray)
        # find the value within the new amplitude array that is closest to the mean amplitude
        #absolute_val_array = np.abs(newAmplitudeArray - meanAmp)
        #smallest_difference_index = absolute_val_array.argmin()
        #closest_element = newAmplitudeArray[smallest_difference_index]
        #idx_3 = (np.where(newAmplitudeArray==closest_element))[0]
        #idx_3 = idx_3
        #latency =(newLatencyArray[idx_3])
        return meanAmp

    lpc_ampfz = (meanAmplitude([500,800],latency,words_fz))
    #lpc_latfz = (meanAmplitude([500,800],latency,words_fz))[1]
    lpc_ampcz = (meanAmplitude([500,800],latency,words_cz))
    #lpc_latcz = (meanAmplitude([500,800],latency,words_cz))[1]
    lpc_amppz = (meanAmplitude([500,800],latency,words_pz))
    #lpc_latpz = (meanAmplitude([500,800],latency,words_pz))[1]
    n200_ampfz = (meanAmplitude([180,400],latency,tones_fz))
    #n200_latfz = (meanAmplitude([180,400],latency,tones_fz))[1]
    n200_ampcz = (meanAmplitude([180,400],latency,tones_cz))
    #n200_latcz = (meanAmplitude([180,400],latency,tones_cz))[1]
    n200_amppz = (meanAmplitude([180,400],latency,tones_pz))
    #n200_latpz = (meanAmplitude([180,400],latency,tones_pz))[1]
    p300_ampfz = (meanAmplitude([250,500],latency,tones_fz))
    #p300_latfz = (meanAmplitude([250,500],latency,tones_fz))[1]
    p300_ampcz = (meanAmplitude([250,500],latency,tones_cz))
    #p300_latcz = (meanAmplitude([250,500],latency,tones_cz))[1]
    p300_amppz = (meanAmplitude([250,500],latency,tones_pz))
    #p300_latpz = (meanAmplitude([250,500],latency,tones_pz))[1]
    n400_ampfz = (meanAmplitude([300,800],latency,words_fz))
    #n400_latfz = (meanAmplitude([300,800],latency,words_fz))[1]
    n400_ampcz = (meanAmplitude([300,800],latency,words_cz))
    #n400_latcz = (meanAmplitude([300,800],latency,words_cz))[1]
    n400_amppz = (meanAmplitude([300,800],latency,words_pz))
    #n400_latpz = (meanAmplitude([300,800],latency,words_pz))[1]
    return lpc_ampfz,lpc_ampcz,lpc_amppz,n200_ampfz,n200_ampcz,n200_amppz,p300_ampfz,p300_ampcz,p300_amppz,n400_ampfz,n400_ampcz,n400_amppz

#%% dementia group LPC extractor
dmn_lpc_ampfz = []
#dmn_lpc_latfz = []
dmn_lpc_ampcz = []
#dmn_lpc_latcz = []
dmn_lpc_amppz = []
#dmn_lpc_latpz = []
dmn_n200_ampfz = []
#dmn_n200_latfz = []
dmn_n200_ampcz = []
#dmn_n200_latcz = []
dmn_n200_amppz = []
#dmn_n200_latpz = []
dmn_p300_ampfz = []
#dmn_p300_latfz = []
dmn_p300_ampcz = []
#dmn_p300_latcz = []
dmn_p300_amppz = []
#dmn_p300_latpz = []
dmn_n400_ampfz = []
#dmn_n400_latfz = []
dmn_n400_ampcz = []
#dmn_n400_latcz = []
dmn_n400_amppz = []
#dmn_n400_latpz = []
for i in range(len(dementiaGroup)):
    print(dementiaGroup[i])
    test = lpcN200P300N400Extractor(version=dg.neurocatchVersion,filename=dementiaGroup[i],
                                    localPath=dg.path,line=dg.line,fs=dg.fs,Q=dg.Q,stimTrig=dg.stimTrig,
                                    lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,clip=dg.clip)

    dmn_lpc_ampfz.append(test[0])
    dmn_lpc_ampcz.append(test[1])
    dmn_lpc_amppz.append(test[2])
    dmn_n200_ampfz.append(test[3])
    dmn_n200_ampcz.append(test[4])
    dmn_n200_amppz.append(test[5])
    dmn_p300_ampfz.append(test[6])
    dmn_p300_ampcz.append(test[7])
    dmn_p300_amppz.append(test[8])
    dmn_n400_ampfz.append(test[9])
    dmn_n400_ampcz.append(test[10])
    dmn_n400_amppz.append(test[11])





#%%
dmn_lpc_ampfz = np.hstack((np.repeat(np.array(dmn_lpc_ampfz),308)))
dmn_lpc_ampcz = np.hstack((np.repeat(np.array(dmn_lpc_ampcz),308)))
dmn_lpc_amppz = np.hstack((np.repeat(np.array(dmn_lpc_amppz),308)))
dmn_n200_ampfz = np.hstack((np.repeat(np.array(dmn_n200_ampfz),308)))
dmn_n200_ampcz = np.hstack((np.repeat(np.array(dmn_n200_ampcz),308)))
dmn_n200_amppz = np.hstack((np.repeat(np.array(dmn_n200_amppz),308)))
dmn_p300_ampfz = np.hstack((np.repeat(np.array(dmn_p300_ampfz),308)))
dmn_p300_ampcz = np.hstack((np.repeat(np.array(dmn_p300_ampcz),308)))
dmn_p300_amppz = np.hstack((np.repeat(np.array(dmn_p300_amppz),308)))
dmn_n400_ampfz = np.hstack((np.repeat(np.array(dmn_n400_ampfz),308)))
dmn_n400_ampcz = np.hstack((np.repeat(np.array(dmn_n400_ampcz),308)))
dmn_n400_amppz = np.hstack((np.repeat(np.array(dmn_n400_amppz),308)))

# replace nan with 0
dmn_lpc_ampfz = np.nan_to_num(dmn_lpc_ampfz)
dmn_lpc_ampcz = np.nan_to_num(dmn_lpc_ampcz)
dmn_lpc_amppz = np.nan_to_num(dmn_lpc_amppz)
dmn_n200_ampfz = np.nan_to_num(dmn_n200_ampfz)
dmn_n200_ampcz = np.nan_to_num(dmn_n200_ampcz)
dmn_n200_amppz = np.nan_to_num(dmn_n200_amppz)
dmn_p300_ampfz = np.nan_to_num(dmn_p300_ampfz)
dmn_p300_ampcz = np.nan_to_num(dmn_p300_ampcz)
dmn_p300_amppz = np.nan_to_num(dmn_p300_amppz)
dmn_n400_ampfz = np.nan_to_num(dmn_n400_ampfz)
dmn_n400_ampcz = np.nan_to_num(dmn_n400_ampcz)
dmn_n400_amppz = np.nan_to_num(dmn_n400_amppz)

#%%
# non-dementia group
ndm_lpc_ampfz = []
ndm_lpc_ampcz = []
ndm_lpc_amppz = []
ndm_n200_ampfz = []
ndm_n200_ampcz = []
ndm_n200_amppz = []
ndm_p300_ampfz = []
ndm_p300_ampcz = []
ndm_p300_amppz = []
ndm_n400_ampfz = []
ndm_n400_ampcz = []
ndm_n400_amppz = []
for i in range(len(non_dementiaGroup)):
    print(non_dementiaGroup[i])
    test = lpcN200P300N400Extractor(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],
                                    localPath=dg.path,line=dg.line,fs=dg.fs,Q=dg.Q,stimTrig=dg.stimTrig,
                                    lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,clip=dg.clip)
    
    ndm_lpc_ampfz.append(test[0])
    ndm_lpc_ampcz.append(test[1])
    ndm_lpc_amppz.append(test[2])
    ndm_n200_ampfz.append(test[3])
    ndm_n200_ampcz.append(test[4])
    ndm_n200_amppz.append(test[5])
    ndm_p300_ampfz.append(test[6])
    ndm_p300_ampcz.append(test[7])
    ndm_p300_amppz.append(test[8])
    ndm_n400_ampfz.append(test[9])
    ndm_n400_ampcz.append(test[10])
    ndm_n400_amppz.append(test[11])


ndm_lpc_ampfz = np.hstack((np.repeat(np.array(ndm_lpc_ampfz),308)))
ndm_lpc_ampcz = np.hstack((np.repeat(np.array(ndm_lpc_ampcz),308)))
ndm_lpc_amppz = np.hstack((np.repeat(np.array(ndm_lpc_amppz),308)))
ndm_n200_ampfz = np.hstack((np.repeat(np.array(ndm_n200_ampfz),308)))
ndm_n200_ampcz = np.hstack((np.repeat(np.array(ndm_n200_ampcz),308)))
ndm_n200_amppz = np.hstack((np.repeat(np.array(ndm_n200_amppz),308)))
ndm_p300_ampfz = np.hstack((np.repeat(np.array(ndm_p300_ampfz),308)))
ndm_p300_ampcz = np.hstack((np.repeat(np.array(ndm_p300_ampcz),308)))
ndm_p300_amppz = np.hstack((np.repeat(np.array(ndm_p300_amppz),308)))
ndm_n400_ampfz = np.hstack((np.repeat(np.array(ndm_n400_ampfz),308)))
ndm_n400_ampcz = np.hstack((np.repeat(np.array(ndm_n400_ampcz),308)))
ndm_n400_amppz = np.hstack((np.repeat(np.array(ndm_n400_amppz),308)))

# replace nan with 0
ndm_lpc_ampfz = np.nan_to_num(ndm_lpc_ampfz)
ndm_lpc_ampcz = np.nan_to_num(ndm_lpc_ampcz)
ndm_lpc_amppz = np.nan_to_num(ndm_lpc_amppz)
ndm_n200_ampfz = np.nan_to_num(ndm_n200_ampfz)
ndm_n200_ampcz = np.nan_to_num(ndm_n200_ampcz)
ndm_n200_amppz = np.nan_to_num(ndm_n200_amppz)
ndm_p300_ampfz = np.nan_to_num(ndm_p300_ampfz)
ndm_p300_ampcz = np.nan_to_num(ndm_p300_ampcz)
ndm_p300_amppz = np.nan_to_num(ndm_p300_amppz)
ndm_n400_ampfz = np.nan_to_num(ndm_n400_ampfz)
ndm_n400_ampcz = np.nan_to_num(ndm_n400_ampcz)
ndm_n400_amppz = np.nan_to_num(ndm_n400_amppz)

#%% stack dementia erps on non-dementia group erps
lpc_ampfz = np.hstack((dmn_lpc_ampfz,ndm_lpc_ampfz))
lpc_ampcz = np.hstack((dmn_lpc_ampcz,ndm_lpc_ampcz))
lpc_amppz = np.hstack((dmn_lpc_amppz,ndm_lpc_amppz))
n200_ampfz = np.hstack((dmn_n200_ampfz,ndm_n200_ampfz))
n200_ampcz = np.hstack((dmn_n200_ampcz,ndm_n200_ampcz))
n200_amppz = np.hstack((dmn_n200_amppz,ndm_n200_amppz))
p300_ampfz = np.hstack((dmn_p300_ampfz,ndm_p300_ampfz))
p300_ampcz = np.hstack((dmn_p300_ampcz,ndm_p300_ampcz))
p300_amppz = np.hstack((dmn_p300_amppz,ndm_p300_amppz))
n400_ampfz = np.hstack((dmn_n400_ampfz,ndm_n400_ampfz))
n400_ampcz = np.hstack((dmn_n400_ampcz,ndm_n400_ampcz))
n400_amppz = np.hstack((dmn_n400_amppz,ndm_n400_amppz))

#%% export  results
df1 = pd.DataFrame({"LPC Amplitude Fz":lpc_ampfz})
df1 = df1.fillna(0)
df3 = pd.DataFrame({"LPC Amplitude Cz":lpc_ampcz})
df3 = df3.fillna(0)
df5 = pd.DataFrame({"LPC Amplitude Pz":lpc_amppz})
df5 = df5.fillna(0)
df7 = pd.DataFrame({"N200 Amplitude Fz":n200_ampfz})
df7 = df7.fillna(0)
df9 = pd.DataFrame({"N200 Amplitude Cz":n200_ampcz})
df9 = df9.fillna(0)
df11 = pd.DataFrame({"N200 Amplitude Pz":n200_amppz})
df11 = df11.fillna(0)
df13 = pd.DataFrame({"P300 Amplitude Fz":p300_ampfz})
df13 = df13.fillna(0)
df15 = pd.DataFrame({"P300 Amplitude Cz":p300_ampcz})
df15 = df15.fillna(0)
df17 = pd.DataFrame({"P300 Amplitude Pz":p300_amppz})
df17 = df17.fillna(0)
df19 = pd.DataFrame({"N400 Amplitude Fz":n400_ampfz})
df19 = df19.fillna(0)
df21 = pd.DataFrame({"N400 Amplitude Cz":n400_ampcz})
df21 = df21.fillna(0)
df23 = pd.DataFrame({"N400 Amplitude Pz":n400_amppz})
df23 = df23.fillna(0)
pd.concat([df1,df3,df5,
            df7,df9,df11,
            df13,df15,df17,
            df19,df21,df23],axis=1).to_csv(r"/Volumes/Backup Plus/eegDatasets/laurel_place/trainingData/erps.csv",index=False, mode='w')
# %%
