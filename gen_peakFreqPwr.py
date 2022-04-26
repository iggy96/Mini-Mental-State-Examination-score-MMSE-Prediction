"""
Peak Frequency Generator for Dementia Prediction
script is used to generate peak frequency per scan for the three channels of neurocatch
The peak frequencies are exported to the training data via csv file
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

#%% functions for generating peak frequency & peak power for each channel
def channelPeakFreqPwr(version,filename,localPath,line,fs,Q,stimTrig,lowcut,highcut,order,win,low,high):
    ncPipeline(version,filename,localPath,line,fs,Q,stimTrig,lowcut,highcut,order,clip=None,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    def peakFrequencyPower(eegData,fs,win,low,high):
        freqs, psd = signal.welch(eegData,fs,nperseg=win)
        idx_freqBands = np.logical_and(freqs >= low, freqs <= high) 
        freqBands = freqs[idx_freqBands]
        powerBands = psd[idx_freqBands]
        maxPower = np.max(powerBands)
        idx_maxPowerBands = np.argmax(powerBands)
        maxFreq = freqBands[idx_maxPowerBands]
        return maxPower, maxFreq    
    
    eeg = ncPipeline.bandPassFOutput
    fz_frq, fz_pwr = peakFrequencyPower(eeg[:,0],fs,win,low,high)
    cz_frq, cz_pwr = peakFrequencyPower(eeg[:,1],fs,win,low,high)
    pz_frq, pz_pwr = peakFrequencyPower(eeg[:,2],fs,win,low,high)
    return fz_frq, fz_pwr, cz_frq, cz_pwr, pz_frq, pz_pwr
#%% generate peak frequency and peak power for dementia group
fz_deltaPkFrq_dmn=[]
fz_deltaPkPwr_dmn=[]
cz_deltaPkFrq_dmn=[]
cz_deltaPkPwr_dmn=[]
pz_deltaPkFrq_dmn=[]
pz_deltaPkPwr_dmn=[]
for i in range(len(dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['delta'][0],
                high=dg.brainWaves['delta'][1])
    fz_deltaPkFrq_dmn.append(load[0])
    fz_deltaPkPwr_dmn.append(load[1])
    cz_deltaPkFrq_dmn.append(load[2])
    cz_deltaPkPwr_dmn.append(load[3])
    pz_deltaPkFrq_dmn.append(load[4])
    pz_deltaPkPwr_dmn.append(load[5])
fz_deltaPkFrq_dmn = np.asarray([item for item in fz_deltaPkFrq_dmn for i in range(308)])
fz_deltaPkPwr_dmn = np.asarray([item for item in fz_deltaPkPwr_dmn for i in range(308)])
cz_deltaPkFrq_dmn = np.asarray([item for item in cz_deltaPkFrq_dmn for i in range(308)])
cz_deltaPkPwr_dmn = np.asarray([item for item in cz_deltaPkPwr_dmn for i in range(308)])
pz_deltaPkFrq_dmn = np.asarray([item for item in pz_deltaPkFrq_dmn for i in range(308)])
pz_deltaPkPwr_dmn = np.asarray([item for item in pz_deltaPkPwr_dmn for i in range(308)])

# generate peak delta frequency and peak power for non-dementia group
fz_deltaPkFrq_ndm=[]
fz_deltaPkPwr_ndm=[]
cz_deltaPkFrq_ndm=[]
cz_deltaPkPwr_ndm=[]
pz_deltaPkFrq_ndm=[]
pz_deltaPkPwr_ndm=[]
for i in range(len(non_dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['delta'][0],
                high=dg.brainWaves['delta'][1])
    fz_deltaPkFrq_ndm.append(load[0])
    fz_deltaPkPwr_ndm.append(load[1])
    cz_deltaPkFrq_ndm.append(load[2])
    cz_deltaPkPwr_ndm.append(load[3])
    pz_deltaPkFrq_ndm.append(load[4])
    pz_deltaPkPwr_ndm.append(load[5])
fz_deltaPkFrq_ndm = np.asarray([item for item in fz_deltaPkFrq_ndm for i in range(308)])
fz_deltaPkPwr_ndm = np.asarray([item for item in fz_deltaPkPwr_ndm for i in range(308)])
cz_deltaPkFrq_ndm = np.asarray([item for item in cz_deltaPkFrq_ndm for i in range(308)])
cz_deltaPkPwr_ndm = np.asarray([item for item in cz_deltaPkPwr_ndm for i in range(308)])
pz_deltaPkFrq_ndm = np.asarray([item for item in pz_deltaPkFrq_ndm for i in range(308)])
pz_deltaPkPwr_ndm = np.asarray([item for item in pz_deltaPkPwr_ndm for i in range(308)])

#%% generate peak theta frequency and peak power for dementia group
fz_thetaPkFrq_dmn=[]
fz_thetaPkPwr_dmn=[]
cz_thetaPkFrq_dmn=[]
cz_thetaPkPwr_dmn=[]
pz_thetaPkFrq_dmn=[]
pz_thetaPkPwr_dmn=[]
for i in range(len(dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['theta'][0],
                high=dg.brainWaves['theta'][1])
    fz_thetaPkFrq_dmn.append(load[0])
    fz_thetaPkPwr_dmn.append(load[1])
    cz_thetaPkFrq_dmn.append(load[2])
    cz_thetaPkPwr_dmn.append(load[3])
    pz_thetaPkFrq_dmn.append(load[4])
    pz_thetaPkPwr_dmn.append(load[5])
fz_thetaPkFrq_dmn = np.asarray([item for item in fz_thetaPkFrq_dmn for i in range(308)])
fz_thetaPkPwr_dmn = np.asarray([item for item in fz_thetaPkPwr_dmn for i in range(308)])
cz_thetaPkFrq_dmn = np.asarray([item for item in cz_thetaPkFrq_dmn for i in range(308)])
cz_thetaPkPwr_dmn = np.asarray([item for item in cz_thetaPkPwr_dmn for i in range(308)])
pz_thetaPkFrq_dmn = np.asarray([item for item in pz_thetaPkFrq_dmn for i in range(308)])
pz_thetaPkPwr_dmn = np.asarray([item for item in pz_thetaPkPwr_dmn for i in range(308)])

# generate peak theta frequency and peak power for non-dementia group
fz_thetaPkFrq_ndm=[]
fz_thetaPkPwr_ndm=[]
cz_thetaPkFrq_ndm=[]
cz_thetaPkPwr_ndm=[]
pz_thetaPkFrq_ndm=[]
pz_thetaPkPwr_ndm=[]
for i in range(len(non_dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['theta'][0],
                high=dg.brainWaves['theta'][1])
    fz_thetaPkFrq_ndm.append(load[0])
    fz_thetaPkPwr_ndm.append(load[1])
    cz_thetaPkFrq_ndm.append(load[2])
    cz_thetaPkPwr_ndm.append(load[3])
    pz_thetaPkFrq_ndm.append(load[4])
    pz_thetaPkPwr_ndm.append(load[5])
fz_thetaPkFrq_ndm = np.asarray([item for item in fz_thetaPkFrq_ndm for i in range(308)])
fz_thetaPkPwr_ndm = np.asarray([item for item in fz_thetaPkPwr_ndm for i in range(308)])
cz_thetaPkFrq_ndm = np.asarray([item for item in cz_thetaPkFrq_ndm for i in range(308)])
cz_thetaPkPwr_ndm = np.asarray([item for item in cz_thetaPkPwr_ndm for i in range(308)])
pz_thetaPkFrq_ndm = np.asarray([item for item in pz_thetaPkFrq_ndm for i in range(308)])
pz_thetaPkPwr_ndm = np.asarray([item for item in pz_thetaPkPwr_ndm for i in range(308)])


#%% generate peak alpha frequency and peak power for dementia group
fz_alphaPkFrq_dmn=[]
fz_alphaPkPwr_dmn=[]
cz_alphaPkFrq_dmn=[]
cz_alphaPkPwr_dmn=[]
pz_alphaPkFrq_dmn=[]
pz_alphaPkPwr_dmn=[]

for i in range(len(dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['alpha'][0],
                high=dg.brainWaves['alpha'][1])
    fz_alphaPkFrq_dmn.append(load[0])
    fz_alphaPkPwr_dmn.append(load[1])
    cz_alphaPkFrq_dmn.append(load[2])
    cz_alphaPkPwr_dmn.append(load[3])
    pz_alphaPkFrq_dmn.append(load[4])
    pz_alphaPkPwr_dmn.append(load[5])
fz_alphaPkFrq_dmn = np.asarray([item for item in fz_alphaPkFrq_dmn for i in range(308)])
fz_alphaPkPwr_dmn = np.asarray([item for item in fz_alphaPkPwr_dmn for i in range(308)])
cz_alphaPkFrq_dmn = np.asarray([item for item in cz_alphaPkFrq_dmn for i in range(308)])
cz_alphaPkPwr_dmn = np.asarray([item for item in cz_alphaPkPwr_dmn for i in range(308)])
pz_alphaPkFrq_dmn = np.asarray([item for item in pz_alphaPkFrq_dmn for i in range(308)])
pz_alphaPkPwr_dmn = np.asarray([item for item in pz_alphaPkPwr_dmn for i in range(308)])

# generate peak alpha frequency and peak power for non-dementia group
fz_alphaPkFrq_ndm=[]
fz_alphaPkPwr_ndm=[]
cz_alphaPkFrq_ndm=[]
cz_alphaPkPwr_ndm=[]
pz_alphaPkFrq_ndm=[]
pz_alphaPkPwr_ndm=[]
for i in range(len(non_dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['alpha'][0],
                high=dg.brainWaves['alpha'][1])
    fz_alphaPkFrq_ndm.append(load[0])
    fz_alphaPkPwr_ndm.append(load[1])
    cz_alphaPkFrq_ndm.append(load[2])
    cz_alphaPkPwr_ndm.append(load[3])
    pz_alphaPkFrq_ndm.append(load[4])
    pz_alphaPkPwr_ndm.append(load[5])
fz_alphaPkFrq_ndm = np.asarray([item for item in fz_alphaPkFrq_ndm for i in range(308)])
fz_alphaPkPwr_ndm = np.asarray([item for item in fz_alphaPkPwr_ndm for i in range(308)])
cz_alphaPkFrq_ndm = np.asarray([item for item in cz_alphaPkFrq_ndm for i in range(308)])
cz_alphaPkPwr_ndm = np.asarray([item for item in cz_alphaPkPwr_ndm for i in range(308)])
pz_alphaPkFrq_ndm = np.asarray([item for item in pz_alphaPkFrq_ndm for i in range(308)])
pz_alphaPkPwr_ndm = np.asarray([item for item in pz_alphaPkPwr_ndm for i in range(308)])


#%% generate peak beta frequency and peak power for dementia group
fz_betaPkFrq_dmn=[]
fz_betaPkPwr_dmn=[]
cz_betaPkFrq_dmn=[]
cz_betaPkPwr_dmn=[]
pz_betaPkFrq_dmn=[]
pz_betaPkPwr_dmn=[]
for i in range(len(dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['beta'][0],
                high=dg.brainWaves['beta'][1])
    fz_betaPkFrq_dmn.append(load[0])
    fz_betaPkPwr_dmn.append(load[1])
    cz_betaPkFrq_dmn.append(load[2])
    cz_betaPkPwr_dmn.append(load[3])
    pz_betaPkFrq_dmn.append(load[4])
    pz_betaPkPwr_dmn.append(load[5])
fz_betaPkFrq_dmn = np.asarray([item for item in fz_betaPkFrq_dmn for i in range(308)])
fz_betaPkPwr_dmn = np.asarray([item for item in fz_betaPkPwr_dmn for i in range(308)])
cz_betaPkFrq_dmn = np.asarray([item for item in cz_betaPkFrq_dmn for i in range(308)])
cz_betaPkPwr_dmn = np.asarray([item for item in cz_betaPkPwr_dmn for i in range(308)])
pz_betaPkFrq_dmn = np.asarray([item for item in pz_betaPkFrq_dmn for i in range(308)])
pz_betaPkPwr_dmn = np.asarray([item for item in pz_betaPkPwr_dmn for i in range(308)])

# generate peak beta frequency and peak power for non-dementia group
fz_betaPkFrq_ndm=[]
fz_betaPkPwr_ndm=[]
cz_betaPkFrq_ndm=[]
cz_betaPkPwr_ndm=[]
pz_betaPkFrq_ndm=[]
pz_betaPkPwr_ndm=[]
for i in range(len(non_dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['beta'][0],
                high=dg.brainWaves['beta'][1])
    fz_betaPkFrq_ndm.append(load[0])
    fz_betaPkPwr_ndm.append(load[1])
    cz_betaPkFrq_ndm.append(load[2])
    cz_betaPkPwr_ndm.append(load[3])
    pz_betaPkFrq_ndm.append(load[4])
    pz_betaPkPwr_ndm.append(load[5])
fz_betaPkFrq_ndm = np.asarray([item for item in fz_betaPkFrq_ndm for i in range(308)])
fz_betaPkPwr_ndm = np.asarray([item for item in fz_betaPkPwr_ndm for i in range(308)])
cz_betaPkFrq_ndm = np.asarray([item for item in cz_betaPkFrq_ndm for i in range(308)])
cz_betaPkPwr_ndm = np.asarray([item for item in cz_betaPkPwr_ndm for i in range(308)])
pz_betaPkFrq_ndm = np.asarray([item for item in pz_betaPkFrq_ndm for i in range(308)])
pz_betaPkPwr_ndm = np.asarray([item for item in pz_betaPkPwr_ndm for i in range(308)])


#%% generate peak gamma frequency and peak power for dementia group
fz_gammaPkFrq_dmn=[]
fz_gammaPkPwr_dmn=[]
cz_gammaPkFrq_dmn=[]
cz_gammaPkPwr_dmn=[]
pz_gammaPkFrq_dmn=[]
pz_gammaPkPwr_dmn=[]
for i in range(len(dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['gamma'][0],
                high=dg.brainWaves['gamma'][1])
    fz_gammaPkFrq_dmn.append(load[0])
    fz_gammaPkPwr_dmn.append(load[1])
    cz_gammaPkFrq_dmn.append(load[2])
    cz_gammaPkPwr_dmn.append(load[3])
    pz_gammaPkFrq_dmn.append(load[4])
    pz_gammaPkPwr_dmn.append(load[5])
fz_gammaPkFrq_dmn = np.asarray([item for item in fz_gammaPkFrq_dmn for i in range(308)])
fz_gammaPkPwr_dmn = np.asarray([item for item in fz_gammaPkPwr_dmn for i in range(308)])
cz_gammaPkFrq_dmn = np.asarray([item for item in cz_gammaPkFrq_dmn for i in range(308)])
cz_gammaPkPwr_dmn = np.asarray([item for item in cz_gammaPkPwr_dmn for i in range(308)])
pz_gammaPkFrq_dmn = np.asarray([item for item in pz_gammaPkFrq_dmn for i in range(308)])
pz_gammaPkPwr_dmn = np.asarray([item for item in pz_gammaPkPwr_dmn for i in range(308)])


# generate peak gamma frequency and peak power for non-dementia group
fz_gammaPkFrq_ndm=[]
fz_gammaPkPwr_ndm=[]
cz_gammaPkFrq_ndm=[]
cz_gammaPkPwr_ndm=[]
pz_gammaPkFrq_ndm=[]
pz_gammaPkPwr_ndm=[]
for i in range(len(non_dementiaGroup)):
    load = channelPeakFreqPwr(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,fs=dg.fs,
                Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,win=dg.win,low=dg.brainWaves['gamma'][0],
                high=dg.brainWaves['gamma'][1])
    fz_gammaPkFrq_ndm.append(load[0])
    fz_gammaPkPwr_ndm.append(load[1])
    cz_gammaPkFrq_ndm.append(load[2])
    cz_gammaPkPwr_ndm.append(load[3])
    pz_gammaPkFrq_ndm.append(load[4])
    pz_gammaPkPwr_ndm.append(load[5])
fz_gammaPkFrq_ndm = np.asarray([item for item in fz_gammaPkFrq_ndm for i in range(308)])
fz_gammaPkPwr_ndm = np.asarray([item for item in fz_gammaPkPwr_ndm for i in range(308)])
cz_gammaPkFrq_ndm = np.asarray([item for item in cz_gammaPkFrq_ndm for i in range(308)])
cz_gammaPkPwr_ndm = np.asarray([item for item in cz_gammaPkPwr_ndm for i in range(308)])
pz_gammaPkFrq_ndm = np.asarray([item for item in pz_gammaPkFrq_ndm for i in range(308)])
pz_gammaPkPwr_ndm = np.asarray([item for item in pz_gammaPkPwr_ndm for i in range(308)])

#%% concatenate dementia groups with non-dementia groups
fz_deltaPkFrq = np.hstack((fz_deltaPkFrq_dmn,fz_deltaPkFrq_ndm))
fz_deltaPkPwr = np.hstack((fz_deltaPkPwr_dmn,fz_deltaPkPwr_ndm))
cz_deltaPkFrq = np.hstack((cz_deltaPkFrq_dmn,cz_deltaPkFrq_ndm))
cz_deltaPkPwr = np.hstack((cz_deltaPkPwr_dmn,cz_deltaPkPwr_ndm))
pz_deltaPkFrq = np.hstack((pz_deltaPkFrq_dmn,pz_deltaPkFrq_ndm))
pz_deltaPkPwr = np.hstack((pz_deltaPkPwr_dmn,pz_deltaPkPwr_ndm))
fz_thetaPkFrq = np.hstack((fz_thetaPkFrq_dmn,fz_thetaPkFrq_ndm))
fz_thetaPkPwr = np.hstack((fz_thetaPkPwr_dmn,fz_thetaPkPwr_ndm))
cz_thetaPkFrq = np.hstack((cz_thetaPkFrq_dmn,cz_thetaPkFrq_ndm))
cz_thetaPkPwr = np.hstack((cz_thetaPkPwr_dmn,cz_thetaPkPwr_ndm))
pz_thetaPkFrq = np.hstack((pz_thetaPkFrq_dmn,pz_thetaPkFrq_ndm))
pz_thetaPkPwr = np.hstack((pz_thetaPkPwr_dmn,pz_thetaPkPwr_ndm))
fz_alphaPkFrq = np.hstack((fz_alphaPkFrq_dmn,fz_alphaPkFrq_ndm))
fz_alphaPkPwr = np.hstack((fz_alphaPkPwr_dmn,fz_alphaPkPwr_ndm))
cz_alphaPkFrq = np.hstack((cz_alphaPkFrq_dmn,cz_alphaPkFrq_ndm))
cz_alphaPkPwr = np.hstack((cz_alphaPkPwr_dmn,cz_alphaPkPwr_ndm))
pz_alphaPkFrq = np.hstack((pz_alphaPkFrq_dmn,pz_alphaPkFrq_ndm))
pz_alphaPkPwr = np.hstack((pz_alphaPkPwr_dmn,pz_alphaPkPwr_ndm))
fz_betaPkFrq = np.hstack((fz_betaPkFrq_dmn,fz_betaPkFrq_ndm))
fz_betaPkPwr = np.hstack((fz_betaPkPwr_dmn,fz_betaPkPwr_ndm))
cz_betaPkFrq = np.hstack((cz_betaPkFrq_dmn,cz_betaPkFrq_ndm))
cz_betaPkPwr = np.hstack((cz_betaPkPwr_dmn,cz_betaPkPwr_ndm))
pz_betaPkFrq = np.hstack((pz_betaPkFrq_dmn,pz_betaPkFrq_ndm))
pz_betaPkPwr = np.hstack((pz_betaPkPwr_dmn,pz_betaPkPwr_ndm))
fz_gammaPkFrq = np.hstack((fz_gammaPkFrq_dmn,fz_gammaPkFrq_ndm))
fz_gammaPkPwr = np.hstack((fz_gammaPkPwr_dmn,fz_gammaPkPwr_ndm))
cz_gammaPkFrq = np.hstack((cz_gammaPkFrq_dmn,cz_gammaPkFrq_ndm))
cz_gammaPkPwr = np.hstack((cz_gammaPkPwr_dmn,cz_gammaPkPwr_ndm))
pz_gammaPkFrq = np.hstack((pz_gammaPkFrq_dmn,pz_gammaPkFrq_ndm))
pz_gammaPkPwr = np.hstack((pz_gammaPkPwr_dmn,pz_gammaPkPwr_ndm))

#%% export  results
df1 = pd.DataFrame({"Delta Peak Freq Fz":fz_deltaPkFrq})
df2 = pd.DataFrame({"Delta Peak Power Fz":fz_deltaPkPwr})
df3 = pd.DataFrame({"Theta Peak Freq Fz":fz_thetaPkFrq})
df4 = pd.DataFrame({"Theta Peak Power Fz":fz_thetaPkPwr})
df5 = pd.DataFrame({"Alpha Peak Freq Fz":fz_alphaPkFrq})
df6 = pd.DataFrame({"Alpha Peak Power Fz":fz_alphaPkPwr})
df7 = pd.DataFrame({"Beta Peak Freq Fz":fz_betaPkFrq})
df8 = pd.DataFrame({"Beta Peak Power Fz":fz_betaPkPwr})
df9 = pd.DataFrame({"Gamma Peak Freq Fz":fz_gammaPkFrq})
df10 = pd.DataFrame({"Gamma Peak Power Fz":fz_gammaPkPwr})
df11 = pd.DataFrame({"Delta Peak Freq Cz":cz_deltaPkFrq})
df12 = pd.DataFrame({"Delta Peak Power Cz":cz_deltaPkPwr})
df13 = pd.DataFrame({"Theta Peak Freq Cz":cz_thetaPkFrq})
df14 = pd.DataFrame({"Theta Peak Power Cz":cz_thetaPkPwr})
df15 = pd.DataFrame({"Alpha Peak Freq Cz":cz_alphaPkFrq})
df16 = pd.DataFrame({"Alpha Peak Power Cz":cz_alphaPkPwr})
df17 = pd.DataFrame({"Beta Peak Freq Cz":cz_betaPkFrq})
df18 = pd.DataFrame({"Beta Peak Power Cz":cz_betaPkPwr})
df19 = pd.DataFrame({"Gamma Peak Freq Cz":cz_gammaPkFrq})
df20 = pd.DataFrame({"Gamma Peak Power Cz":cz_gammaPkPwr})
df21 = pd.DataFrame({"Delta Peak Freq Pz":pz_deltaPkFrq})
df22 = pd.DataFrame({"Delta Peak Power Pz":pz_deltaPkPwr})
df23 = pd.DataFrame({"Theta Peak Freq Pz":pz_thetaPkFrq})
df24 = pd.DataFrame({"Theta Peak Power Pz":pz_thetaPkPwr})
df25 = pd.DataFrame({"Alpha Peak Freq Pz":pz_alphaPkFrq})
df26 = pd.DataFrame({"Alpha Peak Power Pz":pz_alphaPkPwr})
df27 = pd.DataFrame({"Beta Peak Freq Pz":pz_betaPkFrq})
df28 = pd.DataFrame({"Beta Peak Power Pz":pz_betaPkPwr})
df29 = pd.DataFrame({"Gamma Peak Freq Pz":pz_gammaPkFrq})
df30 = pd.DataFrame({"Gamma Peak Power Pz":pz_gammaPkPwr})
pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,
           df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,
           df29,df30],axis=1).to_csv(r"/Volumes/Backup Plus/eegDatasets/laurel_place/trainingData/peakFreqPwr.csv",index=False, mode='w')











