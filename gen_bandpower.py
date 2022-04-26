"""
Generating training data for dementia prediction project using raw EEG data.
Raw eeg is processed for each participant of both dementia and non-dementia groups
rolling window was utilized to extract features from the raw EEG data.
"""
#%%
import sys
sys.path.insert(0, '/Users/joshuaighalo/Documents/BrainNet/Projects/workspace/code_base/laurel_place')
import params as dg 
import sys
sys.path.insert(0, '/Users/joshuaighalo/Documents/BrainNet/Projects/workspace/code_base')
from fn_cfg import*
from df_lib import*

#%% import participant scans from backup

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

#ncPipeline(version=dg.neurocatchVersion,filename=dementiaGroup[0],localPath=dg.path,line=dg.line,fs=dg.fs,
#            Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,clip=None,
#            dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
#eeg = ncPipeline.bandPassFOutput

#q = rolling_window(eeg[:,0], dg.winSize, dg.winFreq)
#%%  function to extract band power from windows extracted from eeg
def bandPowerPerWindow(eeg,windowSize,windowFrequency,samplingFrequency,lowFrequency,highFrequency,bandWindowSize):
    # this function calculates the power in a specific frequency band for each window
    # eeg: the eeg data (samples,channels)
    # windowedEEG =(channels,no of windows,length of window)    
    # windowSize: the length of the window in seconds (corresponding to the time period array)
    # lowFrequency: the low frequency of the brain wave
    # highFrequency: the high frequency of the brain wave
    windowedEEG = []
    for i in range(len(eeg.T)):
        load = rolling_window(eeg[:,i],windowSize,windowFrequency)
        windowedEEG.append(load)
    windowedEEG = np.array(windowedEEG)

    # calculate the power in the frequency band for each window
    bandPower = []
    for i in range(windowedEEG.shape[0]):
        loop_1 = windowedEEG[i,:,:]
        loopPower = []
        for j in range(windowedEEG.shape[1]):
            loop_2 = absBandPower(loop_1[j,:],samplingFrequency,lowFrequency,highFrequency,bandWindowSize,chanTitle='None',dispIMG=None,ylim=None,xlim=None,show_result=None)
            loopPower.append(loop_2)
        loopPower = np.array(loopPower)
        bandPower.append(loopPower)
    bandPower = np.array(bandPower).T
    return bandPower


def ncBandPowerPerWindow(version,filename,localPath,line,Q,stimTrig,lowcut,highcut,order,clip,windowSize,windowFrequency,samplingFrequency,lowFrequency,highFrequency,bandWindowSize,dispIMG,y_lim,figsize,pltclr,titles):
    # this function calculates the power in a specific frequency band for each window for the neurocatch pipeline
    ncPipeline(version,filename,localPath,line,samplingFrequency,Q,stimTrig,lowcut,highcut,order,clip,dispIMG,y_lim,figsize,pltclr,titles)
    eeg = ncPipeline.bandPassFOutput
    bandPower = bandPowerPerWindow(eeg,windowSize,windowFrequency,samplingFrequency,lowFrequency,highFrequency,bandWindowSize)
    return bandPower

#%% full delta power
deltaPower_dementia=[]
for i in range (len(dementiaGroup)):
    deltaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['delta'][0],highFrequency=dg.brainWaves['delta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    deltaPower_dementia.append(deltaPower)
deltaPower_dementia = np.array(deltaPower_dementia)
deltaPower_dementia = deltaPower_dementia.reshape(((deltaPower_dementia.shape[0])*deltaPower_dementia.shape[1]),deltaPower_dementia.shape[2])

deltaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    deltaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['delta'][0],highFrequency=dg.brainWaves['delta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    deltaPower_nondementia.append(deltaPower)
deltaPower_nondementia = np.array(deltaPower_nondementia)
deltaPower_nondementia = deltaPower_nondementia.reshape(((deltaPower_nondementia.shape[0])*deltaPower_nondementia.shape[1]),deltaPower_nondementia.shape[2])


# early delta power
earlyDeltaPower_dementia=[]
for i in range (len(dementiaGroup)):
    deltaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyDelta'][0],highFrequency=dg.segBrainWaves['earlyDelta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyDeltaPower_dementia.append(deltaPower)
earlyDeltaPower_dementia = np.array(earlyDeltaPower_dementia)
earlyDeltaPower_dementia = earlyDeltaPower_dementia.reshape(((earlyDeltaPower_dementia.shape[0])*earlyDeltaPower_dementia.shape[1]),earlyDeltaPower_dementia.shape[2])

earlyDeltaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    deltaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyDelta'][0],highFrequency=dg.segBrainWaves['earlyDelta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyDeltaPower_nondementia.append(deltaPower)
earlyDeltaPower_nondementia = np.array(earlyDeltaPower_nondementia)
earlyDeltaPower_nondementia = earlyDeltaPower_nondementia.reshape(((earlyDeltaPower_nondementia.shape[0])*earlyDeltaPower_nondementia.shape[1]),earlyDeltaPower_nondementia.shape[2])

# mid delta power
midDeltaPower_dementia=[]
for i in range (len(dementiaGroup)):
    deltaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midDelta'][0],highFrequency=dg.segBrainWaves['midDelta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midDeltaPower_dementia.append(deltaPower)
midDeltaPower_dementia = np.array(midDeltaPower_dementia)
midDeltaPower_dementia = midDeltaPower_dementia.reshape(((midDeltaPower_dementia.shape[0])*midDeltaPower_dementia.shape[1]),midDeltaPower_dementia.shape[2])

midDeltaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    deltaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midDelta'][0],highFrequency=dg.segBrainWaves['midDelta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midDeltaPower_nondementia.append(deltaPower)
midDeltaPower_nondementia = np.array(midDeltaPower_nondementia)
midDeltaPower_nondementia = midDeltaPower_nondementia.reshape(((midDeltaPower_nondementia.shape[0])*midDeltaPower_nondementia.shape[1]),midDeltaPower_nondementia.shape[2])

# late delta power
lateDeltaPower_dementia=[]
for i in range (len(dementiaGroup)):
    deltaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateDelta'][0],highFrequency=dg.segBrainWaves['lateDelta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateDeltaPower_dementia.append(deltaPower)
lateDeltaPower_dementia = np.array(lateDeltaPower_dementia)
lateDeltaPower_dementia = lateDeltaPower_dementia.reshape(((lateDeltaPower_dementia.shape[0])*lateDeltaPower_dementia.shape[1]),lateDeltaPower_dementia.shape[2])

lateDeltaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    deltaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateDelta'][0],highFrequency=dg.segBrainWaves['lateDelta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateDeltaPower_nondementia.append(deltaPower)
lateDeltaPower_nondementia = np.array(lateDeltaPower_nondementia)
lateDeltaPower_nondementia = lateDeltaPower_nondementia.reshape(((lateDeltaPower_nondementia.shape[0])*lateDeltaPower_nondementia.shape[1]),lateDeltaPower_nondementia.shape[2])



#%% full theta power
thetaPower_dementia=[]
for i in range (len(dementiaGroup)):
    thetaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['theta'][0],highFrequency=dg.brainWaves['theta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    thetaPower_dementia.append(thetaPower)
thetaPower_dementia = np.array(thetaPower_dementia)
thetaPower_dementia = thetaPower_dementia.reshape(((thetaPower_dementia.shape[0])*thetaPower_dementia.shape[1]),thetaPower_dementia.shape[2])

thetaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    thetaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['theta'][0],highFrequency=dg.brainWaves['theta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    thetaPower_nondementia.append(thetaPower)
thetaPower_nondementia = np.array(thetaPower_nondementia)
thetaPower_nondementia = thetaPower_nondementia.reshape(((thetaPower_nondementia.shape[0])*thetaPower_nondementia.shape[1]),thetaPower_nondementia.shape[2])

# early Theta power
earlyThetaPower_dementia=[]
for i in range (len(dementiaGroup)):
    thetaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyTheta'][0],highFrequency=dg.segBrainWaves['earlyTheta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyThetaPower_dementia.append(thetaPower)
earlyThetaPower_dementia = np.array(earlyThetaPower_dementia)
earlyThetaPower_dementia = earlyThetaPower_dementia.reshape(((earlyThetaPower_dementia.shape[0])*earlyThetaPower_dementia.shape[1]),earlyThetaPower_dementia.shape[2])

earlyThetaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    thetaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyTheta'][0],highFrequency=dg.segBrainWaves['earlyTheta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyThetaPower_nondementia.append(thetaPower)
earlyThetaPower_nondementia = np.array(earlyThetaPower_nondementia)
earlyThetaPower_nondementia = earlyThetaPower_nondementia.reshape(((earlyThetaPower_nondementia.shape[0])*earlyThetaPower_nondementia.shape[1]),earlyThetaPower_nondementia.shape[2])


# mid Theta power
midThetaPower_dementia=[]
for i in range (len(dementiaGroup)):
    thetaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midTheta'][0],highFrequency=dg.segBrainWaves['midTheta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midThetaPower_dementia.append(thetaPower)
midThetaPower_dementia = np.array(midThetaPower_dementia)
midThetaPower_dementia = midThetaPower_dementia.reshape(((midThetaPower_dementia.shape[0])*midThetaPower_dementia.shape[1]),midThetaPower_dementia.shape[2])

midThetaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    thetaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midTheta'][0],highFrequency=dg.segBrainWaves['midTheta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midThetaPower_nondementia.append(thetaPower)
midThetaPower_nondementia = np.array(midThetaPower_nondementia)
midThetaPower_nondementia = midThetaPower_nondementia.reshape(((midThetaPower_nondementia.shape[0])*midThetaPower_nondementia.shape[1]),midThetaPower_nondementia.shape[2])


# late Theta power
lateThetaPower_dementia=[]
for i in range (len(dementiaGroup)):
    thetaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateTheta'][0],highFrequency=dg.segBrainWaves['lateTheta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateThetaPower_dementia.append(thetaPower)
lateThetaPower_dementia = np.array(lateThetaPower_dementia)
lateThetaPower_dementia = lateThetaPower_dementia.reshape(((lateThetaPower_dementia.shape[0])*lateThetaPower_dementia.shape[1]),lateThetaPower_dementia.shape[2])

lateThetaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    thetaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateTheta'][0],highFrequency=dg.segBrainWaves['lateTheta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateThetaPower_nondementia.append(thetaPower)
lateThetaPower_nondementia = np.array(lateThetaPower_nondementia)
lateThetaPower_nondementia = lateThetaPower_nondementia.reshape(((lateThetaPower_nondementia.shape[0])*lateThetaPower_nondementia.shape[1]),lateThetaPower_nondementia.shape[2])


#%% Alpha Power
# full alpha power, early alpha power, mid alpha power, late alpha power
alphaPower_dementia=[]
for i in range (len(dementiaGroup)):
    alphaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['alpha'][0],highFrequency=dg.brainWaves['alpha'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    alphaPower_dementia.append(alphaPower)
alphaPower_dementia = np.array(alphaPower_dementia)
alphaPower_dementia = alphaPower_dementia.reshape(((alphaPower_dementia.shape[0])*alphaPower_dementia.shape[1]),alphaPower_dementia.shape[2])

alphaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    alphaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['alpha'][0],highFrequency=dg.brainWaves['alpha'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    alphaPower_nondementia.append(alphaPower)
alphaPower_nondementia = np.array(alphaPower_nondementia)
alphaPower_nondementia = alphaPower_nondementia.reshape(((alphaPower_nondementia.shape[0])*alphaPower_nondementia.shape[1]),alphaPower_nondementia.shape[2])

# early alpha power
earlyAlphaPower_dementia=[]
for i in range (len(dementiaGroup)):
    alphaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyAlpha'][0],highFrequency=dg.segBrainWaves['earlyAlpha'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyAlphaPower_dementia.append(alphaPower)
earlyAlphaPower_dementia = np.array(earlyAlphaPower_dementia)
earlyAlphaPower_dementia = earlyAlphaPower_dementia.reshape(((earlyAlphaPower_dementia.shape[0])*earlyAlphaPower_dementia.shape[1]),earlyAlphaPower_dementia.shape[2])

earlyAlphaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    alphaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyAlpha'][0],highFrequency=dg.segBrainWaves['earlyAlpha'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyAlphaPower_nondementia.append(alphaPower)
earlyAlphaPower_nondementia = np.array(earlyAlphaPower_nondementia)
earlyAlphaPower_nondementia = earlyAlphaPower_nondementia.reshape(((earlyAlphaPower_nondementia.shape[0])*earlyAlphaPower_nondementia.shape[1]),earlyAlphaPower_nondementia.shape[2])

# mid alpha power
midAlphaPower_dementia=[]
for i in range (len(dementiaGroup)):
    alphaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midAlpha'][0],highFrequency=dg.segBrainWaves['midAlpha'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midAlphaPower_dementia.append(alphaPower)
midAlphaPower_dementia = np.array(midAlphaPower_dementia)
midAlphaPower_dementia = midAlphaPower_dementia.reshape(((midAlphaPower_dementia.shape[0])*midAlphaPower_dementia.shape[1]),midAlphaPower_dementia.shape[2])

midAlphaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    alphaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midAlpha'][0],highFrequency=dg.segBrainWaves['midAlpha'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midAlphaPower_nondementia.append(alphaPower)
midAlphaPower_nondementia = np.array(midAlphaPower_nondementia)
midAlphaPower_nondementia = midAlphaPower_nondementia.reshape(((midAlphaPower_nondementia.shape[0])*midAlphaPower_nondementia.shape[1]),midAlphaPower_nondementia.shape[2])


# late alpha power
lateAlphaPower_dementia=[]
for i in range (len(dementiaGroup)):
    alphaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateAlpha'][0],highFrequency=dg.segBrainWaves['lateAlpha'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateAlphaPower_dementia.append(alphaPower)
lateAlphaPower_dementia = np.array(lateAlphaPower_dementia)
lateAlphaPower_dementia = lateAlphaPower_dementia.reshape(((lateAlphaPower_dementia.shape[0])*lateAlphaPower_dementia.shape[1]),lateAlphaPower_dementia.shape[2])

lateAlphaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    alphaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateAlpha'][0],highFrequency=dg.segBrainWaves['lateAlpha'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateAlphaPower_nondementia.append(alphaPower)
lateAlphaPower_nondementia = np.array(lateAlphaPower_nondementia)
lateAlphaPower_nondementia = lateAlphaPower_nondementia.reshape(((lateAlphaPower_nondementia.shape[0])*lateAlphaPower_nondementia.shape[1]),lateAlphaPower_nondementia.shape[2])


#%% Beta Power
# full beta power, early beta power, mid beta power, late beta power
betaPower_dementia=[]
for i in range (len(dementiaGroup)):
    betaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['beta'][0],highFrequency=dg.brainWaves['beta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    betaPower_dementia.append(betaPower)
betaPower_dementia = np.array(betaPower_dementia)
betaPower_dementia = betaPower_dementia.reshape(((betaPower_dementia.shape[0])*betaPower_dementia.shape[1]),betaPower_dementia.shape[2])

betaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    betaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['beta'][0],highFrequency=dg.brainWaves['beta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    betaPower_nondementia.append(betaPower)
betaPower_nondementia = np.array(betaPower_nondementia)
betaPower_nondementia = betaPower_nondementia.reshape(((betaPower_nondementia.shape[0])*betaPower_nondementia.shape[1]),betaPower_nondementia.shape[2])

# early beta power
earlyBetaPower_dementia=[]
for i in range (len(dementiaGroup)):
    betaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyBeta'][0],highFrequency=dg.segBrainWaves['earlyBeta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyBetaPower_dementia.append(betaPower)
earlyBetaPower_dementia = np.array(earlyBetaPower_dementia)
earlyBetaPower_dementia = earlyBetaPower_dementia.reshape(((earlyBetaPower_dementia.shape[0])*earlyBetaPower_dementia.shape[1]),earlyBetaPower_dementia.shape[2])

earlyBetaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    betaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyBeta'][0],highFrequency=dg.segBrainWaves['earlyBeta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyBetaPower_nondementia.append(betaPower)
earlyBetaPower_nondementia = np.array(earlyBetaPower_nondementia)
earlyBetaPower_nondementia = earlyBetaPower_nondementia.reshape(((earlyBetaPower_nondementia.shape[0])*earlyBetaPower_nondementia.shape[1]),earlyBetaPower_nondementia.shape[2])

# mid beta power
midBetaPower_dementia=[]
for i in range (len(dementiaGroup)):
    betaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midBeta'][0],highFrequency=dg.segBrainWaves['midBeta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midBetaPower_dementia.append(betaPower)
midBetaPower_dementia = np.array(midBetaPower_dementia)
midBetaPower_dementia = midBetaPower_dementia.reshape(((midBetaPower_dementia.shape[0])*midBetaPower_dementia.shape[1]),midBetaPower_dementia.shape[2])

midBetaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    betaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midBeta'][0],highFrequency=dg.segBrainWaves['midBeta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midBetaPower_nondementia.append(betaPower)
midBetaPower_nondementia = np.array(midBetaPower_nondementia)
midBetaPower_nondementia = midBetaPower_nondementia.reshape(((midBetaPower_nondementia.shape[0])*midBetaPower_nondementia.shape[1]),midBetaPower_nondementia.shape[2])

# late beta power
lateBetaPower_dementia=[]
for i in range (len(dementiaGroup)):
    betaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateBeta'][0],highFrequency=dg.segBrainWaves['lateBeta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateBetaPower_dementia.append(betaPower)
lateBetaPower_dementia = np.array(lateBetaPower_dementia)
lateBetaPower_dementia = lateBetaPower_dementia.reshape(((lateBetaPower_dementia.shape[0])*lateBetaPower_dementia.shape[1]),lateBetaPower_dementia.shape[2])

lateBetaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    betaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateBeta'][0],highFrequency=dg.segBrainWaves['lateBeta'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateBetaPower_nondementia.append(betaPower)
lateBetaPower_nondementia = np.array(lateBetaPower_nondementia)
lateBetaPower_nondementia = lateBetaPower_nondementia.reshape(((lateBetaPower_nondementia.shape[0])*lateBetaPower_nondementia.shape[1]),lateBetaPower_nondementia.shape[2])


#%% Gamma Power
# full gamma power, early gamma power, mid gamma power, late gamma power
gammaPower_dementia=[]
for i in range (len(dementiaGroup)):
    gammaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['gamma'][0],highFrequency=dg.brainWaves['gamma'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    gammaPower_dementia.append(gammaPower)
gammaPower_dementia = np.array(gammaPower_dementia)
gammaPower_dementia = gammaPower_dementia.reshape(((gammaPower_dementia.shape[0])*gammaPower_dementia.shape[1]),gammaPower_dementia.shape[2])

gammaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    gammaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.brainWaves['gamma'][0],highFrequency=dg.brainWaves['gamma'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    gammaPower_nondementia.append(gammaPower)
gammaPower_nondementia = np.array(gammaPower_nondementia)
gammaPower_nondementia = gammaPower_nondementia.reshape(((gammaPower_nondementia.shape[0])*gammaPower_nondementia.shape[1]),gammaPower_nondementia.shape[2])

# early gamma power
earlyGammaPower_dementia=[]
for i in range (len(dementiaGroup)):
    gammaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyGamma'][0],highFrequency=dg.segBrainWaves['earlyGamma'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyGammaPower_dementia.append(gammaPower)
earlyGammaPower_dementia = np.array(earlyGammaPower_dementia)
earlyGammaPower_dementia = earlyGammaPower_dementia.reshape(((earlyGammaPower_dementia.shape[0])*earlyGammaPower_dementia.shape[1]),earlyGammaPower_dementia.shape[2])

earlyGammaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    gammaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['earlyGamma'][0],highFrequency=dg.segBrainWaves['earlyGamma'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    earlyGammaPower_nondementia.append(gammaPower)
earlyGammaPower_nondementia = np.array(earlyGammaPower_nondementia)
earlyGammaPower_nondementia = earlyGammaPower_nondementia.reshape(((earlyGammaPower_nondementia.shape[0])*earlyGammaPower_nondementia.shape[1]),earlyGammaPower_nondementia.shape[2])

# mid gamma power
midGammaPower_dementia=[]
for i in range (len(dementiaGroup)):
    gammaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midGamma'][0],highFrequency=dg.segBrainWaves['midGamma'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midGammaPower_dementia.append(gammaPower)
midGammaPower_dementia = np.array(midGammaPower_dementia)
midGammaPower_dementia = midGammaPower_dementia.reshape(((midGammaPower_dementia.shape[0])*midGammaPower_dementia.shape[1]),midGammaPower_dementia.shape[2])

midGammaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    gammaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['midGamma'][0],highFrequency=dg.segBrainWaves['midGamma'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    midGammaPower_nondementia.append(gammaPower)
midGammaPower_nondementia = np.array(midGammaPower_nondementia)
midGammaPower_nondementia = midGammaPower_nondementia.reshape(((midGammaPower_nondementia.shape[0])*midGammaPower_nondementia.shape[1]),midGammaPower_nondementia.shape[2])

# late gamma power
lateGammaPower_dementia=[]
for i in range (len(dementiaGroup)):
    gammaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateGamma'][0],highFrequency=dg.segBrainWaves['lateGamma'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateGammaPower_dementia.append(gammaPower)
lateGammaPower_dementia = np.array(lateGammaPower_dementia)
lateGammaPower_dementia = lateGammaPower_dementia.reshape(((lateGammaPower_dementia.shape[0])*lateGammaPower_dementia.shape[1]),lateGammaPower_dementia.shape[2])

lateGammaPower_nondementia=[]
for i in range (len(non_dementiaGroup)):
    gammaPower = ncBandPowerPerWindow(version=dg.neurocatchVersion,filename=non_dementiaGroup[i],localPath=dg.path,line=dg.line,
                                        Q=dg.Q,stimTrig=dg.stimTrig,lowcut=dg.highPass,highcut=dg.lowPass,order=dg.order,
                                        clip=None,windowSize=dg.winSize,windowFrequency=dg.winFreq,samplingFrequency=dg.fs,
                                        lowFrequency=dg.segBrainWaves['lateGamma'][0],highFrequency=dg.segBrainWaves['lateGamma'][1],
                                        bandWindowSize=dg.win,dispIMG=None,y_lim=None,figsize=None,pltclr=None,titles=None)
    lateGammaPower_nondementia.append(gammaPower)
lateGammaPower_nondementia = np.array(lateGammaPower_nondementia)
lateGammaPower_nondementia = lateGammaPower_nondementia.reshape(((lateGammaPower_nondementia.shape[0])*lateGammaPower_nondementia.shape[1]),lateGammaPower_nondementia.shape[2])


#%% concatenate dementia groups with non-dementia groups
delta_power = np.concatenate((deltaPower_dementia,deltaPower_nondementia),axis=0)
theta_power=np.concatenate((thetaPower_dementia,thetaPower_nondementia),axis=0)
alpha_power=np.concatenate((alphaPower_dementia,alphaPower_nondementia),axis=0)
beta_power=np.concatenate((betaPower_dementia,betaPower_nondementia),axis=0)
gamma_power=np.concatenate((gammaPower_dementia,gammaPower_nondementia),axis=0)

earlyDeltaPower=np.concatenate((earlyDeltaPower_dementia,earlyDeltaPower_nondementia),axis=0)
midDeltaPower=np.concatenate((midDeltaPower_dementia,midDeltaPower_nondementia),axis=0)
lateDeltaPower=np.concatenate((lateDeltaPower_dementia,lateDeltaPower_nondementia),axis=0)

earlyThetaPower=np.concatenate((earlyThetaPower_dementia,earlyThetaPower_nondementia),axis=0)
midThetaPower=np.concatenate((midThetaPower_dementia,midThetaPower_nondementia),axis=0)
lateThetaPower=np.concatenate((lateThetaPower_dementia,lateThetaPower_nondementia),axis=0)

earlyAlphaPower=np.concatenate((earlyAlphaPower_dementia,earlyAlphaPower_nondementia),axis=0)
midAlphaPower=np.concatenate((midAlphaPower_dementia,midAlphaPower_nondementia),axis=0)
lateAlphaPower=np.concatenate((lateAlphaPower_dementia,lateAlphaPower_nondementia),axis=0)

earlyBetaPower=np.concatenate((earlyBetaPower_dementia,earlyBetaPower_nondementia),axis=0)
midBetaPower=np.concatenate((midBetaPower_dementia,midBetaPower_nondementia),axis=0)
lateBetaPower=np.concatenate((lateBetaPower_dementia,lateBetaPower_nondementia),axis=0)

earlyGammaPower=np.concatenate((earlyGammaPower_dementia,earlyGammaPower_nondementia),axis=0)
midGammaPower=np.concatenate((midGammaPower_dementia,midGammaPower_nondementia),axis=0)
lateGammaPower=np.concatenate((lateGammaPower_dementia,lateGammaPower_nondementia),axis=0)

#%% export  results
df1=pd.DataFrame({"Delta Power fz":delta_power[:,0]})
df2=pd.DataFrame({"Delta Power cz":delta_power[:,1]})
df3=pd.DataFrame({"Delta Power pz":delta_power[:,2]})
df4=pd.DataFrame({"Theta Power fz":theta_power[:,0]})
df5=pd.DataFrame({"Theta Power cz":theta_power[:,1]})
df6=pd.DataFrame({"Theta Power pz":theta_power[:,2]})
df7=pd.DataFrame({"Alpha Power fz":alpha_power[:,0]})
df8=pd.DataFrame({"Alpha Power cz":alpha_power[:,1]})
df9=pd.DataFrame({"Alpha Power pz":alpha_power[:,2]})
df10=pd.DataFrame({"Beta Power fz":beta_power[:,0]})
df11=pd.DataFrame({"Beta Power cz":beta_power[:,1]})
df12=pd.DataFrame({"Beta Power pz":beta_power[:,2]})
df13=pd.DataFrame({"Gamma Power fz":gamma_power[:,0]})
df14=pd.DataFrame({"Gamma Power cz":gamma_power[:,1]})
df15=pd.DataFrame({"Gamma Power pz":gamma_power[:,2]})
df16=pd.DataFrame({"Early Delta Power fz":earlyDeltaPower[:,0]})
df17=pd.DataFrame({"Early Delta Power cz":earlyDeltaPower[:,1]})
df18=pd.DataFrame({"Early Delta Power pz":earlyDeltaPower[:,2]})
df19=pd.DataFrame({"Mid Delta Power fz":midDeltaPower[:,0]})
df20=pd.DataFrame({"Mid Delta Power cz":midDeltaPower[:,1]})
df21=pd.DataFrame({"Mid Delta Power pz":midDeltaPower[:,2]})
df22=pd.DataFrame({"Late Delta Power fz":lateDeltaPower[:,0]})
df23=pd.DataFrame({"Late Delta Power cz":lateDeltaPower[:,1]})
df24=pd.DataFrame({"Late Delta Power pz":lateDeltaPower[:,2]})
df25=pd.DataFrame({"Early Theta Power fz":earlyThetaPower[:,0]})
df26=pd.DataFrame({"Early Theta Power cz":earlyThetaPower[:,1]})
df27=pd.DataFrame({"Early Theta Power pz":earlyThetaPower[:,2]})
df28=pd.DataFrame({"Mid Theta Power fz":midThetaPower[:,0]})
df29=pd.DataFrame({"Mid Theta Power cz":midThetaPower[:,1]})
df30=pd.DataFrame({"Mid Theta Power pz":midThetaPower[:,2]})
df31=pd.DataFrame({"Late Theta Power fz":lateThetaPower[:,0]})
df32=pd.DataFrame({"Late Theta Power cz":lateThetaPower[:,1]})
df33=pd.DataFrame({"Late Theta Power pz":lateThetaPower[:,2]})
df34=pd.DataFrame({"Early Alpha Power fz":earlyAlphaPower[:,0]})
df35=pd.DataFrame({"Early Alpha Power cz":earlyAlphaPower[:,1]})
df36=pd.DataFrame({"Early Alpha Power pz":earlyAlphaPower[:,2]})
df37=pd.DataFrame({"Mid Alpha Power fz":midAlphaPower[:,0]})
df38=pd.DataFrame({"Mid Alpha Power cz":midAlphaPower[:,1]})
df39=pd.DataFrame({"Mid Alpha Power pz":midAlphaPower[:,2]})
df40=pd.DataFrame({"Late Alpha Power fz":lateAlphaPower[:,0]})
df41=pd.DataFrame({"Late Alpha Power cz":lateAlphaPower[:,1]})
df42=pd.DataFrame({"Late Alpha Power pz":lateAlphaPower[:,2]})
df43=pd.DataFrame({"Early Beta Power fz":earlyBetaPower[:,0]})
df44=pd.DataFrame({"Early Beta Power cz":earlyBetaPower[:,1]})
df45=pd.DataFrame({"Early Beta Power pz":earlyBetaPower[:,2]})
df46=pd.DataFrame({"Mid Beta Power fz":midBetaPower[:,0]})
df47=pd.DataFrame({"Mid Beta Power cz":midBetaPower[:,1]})
df48=pd.DataFrame({"Mid Beta Power pz":midBetaPower[:,2]})
df49=pd.DataFrame({"Late Beta Power fz":lateBetaPower[:,0]})
df50=pd.DataFrame({"Late Beta Power cz":lateBetaPower[:,1]})
df51=pd.DataFrame({"Late Beta Power pz":lateBetaPower[:,2]})
df52=pd.DataFrame({"Early Gamma Power fz":earlyGammaPower[:,0]})
df53=pd.DataFrame({"Early Gamma Power cz":earlyGammaPower[:,1]})
df54=pd.DataFrame({"Early Gamma Power pz":earlyGammaPower[:,2]})
df55=pd.DataFrame({"Mid Gamma Power fz":midGammaPower[:,0]})
df56=pd.DataFrame({"Mid Gamma Power cz":midGammaPower[:,1]})
df57=pd.DataFrame({"Mid Gamma Power pz":midGammaPower[:,2]})
df58=pd.DataFrame({"Late Gamma Power fz":lateGammaPower[:,0]})
df59=pd.DataFrame({"Late Gamma Power cz":lateGammaPower[:,1]})
df60=pd.DataFrame({"Late Gamma Power pz":lateGammaPower[:,2]})
pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,
              df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,
                df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58,
                df59,df60],axis=1).to_csv(r"/Volumes/Backup Plus/eegDatasets/laurel_place/trainingData/bandPower.csv",index=False, mode='w')

