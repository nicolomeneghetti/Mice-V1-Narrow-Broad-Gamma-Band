# Mice-V1-Narrow-Broad-Gamma-Band
This repository contains the code for running the computational model described in Nicolò Meneghetti, Chiara Cerri, Elena Tantillo, Eleonora Vannini, Matteo Caleo and Alberto Mazzoni "Narrow and broad gamma bands process complementary visual information in mouse primary visual cortex" eNeuro (2021)  https://doi.org/10.1523/ENEURO.0106-21.2021


The simulations require the freely available Brian2 simulator (https://github.com/brian-team/brian2) and Python3 (https://www.python.org/). 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

The main codes are: 

1) MiceV1_BBandNB_model_code.py

This code is divided into two parts: 

-) modulation of sustained thalamic input (which is the mean level of firing rates of thalamic afferent to V1). This simulates the spectral LFPs of mice V1 when presented with vertical gratings visual stimuli of contrast level >30. 

-) modulation of rythmic thalamic input (which reflects the oscillatory firing rate of LGN neurons when mice are presented with low contasts visual stimuli). This simulates the spectral LFPs of mice V1 when presented with vertical gratings visual stimuli of contrast level <30. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

2) Migraine_V1_BB_NB_Brian2_simulations.py

This code build the simulations to replicate the spectral content of FHM1 primary visual cortex awake mice. Full details can be found in the published article: Nicolò Meneghetti, Chiara Cerri, Eleonora Vannini, Elena Tantillo, Angelita Tottene, Daniela Pietrobon, Matteo Caleo, and Alberto Mazzoni "Synaptic alterations in visual cortex reshape contrast-dependent gamma oscillations and inhibition-excitation ratio in a genetic mouse model of migraine" https://doi.org/10.1186/s10194-022-01495-9


Briefly, the code modifies the previous one (developed for replicating awake mice V1 spectral content in healthy condition) by introducing three synaptic alterations known to be represent the (main component of) etiology of a genetic mouse model of familial-hemipleigc-type1 migraine. 

These three alterations are: 

