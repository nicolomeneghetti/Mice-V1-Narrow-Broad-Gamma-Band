from brian2 import *

import numpy, scipy.io
import sys

defaultclock.dt = 0.05*ms
prefs.codegen.target = 'numpy'

from brian2hears import *

class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0
    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1)) # return to start of˓→line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed-self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("\n")




start_scope()


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# POPULATION PARAMETRS
N = 5000
N_E = int(N * 0.8)  # pyramidal neurons
N_I = int(N * 0.2)  # interneurons
duration=5000*ms # duration of simulation
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NEURON PARAMETRS
tau_m_exc=20*ms # excitatory membrane time constant
tau_m_inh=10*ms # inhibitory membrane time constant
g_leak_exc=25*nS # leak conductance excitatory neurons
g_leak_inh=20*nS # leak conductance inhibitory neurons
V_leak=-70*mV # leak potential
V_thr=-52*mV # threshold potential
V_reset=-59*mV # reset potential

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SYNAPSES PARAMETERS

V_syn_AMPA=0.*mV # AMPA synapses reversal potential
V_syn_GABA=-80*mV # GABA synapses reversal potential

ref_Exc=2*ms # excitatory neurons refractory time
ref_Inh=1*ms # inhibitory neurons refractory time

tau_r_AMPAonExc=0.4*ms # rise time of AMPA synapses on excitatory neurons
tau_d_AMPAonExc=2.25*ms # decay time of AMPA synapses on excitatory neurons
tau_r_GABAonExc=1*ms # rise time of GABA synapses on excitatory neurons
tau_d_GABAonExc=5*ms # decay time of GABA synapses on excitatory neurons
tau_r_GABAonInh=1*ms # rise time of GABA synapses on inhibitory neurons
tau_d_GABAonInh=5*ms # decay time of GABA synapses on inhibitory neurons
tau_r_AMPAonInh=0.2*ms # rise time of AMPA synapses on inhibitory neurons
tau_d_AMPAonInh=1.25*ms # decay time of AMPA synapses on inhibitory neurons



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# THALAMIC INPUT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

sound = whitenoise(duration,20000*Hz)
order = 3 #order of the filters
nchannels = 1
center_frequencies = 57 * Hz # thalamic narrow band central freq.
bw = 10 * Hz # bandwidth of the filters
fc = vstack((center_frequencies-bw/2, center_frequencies+bw/2))
filterbank = Butterworth(sound, nchannels, order, fc, 'bandpass')
filterbank_mon = filterbank.process()
filterbank_mon=(filterbank_mon-mean(filterbank_mon))/std(filterbank_mon)
filterbank_mon=np.squeeze(filterbank_mon)
rate_bah = TimedArray(filterbank_mon*Hz, dt=0.05*ms) # rythmic thalamic rate

st_dev_1=100
ratenull=0*Hz

alpha_thalamus=800*Hz
# This version is periodic signal + noise
OU_Periodic_1 = Equations('''
rate_0 =  v_0 - alpha_thalamus : Hz
rate_1 =  rate_0 * int(rate_0 >= ratenull) + ratenull * int(rate_0 < ratenull) : Hz

''')

OU_Periodic_2 = Equations('''
rate_2 =  st_dev_1*rate_exou(t) + alpha_thalamus +  a_0*rate_bah(t): Hz
rate_3 =  rate_2 * int(rate_2 >= ratenull) + ratenull * int(rate_2 < ratenull) : Hz

''')

input_on_Exc = NeuronGroup(N_E, model=OU_Periodic_1, threshold='rand()<rate_1*dt')
input_on_Inh = NeuronGroup(N_I, model=OU_Periodic_1, threshold='rand()<rate_1*dt')

input_on_Exc_otherCorticalAreas = NeuronGroup(N_E, model=OU_Periodic_2, threshold='rand()<rate_3*dt')
input_on_Inh_otherCorticalAreas = NeuronGroup(N_I, model=OU_Periodic_2, threshold='rand()<rate_3*dt')

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# MODEL EQUATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

eqs_Exc = '''
    dV/dt  = (-(V-V_leak) - (IA_rec+IG_onexc+IA_ext+IA_ext_othercortex)/g_leak_exc)/tau_m_exc : volt (unless refractory)

    IA_rec = g_syn_AMPAonExc * (V-V_syn_AMPA) * SA_rec              : ampere
    dSA_rec/dt = (-SA_rec + XA_rec)/tau_d_AMPAonExc                 : 1
    dXA_rec/dt = -XA_rec/tau_r_AMPAonExc                            : 1


    IG_onexc = g_syn_GABAonExc * (V-V_syn_GABA) * SG_onexc          : ampere
    dSG_onexc/dt = (-SG_onexc + XG_onexc)/tau_d_GABAonExc           : 1
    dXG_onexc/dt = -XG_onexc/tau_r_GABAonExc                        : 1

    IA_ext = g_syn_AMPA_ext_onExc * (V-V_syn_AMPA) * SA_ext         : ampere
    dSA_ext/dt = (-SA_ext + XA_ext)/tau_d_AMPAonExc                 : 1
    dXA_ext/dt = -XA_ext/tau_r_AMPAonExc                            : 1

    IA_ext_othercortex = g_syn_AMPA_ext_onExc_othercortex * (V-V_syn_AMPA) * SA_ext_othercortex         : ampere
    dSA_ext_othercortex/dt = (-SA_ext_othercortex + XA_ext_othercortex)/tau_d_AMPAonExc                 : 1
    dXA_ext_othercortex/dt = -XA_ext_othercortex/tau_r_AMPAonExc                            : 1

    LFP = (abs(IG_onexc) + abs(IA_rec) + abs(IA_ext) + abs(IA_ext_othercortex)) / g_leak_exc  : volt

    EPSC_REcurrent_Exc = IA_rec                          : ampere
    IPSC_REcurren_Exc = IG_onexc                        : ampere

'''

eqs_Inh = '''
    dV/dt  = (-(V-V_leak) -(IA_oninh+IG_rec+IA_ext_i+IA_ext_i_othercortex)/g_leak_inh)/tau_m_inh : volt (unless refractory)


    IA_oninh = g_syn_AMPAonInh * (V-V_syn_AMPA) * SA_oninh          : ampere
    dSA_oninh/dt = (-SA_oninh + XA_oninh)/tau_d_AMPAonInh           : 1
    dXA_oninh/dt = -XA_oninh/tau_r_AMPAonInh                        : 1


    IG_rec = g_syn_GABAonInh * (V-V_syn_GABA) * SG_rec              : ampere
    dSG_rec/dt = (-SG_rec + XG_rec)/tau_d_GABAonInh                 : 1
    dXG_rec/dt = -XG_rec/tau_r_GABAonInh                            : 1

    IA_ext_i = g_syn_AMPA_ext_onInh * (V-V_syn_AMPA) * SA_ext_i     : ampere
    dSA_ext_i/dt = (-SA_ext_i + XA_ext_i)/tau_d_AMPAonInh           : 1
    dXA_ext_i/dt = -XA_ext_i/tau_r_AMPAonInh                        : 1

    IA_ext_i_othercortex = g_syn_AMPA_ext_onInh_othercortex * (V-V_syn_AMPA) * SA_ext_i_othercortex     : ampere
    dSA_ext_i_othercortex/dt = (-SA_ext_i_othercortex + XA_ext_i_othercortex)/tau_d_AMPAonInh           : 1
    dXA_ext_i_othercortex/dt = -XA_ext_i_othercortex/tau_r_AMPAonInh                        : 1

    EPSC_REcurrent_Inh = IA_oninh                          : ampere
    IPSC_REcurren_Inh = IG_rec                            : ampere

'''


Exc = NeuronGroup(N_E, eqs_Exc, threshold='V > V_thr', reset='V=V_reset', refractory=ref_Exc, method='rk2')
Inh = NeuronGroup(N_I, eqs_Inh, threshold='V > V_thr', reset='V=V_reset', refractory=ref_Inh, method='rk2')
Exc.V = V_leak + (V_reset - V_leak) * rand(len(Exc))
Inh.V = V_leak + (V_reset - V_leak) * rand(len(Inh))

# Input to PYRAMIDAL neurons !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Wee = (tau_m_exc)/tau_r_AMPAonExc
C_E_E = Synapses(Exc, Exc, on_pre='XA_rec += Wee', delay=2*ms)
C_E_E.connect(condition='i != j',p=0.2)

Wie = (tau_m_exc)/tau_r_GABAonExc
C_I_E = Synapses(Inh, Exc, on_pre='XG_onexc += Wie', delay=1*ms)
C_I_E.connect(p=0.2)

# Input to INTERNEURONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Wii = (tau_m_inh)/tau_r_GABAonInh
C_I_I = Synapses(Inh, Inh, on_pre='XG_rec += Wii', delay=1*ms)
C_I_I.connect(condition='i != j',p=0.2)

Wei = (tau_m_inh)/tau_r_AMPAonInh
C_E_I = Synapses(Exc, Inh, on_pre='XA_oninh += Wei', delay=2*ms)
C_E_I.connect(p=0.2)

# Thalamic to network  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WextE = (tau_m_exc)/tau_r_AMPAonExc
C_P_E = Synapses(input_on_Exc, Exc, on_pre='XA_ext += WextE', delay=2*ms)
C_P_E.connect('i==j')

WextI = (tau_m_inh)/tau_r_AMPAonInh
C_P_I = Synapses(input_on_Inh, Inh, on_pre='XA_ext_i += WextI', delay=2*ms)
C_P_I.connect('i==j')

WextE_othercortex = (tau_m_exc)/tau_r_AMPAonExc
C_P_E_othercortex = Synapses(input_on_Exc_otherCorticalAreas, Exc, on_pre='XA_ext_othercortex += WextE_othercortex', delay=2*ms)
C_P_E_othercortex.connect('i==j')

WextI_othercortex = (tau_m_inh)/tau_r_AMPAonInh
C_P_I_othercortex = Synapses(input_on_Inh_otherCorticalAreas, Inh, on_pre='XA_ext_i_othercortex += WextI_othercortex', delay=2*ms)
C_P_I_othercortex.connect('i==j')

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# MONITORS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

LFP_e = StateMonitor(Exc, 'LFP', record=True)
rate_e = PopulationRateMonitor(Exc)
rate_i = PopulationRateMonitor(Inh)

EPSC_onExc = StateMonitor(Exc, 'EPSC_REcurrent_Exc', record=True)
IPSC_onExc = StateMonitor(Exc, 'IPSC_REcurren_Exc', record=True)





###############################################################################
# MAIN CODE PORTION
index_for_loop=0
numero_trials=5 # number of trials
parametri_v0_fit=[1000,1050,1025,1075] # levels of sustained thalamic inputs
parametri_a0_fit=0 # levels of rythmic thalamic inputs


# here's the three synaptic alterations for simulating FHM1 spectral content
thalamic_migraine_increase=[0.05,0.1,0.15,0.2,0.25] # migraine thalamic increase
cortical_migraine_increase=[0.05,0.1,0.15,0.2,0.25] # migraine glutamatergic synaptic increase
thalamic_increase_migraine_asymmetry=[0.8,1,1.1,1.2,1.3,1.4] # migraine thalamic increase asymmetry

number_of_iterations=size(parametri_v0_fit)*numero_trials*size(thalamic_migraine_increase)*size(cortical_migraine_increase)*size(thalamic_increase_migraine_asymmetry)


store('modified')
for sustained_level_thal_input in parametri_v0_fit:
    restore('modified')
    v_0=(sustained_level_thal_input)*Hz # level of sustained thalamic input
    a_0=0 # level of rythmic thalamic input
    store('modified2')

    for inp_thalamic_incr in thalamic_migraine_increase:
        restore('modified2')
        store('modified3')

        for inp_cortical_incr in cortical_migraine_increase:
            restore('modified3')
            store('modified4')

            for inp_silanc_thalamico in thalamic_increase_migraine_asymmetry:
                restore('modified4')
                store('modified5')

                for lp in range(numero_trials): # for loop across trials (set to 5)
                    restore('modified5')
                    index_for_loop +=1
                    print('\n -----------------------------------------------------------------------------------------------------------------------\n')
                    print('processing iteration '+str(index_for_loop)+' of '+str(number_of_iterations))
                    print('\n -----------------------------------------------------------------------------------------------------------------------\n')





                    #SYNAPSES PARAMETERS
                    g_syn_AMPAonExc=(0.178)*nS # conductance of AMPA synapses on excitatory neurons
                    g_syn_AMPAonExc=g_syn_AMPAonExc*(1+inp_cortical_incr) # migraine modification
                    g_syn_GABAonExc=2.01*nS # conductance of GABA synapses on excitatory neurons
                    g_syn_AMPA_ext_onExc=0.234*nS # conductance of AMPA synapses on excitatory neurons coming from other cortical areas
                    g_syn_AMPA_ext_onExc_othercortex=0.234*nS # conductance of AMPA synapses on excitatory neurons coming from other cortical areas

                    TC_pyr = (2*inp_thalamic_incr)/(1+sbilanciamento_parametrico) # migraine paramter of thalamic synaptic increase on pyramidal neurons
                    TC_int = sbilanciamento_parametrico*TC_pyr # migraine paramter of thalamic synaptic increase on inhibitory neurons
                    g_syn_AMPA_ext_onExc=(1+TC_pyr)*g_syn_AMPA_ext_onExc # conductance of thalamo-cortical AMPA synapses on excitatory neurons
                    g_syn_AMPAonInh=0.233*nS # conductance of AMPA synapses on inhibitory neurons
                    g_syn_AMPAonInh=g_syn_AMPAonInh*(1+inp_cortical_incr) # migraine modifications
                    g_syn_GABAonInh=2.7*nS # conductance of GABA synapses on inhibitory neurons
                    g_syn_AMPA_ext_onInh=0.317*nS # conductance of thalamo-cortical AMPA synapses on inh. neurons
                    g_syn_AMPA_ext_onInh_othercortex=0.317*nS # conductance of AMPA synapses on inhibitory neurons coming from other cortical areas
                    g_syn_AMPA_ext_onInh=(1+TC_int)*g_syn_AMPA_ext_onInh  # migraine modifications


                    # import colored noise for driving noise input simulating inputs coming from other cortical areas
                    mat = scipy.io.loadmat('Colored_Noise_trial'+str(lp+1)+'.mat')
                    colored_noise=mat['colored_noise']
                    colored_noise=np.squeeze(colored_noise)
                    rate_exou = TimedArray(colored_noise*Hz, dt=0.05*ms)


                    run(duration, report=ProgressBar(), report_period=1*second) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


                    ################################################################################
                    ################################################################################
                    ################################################################################
                    ################################################################################
                    ################################################################################
                    # SAVING OUTPUTS
                    LFP_e2=LFP_e.LFP.T
                    total_LFP = []
                    for lfp in range(len(LFP_e2)):
                        total_LFP.append(sum(LFP_e2[lfp,:]))

                    tmp_exc_rate = rate_e.smooth_rate(window='flat', width=1*ms)/Hz
                    tmp_inh_rate = rate_i.smooth_rate(window='flat', width=1*ms)/Hz

                    scipy.io.savemat('LFP_v0'+str(int(v_0))+'_a0'+str(int(a_0))+'thal_inc'+str(inp_thalamic_incr)+'cor_inc'+str(inp_cortical_incr)+'thal_asymmetry'+str(sbilanciamento_parametrico)+'_trial'+str(lp+1)+'.mat', mdict={'LFP': total_LFP})
                    scipy.io.savemat('Exc_rate_v0'+str(int(v_0))+'_a0'+str(int(a_0))+'thal_inc'+str(inp_thalamic_incr)+'cor_inc'+str(inp_cortical_incr)+'thal_asymmetry'+str(sbilanciamento_parametrico)+'_trial'+str(lp+1)+'.mat', mdict={'e_rate': tmp_exc_rate})
                    scipy.io.savemat('Inh_rate_v0'+str(int(v_0))+'_a0'+str(int(a_0))+'thal_inc'+str(inp_thalamic_incr)+'cor_inc'+str(inp_cortical_incr)+'thal_asymmetry'+str(sbilanciamento_parametrico)+'_trial'+str(lp+1)+'.mat', mdict={'i_rate': tmp_inh_rate})


