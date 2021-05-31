
# simulations dependencies
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


def NB_BB_mouseV1(trial,a_0,v_0):

    defaultclock.dt = 0.05*ms

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
    g_syn_AMPAonExc=0.178*nS # conductance of AMPA synapses on excitatory neurons
    g_syn_GABAonExc=2.01*nS # conductance of GABA synapses on excitatory neurons
    g_syn_AMPA_ext_onExc=0.234*nS # conductance of thalamo-cortical AMPA synapses on excitatory neurons

    g_syn_AMPAonInh=0.233*nS # conductance of AMPA synapses on inhibitory neurons
    g_syn_GABAonInh=2.7*nS # conductance of GABA synapses on inhibitory neurons
    g_syn_AMPA_ext_onInh=0.317*nS # conductance of thalamo-cortical AMPA synapses on inhibitory neurons

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



    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # THALAMIC INPUT = a_0*(thalamic_NB)+v0+noise

    # thalamic_NB: Thalamic Narrow Band
    fs=20000 # frequency rate
    sound = whitenoise(duration,20000*Hz) #white noise
    order = 3 #order of the filters
    nchannels = 1
    center_frequencies = 57 * Hz
    bw = 10 * Hz # bandwidth of the filters
    fc = vstack((center_frequencies-bw/2, center_frequencies+bw/2))
    filterbank = Butterworth(sound, nchannels, order, fc, 'bandpass')
    filterbank_mon = filterbank.process()
    filterbank_mon=np.squeeze((filterbank_mon-mean(filterbank_mon))/std(filterbank_mon))
    thalamic_NB = TimedArray(filterbank_mon*Hz, dt=0.05*ms)

    #noise
    st_dev=400
    ratenull=0*Hz
    colored_noise = powerlawnoise(duration, 1.3, samplerate=20000*Hz)
    colored_noise = (colored_noise - mean(colored_noise))/(std(colored_noise))
    colored_noise = np.squeeze(colored_noise)
    sustained_noise = TimedArray(colored_noise*Hz, dt=0.05*ms)

    # THALAMIC INPUT model = a_0*(thalamic_NB)+v0+noise
    Thalamic_Input_model = Equations('''
    rate_0 =  st_dev*sustained_noise(t) + a_0*thalamic_NB(t) + v_0 : Hz
    rate_1 =  rate_0 * int(rate_0 >= ratenull) + ratenull * int(rate_0 < ratenull) : Hz
    ''')

    # thalamic input rate on excitatory neurons
    input_on_Exc = NeuronGroup(N_E, model=Thalamic_Input_model, threshold='rand()<rate_1*dt')
    # thalamic input rate on inhibitory neurons
    input_on_Inh = NeuronGroup(N_I, model=Thalamic_Input_model, threshold='rand()<rate_1*dt')



    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # MODEL EQUATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # excitatory neurons equation
    eqs_Exc = '''
        dV/dt  = (-(V-V_leak) - (IA_rec+IG_onexc+IA_ext)/g_leak_exc)/tau_m_exc : volt (unless refractory)

        IA_rec = g_syn_AMPAonExc * (V-V_syn_AMPA) * SA_rec              : ampere
        dSA_rec/dt = (-SA_rec + XA_rec)/tau_d_AMPAonExc                 : 1
        dXA_rec/dt = -XA_rec/tau_r_AMPAonExc                            : 1


        IG_onexc = g_syn_GABAonExc * (V-V_syn_GABA) * SG_onexc          : ampere
        dSG_onexc/dt = (-SG_onexc + XG_onexc)/tau_d_GABAonExc           : 1
        dXG_onexc/dt = -XG_onexc/tau_r_GABAonExc                        : 1

        IA_ext = g_syn_AMPA_ext_onExc * (V-V_syn_AMPA) * SA_ext         : ampere
        dSA_ext/dt = (-SA_ext + XA_ext)/tau_d_AMPAonExc                 : 1
        dXA_ext/dt = -XA_ext/tau_r_AMPAonExc                            : 1


        LFP = (abs(IG_onexc) + abs(IA_rec) + abs(IA_ext)) / g_leak_exc  : volt

    '''
    # inhibitory neurons equation
    eqs_Inh = '''
        dV/dt  = (-(V-V_leak) -(IA_oninh+IG_rec+IA_ext_i)/g_leak_inh)/tau_m_inh : volt (unless refractory)


        IA_oninh = g_syn_AMPAonInh * (V-V_syn_AMPA) * SA_oninh          : ampere
        dSA_oninh/dt = (-SA_oninh + XA_oninh)/tau_d_AMPAonInh           : 1
        dXA_oninh/dt = -XA_oninh/tau_r_AMPAonInh                        : 1


        IG_rec = g_syn_GABAonInh * (V-V_syn_GABA) * SG_rec              : ampere
        dSG_rec/dt = (-SG_rec + XG_rec)/tau_d_GABAonInh                 : 1
        dXG_rec/dt = -XG_rec/tau_r_GABAonInh                            : 1

        IA_ext_i = g_syn_AMPA_ext_onInh * (V-V_syn_AMPA) * SA_ext_i     : ampere
        dSA_ext_i/dt = (-SA_ext_i + XA_ext_i)/tau_d_AMPAonInh           : 1
        dXA_ext_i/dt = -XA_ext_i/tau_r_AMPAonInh                        : 1

    '''

    # excitatory neuronal model
    Exc = NeuronGroup(N_E, eqs_Exc, threshold='V > V_thr', reset='V=V_reset', refractory=ref_Exc, method='rk2')

    # inhibitory neuronal model
    Inh = NeuronGroup(N_I, eqs_Inh, threshold='V > V_thr', reset='V=V_reset', refractory=ref_Inh, method='rk2')

    #setting the initial conditions
    Exc.V = V_leak + (V_reset - V_leak) * rand(len(Exc))
    Inh.V = V_leak + (V_reset - V_leak) * rand(len(Inh))

    ############################################################################
    # Input to excitatory neurons
    # E-> E
    Wee = (tau_m_exc)/tau_r_AMPAonExc
    C_E_E = Synapses(Exc, Exc, on_pre='XA_rec += Wee', delay=2*ms)
    C_E_E.connect(condition='i != j',p=0.2)
    # I -> E
    Wie = (tau_m_exc)/tau_r_GABAonExc
    C_I_E = Synapses(Inh, Exc, on_pre='XG_onexc += Wie', delay=1*ms)
    C_I_E.connect(p=0.2)

    ############################################################################
    # Input to inhibitory neurons
    # I-> I
    Wii = (tau_m_inh)/tau_r_GABAonInh
    C_I_I = Synapses(Inh, Inh, on_pre='XG_rec += Wii', delay=1*ms)
    C_I_I.connect(condition='i != j',p=0.2)
    # E-> I
    Wei = (tau_m_inh)/tau_r_AMPAonInh
    C_E_I = Synapses(Exc, Inh, on_pre='XA_oninh += Wei', delay=2*ms)
    C_E_I.connect(p=0.2)

    ############################################################################
    # Thalamic input to model
    # ext -> E
    WextE = (tau_m_exc)/tau_r_AMPAonExc
    C_P_E = Synapses(input_on_Exc, Exc, on_pre='XA_ext += WextE', delay=2*ms)
    C_P_E.connect('i==j')
    # ext -> I
    WextI = (tau_m_inh)/tau_r_AMPAonInh
    C_P_I = Synapses(input_on_Inh, Inh, on_pre='XA_ext_i += WextI', delay=2*ms)
    C_P_I.connect('i==j')


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # MONITORS for output
    LFP_simulated = StateMonitor(Exc, 'LFP', record=True)

    run(duration, report=ProgressBar(), report_period=1*second)

    LFP_simulated=LFP_simulated.LFP.T
    total_LFP = []
    for lfp in range(len(LFP_simulated)):
        total_LFP.append(sum(LFP_simulated[lfp,:]))


    scipy.io.savemat('LFP_a0'+str(a_0)+'_v0'+str(int(v0))+'_trial'+str(trial)+'.mat', mdict={'LFP': total_LFP})



# main loop of simulations across trials
number_trials = 50
sustained_thalamic_input=[500,600,700,800,900]
store('modified')

for v0 in sustained_thalamic_input:
    restore('modified')
    v_0 = v0 * Hz # sustained component of the thalamic input
    store('modified1')
    for trial in range(number_trials):
        restore('modified1')
        a_0 = 0 # amplitude of thalamic periodic NB input

        NB_BB_mouseV1(trial,a_0,v_0)
