from mpmath import *
import numpy as np
import cmath
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker
from matplotlib import rc

# An instance of RcParams for handling default Matplotlib values.
params = {
    'figure.figsize': [16, 9],
    'axes.labelsize' : 12,
    'figure.autolayout': True,
    'mathtext.rm': 'cm',
    'font.family': 'Times New Roman',
    'font.serif': "Times New Roman",
    }
matplotlib.rcParams.update(params)
# Colour code for the plot
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
columns = ['frequency', 'Thermal_freq']
xlabels = ['Frequency (Hz)', 'Thermal excitation frequency (Hz)']
linestyles = ['-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted']

# Resolution complex number
mp.dps = 10                              # Mpmath setting, dps is the decimal precision
mp.pretty = True                            # Setting the mp.pretty option will use the str()-style output for repr() as well
a = mpf(0.25)                               # a = 0.25
b = mpf(0.25)                               # b = 0.25
z = mpf(0.75)                               # z = 0.25

# 3 omega measuresure Parameters
D=float(input("entrer la valeur de la diffusivité: "))
#D = 9.07e-5                                 # Thermal diffusivity. 1.14e-3 diamond diffusivity, 200 -1000 e-8 for BN - CNT D = 4.6e-4, 0.025e4 polymers PVP aqueous, 1.3e-7
k=float(input("entrer la valeur de la conductivité: "))
#k = 140                                     # Thermal conductivity. Unit: W/mK (140-160 Si, Diamond - 220 - 420 for BN) CNT 750 W/mK  or from 50 to 80 W/mK, 2000 W/mK, 0.27 PVP
gamma = 0.5772  #Euler constant
V0=float(input("entrer la valeur de V0: "))
#V0 = 1
R0=float(input("entrer la valeur de la résistance: "))                                     # Fundamental peak value
#R0 =80
TCR=float(input("entrer la valeur du TCR: "))                                # Room temp resistance of the heater
#TCR = 0.00253                               # Temperature Coefficient of the electrical Resistivity (TRC), usually in the ~10^-3 /K. (Au TCR = 0.002535 /K)
L=float(input("entrer la valeur de la longeur de l'élément chauffant: '"))
#L = 0.001                                   # Length of the heater
P = V0**2/(2*R0)                            # W/m power per unit length
frequency = np.arange(0.3, 30001, 1000)   # Frequency range
bh = np.array([0.2e-6, 0.4e-6, 1e-6])       # Heater half width
ts = np.array([380e-6])                   # Substrate thickness
linear_limit = 25*D/(4*(math.pi)*(ts**2))   # Limits linear regimes
planar_limit = np.array(D/(100*(math.pi)*(bh**2)))      # Limits planar regimes

# Cartesian product of input variables
idx = pd.MultiIndex.from_product([bh, ts,  frequency], names=["bh", 'ts', "frequency"])
# Reset the index of the DataFrame, and use the default one instead. If the DataFrame has a MultiIndex, this method can remove one or more levels.
df = pd.DataFrame(index=idx).reset_index()

omega = (df["bh"].values**2 * df["frequency"].values) * (2 * math.pi / D)           # Omega is vectorized naturally.
thermal_freq = (2 * df['frequency'])                                                # Thermal frequency is 2nd harmonic
T_depth = (np.sqrt(2*D / (2 * (math.pi) * thermal_freq)))/1e-6                      # Thermal penetration depth function of omega themal not electrical - Cahill
                                                                                    # sqrt(2D/wt)

def f_u(omega_elem):
    # The function meijerg is evaluated as a combination of hypergeometric series
    val1 = (-j*P / (4*L*k*math.pi * omega_elem)) * meijerg([[1, 3 / 2], []], [[1, 1], [0.5, 0]], j * omega_elem)
    asympt = (P / (k*L*math.pi)) * (-(1 / 2) * np.log(omega_elem) + 3 / 2 - gamma - j * ((math.pi) / 4))
    amplitude = math.sqrt(np.real(val1) ** 2 + np.imag(val1) ** 2)
    phase = math.degrees(math.atan(np.imag(val1) / np.real(val1)))
    V3omega_asympt = 0.5 * V0 * TCR * asympt                                        # calculate thrid harmonic from DT
    V3omega = 0.5 * V0 * TCR * val1                                                 # calculate thrid harmonic from DT

    return val1, asympt, amplitude, phase, V3omega_asympt, V3omega

# return a tuple of array. Remember to assign two otypes.
f_u_vec = np.vectorize(f_u, otypes=[np.complex128,      # val1 is complex number
                                    np.complex128,      # asympt is complex number
                                    np.ndarray,         # amplitude is array
                                    np.ndarray,         # phase is array
                                    np.complex128,      # V3omega_asympt is complex number
                                    np.complex128]      # V3omega is complex number
                                    )

tup = f_u_vec(omega)                    # tuple of arrays: (val1, asympt, amplitude, phase, V3omega_asympt, V3omega)
df['Thermal_freq'] = thermal_freq       # Insert thermal frequency into data frame
df['T_depth'] = T_depth                 # Insert thermal penetration depth into data frame
df["Re"] = np.real(tup[0])              # Insert val1 real part into data frame
df["Im"] = np.imag(tup[0])              # Insert val1 imagine part into data frame
df["Asympt_Re"] = np.real(tup[1])       # Insert asympt real part into data frame
df["Asympt_Im"] = np.imag(tup[1])       # Insert asympt imagine part into data frame
df['Amplitude'] = tup[2]                # Insert amplitude into data frame
df['Phase'] = tup[3]                    # Insert phase into data frame
df["V3asympt_Re"] = np.real(tup[4])     # Insert V3omega_asympt real part into data frame
df["V3asympt_Im"] = np.imag(tup[4])     # Insert V3omega_asympt imagine part into data frame
df["V3_Re"] = np.real(tup[5])           # Insert V3omega real part into data frame
df["V3_Im"] = np.imag(tup[5])           # Insert V3omega imagine part into data frame

# Result in dataframes
# Generated CSV file based on the bh list.
for bh_elem in bh:
    fname = f"bh={bh_elem:.4e}.csv"
    df_save = df[(df["bh"] == bh_elem)]
    df_save.to_csv(fname)

fig, axs = plt.subplots(1,2)                      # Initial Figure 2x2

locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1, ),numticks=100)
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(2, 10) * .1,numticks=100)
for i in range(2):

    ########################
    # Plot Column 2 Figure #
    ########################
    if i==0:
        axs[i].tick_params(which="both", axis="both",direction="in")
        axs_top2 = axs[i].twiny()
        axs_top2.invert_xaxis()
        axs_top2.tick_params(which="both", axis="both",direction="in")
        axs[i].minorticks_on()

        # Plot the second figure
        axs[i].set_ylabel('Third harmonic voltage V3$_\mathrm{\u03C9}$ (V)')
        axs[i].set_xlabel('Thermal excitation frequency (Hz)')
        axs[i].set_xscale('log')
        axs_top2.set_xticks(df['T_depth'])
        axs_top2.set_xscale('log')
        axs_top2.set_xlabel('Thermal penetration depth: q$^\mathrm{-1}$ = (2D/\u03C9$_\mathrm{t}$)$^\mathrm{1/2}$' + ' (\u03BCm)', labelpad=10)
        axs[i].xaxis.set_major_locator(locmaj)
        axs_top2.xaxis.set_major_locator(locmaj)
        axs[i].xaxis.set_minor_locator(locmin)
        axs_top2.xaxis.set_minor_locator(locmin)
        axs[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs_top2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for axis in [axs[i].xaxis, axs[i].yaxis, axs_top2.xaxis, axs_top2.yaxis]:
            formatter = FuncFormatter(lambda x, _: '{:.10g}'.format(x))
            axis.set_major_formatter(formatter)
        axs[i].margins(x=0)
        axs_top2.margins(x=0)
        axs[i].text(20, -0.004, r'Out-of-phase', fontsize=10)
        axs[i].text(20, 0.04, r'In-phase', fontsize=10)

        for bh_array2 in bh:
            # Re and Imag
            # V3 omega Real part Vs Thermal Frequency, bh = to bh[1]
            axs[i].plot(
                df[(df["bh"] == bh_array2) & (df["ts"] == ts[0])]['Thermal_freq'],
                df[(df["bh"] == bh_array2) & (df["ts"] == ts[0])]['V3_Re'],
                colours[np.where(bh == bh_array2)[0][0]],
                label='bh=' + '{:.1f}'.format((bh_array2/1e-6)) + '\u03BCm - ' + r't$_{\rm s}$' + ' = ' + '{:.0f}'.format((ts[0]/1e-6)) + 'um')
            # V3 omega Real part Vs Thermal penetration depth, bh = to bh[1]
            axs_top2.plot(
                df[(df["bh"] == bh_array2) & (df["ts"] == ts[0])]['T_depth'],
                df[(df["bh"] == bh_array2) & (df["ts"] == ts[0])]['V3_Re'],
                colours[np.where(bh == bh_array2)[0][0]],
                label='bh=' + '{:.1f}'.format((bh_array2/1e-6)) + '\u03BCm - ' + r't$_{\rm s}$' + ' = ' + '{:.0f}'.format((ts[0]/1e-6)) + 'um')
            # V3 omega Imagine part Vs Thermal Frequency, bh = to bh[1]
            axs[i].plot(
                df[(df["bh"] == bh_array2) & (df["ts"] == ts[0])]['Thermal_freq'],
                df[(df["bh"] == bh_array2) & (df["ts"] == ts[0])]['V3_Im'],
                colours[np.where(bh == bh_array2)[0][0]],
                label='_nolegend_')
            # V3 omega Imagine part Vs Thermal penetration depth, bh = to bh[1]
            axs_top2.plot(
                df[(df["bh"] == bh_array2) & (df["ts"] == ts[0])]['T_depth'],
                df[(df["bh"] == bh_array2) & (df["ts"] == ts[0])]['V3_Im'],
                colours[np.where(bh == bh_array2)[0][0]],
                label='_nolegend_')
        # V3asympt Real part Vs Thermal Frequency, bh = to bh[1]
        axs[i].plot(
            df[(df["bh"] == bh[bh.size-1]) & (df["ts"] == ts[0])]['Thermal_freq'],
            df[(df["bh"] == bh[bh.size-1]) & (df["ts"] == ts[0])]['V3asympt_Re'],
            color= 'black',
            linestyle='--',
            label='bh=' + '{0:.1f}'.format((bh[bh.size-1]/1e-6)) + '\u03BCm -  ' + r't$_{\rm s}$' + ' = ' + '{:.0f}'.format((ts[0]/1e-6)) + 'um (asympt)')
        # V3asympt Imagine part Vs Thermal Frequency, bh = to bh[1]
        axs[i].plot(
            df[(df["bh"] == bh[bh.size-1]) & (df["ts"] == ts[0])]['Thermal_freq'],
            df[(df["bh"] == bh[bh.size-1]) & (df["ts"] == ts[0])]['V3asympt_Im'],
            color= 'black',
            linestyle='--',
            label='_nolegend_')

        axs[i].legend(loc='center left', frameon=False, prop={'family': 'Times New Roman'})
    else:
        axs[i].tick_params(which="both", axis="both",direction="in")
        axs_top3 = axs[i].twinx()
        axs_top3.tick_params(which="both", axis="both",direction="in")
        # The fourth figure setting
        axs[i].set_ylabel('Amplitude', fontsize=12)
        axs[i].yaxis.label.set_color('blue')
        axs[i].tick_params(axis='y', colors='blue')
        axs_top3.set_ylabel('Phase (${^o}$)', fontsize=12)
        axs_top3.yaxis.label.set_color('red')
        axs_top3.tick_params(axis='y', colors='red')
        axs[i].set_xlabel('Frequency (Hz)', fontsize=12)
        axs[i].set_xscale('log')
        axs[i].xaxis.set_major_locator(locmaj)
        axs[i].xaxis.set_minor_locator(locmin)
        for axis in [axs[i].xaxis, axs[i].yaxis]:
            formatter = FuncFormatter(lambda x, _: '{:.16g}'.format(x))
            axis.set_major_formatter(formatter)
        axs[i].margins(x=0)
        axs_top3.margins(x=0)

        axs[i].text(60, 28, r'Amplitude', color="blue", fontsize=10)
        axs_top3.text(60, 0.4, r'Phase', color="red", fontsize=10)

        for bh_array3 in bh:
            # Amplitude and phase
            # Amplitude Vs Frequency, bh = to bh[1]
            axs[i].plot(
                df[(df["bh"] == bh_array3) & (df["ts"] == ts[0])]['frequency'],
                df[(df["bh"] == bh_array3) & (df["ts"] == ts[0])]['Amplitude'],
                color= 'blue',
                linestyle=linestyles[np.where(bh == bh_array3)[0][0]],
                label='bh=' + '{:.0f}'.format((bh_array3/1e-6)) + '\u03BCm - ' + r't$_{\rm s}$' + ' = ' + '{:.0f}'.format((ts[0]/1e-6)) + 'um')
            # Phase Vs Frequency, bh = to bh[1]
            axs[i].plot(
                df[(df["bh"] == bh_array3) & (df["ts"] == ts[0])]['frequency'],
                df[(df["bh"] == bh_array3) & (df["ts"] == ts[0])]['Phase'],
                color= 'red',
                linestyle=linestyles[np.where(bh == bh_array3)[0][0]],
                label='bh=' + '{:.0f}'.format((bh_array3/1e-6)) + '\u03BCm - ' + r't$_{\rm s}$' + ' = ' + '{:.0f}'.format((ts[0]/1e-6)) + 'um')

        axs[i].legend(loc='best', frameon=False)

plt.show()