from mpmath import *
import numpy as np
import pandas as pd
import math
# import csv
import matplotlib.pyplot as plt

mp.dps = 15; mp.pretty = True
a = mpf(0.25)
b = mpf(0.25)
z = mpf(0.75)

# df = pd.read_excel('Sample3.xlsx')
df = pd.read_excel('SiN50nm.xlsx')  # for 50 nm took data at 90 mA
# df = pd.read_excel('SiN100nm.xlsx')  # for 50 nm took data at 90 mA

# df = pd.read_excel('Sample3.xlsx', index_col ='Frequency')

bh=5e-6 # half width heater
tf = 50e-9 # thin film thickness
diffusivity_Si = 0.00008  # diffusivity of substrate material
kappa_Si = 138.2 # Si substrate from data at 60 mA
V0 = 1.09 # rms voltage fundamental at the heater
R0 = 12.4 # heater resistance
I0 = V0/R0 # rms current fundamental at the heater
TCR = 0.002535 # temperature coefficient of heater Au
L = 0.001 # heater length
Prms = R0 * (I0**2)/L # power per unit length

#DTac for substrate using Cahill asymptote
Real_DTac_Si = -Prms/(2 * math.pi * kappa_Si)*(np.log(bh**2 / diffusivity_Si) + np.log(4*math.pi*df['Frequency']) - 1.844)
Imag_DTac_Si = -Prms/(4 * kappa_Si)

#DTac for thin film using V3omega
DTac_thin_film = 2 * df['X'] / (V0 * TCR)

# DT differential
DT = DTac_thin_film - Real_DTac_Si

# create dataframe
file = {
    'frequency': df['Frequency'],
    'Real_DTac_Si': Real_DTac_Si,
    'Imag_DTac_Si': Imag_DTac_Si,
    'DTac_thin_film': DTac_thin_film,
    'DT':DTac_thin_film - Real_DTac_Si,
    # 'kappa thin film': Prms * tf / (2 * bh * float('DT'))
}
data = pd.DataFrame(file)

data['kappa thin film'] = Prms * tf / (2 * bh * data['DT'])

print(data)

#
# plt.figure(1)
#plt.subplot(1, 2, 1)
# plt.plot(np.log(4*math.pi * data['frequency']), data['Real_DTac_Si'], 'r', label='Si substrate')
# plt.plot(np.log(4*math.pi * data['frequency']), data['DTac_thin_film'], 'b', label='SiN(50 nm) / Si')
# plt.xlabel('ln(2\u03C9)')
# plt.ylabel('In phase temperature oscillation (K)')
# # plt.xscale('log')
# plt.margins(x=0)
# plt.legend(loc='best')

# plt.figure(2)
# plt.subplot(1, 2, 2)
plt.plot(data['frequency'], data['kappa thin film'], 'b', label='SiN 50 nm')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Thermal conductivity (W/m.K)')
plt.ylim((1.2,1.5))
plt.margins(x=0)
plt.legend(loc='best')
plt.show()

mean_kapp = data['kappa thin film'].mean()

print(mean_kapp)