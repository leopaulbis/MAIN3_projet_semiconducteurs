import numpy as np
from matplotlib import pyplot as plt
from math import *

V0 = 1 # fundamental peak value
R0 = 12.983
I0=12.983 # room temp resistance of the heater
tcr = 0.00253 # Au tcr
D=9.07e-5 #diffusivité thermique
b=10e-6 #caractéristique fil
const=9.23e-1#constante d'ajustement
l=1e-3  #caractéristique fil
k=160 #conductivité thermique
C=log(pow(b,2)/D)-2*const
A=-1/(2*pi*k)
B=C*A

x=np.linspace(5.5e-3,1490,200)
y=np.log(2*x)
z=A*y+B
t=np.ones(200)


plt.xlabel('$ln(2\omega)$',fontsize=12)
plt.ylabel('amplitude thermique',fontsize=12)
plt.plot(y,z,label="phase")
plt.plot(y,-1/(4*k)*t,label="quadrature de phase",linestyle="--")
plt.legend()
plt.show()


