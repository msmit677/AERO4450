"""
Author: Matthew Smith (45326242)
Date: 15/04/2021
Title: AERO4450 Design Report Progress Check
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.sparse as sps
import scipy.sparse.linalg as splinalg

# Parameters
M_N2 = 28 # g/mol
M_F2 = 44 # g/mol
M_O2 = 32 # g/mol
mdot_a = 900 # g/s
mdot_f = 30 # g/s
T_in = 800 # K
T_act = 40000 # K
A = 5*10**11 # (1/s)
T_ref = 298.15 # K

# Thermodynamic Properties (formation enthalpy,Cp)
P = {'F2': (0,2000),
     'FO2': (-12,2200),
     'O2': (0,1090),
     'N2': (0,1170)
        }

# Preliminaries

# Task 1

b = (0.77*M_O2)/(0.23*M_N2)
m_air = 2*M_O2/0.23
Zst = M_F2/(M_F2 + m_air)
AFst = m_air/M_F2
Zavg = mdot_f/(mdot_a + mdot_f)
AFavg = mdot_a/mdot_f

print("Zst =",Zst)
print("AFst =",AFst)
print("Zavg =",Zavg)
print("AFavg =",AFavg)

# Task 2

Y_pc_max = 2*(M_F2/2 + M_O2)/(2*b*M_N2 + 2*(M_F2/2 + M_O2))

# Define the piecewise function Ypc(Z)
def Y_pc(Z):
    if Z <= Zst:
        grad = Y_pc_max/Zst
        c = 0
        Y = grad*Z + c
    if Z > Zst:
        grad = -Y_pc_max/(1-Zst)
        c = -grad
        Y = grad*Z + c
    return Y

# Plot Y_pc(Z)
plt.figure(figsize=(10,8))
plt.plot([0,Zst],[0,Y_pc_max],'b-')
plt.plot([Zst,1],[Y_pc_max,0],'b-')
plt.plot([0,Zst],[Y_pc_max,Y_pc_max],'r--')
plt.plot([Zst,Zst],[0,Y_pc_max],'r--')
plt.xticks([0.0,0.137,0.2,0.4,0.6,0.8,1.0])
plt.yticks([0.0,0.2,0.335,0.4,0.6,0.8,1.0])
plt.xlabel("Mixture Fraction (Z)")
plt.ylabel("Mass Fraction (Y)")
plt.title("Mass Fraction of FO2 vs. Mixture Fraction")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print("Ymax =",Y_pc_max)

# Task 4

# Find ao and af
ao = M_O2/(M_F2/2 + M_O2)
af = 0.5*M_F2/(M_F2/2 + M_O2)
print("ao =",ao)
print("af =",af)

def Y_O2(Z,Y_FO2):
    Y = 0.23*(1-Z) - ao*Y_FO2
    # Ensure that Y is non-negative
    if Y < 0:
        return 0
    else:
        return Y

def Y_F2(Z,Y_FO2):
    Y = Z - af*Y_FO2
    # Ensure that Y is non-negative
    if Y < 0:
        return 0
    else:
        return Y

# YN2 is a conserved scalar
def Y_N2(Z):
    return 0.77*(1-Z)

# Sum of all Y's should be 1
def Y_Total(Z,Y_FO2):
    return Y_O2(Z,Y_FO2) + Y_N2(Z) + Y_F2(Z,Y_FO2) + Y_FO2

# Create lists for all mass fractions
Zs = np.linspace(0,1,200)
O2 = [Y_O2(Z,Y_pc(Z)) for Z in Zs]
F2 = [Y_F2(Z,Y_pc(Z)) for Z in Zs]
N2 = [Y_N2(Z) for Z in Zs]
FO2 = [Y_pc(Z) for Z in Zs]
Total = [Y_Total(Z,Y_pc(Z)) for Z in Zs]

# Plot the mass fractions vs. Z
plt.figure(figsize=(10,8))
plt.plot(Zs,O2,'c-',label='O2')
plt.plot(Zs,F2,'m-',label='F2')
plt.plot(Zs,N2,'g-',label='N2')
plt.plot(Zs,Total,'k-',label='Sum')
plt.plot(Zs,FO2,'b-',label='FO2')
plt.plot([Zst,Zst],[0,1],'r--',label='Zst')
plt.xlabel("Mixture Fraction (Z)")
plt.ylabel("Mass Fraction (Y)")
plt.xlim(0,1)
plt.ylim(0,1.1)
plt.yticks([0.0,0.2,0.23,0.4,0.6,0.77,0.8,1.0])
plt.legend()
plt.show()

# Task 5

def phi(prop,Z,c):
    # Y_FO2 depends on combustion progress
    Y_FO2 = c*Y_pc(Z)
    # Define formation enthalpy
    if prop == 'h_f':
        val = (P['F2'][0]*Y_F2(Z,Y_FO2) + P['FO2'][0]*Y_FO2 + 
               P['O2'][0]*Y_O2(Z,Y_FO2) + P['N2'][0]*Y_N2(Z))*10**6
    # Define heat capacity
    if prop == 'Cp':
        val = (P['F2'][1]*Y_F2(Z,Y_FO2) + P['FO2'][1]*Y_FO2 + 
               P['O2'][1]*Y_O2(Z,Y_FO2) + P['N2'][1]*Y_N2(Z))
    # Define total enthalpy
    if prop == 'h':
        val = phi('h_f',Z,c)*Y_FO2 + (T_in - T_ref)*phi('Cp',Z,c)
    return val

# Task 6

# TotaL enthalpy is a conserved scalar
def h(Z,c):
    return phi('h',0,c) + Z*(phi('h',1,c) - phi('h',0,c))

def T(Z,c):
    return T_ref + (h(Z,c) - phi('h_f',Z,c))/phi('Cp',Z,c)

def W(Z,c):
    Y_FO2 = c*Y_pc(Z)
    return Y_F2(Z,Y_FO2)*Y_O2(Z,Y_FO2)*A*np.exp(-T_act/T(Z,c))

# Task 7
    
Zs = np.linspace(0,1,500)

# Plot the temperature vs. Z for different combustion progresses
plot1 = []
plot2 = []
plot3 = []
plot4 = []
for z in Zs:
    for c in [0,1/3,2/3,1]:
        if c == 1/3:
            plot1.append(T(z,c))
        if c == 2/3:
            plot2.append(T(z,c))
        if c == 0:
            plot3.append(T(z,c))
        if c == 1:
            plot4.append(T(z,c))
plt.figure(figsize=(10,8))
plt.plot(Zs,plot1,'r-',label='c = 1/3')
plt.plot(Zs,plot2,'b-',label='c = 2/3')
plt.plot(Zs,plot3,'g-',label='c = 0')
plt.plot(Zs,plot4,'m-',label='c = 1')
plt.title('Temperature vs. Z for Different c Values')
plt.xlabel('Mixture Fraction (Z)')
plt.ylabel('Temperature (K)')
plt.xlim(0,1)
plt.ylim(500,3500)
plt.yticks([500,800,1000,1500,2000,2500,3000,3500])
plt.legend()
plt.show()

# Plot the reaction rate vs. Z for different combustion progresses
plot1 = []
plot2 = []
for z in Zs:
    for c in [1/3,2/3]:
        if c == 1/3:
            plot1.append(W(z,c))
        if c == 2/3:
            plot2.append(W(z,c))
plt.figure(figsize=(10,8))
plt.plot(Zs,plot1,'r-',label='c = 1/3')
plt.plot(Zs,plot2,'b-',label='c = 2/3')
plt.title('Reaction Rate vs. Z for Different c Values')
plt.xlabel('Mixture Fraction (Z)')
plt.ylabel('W (1/s)')
plt.xlim(0,1)
plt.legend()
plt.show()

# Flamelet Model

# Task 1

nZ = 101
dZ = 1/(nZ-1)
Z_values = np.linspace(0,1,nZ)

# Define flamelet model that output the steady-state mass fractions for a given
# Nst
def flamelet_model(Nst):
    
    W_max = 500
    # Set time-step and CFL number
    dt = 0.01/W_max
    CFL = dt*Nst/(dZ**2)
    t = 0
    
    # Initial conditions
    current_Y = np.array([Y_pc(z) for z in Z_values])
    
    # Initial reaction rates
    current_W = np.zeros(nZ)
    for i in range(1,nZ-1):
        c = current_Y[i]/Y_pc(i*dZ)
        current_W[i] =  W(i*dZ,c)
    
    # Define implicit coefficient matrix
    implicit_matrix = ((1+2*CFL) * sps.eye(nZ, k=0)
                    -CFL * sps.eye(nZ, k=-1)
                    -CFL * sps.eye(nZ, k=+1))
                    
    # Dirichlet boundary conditions
    B = implicit_matrix.tolil()
    
    B[0,:], B[nZ-1,:] = 0, 0
    B[0,0], B[nZ-1,nZ-1] = 1, 1
    
    implicit_matrix = B.tocsr()
    
    # Begin general updates until steady-state solution is achieved or FO2 goes
    # extinct
    previous_Y = np.zeros(nZ)
    while abs(np.amax(current_Y) - np.amax(previous_Y)) > 1*10**-7:
        t += dt
        previous_Y = current_Y.copy()
        # Use sparse matrix solver
        current_Y = splinalg.spsolve(implicit_matrix,(previous_Y+current_W*dt))
        
        # Update reaction rates
        for i in range(1,nZ-1):
            c = current_Y[i]/Y_pc(i*dZ)
            current_W[i] = W(i*dZ,c)
    print('Number of time steps used =', t/dt)
    return current_Y

# Task 2

# Show steady-state solution for Nst = 30 (subcritical)
Y_ss = flamelet_model(30)
Ypc = [Y_pc(Z) for Z in Z_values]
plt.figure(figsize=(10,8))
plt.plot(Z_values,Y_ss,'b-',label='Steady-State Solution')
plt.plot(Z_values,Ypc,'r--',label='Y_pc(Z)')
plt.title('Mass Fraction of FO2 vs. Mixture Fraction for Nst = 30')
plt.xlabel('Mixture Fraction (Z)')
plt.ylabel('Mass Fraction (Y)')
plt.xlim(0,1)
plt.ylim(0,0.4)
plt.legend()
plt.show()

# Task 3

# Golden ratio
gr = (math.sqrt(5) + 1) / 2

# Define Golden-Section Search function
def gss(f, a, b, tol=0.01):
    # Find initial c and d values
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        # If f(c) goes to extinction, return 100
        if np.amax(f(c)) < 10**-3:
            x = 100
        # If f(c) reaches steady-state, return max Y_FO2
        else:
            x = np.amax(f(c))
        # If f(d) goes to extinction, return 100
        if np.amax(f(d)) < 10**-3:
            y = 100
        # If f(d) reaches steady-state, return max Y_FO2
        else:
            y = np.amax(f(d))
        # When f(c) and f(d) go to extinction, a = a, b = c
        if x and y == 100:
            b = c
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            continue
        # When f(c) and f(d) both have a steady solution, a = d, b = b
        if x and y > 10**-3:
            a = d
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            continue
        # If f(c) < f(d), b = d, a = a
        if x < y:
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2

#print(gss(flamelet_model,50.5,51,tol=0.01))

# ^^ uncomment this if you want to see the golden search result
# It takes roughly 5 mins to run though

# Critical value found form golden search
Ncr = 50.64387
print('Ncr =',Ncr)

# Plot critical solution
Y_ss = flamelet_model(Ncr)
Ypc = [Y_pc(Z) for Z in Z_values]
plt.figure(figsize=(10,8))
plt.plot(Z_values,Y_ss,'b-',label='Steady-State Solution')
plt.plot(Z_values,Ypc,'r--',label='Y_pc(Z)')
plt.title('Mass Fration of FO2 vs. Mixture Fraction for Ncr')
plt.xlabel('Mixture Fraction (Z)')
plt.ylabel('Mass Fraction (Y)')
plt.xlim(0,1)
plt.ylim(0,0.4)
plt.legend()
plt.show()

# Plot critical temperatures
Temps = np.zeros(nZ)
Temps[0] = T(0,0)
Temps[nZ-1] = T(1,0)
for i in range(1,nZ-1):
    c = Y_ss[i]/Y_pc(i*dZ)
    Temps[i] = T(i*dZ,c)
T_a = np.amax(Temps)
plt.figure(figsize=(10,8))
plt.plot(Z_values,Temps,'b-')
plt.plot([0,1],[T_a,T_a],'r--')
plt.title('Temperature vs. Mixture Fraction')
plt.xlabel('Mixture Fraction (Z)')
plt.ylabel('Temperature (K)')
plt.xlim(0,1)
plt.ylim(750,3000)
plt.yticks([750,1000,1250,1500,1750,2000,2250,2500,2750,2812.34,3000])
plt.show()

print('Adiabatic Temp =',T_a)
    
# Task 4

# Find residence time
t_res = (Zavg - Zavg**2)/(2*Ncr)

print('Residence Time =',t_res)
