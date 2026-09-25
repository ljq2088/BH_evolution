import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import os
import sys
sys.path.append('../..')
import my_module.my_module as my
import static_solution_symbol

plt.rc('font',family='Times New Roman')
plt.rc('mathtext',fontset='stix')

""" Spectrum
--------------------------------------------
"""
N=100
z,D1,D2=my.chebyshev_spectrum(N,0,6)

""" Initial value of the first iteration
--------------------------------------------
"""
'''1.Schwarzschild-AdS'''
f_tilde_Sch=1-2*z
chi_Sch=np.zeros(N)
phi_tilde_Sch=np.zeros(N)
first_iteration_initial_value=np.array([f_tilde_Sch,chi_Sch,phi_tilde_Sch])

'''2.从演化得到初值'''

""" Iteration configuration
--------------------------------------------
"""
df_tilde0=0.1
min_f_tilde0=-60
max_f_tilde0=-1/z[0]**2
f_tilde0=np.append(np.arange(min_f_tilde0,max_f_tilde0,df_tilde0),max_f_tilde0)
solution_number=len(f_tilde0)
static_solution=np.zeros((solution_number,3,N))
error=np.zeros((solution_number,N))
temperature=np.zeros(solution_number)
entropy_density=np.zeros(solution_number)
energy_density=np.zeros(solution_number)
z_H=np.zeros(solution_number)
r_H=np.zeros(solution_number)
phi_H=np.zeros(solution_number)

""" Iteration
--------------------------------------------
"""
arg_parameter=[]
source=1.889
static_solution[-1]=first_iteration_initial_value.copy()

time_start=time.time()
for i in range(0,solution_number):
    print(i)
    iteration_initial_value=static_solution[i-1].copy()
    boundary_condition=[(1,0,0,1,f_tilde0[i]),(2,2,-1,1,source)]
    static_solution[i]=my.newton_iteration(static_solution_symbol.variation_function,static_solution_symbol.equation_function,arg_parameter,z,iteration_initial_value,boundary_condition,D1,D2,1e-10)
    error[i]=static_solution_symbol.error_function(*arg_parameter,z,*my.numeric_arg_field(static_solution[i],D1,D2))
    print('max(error):',np.abs(error[i]).max())

    f=1+z**2*static_solution[i,0]
    f_fit=np.polynomial.Chebyshev.fit(z,f,N-1)
    phi_tilde_fit=np.polynomial.Chebyshev.fit(z,static_solution[i,2],N-1)
    z_H[i]=scipy.optimize.root(f_fit,z[np.abs(f).argmin()]).x[0]
    temperature[i]=static_solution_symbol.T_function(scipy.misc.derivative(f_fit,z_H[i],1e-6))
    entropy_density[i]=static_solution_symbol.s_function(z_H[i])
    energy_density[i]=-D1[-1].dot(static_solution[i,0])+source*D1[-1].dot(static_solution[i,2])/6
    r_H[i]=1/z_H[i]
    phi_H[i]=z_H[i]*phi_tilde_fit(z_H[i])
    print(f'T={temperature[i]}, s={entropy_density[i]}, epsilon={energy_density[i]}')
    print('max(phi_tilde):',np.abs(static_solution[i,2]).max())
    print('time spent(s):',time.time()-time_start)

print('The maximum value of the error:',np.abs(error).max())

plt.plot(temperature,energy_density)
plt.xlabel('$T$',fontsize=12)
plt.ylabel('$\epsilon$',fontsize=12)
plt.show()
plt.close()

""" EOS ($\kappa_4^2=1$)
--------------------------------------------
"""
file_location=f'./EOS/{source=}/z0={z[0]}/{N=}'
os.makedirs(file_location+'/1.data',exist_ok=True)
os.makedirs(file_location+'/2.picture',exist_ok=True)

plt.plot(temperature,energy_density)
plt.xlabel('$T$',fontsize=12)
plt.ylabel('$\epsilon$',fontsize=12)
plt.savefig(file_location+'/2.picture/0.energy_density_vs_T.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/0.energy_density_vs_T.pdf',bbox_inches='tight')
plt.show()
plt.close()

plt.plot(temperature,entropy_density)
plt.xlabel('$T$',fontsize=12)
plt.ylabel('$s$',fontsize=12)
plt.savefig(file_location+'/2.picture/1.entropy_density_vs_T.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/1.entropy_density_vs_T.pdf',bbox_inches='tight')
plt.show()
plt.close()

plt.plot(temperature,entropy_density/temperature**2)
plt.xlabel('$T$',fontsize=12)
plt.ylabel('$s/T^{2}$',fontsize=12)
plt.savefig(file_location+'/2.picture/2.EOS.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/2.EOS.pdf',bbox_inches='tight')
plt.close()

plt.plot(temperature,r_H)
plt.xlabel('$T$',fontsize=12)
plt.ylabel('$r_{H}$',fontsize=12)
plt.savefig(file_location+'/2.picture/3.r_H_vs_T.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/3.r_H_vs_T.pdf',bbox_inches='tight')
plt.close()

plt.plot(temperature,phi_H)
plt.xlabel('$T$',fontsize=12)
plt.ylabel('$\phi_{H}$',fontsize=12)
plt.savefig(file_location+'/2.picture/4.phi_H_vs_T.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/4.phi_H_vs_T.pdf',bbox_inches='tight')
plt.close()

plt.plot(f_tilde0,error[:,0])
plt.xlabel(r'$\tilde{f}[0]$',fontsize=12)
plt.ylabel('error at inner boundary')
plt.savefig(file_location+'/2.picture/6.error_inner_boundary.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/6.error_inner_boundary.pdf',bbox_inches='tight')
plt.close()

error_fit=[np.polynomial.Chebyshev.fit(z,error[i],N-1) for i in range(0,solution_number)]
error_H=[error_fit[i](z_H[i]) for i in range(0,solution_number)]
plt.plot(f_tilde0,error_H)
plt.xlabel(r'$\tilde{f}[0]$',fontsize=12)
plt.ylabel('error at horizon')
plt.savefig(file_location+'/2.picture/7.error_horizon.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/7.error_horizon.pdf',bbox_inches='tight')
plt.close()

plt.plot(f_tilde0,error[:,-1])
plt.xlabel(r'$\tilde{f}[0]$',fontsize=12)
plt.ylabel('error at boundary')
plt.savefig(file_location+'/2.picture/8.error_boundary.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/8.error_boundary.pdf',bbox_inches='tight')
plt.close()

plt.plot(f_tilde0,np.abs(error).max(axis=1))
plt.xlabel(r'$\tilde{f}[0]$',fontsize=12)
plt.ylabel('max error')
plt.savefig(file_location+'/2.picture/9.max_error.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/9.max_error.pdf',bbox_inches='tight')
plt.close()

""" Save data
--------------------------------------------
"""
np.save(file_location+'/1.data/1.static_solution.npy',static_solution)
np.save(file_location+'/1.data/2.temperature.npy',temperature)
np.save(file_location+'/1.data/3.entropy_density.npy',entropy_density)
np.save(file_location+'/1.data/4.energy_density.npy',energy_density)
np.save(file_location+'/1.data/5.error.npy',error)
np.save(file_location+'/1.data/6.z_H.npy',z_H)
np.save(file_location+'/1.data/7.r_H.npy',r_H)
np.save(file_location+'/1.data/8.phi_H.npy',phi_H)