import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import os
import sys
sys.path.append('..')
import my_module.my_module as my
import stationary_solution_symbol as symbol

plt.rc('font',family='Times New Roman')
plt.rc('mathtext',fontset='stix')

def numeric_arg_field(fields,differentiation_matrixes):
    field_number=len(fields)
    result=[]
    for i in range(0,field_number):
        result=result+[fields[i]]+[differentiation_matrixes[j].dot(fields[i]) for j in range(0,len(differentiation_matrixes))]
    return np.array(result)

def newton_iteration(variation_function,equation_function,parameters,variables,point_numbers,fields_initial,bc,differentiation_matrixes,precision=1e-10,Print=False):
    """
    Function
    ----------
    Solve nonlinear second-order ordinary differential equations by fixed point iteration method.

    Parameters
    ----------
    variation_function : function
        Jacobian matrix variational function
    equation_function : function
        nonlinear second-order ordinary differential equations
        !!! Note : Not the opposite
    parameters : tuple or list
        parameters in the equations
    variables : numpy.array
        the spectral points of the independent variables of the unit function
    fields_initial : two-dimension numpy.array
        the initial value of each field function
        !!! Note : The order is consistent with the variational order
    bc : two-dimension tuple
        bc[n] is a tuple whose size is 5, corresponding a boundary condition:
        bc[n]=(i-th equation,j-th field,location:0/-1,type:1/2/3,boundary value)
        !!! Note : Boundary value is not the opposite
        !!! Example: 第一类：f(0)=a
                            \delta f=-(f(0)-a)
                    第二类：f'(0)=b
                            \delta f'=-(f'(0)-b)
                    第三类：f''(0)=c
                            \delta f''=-(f''(0)-c)
                    第四类：允许有多个场，允许有两阶及以下的导数，此时bc[n][4]表示的boundary value需要输入一个表示边界条件并且等于0的sympy表达式：eq=a f''+b f'+c f+d g+e在lambdify之后的函数，而bc[n][1]不再表示第几个场，而是输入eq的variation在lambdify之后的函数。
    Dz : numpy.array
        first-order differentiation matrix
    Dz2 : numpy.array
        second-order differentiation matrix
    precision : float
        Accuracy reached by iteration
        the default is 1e-10
        !!! Note : The sum of the accuracy of each component
        !!! 现在用的是max

    Returns
    -------
    result : two-dimension numpy.array
        the value of each field function
        !!! Note : The order is consistent with the initialvalue

    """
    variables=np.array(variables)
    if variables.ndim ==1:
        variables=[variables]
    else:
        variables=list(variables)

    fields=fields_initial.copy()
    total_point_number=len(variables[0])
    field_number=len(fields)
    boundary_number=len(bc)
    precision_old=precision
    ones=np.ones(total_point_number)

    delta_f=np.ones(field_number*total_point_number)
    n=1
    while np.abs(delta_f).max() > precision:
        arguments=list(parameters)+variables+list(numeric_arg_field(fields,differentiation_matrixes))
        variation_value=variation_function(*arguments)
        equation=equation_function(*arguments)

        M=np.zeros((field_number,field_number,total_point_number,total_point_number))
        for i in range(0,field_number):
            for j in range(0,field_number):
                M[i,j]=np.diag(variation_value[i][j][0]*ones)
                for k in range(0,len(differentiation_matrixes)):
                    M[i,j]=M[i,j]+np.diag(variation_value[i][j][k+1]*ones).dot(differentiation_matrixes[k])
        E=np.array(equation).flatten()
        if Print:
            print('max(equation):',np.abs(E).max())
        A=np.vstack(tuple([np.hstack(tuple(M[i,:])) for i in range(0,field_number)]))

        extra_point_number=int(total_point_number/point_numbers[0])
        for i in range(0,boundary_number):
            row_index1=bc[i][0]*total_point_number-bc[i][2]*(total_point_number-extra_point_number)
            row_index2=row_index1+extra_point_number
            if type(bc[i][1]) == int:
                column_index1=bc[i][1]*total_point_number
                column_index2=column_index1+total_point_number
            row_index3=-bc[i][2]*(total_point_number-extra_point_number)
            row_index4=row_index3+extra_point_number
            if (bc[i][3] != 5) and (bc[i][3] != 6):
                A[row_index1:row_index2,:]=0
            else:
                A[row_index2-1,:]=0
            if bc[i][3] == 1:
                A[row_index1:row_index2,column_index1:column_index2]=np.eye(total_point_number)[row_index3:row_index4]
                E[row_index1:row_index2]=fields[bc[i][1]][row_index3:row_index4]-bc[i][4]
                '''
            elif bc[i][3] == 2:
                A[row_index1:row_index2,column_index1:column_index2]=Dz[bc[i][2]]
                E[row_index1:row_index2]=differentiation_matrixes[0].dot(fields[bc[i][1]])[bc[i][2]]-bc[i][4]
            elif bc[i][3] == 3:
                A[row_index1:row_index2,column_index1:column_index2]=Dz2[bc[i][2]]
                E[row_index1:row_index2]=differentiation_matrixes[2].dot(fields[bc[i][1]])[bc[i][2]]-bc[i][4]
                '''
            elif bc[i][3] == 4:
                variation_bc=bc[i][1](*arguments)
                equation_bc=bc[i][4](*arguments)
                M_bc=np.zeros((1,field_number,total_point_number,total_point_number))
                for j in range(0,1):
                        for k in range(0,field_number):
                            M_bc[j,k]=np.diag(variation_bc[j][k][0]*ones)
                            for l in range(0,len(differentiation_matrixes)):
                                M_bc[j,k]=M_bc[j,k]+np.diag(variation_bc[j][k][l+1]*ones).dot(differentiation_matrixes[l])
                A_bc=np.vstack(tuple([np.hstack(tuple(M_bc[j,:])) for j in range(0,1)]))

                A[row_index1:row_index2,:]=A_bc[row_index3:row_index4]
                E[row_index1:row_index2]=equation_bc[row_index3:row_index4]
            elif bc[i][3] == 5:
                A[row_index2-1,column_index1:column_index2]=Ix[row_index4-1]
                E[row_index2-1]=Ix.dot(fields[bc[i][1]])[row_index4-1]-bc[i][4]
            elif bc[i][3] == 6:
                A[row_index2-1,column_index1:column_index2]=Ix.dot(Pz)[row_index4-1]
                E[row_index2-1]=Ix.dot(Pz).dot(fields[bc[i][1]])[row_index4-1]-bc[i][4]
            else:
                raise ValueError(f'there are no {bc[i][3]}th kind of boundary conditions')
        delta_f=np.linalg.solve(A,-E)
        fields=fields+delta_f.reshape(field_number,total_point_number)
        if n >= 500:
            Print=True
            precision=precision+0.1*precision_old
        if Print:
            print(n)
            print('max(delta_f):',np.abs(delta_f).max())
        n=n+1
    
    arguments=list(parameters)+variables+list(numeric_arg_field(fields,differentiation_matrixes))
    equation=equation_function(*arguments)
    E=np.array(equation).flatten()
    print('max(equation):',np.abs(E).max())
    if n > 500:
        print('precision:',precision)
    return fields

def horizon_condition(h,f,chi,xi,A,B):
    eAchi=np.exp(A+chi)
    coshB=np.cosh(B)
    px_h=Dx_even.dot(h)
    interp_h=my.chebyshev_interp(h,z0,Warn=False)
    f_h=np.array([interp_h[i].dot(f[:,i]) for i in range(0,m)])
    xi_h=np.array([interp_h[i].dot(xi[:,i]) for i in range(0,m)])
    eAchi_h=np.array([interp_h[i].dot(eAchi[:,i]) for i in range(0,m)])
    coshB_h=np.array([interp_h[i].dot(coshB[:,i]) for i in range(0,m)])

    print(xi_h+px_h/eAchi_h*coshB_h)
    print(f_h+xi_h*px_h)
    print(f_h-px_h**2/eAchi_h*coshB_h)
    print(f_h-xi_h**2*eAchi_h/coshB_h)

    return Dx_odd.dot(xi_h+px_h/eAchi_h*coshB_h)+1/np.tan(x0)*(xi_h+px_h/eAchi_h*coshB_h)+(f_h-px_h**2/eAchi_h*coshB_h)/h

def get_temperature(h,chi,xi,eta,f,A,B):
    interp_h=my.chebyshev_interp(h,z0,Warn=False)
    # chi_h=np.array([interp_h[i].dot(chi[:,i]) for i in range(0,m)])
    # xi_h=np.array([interp_h[i].dot(xi[:,i]) for i in range(0,m)])
    # eta_h=np.array([interp_h[i].dot(eta[:,i]) for i in range(0,m)])
    # f_h=np.array([interp_h[i].dot(f[:,i]) for i in range(0,m)])
    # A_h=np.array([interp_h[i].dot(A[:,i]) for i in range(0,m)])
    # B_h=np.array([interp_h[i].dot(B[:,i]) for i in range(0,m)])
    # pz_chi_h=np.array([interp_h[i].dot(Dz.dot(chi)[:,i]) for i in range(0,m)])
    # pz_xi_h=np.array([interp_h[i].dot(Dz.dot(xi)[:,i]) for i in range(0,m)])
    # pz_eta_h=np.array([interp_h[i].dot(Dz.dot(eta)[:,i]) for i in range(0,m)])
    # pz_f_h=np.array([interp_h[i].dot(Dz.dot(f)[:,i]) for i in range(0,m)])
    # pz_A_h=np.array([interp_h[i].dot(Dz.dot(A)[:,i]) for i in range(0,m)])
    # pz_B_h=np.array([interp_h[i].dot(Dz.dot(B)[:,i]) for i in range(0,m)])

    I=f-xi**2*np.exp(A+chi)/np.cosh(B)
    pz_I=Dz.dot(I)
    pz_I_h=np.array([interp_h[i].dot(pz_I[:,i]) for i in range(0,m)])

    temperature=pz_I_h/2/(2*np.pi)
    print(temperature)

    return np.average(-temperature)

""" Spectrum
--------------------------------------------
"""
N=50
z0,Dz,Dz2=my.chebyshev_spectrum(N,0,1.3)
m=30
M=2*m
x0_full,Dx,Dx2,Dx3=my.fourier_spectrum(M,-np.pi+np.pi/M,2*np.pi)

x0=x0_full[m:]
Dx_even=Dx[m:,m-1::-1]+Dx[m:,m:]
Dx_odd=-Dx[m:,m-1::-1]+Dx[m:,m:]
Dx2_even=Dx2[m:,m-1::-1]+Dx2[m:,m:]
# Dx2_even=Dx_odd.dot(Dx_even)

z=np.kron(z0,np.ones(m))
x=np.kron(np.ones(N),x0)

Pz=np.kron(Dz,np.eye(m))
Px=np.kron(np.eye(N),Dx_even)
Pz2=np.kron(Dz2,np.eye(m))
PzPx=np.kron(Dz,Dx_even)
Px2=np.kron(np.eye(N),Dx2_even)
differentiation_matrixes=[Pz,Px,Pz2,PzPx,Px2]

ix=np.outer(np.ones(m),np.sin(x0)*np.pi/m/2)
Ix=np.kron(np.eye(N),ix)
# Ix=np.eye(N*m)

""" Initial value of the first iteration
--------------------------------------------
"""
'''1.Schwarzschild-AdS'''
f_tilde_Sch=1-2*z
chi_Sch=np.zeros(N*m)
xi_tilde_Sch=np.zeros(N*m)
eta_tilde_Sch=np.zeros(N*m)
A_Sch=np.zeros(N*m)
B_Sch=np.zeros(N*m)
phi_tilde_Sch=np.zeros(N*m)

first_iteration_initial_value=np.array([chi_Sch,xi_tilde_Sch,f_tilde_Sch,A_Sch,phi_tilde_Sch])

'''2.static_solution'''
'''
initial_static_solution=np.load('./1.static_solution.npy')[::-5]
initial_static_solution=np.load('./1.static_solution.npy')[-1].reshape(1,3,N)
f_tilde0=initial_static_solution[:,0,0]
eta3=np.arange(0,0.1,0.1)

f_tilde_static=np.kron(initial_static_solution[i,0],np.ones(m))
chi_static=np.kron(initial_static_solution[i,1],np.ones(m))
phi_tilde_static=np.kron(initial_static_solution[i,2],np.ones(m))
first_iteration_initial_value=np.array([chi_Sch,xi_tilde_Sch,f_tilde_Sch,A_Sch,phi_tilde_Sch])
'''

'''3.final_state_no_spin'''
constraint_fields=np.load('./final_state_no_spin/constraint_fields.npy')
evolution_fields=np.load('./final_state_no_spin/evolution_fields.npy',allow_pickle=True)
h=np.load('./final_state_no_spin/h.npy')

t_count=2000+1
chi=np.full((t_count,N,m),np.nan)
xi=np.full((t_count,N,m),np.nan)
f=np.full((t_count,N,m),np.nan)
f_tilde=np.full((t_count,N,m),np.nan)
A=np.full((t_count,N,m),np.nan)
phi=np.full((t_count,N,m),np.nan)
xi3=np.full((t_count,m),np.nan)
f3=np.full((t_count,m),np.nan)
t_last=2000
A,phi,xi3,f3=evolution_fields[-1]
chi,xi,f,f_tilde=constraint_fields[-1]

xi_tilde=xi/z0[:,None]**3*np.sin(x0)
xi_tilde[-1]=xi3*np.sin(x0)
A_tilde=A/z0[:,None]**3
A_tilde[-1]=Dz[-1].dot(Dz2).dot(A)/6
phi_tilde=phi/z0[:,None]
phi_tilde[-1]=Dz[-1].dot(phi)

first_iteration_initial_value=np.array([chi.reshape(N*m),xi_tilde.reshape(N*m),f_tilde.reshape(N*m),A_tilde.reshape(N*m),phi_tilde.reshape(N*m)])

f_tilde0=[f_tilde[0,0]]
eta3=np.arange(0,0.1,0.1)

'''4.final_state_spin'''
'''
constraint_fields=np.load('./final_state_spin/constraint_fields.npy')
evolution_fields=np.load('./final_state_spin/evolution_fields.npy',allow_pickle=True)

A_spherical,B0,phi,xi3,eta3,f3=evolution_fields[-1]
chi0,xi,eta,f,f_tilde=constraint_fields[-1]
xi3_spherical=xi3/np.sin(x0)
xi_spherical=xi/np.sin(x0)

xi_tilde0=xi_spherical/z0[:,None]**3*np.sin(x0)
xi_tilde0[-1]=xi3_spherical*np.sin(x0)
eta_tilde0=eta/z0[:,None]**3
eta_tilde0[-1]=eta3

first_iteration_initial_value=np.array([chi0.reshape(N*m),xi_tilde0.reshape(N*m),eta_tilde0.reshape(N*m),f_tilde.reshape(N*m),A_spherical.reshape(N*m),B0.reshape(N*m),phi_tilde_Sch])
'''

""" Iteration
--------------------------------------------
"""
arg_parameter=[]
source=2

f_tilde0_number=len(f_tilde0)
eta3_number=len(eta3)
stationary_solution=np.zeros((eta3_number,f_tilde0_number,5,N*m))
error=np.zeros((eta3_number,f_tilde0_number,N*m))
temperature=np.zeros((eta3_number,f_tilde0_number))
entropy=np.zeros((eta3_number,f_tilde0_number))
energy_density=np.zeros((eta3_number,f_tilde0_number))
h=np.zeros((eta3_number,f_tilde0_number,m))

for i in range(0,f_tilde0_number):
    print(f'f_tilde0={f_tilde0[i]}')
    
    stationary_solution[-1,i]=first_iteration_initial_value.copy()

    '''2.从演化得到初值'''

    """ Iteration configuration
    --------------------------------------------
    """
    # df_tilde0=0.1
    # min_f_tilde0=-15
    # max_f_tilde0=-1/z[0]**2
    # f_tilde0=np.arange(min_f_tilde0,max_f_tilde0,df_tilde0)
    # f_tilde0=[f_tilde_Sch[0]]
    # solution_number=len(f_tilde0)
    # stationary_solution=np.zeros((solution_number,len(first_iteration_initial_value),N*m))
    # error=np.zeros((solution_number,N*m))
    # temperature=np.zeros(solution_number)
    # entropy=np.zeros(solution_number)
    # energy_density=np.zeros(solution_number)
    # h=np.zeros((solution_number,m))

    for j in range(0,eta3_number):
        print(f'eta3={eta3[j]}')
        """ Iteration
        --------------------------------------------
        """
        time_start=time.time()
        iteration_initial_value=stationary_solution[j-1,i].copy()
        boundary_condition=[(1,1,-1,1,0),(4,4,-1,1,source),(2,2,0,1,f_tilde[0]),(3,3,-1,1,0)]
        # boundary_condition=[(1,1,-1,1,xi_tilde[-1]),(4,4,-1,1,source),(2,symbol.boundary_condition_f_variation_function,-1,4,symbol.boundary_condition_f_function),(2,2,0,5,f_tilde0[0])]
        stationary_solution[j,i]=newton_iteration(symbol.variation_function,symbol.equations_function,arg_parameter,[z,x],[N,m],iteration_initial_value,boundary_condition,differentiation_matrixes,1e-9,True)
        error[j,i]=symbol.error_function(*arg_parameter,*[z,x],*numeric_arg_field(stationary_solution[j,i],differentiation_matrixes))
        print('max(error):',np.abs(error[j,i]).max())

        # chi,xi_tilde,eta_tilde,f_tilde,A_tilde,B_tilde,phi_tilde=stationary_solution[j,i].reshape(7,N,m)
        # f=1+z0[:,None]**2*f_tilde
        # xi=z0[:,None]**3*xi_tilde/np.sin(x0)
        # eta=z0[:,None]**3*eta_tilde
        # A=A_tilde
        # B=B_tilde
        # phi=z0[:,None]*phi_tilde

        # h[j,i]=scipy.optimize.root(horizon_condition,np.ones(m),(f,chi,xi,A,B)).x
        # temperature[j,i]=get_temperature(h[j,i],chi,xi,eta,f,A,B)
        # entropy[j,i]=(2*np.pi/h[j,i]**2*np.sin(x0)).sum(axis=-1)*np.pi/m/2*4*np.pi

        # # energy_density[i]=-Dz[-1].dot(stationary_solution[i,0])+source*Dz[-1].dot(stationary_solution[i,2])/6
        # print(f'T={temperature[j,i]}, S={entropy[j,i]}, epsilon={energy_density[j,i]}')
        print('time spent(s):',time.time()-time_start)

    print('The maximum value of the error:',np.abs(error).max())

""" Test
--------------------------------------------
"""
'''
Dx_even_T=Dx_even.T
f3=Dz[-1].dot(f_tilde)
# A3=Dz[-1].dot(Dz2).dot(A_spherical)/6
# B3=Dz[-1].dot(Dz2).dot(B0)/6
A3=A_tilde[-1]
B3=B_tilde[-1]
phi2=Dz[-1].dot(phi_tilde)
f3.dot(Dx_even_T)/2-3*A3.dot(Dx_even_T)/2-3*A3*np.cos(x0)/np.sin(x0)-source*phi2.dot(Dx_even_T)/3
B3.dot(Dx_even_T)+2*B3*np.cos(x0)/np.sin(x0)

Dtest=Dx_even+2*np.diag(np.cos(x0)/np.sin(x0))
# Dtest=Dx_even.copy()
Dtest[-1]=np.eye(m)[-1]
b=np.zeros(m)
b[-1]=1
np.linalg.inv(Dtest).dot(b)
'''

""" EOS ($\kappa_4^2=1$)
--------------------------------------------
"""
file_location=f'./EOS/{source=}/z[0]={z[0]}/{N=}_{m=}'
os.makedirs(file_location+'/1.data',exist_ok=True)
os.makedirs(file_location+'/2.picture',exist_ok=True)

plt.plot(temperature,entropy)
plt.xlabel('$T$',fontsize=12)
plt.ylabel('$S$',fontsize=12)
plt.savefig(file_location+'/2.picture/1.entropy_vs_T.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/1.entropy_vs_T.pdf',bbox_inches='tight')
plt.show()
plt.close()

plt.plot(f_tilde0,error[:,0])
plt.xlabel(r'$\tilde{f}[0]$',fontsize=12)
plt.ylabel('error in horizon',fontsize=12)
plt.savefig(file_location+'/2.picture/7.error_horizon.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/7.error_horizon.pdf',bbox_inches='tight')
plt.close()

plt.plot(f_tilde0,error[:,-1])
plt.xlabel(r'$\tilde{f}[0]$',fontsize=12)
plt.ylabel('error in boundary',fontsize=12)
plt.savefig(file_location+'/2.picture/8.error_boundary.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/8.error_boundary.pdf',bbox_inches='tight')
plt.close()

plt.plot(f_tilde0,np.abs(error).max(axis=1))
plt.xlabel(r'$\tilde{f}[0]$',fontsize=12)
plt.ylabel('max error',fontsize=12)
plt.savefig(file_location+'/2.picture/9.max_error.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/9.max_error.pdf',bbox_inches='tight')
plt.close()

""" Save data
--------------------------------------------
"""
np.save(file_location+'/1.data/1.stationary_solution.npy',stationary_solution)
np.save(file_location+'/1.data/2.h.npy',h)
np.save(file_location+'/1.data/3.temperature.npy',temperature)
np.save(file_location+'/1.data/4.entropy.npy',entropy)
np.save(file_location+'/1.data/5.energy_density.npy',energy_density)
np.save(file_location+'/1.data/6.error.npy',error)

""" Load data
--------------------------------------------
"""
# stationary_solution=np.load(file_location+'/1.data/1.stationary_solution.npy')
# h=np.load(file_location+'/1.data/2.h.npy')
# temperature=np.load(file_location+'/1.data/3.temperature.npy',)
# entropy=np.load(file_location+'/1.data/4.entropy.npy')
# energy_density=np.load(file_location+'/1.data/5.energy_density.npy')
# error=np.load(file_location+'/1.data/6.error.npy')