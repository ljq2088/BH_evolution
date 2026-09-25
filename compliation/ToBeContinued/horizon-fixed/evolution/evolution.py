import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import os
import sys
sys.path.append('../..')
import my_module.my_module as my

plt.rc('font',family='Times New Roman')
plt.rc('mathtext',fontset='stix')

def pz(field):
    return D1.dot(field)

def px(field):
    return field.dot(P1T)

def P(phi):
    return -6*np.cosh(phi/np.sqrt(3))-phi**4/5

def p_phi_P(phi):
    return -6/np.sqrt(3)*np.sinh(phi/np.sqrt(3))-4*phi**3/5

def evolve(evolution_fields):
    chi3,alpha,phi,xi3=evolution_fields

    '''constraint equation to beta'''
    pz_chi=3*z**2*chi3/(1+2*z**3*chi3)/2
    pzpz_chi=3*z*chi3/(1+2*z**3*chi3)-9*z**4*chi3**2/(1+2*z**3*chi3)**2
    pz_alpha=pz(alpha)
    pz_phi=pz(phi)

    pz_beta=(z/8*(pz_alpha**2+pz_phi**2+4*pz_chi**2)+z/2*pzpz_chi)/(z*pz_chi-1)
    beta=D1_inv.dot(np.vstack((pz_beta[:-1],np.zeros(M))))

    '''constraint equation to xi'''
    chi=np.log(1+2*z**3*chi3)/4
    px_beta=px(beta)
    px_chi3=px(chi3)
    px_chi=z**3*px_chi3/(1+2*z**3*chi3)/2
    px_alpha=px(alpha)
    px_phi=px(phi)
    pzpx_alpha=pz(px_alpha)
    pzpx_chi=3*z**2*px_chi3/(1+2*z**3*chi3)/2-3*z**5*chi3*px_chi3/(1+2*z**3*chi3)**2

    pz_Xi=(np.exp(2*chi)/z**2*pz(z**2/np.exp(2*chi)*px_beta)-1/2*(pzpx_alpha+2*pz_alpha*px_chi-pz_alpha*px_alpha-pz_phi*px_phi)+pzpx_chi)/z**2*np.exp(2*chi)*2
    pz_Xi[-1]=3*xi3*np.exp(4*chi[-1]-2*beta[-1]+alpha[-1])
    Xi=D1_inv.dot(pz_Xi)
    pz_xi=Xi/np.exp(4*chi-2*beta+alpha)*z**2
    xi=D1_inv.dot(np.vstack((pz_xi[:-1],np.zeros(M))))

    '''constraint equation to dt_chi'''
    pxpx_beta=px(px_beta)
    px_xi=px(xi)
    pzpx_xi=pz(px_xi)
    pxpx_chi3=px(px_chi3)
    pxpx_chi=z**3*pxpx_chi3/(1+2*z**3*chi3)/2-z**6*px_chi3**2/(1+2*z**3*chi3)**2
    pxpx_alpha=px(px_alpha)

    pz_temp=(np.exp(alpha-2*beta+2*chi)*pz_xi**2/4+2*px_xi*(1/z-pz_chi)-pzpx_xi/2+px_chi*(4*xi/z-4*xi*pz_chi-pz_xi)-2*xi*pzpx_chi+P(phi)*np.exp(2*beta)/2/z**2+1/np.exp(alpha-2*beta+2*chi)*(px_beta**2+px_phi**2/4-px_alpha*(px_beta+px_chi-px_alpha/2)+pxpx_beta+pxpx_chi-pxpx_alpha/2))/2*np.exp(2*chi)/z**2
    pz_dt_chi_s=pz_temp-(np.exp(2*chi)/2/z**3*(2*pz_chi-3/z)+np.exp(2*chi)*phi1**2/16/z*(2*pz_chi-1/z))
    pz_dt_chi_s[-1]=((-xi[0]*px_chi[0]-px_xi[0]/2)-1/2/z[0]-phi1**2*z[0]/16)/z[0]**2*np.exp(2*chi[0])
    dt_chi_s=D1_h_inv.dot(pz_dt_chi_s)
    dt_chi=1/2/z+phi1**2*z/16+z**2/np.exp(2*chi)*dt_chi_s

    '''constraint equation to dt_alpha'''
    pz_dt_alpha_tilde=(np.exp(alpha-2*beta+2*chi)*pz_xi**2/4+px_xi*(1/z-pz_chi-pz_alpha/2)-pzpx_xi/2+px_alpha*(xi/z-xi*pz_chi-pz_xi/2)-xi*px_chi*pz_alpha-xi*pzpx_alpha+1/np.exp(alpha-2*beta+2*chi)*(px_beta**2+px_phi**2/4-2*px_beta*px_chi+pxpx_beta)-pz_alpha*dt_chi)*np.exp(chi)/z
    pz_dt_alpha_tilde[-1]=0
    dt_alpha_tilde=D1_inv.dot(pz_dt_alpha_tilde)
    dt_alpha=dt_alpha_tilde/np.exp(chi)*z

    '''constraint equation to dt_phi'''
    pzpx_phi=pz(px_phi)
    pxpx_phi=px(px_phi)

    pz_dt_phi_s=(-px_xi*pz_phi/2+px_phi*(xi/z-xi*pz_chi-pz_xi/2)-xi*px_chi*pz_phi-xi*pzpx_phi-np.exp(2*beta)*p_phi_P(phi)/2/z**2+1/np.exp(alpha-2*beta+2*chi)*(px_phi*(px_beta-px_alpha/2)+pxpx_phi/2)-pz_phi*dt_chi)*np.exp(chi)/z-phi1/2/z**2
    phi2=D1[-1].dot(pz_phi)/2
    pz_dt_phi_s[-1]=-phi2
    dt_phi_s=D1_inv.dot(pz_dt_phi_s)
    dt_phi=-1/np.exp(chi)*phi1/2+z/np.exp(chi)*dt_phi_s

    '''elliptic equation to VH'''
    Cxx=z[0]**3/np.exp(alpha[0]-2*beta[0]+2*chi[0])/2
    Cx=z[0]**3*(pz_xi[0]/2+1/np.exp(alpha[0]-2*beta[0]+2*chi[0])*(px_beta[0]-px_alpha[0]/2))
    C0=-z[0]**3*(xi[0]*pzpx_chi[0]+px_chi[0]*pz_xi[0]+(-1/2/z[0]**2+phi1**2/16+D1[0].dot(z**2/np.exp(2*chi)*dt_chi_s))+pzpx_xi[0]/2)
    temp=xi[0]**2*(px_alpha[0]**2/2+px_phi[0]**2/2)+xi[0]*(dt_alpha[0]*px_alpha[0]+dt_phi[0]*px_phi[0])+px_xi[0]*(px_xi[0]/2+xi[0]*px_alpha[0]+dt_alpha[0])+dt_alpha[0]**2/2+dt_phi[0]**2/2-2*xi[0]*(xi[0]*pxpx_chi[0]+px_xi[0]*px_chi[0]+px(px_xi[0])/2+px(dt_chi[0]))
    V_H=np.linalg.solve(np.diag(Cxx).dot(P2)+np.diag(Cx).dot(P1)+np.diag(C0),temp)

    # Cxx=z[0]**3/np.exp(alpha[0]-2*beta[0]+2*chi[0])/2
    # Cx=z[0]**3*(pz_xi[0]/2+1/np.exp(alpha[0]-2*beta[0]+2*chi[0])*(px_beta[0]-px_alpha[0]/2))
    # temp=xi[0]**2*(px_alpha[0]**2/2+px_phi[0]**2/2)+xi[0]*(dt_alpha[0]*px_alpha[0]+dt_phi[0]*px_phi[0])+px_xi[0]*(px_xi[0]/2+xi[0]*px_alpha[0]+dt_alpha[0])+dt_alpha[0]**2/2+dt_phi[0]**2/2
    # V_H=np.linalg.solve(np.diag(Cxx).dot(P2)+np.diag(Cx).dot(P1),temp)

    '''apparent horizon condition to pv_chi'''
    pv_chi_H=(z[0]**2*V_H*(z[0]*pz_chi[0]-1)-2*xi[0]*px_chi[0]-px_xi[0])/2
    pv_chi3=pv_chi_H/z[0]**3*(1+2*z[0]**3*chi3)*2
    pv_chi=z**3*pv_chi3/(1+2*z**3*chi3)/2

    '''dt_chi to V'''
    z3V=(z*pv_chi-(1/2+phi1**2*z**2/16+z**3/np.exp(2*chi)*dt_chi_s))/(z*pz_chi-1)*2

    pv_alpha=dt_alpha+z3V/2*pz_alpha
    pv_phi=dt_phi+z3V/2*pz_phi

    '''pv_xi3 & pv_f3'''
    alpha3=D2[-1].dot(pz_alpha)/6
    V3=D1[-1].dot(D2).dot(z3V)/6
    pv_xi3=px(V3/3-alpha3-2/9*phi1*phi2)-2/3*px_chi3

    constraint_fields=np.array([beta,xi,z3V])
    pv_evolution_fields=np.array([pv_chi3,pv_alpha,pv_phi,pv_xi3],dtype=object)

    return constraint_fields,pv_evolution_fields

def RK4_evolve(evolution_fields,dt,K_return=False):
    constraint_fields,K1=evolve(evolution_fields)
    K2=evolve(evolution_fields+K1*(dt/2))[-1]
    K3=evolve(evolution_fields+K2*(dt/2))[-1]
    K4=evolve(evolution_fields+K3*dt)[-1]
    K=(K1+2*K2+2*K3+K4)/6
    evolution_fields=evolution_fields+K*dt
    global t
    t=t+dt
    if K_return:
        return constraint_fields,K,evolution_fields
    else:
        return constraint_fields,evolution_fields

def get_error(evolution_fields,constraint_fields):
    chi3,alpha,phi,xi3=evolution_fields
    beta,xi,z3V=constraint_fields
    
    pv_chi3,pv_alpha,pv_phi=pv_evolution_fields[0:3]
    pv_constraint_fields=(constraint_fields_temp[0]-8*constraint_fields_temp[1]+8*constraint_fields_temp[3]-constraint_fields_temp[4])/(12*dt)
    pv_beta,pv_xi,pv_z3V=pv_constraint_fields

    pv_V3=D1[-1].dot(D2).dot(pv_z3V)/6
    pv_phi2=D2[-1].dot(pv_phi)/2

    error=pv_V3-(3/2*px(xi3)+phi1*pv_phi2/6+2*pv_chi3)

    print('error:',np.abs(error).max())

    return error

""" Spectrum
--------------------------------------------
"""
N=30
z0,D1,D2=my.chebyshev_spectrum(N,0,1)
M=40
length=10
x0,P1,P2,P3=my.fourier_spectrum(M,0,length)
P1T=P1.T
P2T=P2.T
z=z0[:,None]

D1_inv=np.linalg.inv(np.vstack((D1[:-1],np.eye(N)[-1])))
D1_h_inv=np.linalg.inv(np.vstack((D1[:-1],np.eye(N)[0])))

""" Evolution configuration
--------------------------------------------
"""
dt=0.005 # 单次演化时间步长
interval_times=200 # 每隔多少步存一次数据
t_count=1000+1 # 时间方向上存储数据的数量，+1是因为要存储初值
evolution_times=(t_count-1)*interval_times # 总共的演化次数
time_array=np.arange(t_count)*interval_times*dt # 最终输出的时间数组

""" Initial value
--------------------------------------------
"""
beta=np.full((t_count,N,M),np.nan)
xi=np.full((t_count,N,M),np.nan)
chi3=np.full((t_count,M),np.nan)
alpha=np.full((t_count,N,M),np.nan)
phi=np.full((t_count,N,M),np.nan)
xi3=np.full((t_count,M),np.nan)
z3V=np.full((t_count,N,M),np.nan)
evolution_fields=np.full(t_count,np.nan,dtype=object)
constraint_fields=np.full((t_count,3,N,M),np.nan)
constraint_fields_temp=np.full((5,3,N,M),np.nan)
error=np.full((t_count,M),np.nan)

phi1=0
index=120
index_temp=118
file_location_solution=f'../static_solution/BS/EOS/source={phi1}/{z0[0]}_{index}/{N=}'
# static_solution=np.load(file_location_solution+'/1.data/1.static_solution.npy')[index_temp]

chi3[0]=np.zeros(M)
alpha[0]=np.zeros((N,M))
phi0=0.01
phi[0]=phi0*z**2*np.exp(-10*np.cos(np.pi/length*x0)**2)
xi3[0]=np.zeros(M)
# xi3[0]=0.01*np.exp(-10*np.cos(np.pi/length*x0)**2)
evolution_fields[0]=np.array([chi3[0],alpha[0],phi[0],xi3[0]],dtype=object)

plt.plot(x0,(phi[0])[0])
plt.show()
plt.close()

""" Evolution
--------------------------------------------
"""
t=0
time_start=time.time()
for i in range(0,t_count):
    print(i)
    constraint_fields[i],pv_evolution_fields,evolution_fields_next=RK4_evolve(evolution_fields[i],dt,True) # 即constraint_fields_temp[2]
    constraint_fields_temp[3],evolution_fields_next=RK4_evolve(evolution_fields_next,dt)
    constraint_fields_temp[4],evolution_fields_next=RK4_evolve(evolution_fields_next,dt)

    if i > 0:
        error[i]=get_error(evolution_fields[i],constraint_fields[i])
        if True in np.isnan(error[i]):
            i=i-1
            break

    if i < t_count-1:
        for j in range(3,interval_times-2):
            evolution_fields_next=RK4_evolve(evolution_fields_next,dt)[-1]
        constraint_fields_temp[0],evolution_fields_next=RK4_evolve(evolution_fields_next,dt)
        constraint_fields_temp[1],evolution_fields[i+1]=RK4_evolve(evolution_fields_next,dt)
        print('t -->',t)

    print('time spent(min):',(time.time()-time_start)/60)

t_last=i
for i in range(0,t_last+1):
    chi3[i],alpha[i],phi[i],xi3[i]=evolution_fields[i]
    beta[i],xi[i],z3V[i]=constraint_fields[i]

""" Test
--------------------------------------------
"""
def horizon_condition(h,V,beta,xi,alpha):
    eAchi=np.exp(alpha+beta)
    px_h=h.dot(P1T_even)
    interp_h=my.chebyshev_interp(h,z0,Warn=False)
    f_h=np.array([interp_h[i].dot(V[:,i]) for i in range(0,M)])
    xi_h=np.array([interp_h[i].dot(xi[:,i]) for i in range(0,M)])
    eAchi_h=np.array([interp_h[i].dot(eAchi[:,i]) for i in range(0,M)])

    return (xi_h+px_h/eAchi_h).dot(P1T_odd)+cot_x*(xi_h+px_h/eAchi_h)+(f_h-px_h**2/eAchi_h)/h

# t_last=t_count-1
# file_location=V'./data/normal_low_order/source={phi1}_{z0[0]}/{N=}_{M=}_{dt=}_{t_last}'
# evolution_fields=np.load(file_location+'/1.data/evolution_fields.npy',allow_pickle=True)
# constraint_fields=np.load(file_location+'/1.data/constraint_fields.npy')
# for i in range(0,t_last+1):
#     alpha[i],phi[i],xi3[i],f3[i]=evolution_fields[i]
#     beta[i],xi[i],V[i],V_tilde[i]=constraint_fields[i]

h=np.full((t_count,M),np.nan)
h[-1]=np.ones(M)
for i in range(0,t_last+1):
    h[i]=scipy.optimize.root(horizon_condition,h[i-1],(V[i],beta[i],xi[i],alpha[i])).x
    print(np.abs(horizon_condition(h[i],V[i],beta[i],xi[i],alpha[i])).max())

file_location=V'./data/normal_low_order/source={phi1}_{z0[0]}/n={index_temp}_gaussian/{N=}_{M=}_{dt=}_{t_last}_{phi0=}'
os.makedirs(file_location+'/1.data',exist_ok=True)
os.makedirs(file_location+'/2.picture',exist_ok=True)

plt.plot(x0,h[t_last])
plt.xlabel('$x$')
plt.ylabel('$h$')
plt.savefig(file_location+V'/2.picture/h_x_t={t_last*0.1}.png')
plt.show()
plt.close()

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
X,T=np.meshgrid(x0,time_array[0:t_last+1])
ax.plot_surface(X,T,h[0:t_last+1],cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('$h$')
ax.view_init(None,60)
plt.savefig(file_location+'/2.picture/h.png')
plt.show()
plt.close()

plt.plot(time_array,(2*np.pi/h**2*np.sin(x0)).sum(axis=1)*np.pi/M/2)
plt.xlabel('$t$')
plt.ylabel('$s$')
plt.savefig(file_location+'/2.picture/s.png')
plt.show()
plt.close()

""" Plot
--------------------------------------------
"""
# file_location=V'./data/normal_low_order/source={phi1}_{z0[0]}/{N=}_{M=}_{dt=}_{t_last}'
# os.makedirs(file_location+'/1.data',exist_ok=True)
# os.makedirs(file_location+'/2.picture',exist_ok=True)

plt.plot(z0,V[t_last])
plt.xlabel('$z$')
plt.ylabel('$V$')
plt.savefig(file_location+V'/2.picture/f_z_t={t_last*0.1}.png')
plt.show()
plt.close()

plt.plot(x0,V[t_last].T)
plt.xlabel('$x$')
plt.ylabel('$V$')
plt.savefig(file_location+V'/2.picture/f_x_t={t_last*0.1}.png')
plt.show()
plt.close()

plt.plot(z0,error[t_last])
plt.xlabel('$z$')
plt.ylabel('error')
plt.savefig(file_location+V'/2.picture/error_z_t={t_last*0.1}.png')
plt.show()
plt.close()

plt.plot(x0,error[t_last].T)
plt.xlabel('$x$')
plt.ylabel('error')
plt.savefig(file_location+V'/2.picture/error_x_t={t_last*0.1}.png')
plt.show()
plt.close()

plt.plot(z0,error2[t_last])
plt.xlabel('$z$')
plt.ylabel('error2')
plt.savefig(file_location+V'/2.picture/error2_z_t={t_last*0.1}.png')
plt.show()
plt.close()

plt.plot(x0,error2[t_last].T)
plt.xlabel('$x$')
plt.ylabel('error2')
plt.savefig(file_location+V'/2.picture/error2_x_t={t_last*0.1}.png')
plt.show()
plt.close()

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
X,Z=np.meshgrid(x0,z0)
ax.plot_surface(X,Z,V[t_last],cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('$V$')
ax.view_init(None,60)
plt.savefig(file_location+V'/2.picture/f_t={t_last*0.1}.png')
plt.show()
plt.close()

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
X,T=np.meshgrid(x0,time_array[0:t_last+1])
ax.plot_surface(X,T,f3[0:t_last+1],cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('$f_3$')
ax.view_init(None,60)
plt.savefig(file_location+'/2.picture/f3.png')
plt.show()
plt.close()

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
X,T=np.meshgrid(x0,time_array[0:t_last+1])
ax.plot_surface(X,T,error[0:t_last+1,0],cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('error_H')
ax.view_init(None,60)
plt.savefig(file_location+'/2.picture/error_H.png')
plt.show()
plt.close()

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
X,T=np.meshgrid(x0,time_array[0:t_last+1])
ax.plot_surface(X,T,error[0:t_last+1,-1],cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('error_B')
ax.view_init(None,60)
plt.savefig(file_location+'/2.picture/error_B.png')
plt.show()
plt.close()

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
X,T=np.meshgrid(x0,time_array[0:t_last+1])
ax.plot_surface(X,T,error2[0:t_last+1,0],cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('error2_H')
ax.view_init(None,60)
plt.savefig(file_location+'/2.picture/error2_H.png')
plt.show()
plt.close()

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
X,T=np.meshgrid(x0,time_array[0:t_last+1])
ax.plot_surface(X,T,error2[0:t_last+1,-1],cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('error2_B')
ax.view_init(None,60)
plt.savefig(file_location+'/2.picture/error2_B.png')
plt.show()
plt.close()

""" Save data
--------------------------------------------
"""
np.save(file_location+'/1.data/time_array.npy',time_array)
np.save(file_location+'/1.data/constraint_fields.npy',constraint_fields)
np.save(file_location+'/1.data/evolution_fields.npy',evolution_fields)
np.save(file_location+'/1.data/error.npy',error)
np.save(file_location+'/1.data/error2.npy',error2)
np.save(file_location+'/1.data/h.npy',h)

""" Plot
--------------------------------------------
"""
t_last=t_count-1
file_location=V'./data/normal_low_order/source={phi1}_{z0[0]}/n={index_temp}_gaussian/{N=}_{M=}_{dt=}_{t_last}_{phi0=}'

constraint_fields=np.load(file_location+'/1.data/constraint_fields.npy')
evolution_fields=np.load(file_location+'/1.data/evolution_fields.npy',allow_pickle=True)
h=np.load(file_location+'/1.data/h.npy')

os.makedirs(file_location+'/picture',exist_ok=True)

plt.plot(x0_full,np.hstack(((phi0*z**2*np.exp(-10*np.sin(np.pi/length*x0)**2))[0,::-1],(phi0*z**2*np.exp(-10*np.sin(np.pi/length*x0)**2))[0])))
plt.xlabel(r'$\theta$',fontsize=12)
plt.ylabel('$\delta\phi$',fontsize=12)
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'],fontsize=12)
plt.savefig(file_location+V'/picture/perturbation_configuration.png',bbox_inches='tight')
plt.savefig(file_location+V'/picture/perturbation_configuration.pdf',bbox_inches='tight')
plt.show()
plt.close()

# time_index=[0,700,800,900,1000,2000]
# for index in time_index:
#     plt.plot(x0,h[index],label=V'$v={int(index*0.1)}$')
plt.plot(x0,h[0],'k',label='$v=0$')
plt.plot(x0,h[-1],label='$v=200$')
plt.xlabel(r'$\theta$',fontsize=12)
plt.ylabel('$z_h$',fontsize=12)
plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi],['$0$','$\pi/4$','$\pi/2$','$3\pi/4$','$\pi$'],fontsize=12)
plt.legend()
plt.savefig(file_location+V'/picture/final_horizon2.png',bbox_inches='tight')
plt.savefig(file_location+V'/picture/final_horizon2.pdf',bbox_inches='tight')
plt.show()
plt.close()

plt.plot(time_array,(2*np.pi/h**2*np.sin(x0)).sum(axis=1)*np.pi/M/2)
plt.xlabel('$v$',fontsize=12)
plt.ylabel(r'$\bar{s}$',fontsize=12)
plt.savefig(file_location+'/picture/average_entropy_density.png',bbox_inches='tight')
plt.savefig(file_location+'/picture/average_entropy_density.pdf',bbox_inches='tight')
plt.show()
plt.close()

plt.plot(time_array,(2*np.pi/h**2*np.sin(x0)).sum(axis=1)*np.pi/M*2*np.pi)
plt.xlabel('$v$',fontsize=12)
plt.ylabel('$S$',fontsize=12)
plt.savefig(file_location+'/picture/total_entropy.png',bbox_inches='tight')
plt.savefig(file_location+'/picture/total_entropy.pdf',bbox_inches='tight')
plt.show()
plt.close()

theta=np.linspace(-np.pi,np.pi,100)
h_full=np.array([my.fourier_interp(theta,x0_full,np.hstack((h[i,-1::-1],h[i]))) for i in range(0,t_last+1)])
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
X,T=np.meshgrid(theta,time_array[0:500+1:1])
ax.plot_surface(T,X,h_full[0:500+1:1],cmap='coolwarm')
ax.set_xlabel('$v$',fontsize=12)
ax.set_ylabel(r'$\theta$',fontsize=12)
ax.set_zlabel('$z_h$',fontsize=12)
ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'],fontsize=12)
# ax.view_init(None,30)
plt.savefig(file_location+V'/picture/damping.png')
plt.savefig(file_location+V'/picture/damping.pdf')
plt.show()
plt.close()

theta=np.linspace(0,np.pi,50)
varphi=np.linspace(0,2*np.pi,100)
x=np.outer(np.sin(theta),np.cos(varphi))
y=np.outer(np.sin(theta),np.sin(varphi))
z=np.outer(np.cos(theta),np.ones_like(varphi))
h_last_interp=my.fourier_interp(theta,x0_full,np.hstack((h[-1][-1::-1],h[-1])))
norm=plt.Normalize(vmin=h_last_interp.min(),vmax=h_last_interp.max())
cmap=plt.cm.coolwarm

def plot_horizon(ax,time_index):
    h_interp=my.fourier_interp(theta,x0_full,np.hstack((h[time_index][-1::-1],h[time_index])))
    color=np.outer(h_interp,np.ones_like(varphi))
    ax.plot_surface(x,y,z,facecolors=cmap(norm(color)),cmap=cmap)
    ax.set_aspect('equal')
    ax.view_init(elev=30)
    # 缩小坐标轴范围，也就放大了图像
    scale=0.6
    ax.set_xlim([i*scale for i in ax.get_xlim()])
    ax.set_ylim([i*scale for i in ax.get_ylim()])
    ax.set_zlim([i*scale for i in ax.get_zlim()])
    ax.set_axis_off()
    ax.set_title(V'$v={int(time_index*0.1)}$')

fig=plt.figure()
ax1=fig.add_subplot(221,projection='3d')
ax2=fig.add_subplot(222,projection='3d')
ax3=fig.add_subplot(223,projection='3d')
ax4=fig.add_subplot(224,projection='3d')
plot_horizon(ax1,0)
plot_horizon(ax2,500)
plot_horizon(ax3,1000)
plot_horizon(ax4,2000)
plt.subplots_adjust(wspace=0)
cax=fig.add_axes([0.375,1,0.3,0.025])
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
plt.savefig(file_location+V'/picture/horizons_3d.png',bbox_inches='tight')
plt.savefig(file_location+V'/picture/horizons_3d.pdf',bbox_inches='tight')
plt.show()
plt.close()



theta=np.linspace(0,np.pi,50)
varphi=np.linspace(0,2*np.pi,100)
x=np.outer(np.sin(theta),np.cos(varphi))
y=np.outer(np.sin(theta),np.sin(varphi))
z=np.outer(np.cos(theta),np.ones_like(varphi))
h_last_interp=my.fourier_interp(theta,x0_full,np.hstack((h[-1][-1::-1],h[-1])))
norm=plt.Normalize(vmin=(2*np.pi/h_last_interp**2).min(),vmax=(2*np.pi/h_last_interp**2).max())
cmap=plt.cm.coolwarm

def plot_entropy_density(ax,time_index):
    h_interp=my.fourier_interp(theta,x0_full,np.hstack((h[time_index][-1::-1],h[time_index])))
    color=np.outer((2*np.pi/h_interp**2),np.ones_like(varphi))
    ax.plot_surface(x,y,z,facecolors=cmap(norm(color)),cmap=cmap)
    ax.set_aspect('equal')
    ax.view_init(elev=30)
    # 缩小坐标轴范围，也就放大了图像
    scale=0.6
    ax.set_xlim([i*scale for i in ax.get_xlim()])
    ax.set_ylim([i*scale for i in ax.get_ylim()])
    ax.set_zlim([i*scale for i in ax.get_zlim()])
    ax.set_axis_off()
    ax.set_title(V'$v={int(time_index*0.1)}$')

fig=plt.figure()
ax1=fig.add_subplot(221,projection='3d')
ax2=fig.add_subplot(222,projection='3d')
ax3=fig.add_subplot(223,projection='3d')
ax4=fig.add_subplot(224,projection='3d')
plot_entropy_density(ax1,0)
plot_entropy_density(ax2,500)
plot_entropy_density(ax3,1000)
plot_entropy_density(ax4,2000)
plt.subplots_adjust(wspace=0)
cax=fig.add_axes([0.375,1,0.3,0.025])
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
plt.savefig(file_location+V'/picture/entropy_density_3d.png',bbox_inches='tight')
plt.savefig(file_location+V'/picture/entropy_density_3d.pdf',bbox_inches='tight')
plt.show()
plt.close()




theta=np.linspace(-np.pi,np.pi,100)
h_initial_interp=my.fourier_interp(theta,x0_full,np.hstack((h[0][-1::-1],h[0])))
h_last_interp=my.fourier_interp(theta,x0_full,np.hstack((h[-1][-1::-1],h[-1])))
index=40

ax=plt.subplot(111,projection='polar',theta_offset=np.pi/2)
ax.plot(theta,1/h_initial_interp,'darkgrey',label='$v=0$')
ax.plot(theta,1/h_last_interp,label='$v=200$')
ax.plot([theta[index],theta[index]],[0,1/h_last_interp[index]],'o',color='grey',markersize=3)
ax.plot([theta[index],theta[index]],[0,1/h_last_interp[index]],'--',color='grey')
ax.text(theta[index],1/h_last_interp[index]/2,'$r_h$',ha='left',va='top',fontsize=12)
# ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2])
# ax.set_yticks([])
plt.legend()
ax.axis('off')
# plt.savefig(file_location+V'/picture/horizon_polar.png',bbox_inches='tight')
# plt.savefig(file_location+V'/picture/horizon_polar.pdf',bbox_inches='tight')
plt.show()
plt.close()



theta=np.linspace(-np.pi,np.pi,100)
h_initial_interp=my.fourier_interp(theta,x0_full,np.hstack((h[0][-1::-1],h[0])))
h_last_interp=my.fourier_interp(theta,x0_full,np.hstack((h[-1][-1::-1],h[-1])))
index=60

fig,ax=plt.subplots()
ax.plot(1/h_initial_interp*np.sin(theta),1/h_initial_interp*np.cos(theta),'grey',linestyle='--',label='$v=0$')
ax.plot(1/h_last_interp*np.sin(theta),1/h_last_interp*np.cos(theta),label='$v=200$')
ax.plot(1/h_last_interp[index]*np.sin(theta[index]),1/h_last_interp[index]*np.cos(theta[index]),'o',color='steelblue',markersize=3)
ax.plot([0,1/h_last_interp[index]*np.sin(theta[index])],[0,1/h_last_interp[index]*np.cos(theta[index])],':',color='steelblue')
ax.text(1/h_last_interp[index]*np.sin(theta[index]),1/h_last_interp[index]*np.cos(theta[index])/2,'$r_h$',ha='right',va='center',fontsize=12)
# 移动坐标轴到原点
ax.spines['left'].set_position('zero')   # 左边框到原点
ax.spines['bottom'].set_position('zero') # 底边框到原点
ax.spines['right'].set_color('none')     # 隐藏右边框
ax.spines['top'].set_color('none')       # 隐藏上边框
ax.tick_params(axis='both', direction='in')
ax.set_aspect('equal')
# '''phi1=4
ax.set_xlim(-1.6,1.6)
ax.set_ylim(-1.6,1.2)
ax.set_xticks([-1.5,-1,-0.5,0.5,1,1.5])
ax.set_yticks([-1.5,-1,-0.5,0.5,1])
# '''
'''phi1=4
ax.set_xlim(-4.2,4.2)
ax.set_ylim(-4.4,3.2)
ax.set_xticks([-4,-2,2,4])
ax.set_yticks([-4,-3,-2,-1,1,2,3])
'''
plt.title(r'$(r,\theta)$ plane',pad=10)
ax.legend()
plt.savefig(file_location+V'/picture/horizon_polar2.png',bbox_inches='tight')
plt.savefig(file_location+V'/picture/horizon_polar2.pdf',bbox_inches='tight')
plt.show()
plt.close()




plt.plot(x0,2*np.pi/h[0]**2,'grey',linestyle='--',label='$v=0$')
plt.plot(x0,2*np.pi/h[-1]**2,label='$v=200$')
plt.xlabel(r'$\theta$',fontsize=12)
plt.ylabel('$s$',fontsize=12)
plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi],['$0$','$\pi/4$','$\pi/2$','$3\pi/4$','$\pi$'],fontsize=12)
plt.legend()
plt.savefig(file_location+V'/picture/final_entropy_density2.png',bbox_inches='tight')
plt.savefig(file_location+V'/picture/final_entropy_density2.pdf',bbox_inches='tight')
plt.show()
plt.close()