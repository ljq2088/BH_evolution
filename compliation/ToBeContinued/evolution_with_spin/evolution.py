import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import os
import sys
sys.path.append('..')
import my_module.my_module as my

plt.rc('font',family='Times New Roman')
plt.rc('mathtext',fontset='stix')

def pz(field):
    return D1.dot(field)

def px(field):
    return field.dot(P1T_even)

def px_odd(field):
    return field.dot(P1T_odd)

def V(phi):
    return -6*np.cosh(phi/np.sqrt(3))-phi**4/5

def p_phi_V(phi):
    return -6/np.sqrt(3)*np.sinh(phi/np.sqrt(3))-4*phi**3/5

def evolve(evolution_fields):
    A,B,phi,xi3,eta3,f3=evolution_fields
    sinhB=np.sinh(B)
    coshB=np.cosh(B)
    p0=3*xi3
    q_tilde0=3*eta3*sin_x

    '''constrain equation to chi'''
    pz_A=pz(A)
    pz_B=pz(B)
    pz_phi=pz(phi)

    pz_chi=z/4*(pz_A**2*coshB**2+pz_B**2+pz_phi**2)
    chi=D1_operator_inv.dot(np.vstack((pz_chi[:-1],np.zeros(m))))

    '''constrain equation to xi, eta'''
    px_A=px(A)
    px_B=px(B)
    px_phi=px(phi)
    px_chi=px(chi)
    eA=np.exp(A)
    echi=np.exp(chi)

    pz_p=-1/z**2*(px(pz_A*coshB**2+pz_chi)+2*px_chi/z-(pz_A*px_A*coshB**2+pz_B*px_B+pz_phi*px_phi)+2*cot_x*pz_A*coshB**2)
    p=D1_operator_inv.dot(np.vstack((pz_p[:-1],p0)))
    pz_q_tilde=-1/z**2*(px(1/eA*(pz_B+pz_A*sinhB*coshB))+2/eA*cot_x*(pz_B+pz_A*sinhB*coshB))
    # pz_q_tilde=-1/z**2/eA*(-px_A*(pz_B+pz_A*sinhB*coshB)+px(pz_B+pz_A*sinhB*coshB))
    q_tilde=D1_operator_inv.dot(np.vstack((pz_q_tilde[:-1],q_tilde0)))

    q=q_tilde*sin_x

    pz_xi=z**2/echi*(1/eA*coshB*p-sinhB*q_tilde)
    pz_eta=z**2/echi*(-sinhB*p+eA*coshB*q_tilde)/sin_x
    xi=D1_operator_inv.dot(np.vstack((pz_xi[:-1],np.zeros(m))))
    eta=D1_operator_inv.dot(np.vstack((pz_eta[:-1],np.zeros(m))))

    '''constrain equation to f'''
    px_xi=px_odd(xi)
    px_eta=px(eta)
    Xi=pz(px_xi)/2/z**2-px_xi/z**3
    Eta=pz(px_eta)/2/z**2-px_eta/z**3
    com1=1/eA/echi/4/z*(px_chi**2+px_phi**2)

    temp=px_xi/z**2+(3+np.exp(-chi)*V(phi)/2)/z**3-z*Xi+z*(pz_xi*p+pz_eta*q)/4+1/echi/2/z*px_odd(px(1/eA*coshB)-1/eA*coshB*px_chi)+com1*coshB-cot_x/2/z*(pz_xi+1/eA/echi*(3*px_A*coshB+px_chi*coshB-3*px_B*sinhB)-4*xi/z)-1/eA/echi*coshB/z
    temp[-1]=f3
    f_tilde=D_f_tilde_operator_inv.dot(temp)
    f=1+z**2*f_tilde

    '''evolution equation to pv_A, pv_B'''
    pxpx_chi=px_odd(px_chi)

    sA=-f*pz_A*coshB/2/z**2+z**3/echi*(p**2/eA-q_tilde**2*eA)/4+(pz_xi*px_A+px_xi*pz_A)*coshB/2/z-z/eA*(Xi*eA*coshB+Eta*sinhB*sin_x)-1/eA/z*(pz(eA*coshB)*px_xi+pz(sinhB*sin_x)*px_eta)+com1-1/eA/echi/4/z*2*pxpx_chi+cot_x/2/z*(coshB*(pz_xi-xi*pz_A-2*xi/z)+1/eA/echi*px_chi+2*xi*pz_B*sinhB)
    sB=-f*pz_B/2/z**2+z**3/echi*(2*p*q_tilde*coshB-(p**2/eA+q_tilde**2*eA)*sinhB)/4+(pz_xi*px_B+px_xi*pz_B)/2/z-z/eA*Eta*sin_x-1/eA/z*(eA*(pz_B-pz_A*sinhB*coshB)*px_xi-pz_A*coshB**2*px_eta*sin_x)-(com1-1/eA/echi/4/z*2*pxpx_chi)*sinhB-cot_x/2/z*(xi*pz_B+2*xi*pz_A*sinhB*coshB+1/eA/echi*px_chi*sinhB)

    temp=pz_A*sinhB
    temp[-1]=np.zeros(m)
    temp=D1_operator_inv.dot(temp)
    TAA=np.cos(temp)
    TBB=TAA
    TAB=np.sin(temp)
    TBA=-TAB

    pz_PiA0=TAA*sA+TAB*sB
    pz_PiB0=TBA*sA+TBB*sB
    PiA0=D1_operator_inv.dot(np.vstack((pz_PiA0[:-1],np.zeros(m))))
    PiB0=D1_operator_inv.dot(np.vstack((pz_PiB0[:-1],np.zeros(m))))

    PiA=TBB*PiA0-TAB*PiB0
    PiB=-TBA*PiA0+TAA*PiB0

    pv_A=z*PiA/coshB+f*pz_A/2-xi*px_A
    pv_B=z*PiB+f*pz_B/2-xi*px_B

    '''evolution equation to pv_phi'''
    temp=f*pz_phi-xi*px_phi
    temp=(pz(temp)/2-temp/z+px_odd(1/eA/echi*px_phi*coshB-xi*pz_phi)/2-1/2/z**2/echi*p_phi_V(phi)+cot_x/2*(1/eA/echi*px_phi*coshB-xi*pz_phi))/z
    temp[-1]=0
    pv_phi=D1_operator_inv.dot(temp)*z

    '''pv_xi3 & pv_f3'''
    A3=D2[-1].dot(pz_A)/6
    B3=D2[-1].dot(pz_B)/6
    phi1=pz_phi[-1]
    phi2=D1[-1].dot(pz_phi)/2
    pv_phi2=D2[-1].dot(pv_phi)/2
    pv_xi3=px(f3/3-A3-2*phi1*phi2/9)-2*cot_x*A3
    pv_eta3=(-px(B3)-2*cot_x*B3)/sin_x
    pv_f3=3*px_odd(xi3)/2+phi1*pv_phi2/6+cot_x*3*xi3/2

    constraint_fields=np.array([chi,xi,eta,f,f_tilde])
    pv_evolution_fields=np.array([pv_A,pv_B,pv_phi,pv_xi3,pv_eta3,pv_f3],dtype=object)

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
    A,B,phi,xi3,eta3,f3=evolution_fields
    chi,xi,eta,f,f_tilde=constraint_fields
    
    pv_A,pv_B,pv_phi=pv_evolution_fields[0:3]
    pv_constraint_fields=(constraint_fields_temp[0]-8*constraint_fields_temp[1]+8*constraint_fields_temp[3]-constraint_fields_temp[4])/(12*dt)
    pv_chi,pv_xi,pv_eta,pv_f,pv_f_tilde=pv_constraint_fields

    eAchi=np.exp(A+chi)
    pz_A=pz(A)
    px_A=px(A)
    pz_phi=pz(phi)
    px_phi=px(phi)
    pz_chi=pz(chi)
    px_chi=px(chi)
    pz_xi=pz(xi)
    px_xi=px_odd(xi)
    pz_f=pz(f)
    px_f=px(f)
    pv_Achi=pv_A+pv_chi
    dn_A=pv_A-f*pz_A+xi*px_A
    dn_phi=pv_phi-f*pz_phi+xi*px_phi
    dn_chi=pv_chi-f*pz_chi+xi*px_chi
    dn_xi=pv_xi-f*pz_xi+xi*px_xi
    Xi=eAchi*pz_xi
    pv_Xi=eAchi*(pz_xi*pv_Achi+pz(pv_xi))

    '''evolution equation of xi'''
    temp=2/z*(px_f+f*px_chi)
    temp[-1]=0
    error=temp+px(2*xi*Xi-pz_f+f*(pz_A+pz_chi)-pv_Achi)-xi*px_odd(Xi+px_A+px_chi)+px_A*dn_A+px_phi*dn_phi+pv_Xi+cot_x*(xi*(Xi-3*px_A+px_chi)+2*f*pz_A-2*pv_A)-2*xi

    '''evolution equation of f'''
    temp=2/z*(pv_f+f*pv_chi)
    temp[-1]=0
    error2=px_odd(px_f/eAchi-pz(f*xi)+xi**2*Xi-xi*(dn_A+dn_chi)-2*dn_xi)-px_xi*pv_Achi-temp-pv_A*dn_A-pv_phi*dn_phi+xi*pv_Xi+cot_x*(xi**2*(Xi-(px_A+px_chi))+xi*(f*(pz_A+pz_chi)-2*pv_chi-2*px_xi-pz_f)+f*pz_xi-2*pv_xi+px_f/eAchi)

    print('error:',np.abs(error).max())
    print('error2:',np.abs(error2).max())
    # print('xi3_error:',np.abs(xi3-D1[-1].dot(D2).dot(xi)/6).max())
    print('f3_error:',np.abs(f3-D1[-1].dot(D2).dot(f)/6).max())
    # print('f3_error2:',np.abs(z**2*D1.dot(f_tilde)+2*z*f_tilde-D1.dot(f)).max())
    # print('f3_f_tilde_error:',np.abs(f3-D1[-1].dot(f_tilde)).max())

    return error,error2

""" Spectrum
--------------------------------------------
"""
N=50
z0,D1,D2=my.chebyshev_spectrum(N,0,1.3)
m=20
M=2*m
x0_full,P1,P2,P3=my.fourier_spectrum(M,-np.pi+np.pi/M,2*np.pi)
x0=x0_full[m:]
P1T=P1.T
P2T=P2.T
P1T_even=P1T[m-1::-1,m:]+P1T[m:,m:]
P1T_odd=-P1T[m-1::-1,m:]+P1T[m:,m:]
z=z0[:,None]

cot_x=1/np.tan(x0)
sin_x=np.sin(x0)

D1_operator=np.vstack((D1[:-1],np.eye(N)[-1]))
D1_operator_inv=np.linalg.inv(D1_operator)

D_f_tilde_operator=-np.diag(1/z0)+D1
D_f_tilde_operator[-1]=D1[-1]
D_f_tilde_operator_inv=np.linalg.inv(D_f_tilde_operator)

""" Evolution configuration
--------------------------------------------
"""
dt=0.001 # 单次演化时间步长
interval_times=100 # 每隔多少步存一次数据
t_count=1000+1 # 时间方向上存储数据的数量，+1是因为要存储初值
evolution_times=(t_count-1)*interval_times # 总共的演化次数
time_array=np.arange(t_count)*interval_times*dt # 最终输出的时间数组

for eta3_0 in np.arange(0,2,0.1):
    """ Initial value
    --------------------------------------------
    """
    chi=np.full((t_count,N,m),np.nan)
    xi=np.full((t_count,N,m),np.nan)
    eta=np.full((t_count,N,m),np.nan)
    f=np.full((t_count,N,m),np.nan)
    f_tilde=np.full((t_count,N,m),np.nan)
    A=np.full((t_count,N,m),np.nan)
    B=np.full((t_count,N,m),np.nan)
    phi=np.full((t_count,N,m),np.nan)
    xi3=np.full((t_count,m),np.nan)
    eta3=np.full((t_count,m),np.nan)
    f3=np.full((t_count,m),np.nan)
    evolution_fields=np.full(t_count,np.nan,dtype=object)
    constraint_fields=np.full((t_count,5,N,m),np.nan)
    constraint_fields_temp=np.full((5,5,N,m),np.nan)
    error=np.full((t_count,N,m),np.nan)
    error2=np.full((t_count,N,m),np.nan)

    phi1=2
    index_temp=118
    static_solution=np.load('./1.static_solution.npy')[index_temp]

    A[0]=np.zeros((N,m))
    B[0]=np.zeros((N,m))
    phi0=0.1
    # phi[0]=z*static_solution[2,:,None]+phi0*z**2*np.cos(x0)
    # phi[0]=z*static_solution[2,:,None]+phi0*z**2*(3*np.cos(x0)**2-1)/2
    phi[0]=z*static_solution[2,:,None]+phi0*z**2*np.exp(-10*np.sin(x0/2)**2)
    # phi[0]=z*static_solution[2,:,None]
    xi3[0]=np.zeros(m)
    eta3[0]=np.zeros(m)+eta3_0
    f3[0]=D1[-1].dot(static_solution[0,:,None])
    evolution_fields[0]=np.array([A[0],B[0],phi[0],xi3[0],eta3[0],f3[0]],dtype=object)

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
            error[i],error2[i]=get_error(evolution_fields[i],constraint_fields[i])
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
        A[i],B[i],phi[i],xi3[i],eta3[i],f3[i]=evolution_fields[i]
        chi[i],xi[i],eta[i],f[i],f_tilde[i]=constraint_fields[i]

    """ Test
    --------------------------------------------
    """
    def horizon_condition(h,f,chi,xi,A,B):
        eAchi=np.exp(A+chi)
        coshB=np.cosh(B)
        px_h=h.dot(P1T_even)
        interp_h=my.chebyshev_interp(h,z0,Warn=False)
        f_h=np.array([interp_h[i].dot(f[:,i]) for i in range(0,m)])
        xi_h=np.array([interp_h[i].dot(xi[:,i]) for i in range(0,m)])
        eAchi_h=np.array([interp_h[i].dot(eAchi[:,i]) for i in range(0,m)])
        coshB_h=np.array([interp_h[i].dot(coshB[:,i]) for i in range(0,m)])

        return (xi_h+px_h/eAchi_h*coshB_h).dot(P1T_odd)+cot_x*(xi_h+px_h/eAchi_h*coshB_h)+(f_h-px_h**2/eAchi_h*coshB_h)/h

    # t_last=t_count-1
    # file_location=f'./data/normal_low_order/source={phi1}_{z0[0]}/{N=}_{m=}_{dt=}_{t_last}'
    # evolution_fields=np.load(file_location+'/1.data/evolution_fields.npy',allow_pickle=True)
    # constraint_fields=np.load(file_location+'/1.data/constraint_fields.npy')
    # for i in range(0,t_last+1):
    #     A[i],phi[i],xi3[i],f3[i]=evolution_fields[i]
    #     chi[i],xi[i],f[i],f_tilde[i]=constraint_fields[i]

    h=np.full((t_count,m),np.nan)
    h[-1]=np.ones(m)
    for i in range(0,t_last+1):
        h[i]=scipy.optimize.root(horizon_condition,h[i-1],(f[i],chi[i],xi[i],A[i],B[i])).x
        print(np.abs(horizon_condition(h[i],f[i],chi[i],xi[i],A[i],B[i])).max())

    file_location=f'./data/source={phi1}_{z0[0]}_n={index_temp}_gaussian/{eta3_0=}/{N=}_{m=}_{dt=}_{t_last}_{phi0=}'
    os.makedirs(file_location+'/1.data',exist_ok=True)
    os.makedirs(file_location+'/2.picture',exist_ok=True)

    plt.plot(x0,h[t_last])
    plt.xlabel('$x$')
    plt.ylabel('$h$')
    plt.savefig(file_location+f'/2.picture/h_x_t={t_last*0.1}.png')
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

    plt.plot(time_array,(2*np.pi/h**2*np.sin(x0)).sum(axis=1)*np.pi/m/2)
    plt.xlabel('$t$')
    plt.ylabel('$s$')
    plt.savefig(file_location+'/2.picture/s.png')
    plt.show()
    plt.close()

    """ Plot
    --------------------------------------------
    """
    # file_location=f'./data/normal_low_order/source={phi1}_{z0[0]}/{N=}_{m=}_{dt=}_{t_last}'
    # os.makedirs(file_location+'/1.data',exist_ok=True)
    # os.makedirs(file_location+'/2.picture',exist_ok=True)

    plt.plot(z0,f[t_last])
    plt.xlabel('$z$')
    plt.ylabel('$f$')
    plt.savefig(file_location+f'/2.picture/f_z_t={t_last*0.1}.png')
    plt.show()
    plt.close()

    plt.plot(x0,f[t_last].T)
    plt.xlabel('$x$')
    plt.ylabel('$f$')
    plt.savefig(file_location+f'/2.picture/f_x_t={t_last*0.1}.png')
    plt.show()
    plt.close()

    plt.plot(z0,error[t_last])
    plt.xlabel('$z$')
    plt.ylabel('error')
    plt.savefig(file_location+f'/2.picture/error_z_t={t_last*0.1}.png')
    plt.show()
    plt.close()

    plt.plot(x0,error[t_last].T)
    plt.xlabel('$x$')
    plt.ylabel('error')
    plt.savefig(file_location+f'/2.picture/error_x_t={t_last*0.1}.png')
    plt.show()
    plt.close()

    plt.plot(z0,error2[t_last])
    plt.xlabel('$z$')
    plt.ylabel('error2')
    plt.savefig(file_location+f'/2.picture/error2_z_t={t_last*0.1}.png')
    plt.show()
    plt.close()

    plt.plot(x0,error2[t_last].T)
    plt.xlabel('$x$')
    plt.ylabel('error2')
    plt.savefig(file_location+f'/2.picture/error2_x_t={t_last*0.1}.png')
    plt.show()
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    X,Z=np.meshgrid(x0,z0)
    ax.plot_surface(X,Z,f[t_last],cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('$f$')
    ax.view_init(None,60)
    plt.savefig(file_location+f'/2.picture/f_t={t_last*0.1}.png')
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

    """ Animation
    --------------------------------------------
    """
    from matplotlib.animation import FuncAnimation
    theta=x0
    varphi=np.linspace(0,2*np.pi,2*m)

    # z坐标
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    def plot_surface(t):
        plt.cla()
        r=h[t]
        x=r[:,None]*np.outer(np.sin(theta),np.cos(varphi))
        y=r[:,None]*np.outer(np.sin(theta),np.sin(varphi))
        z=r[:,None]*np.outer(np.cos(theta),np.ones_like(varphi))
        plt.title(f't={0.1*t}')
        ax.plot_surface(x,y,z,cmap='Blues')
        ax.set_xlim([-0.6,0.6])
        ax.set_ylim([-0.6,0.6])
        ax.set_zlim([-0.4,0.8])
        ax.set_axis_off()
    plot_surface(0)
    ani=FuncAnimation(fig,plot_surface,frames=range(10,t_last+10,10))
    ani.save(file_location+'/z坐标.gif')
    plt.show()
    plt.close()

    # r坐标
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    def plot_surface(t):
        plt.cla()
        r=1/h[t]
        x=r[:,None]*np.outer(np.sin(theta),np.cos(varphi))
        y=r[:,None]*np.outer(np.sin(theta),np.sin(varphi))
        z=r[:,None]*np.outer(np.cos(theta),np.ones_like(varphi))
        plt.title(f't={0.1*t}')
        ax.plot_surface(x,y,z,cmap='Blues')
        ax.set_xlim([-3,3])
        ax.set_ylim([-3,3])
        ax.set_zlim([-3,2])
        ax.set_axis_off()
    plot_surface(0)
    ani=FuncAnimation(fig,plot_surface,frames=range(10,t_last+10,10))
    ani.save(file_location+'/r坐标.gif')
    plt.show()
    plt.close()

""" Plot
--------------------------------------------
"""
t_last=t_count-1
file_location=f'./data/normal_low_order/source={phi1}_{z0[0]}/n={index_temp}_gaussian/{N=}_{m=}_{dt=}_{t_last}_{phi0=}'

constraint_fields=np.load(file_location+'/1.data/constraint_fields.npy')
evolution_fields=np.load(file_location+'/1.data/evolution_fields.npy',allow_pickle=True)
h=np.load(file_location+'/1.data/h.npy')

os.makedirs(file_location+'/picture',exist_ok=True)

# time_index=[0,700,800,900,1000,2000]
# for index in time_index:
#     plt.plot(x0,h[index],label=f'$v={int(index*0.1)}$')
plt.plot(x0,h[0],'k',label='$v=0$')
plt.plot(x0,h[-1],label='$v=200$')
plt.xlabel(r'$\theta$',fontsize=12)
plt.ylabel('$z_h$',fontsize=12)
plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi],['$0$','$\pi/4$','$\pi/2$','$3\pi/4$','$\pi$'],fontsize=12)
plt.legend()
plt.savefig(file_location+f'/picture/final_horizon2.png',bbox_inches='tight')
plt.savefig(file_location+f'/picture/final_horizon2.pdf',bbox_inches='tight')
plt.show()
plt.close()

plt.plot(time_array,(2*np.pi/h**2*np.sin(x0)).sum(axis=1)*np.pi/m/2)
plt.xlabel('$v$',fontsize=12)
plt.ylabel(r'$\bar{s}$',fontsize=12)
plt.savefig(file_location+'/picture/average_entropy_density.png',bbox_inches='tight')
plt.savefig(file_location+'/picture/average_entropy_density.pdf',bbox_inches='tight')
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
plt.savefig(file_location+f'/picture/damping.png')
plt.savefig(file_location+f'/picture/damping.pdf')
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
    ax.set_title(f'$v={int(time_index*0.1)}$')

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
plt.savefig(file_location+f'/picture/horizons_3d.png',bbox_inches='tight')
plt.savefig(file_location+f'/picture/horizons_3d.pdf',bbox_inches='tight')
plt.show()
plt.close()