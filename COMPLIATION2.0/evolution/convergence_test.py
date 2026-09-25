import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')
import my_module.my_module as my

plt.rc('font',family='Times New Roman')
plt.rc('mathtext',fontset='stix')

dt=0.0005 # 单次演化时间步长
interval_times=200 # 每隔多少步存一次数据
t_count=5000+1 # 时间方向上存储数据的数量，+1是因为要存储初值
evolution_times=(t_count-1)*interval_times # 总共的演化次数
time_array=np.arange(t_count)*interval_times*dt # 最终输出的时间数组

phi1=2
index_temp=118
t_last=t_count-1
phi0=0.001



""" Vary m
--------------------------------------------
"""
N=50
z0,D1,D2=my.chebyshev_spectrum(N,0,1.3)

data_list=[4,6,8,10,12,14,16,18,20]
error=np.full(len(data_list),np.nan,dtype=object)
error2=np.full(len(data_list),np.nan,dtype=object)
h=np.full(len(data_list),np.nan,dtype=object)
E=np.full((len(data_list),t_last+1),np.nan)
E2=np.full((len(data_list),t_last+1),np.nan)

for i in range(0,len(data_list)):
    m=data_list[i]
    file_location=f'./data/normal_low_order/source={phi1}_{z0[0]}/n={index_temp}_gaussian/{N=}_{m=}_{dt=}_{t_last}_{phi0=}'

    error[i]=np.load(file_location+'/1.data/error.npy')
    error2[i]=np.load(file_location+'/1.data/error2.npy')
    h[i]=np.load(file_location+'/1.data/h.npy')

    for j in range(0,t_last+1):
        interp_h=my.chebyshev_interp(h[i][j],z0)
        E[i,j]=np.average(np.array([interp_h[k].dot(error[i][j,:,k]) for k in range(0,m)]))

file_location=f'./data/normal_low_order/source={phi1}_{z0[0]}/n={index_temp}_gaussian/{N=}_{dt=}_{t_last}_{phi0=}_covergence_test'
os.makedirs(file_location,exist_ok=True)

colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
excluded_colors=['#2ca02c','#d62728']
available_colors=[color for color in colors if color not in excluded_colors]
plt.rcParams['axes.prop_cycle']=plt.cycler(color=available_colors)

plt.plot(time_array[::10],np.array([np.log(np.abs(E[i][::10])) for i in range(0,len(data_list))]).T,label=[f'$M={2*m}$' for m in data_list])
plt.xlabel('$v$',fontsize=12)
plt.ylabel(r'$\ln|\bar{E}|$',fontsize=12)
plt.legend()
plt.savefig(file_location+f'/convergence_test1_M.png',bbox_inches='tight')
plt.savefig(file_location+f'/convergence_test1_M.pdf',bbox_inches='tight')
plt.show()
plt.close()

plt.plot(np.array(data_list)*2,np.array([np.log(np.abs(E[i,-1])) for i in range(0,len(data_list))]),'o',markersize=4)
plt.xlabel('$M$',fontsize=12)
plt.ylabel(r'$\ln|\bar{E}|$',fontsize=12)
plt.savefig(file_location+f'/convergence_test2_M.png',bbox_inches='tight')
plt.savefig(file_location+f'/convergence_test2_M.pdf',bbox_inches='tight')
plt.show()
plt.close()



""" Vary N'
--------------------------------------------
"""
m=20
M=2*m
length=2*np.pi
x0,P1,P2,P3=my.fourier_spectrum(M,-length/2+length/2/M,length)
x0=x0[m:]

data_list=[35,40,45,50,55,60,65,70]
error=np.full(len(data_list),np.nan,dtype=object)
error2=np.full(len(data_list),np.nan,dtype=object)
h=np.full(len(data_list),np.nan,dtype=object)
E=np.full((len(data_list),t_last+1),np.nan)
E2=np.full((len(data_list),t_last+1),np.nan)

for i in range(0,len(data_list)):
    N=data_list[i]
    z0,D1,D2=my.chebyshev_spectrum(N,0,1.3)
    file_location=f'./data/normal_low_order/source={phi1}_{z0[0]}/n={index_temp}_gaussian/{N=}_{m=}_{dt=}_{t_last}_{phi0=}'

    error[i]=np.load(file_location+'/1.data/error.npy')
    error2[i]=np.load(file_location+'/1.data/error2.npy')
    h[i]=np.load(file_location+'/1.data/h.npy')

    for j in range(0,t_last+1):
        interp_h=my.chebyshev_interp(h[i][j],z0)
        E[i,j]=np.average(np.array([interp_h[k].dot(error[i][j,:,k]) for k in range(0,m)]))

file_location=f'./data/normal_low_order/source={phi1}_{z0[0]}/n={index_temp}_gaussian/{m=}_{dt=}_{t_last}_{phi0=}_covergence_test'
os.makedirs(file_location,exist_ok=True)

colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
excluded_colors=['#2ca02c','#d62728']
available_colors=[color for color in colors if color not in excluded_colors]
plt.rcParams['axes.prop_cycle']=plt.cycler(color=available_colors)

plt.plot(time_array[::10],np.array([np.log(np.abs(E[i][::10])) for i in range(0,len(data_list))]).T,label=[f'${N=}$' for N in data_list])
plt.xlabel('$v$',fontsize=12)
plt.ylabel(r'$\ln|\bar{E}|$',fontsize=12)
plt.legend()
plt.savefig(file_location+f'/convergence_test1_N.png',bbox_inches='tight')
plt.savefig(file_location+f'/convergence_test1_N.pdf',bbox_inches='tight')
plt.show()
plt.close()

plt.plot(data_list,np.array([np.log(np.abs(E[i,-1])) for i in range(0,len(data_list))]),'o',markersize=4)
plt.xlabel('$N$',fontsize=12)
plt.ylabel(r'$\ln|\bar{E}|$',fontsize=12)
plt.savefig(file_location+f'/convergence_test2_N.png',bbox_inches='tight')
plt.savefig(file_location+f'/convergence_test2_N.pdf',bbox_inches='tight')
plt.show()
plt.close()