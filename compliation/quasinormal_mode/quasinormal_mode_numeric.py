import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')
import my_module.my_module as my
import quasinormal_mode_symbol

# plt.rc('font',family='Times New Roman')
# plt.rc('mathtext',fontset='stix')

""" Schwarzschild-AdS
--------------------------------------------
"""
"""
'''1.get modes'''
l=0
N=[40,50,60]
w=len(N)*[None]
v=len(N)*[None]
error=len(N)*[None]
boundary_condition=[(0,0,-1,2,0)]

for i in range(0,len(N)):
    print(f'N={N[i]}')
    z,D1,D2=my.chebyshev_spectrum(N[i],0,1)
    f_Sch=z**2+1-2*z**3
    chi_Sch=np.zeros(N[i])
    phi_Sch=np.zeros(N[i])
    fields_Sch=np.array([f_Sch,chi_Sch,phi_Sch])

    w[i],v[i],error[i]=my.get_modes(fields_Sch,quasinormal_mode_symbol.equation_bg_function,quasinormal_mode_symbol.variation_phi_function,quasinormal_mode_symbol.error_variation_phi_function,[l],z,D1,D2,boundary_condition,-1e-12,False)
    error[i]=np.abs(error[i]).max(axis=0)

'''2.filter'''
error_max=1e-5
mode_base=w[0][np.where(error[0]<error_max)]
mode_ref=[w[i][np.where(error[i]<error_max)] for i in range(1,len(w))]
precision_list=[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]+[1e-2+1e-3*i for i in range(1,991)]
mode_number=6
index=my.get_accurate_mode_index(precision_list,mode_number,mode_base,*mode_ref)

'''3.sort'''
# 先按实部绝对值（保留两位小数）从小到大，再按虚部（保留五位小数）从大到小，再按实部从小到大
argsort=np.lexsort((np.real(mode_base[index]),np.round(-np.imag(mode_base[index]),5),np.round(np.abs(np.real(mode_base[index])),2)))
mode_Sch=mode_base[index][argsort]
print(mode_Sch)

'''4.plot'''
plt.plot(np.real(mode_base[index]),np.imag(mode_base[index]),'o',markersize=3)
plt.axhline(0,color='grey')
plt.xlabel('$\mathrm{Re}(\omega)$',fontsize=12)
plt.ylabel('$\mathrm{Im}(\omega)$',fontsize=12)
plt.show()
plt.close()
"""

""" spherical static solution from iteration
--------------------------------------------
"""
'''1.background solution'''
source=1.889
N=100
z_max=6.0
z=my.chebyshev_spectrum(N,0,z_max)[0]
n=120
n_temp=118
# if not os.path.isdir('checkpoint'):
#     os.mkdir('checkpoint')
file_location_solution=f'../static_solution/BS/EOS/{source=}/z0={z[0]}/{N=}'
static_solution=np.load(file_location_solution+'/1.data/1.static_solution.npy')[n_temp]
fields_original=np.array([1+z**2*static_solution[0],static_solution[1],z*static_solution[2]])
fields_fit=[np.polynomial.Chebyshev.fit(z,fields_original[i],N-1) for i in range(0,3)]

temperature=np.load(file_location_solution+'/1.data/2.temperature.npy')
entropy_density=np.load(file_location_solution+'/1.data/3.entropy_density.npy')
plt.plot(temperature,entropy_density)
plt.plot(temperature[n_temp],entropy_density[n_temp],'o')
plt.xlabel('$T$',fontsize=12)
plt.ylabel('$s$',fontsize=12)
plt.show()
plt.close()

z_H=np.load(file_location_solution+'/1.data/6.z_H.npy')[n_temp]
z_max=z_H

'''2.get modes & filter & plot'''
mode=[]
boundary_condition=[(1,1,-1,1,0),(2,2,-1,1,0),(3,3,-1,1,0),(4,4,-1,2,0)]
# # l=0时
# boundary_condition=[(1,1,-1,1,0),(2,2,-1,2,0)]
file_location=f'./QNM/{source=}/{z[0]}_{n}_{n_temp}_new'
os.makedirs(file_location+'/1.data',exist_ok=True)
os.makedirs(file_location+'/2.picture',exist_ok=True)

for l in range(0,5):
    print(f'{l=}')
    N=[30,40,50]
    w=len(N)*[None]
    v=len(N)*[None]
    error=len(N)*[None]
    for i in range(0,len(N)):
        print(f'N={N[i]}')
        z,D1,D2=my.chebyshev_spectrum(N[i],0,z_max)
        fields=np.array([fields_fit[i](z) for i in range(0,3)])
        w[i],v[i],error[i]=my.get_modes(fields,quasinormal_mode_symbol.equation_bg_function,quasinormal_mode_symbol.variation_function,quasinormal_mode_symbol.error_variation_function,[l],z,D1,D2,boundary_condition,-1e-12,False)
        error[i]=np.abs(error[i]).max(axis=0)
        # error[i]=error[i][0]

    error_max=1e-4
    mode_base=w[0][np.where(error[0]<error_max)]
    mode_ref=[w[i][np.where(error[i]<error_max)] for i in range(1,len(w))]
    precision_list=[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]+[1e-2+1e-3*i for i in range(1,991)]
    mode_number=8
    index=my.get_accurate_mode_index(precision_list,mode_number,mode_base,*mode_ref)

    # 先按实部绝对值（保留两位小数）从小到大，再按虚部（保留两位小数）从大到小，再按实部从小到大
    argsort=np.lexsort((np.real(mode_base[index]),np.round(-np.imag(mode_base[index]),2),np.round(np.abs(np.real(mode_base[index])),2)))
    # # 先按虚部（保留两位小数）从大到小，再按实部从小到大
    # argsort=np.lexsort((np.real(mode_base[index]),np.round(-np.imag(mode_base[index]),2)))
    print(mode_base[index][argsort])
    if l == 0:
        mode.append(np.hstack((0+0j,0+0j,mode_base[index][argsort][1:5])))
    else:
        mode.append(mode_base[index][argsort][0:6])

    plt.axhline(0,color='grey')
    # plt.plot(np.real(mode_base),np.imag(mode_base),'o',markersize=5)
    plt.plot(np.real(mode_base[index][argsort]),np.imag(mode_base[index][argsort]),'or',markersize=5)
    plt.xlabel('$\mathrm{Re}(\omega)$',fontsize=12)
    plt.ylabel('$\mathrm{Im}(\omega)$',fontsize=12)
    # plt.title(f'${l=}$')
    # plt.savefig(file_location+f'/2.picture/{l=}.png',bbox_inches='tight')
    # plt.savefig(file_location+f'/2.picture/{l=}.pdf',bbox_inches='tight')
    plt.show()
    plt.close()

error[0][np.where(error[0]<error_max)][index][argsort]

z,D1,D2=my.chebyshev_spectrum(N[0],0,z_max)
plt.plot(z,np.real(v[0][:,0,:][np.where(error[0]<error_max)][index][argsort][0]).T,label='$\mathrm{Re}$')
plt.plot(z,np.imag(v[0][:,0,:][np.where(error[0]<error_max)][index][argsort][0]).T,label='$\mathrm{Im}$')
plt.axhline(0,color='grey')
plt.xlabel('$z$',fontsize=12)
plt.ylabel('$\delta f$',fontsize=12)
plt.legend()
plt.show()
plt.close()

'''3.save data'''
# mode[0][[0,1]]=0
mode=np.array(mode)
np.save(file_location+'/1.data/mode.npy',mode)

""" Plot
--------------------------------------------
"""
mode_number=4
l_max=3
colors=['tab:blue','tab:orange','tab:purple','tab:purple','tab:brown','tab:brown']
markers=['o','^','v','s','D']

def ax_plot(ax):
    ax.axhline(0,color='grey')
    for i in range(mode_number-1,-1,-1):
        ax.plot(np.real(mode[0:l_max+1,i]),np.imag(mode[0:l_max+1,i]),':',color=colors[i],linewidth=0.8)
        for j in range(0,l_max):
            if (i==0) and (j==0):
                pass
            else:
                ax.annotate('',xy=(np.average(np.real(mode[j:j+2,i]))+(np.real(mode[j+1,i])-np.real(mode[j,i]))/100,scipy.interpolate.interp1d(np.real(mode[j:j+2,i]),np.imag(mode[j:j+2,i]))(np.average(np.real(mode[j:j+2,i]))+(np.real(mode[j+1,i])-np.real(mode[j,i]))/100)),xytext=(np.average(np.real(mode[j:j+2,i]))-(np.real(mode[j+1,i])-np.real(mode[j,i]))/100,scipy.interpolate.interp1d(np.real(mode[j:j+2,i]),np.imag(mode[j:j+2,i]))(np.average(np.real(mode[j:j+2,i]))-(np.real(mode[j+1,i])-np.real(mode[j,i]))/100)),size=8,arrowprops=dict(color=colors[i],arrowstyle="->"))
        for j in range(0,l_max+1):
            ax.plot(np.real(mode[j,i]),np.imag(mode[j,i]),marker=markers[j],color=colors[i],markersize=4)
    ax.plot(0,0,'k',marker='o',markersize=4)

fig,ax=plt.subplots()
ax_plot(ax)

# '''118
axins=ax.inset_axes((0.7,0.3,0.27,0.27))
ax_plot(axins)
axins.annotate('',xy=(-0.06,scipy.interpolate.interp1d(np.real(mode[1:3,0]),np.imag(mode[1:3,0]))(-0.06)),xytext=(-0.05,scipy.interpolate.interp1d(np.real(mode[1:3,0]),np.imag(mode[1:3,0]))(-0.05)),size=8,arrowprops=dict(color=colors[0],arrowstyle="->"))
axins.annotate('',xy=(0,0.05),xytext=(0,0.04),size=8,arrowprops=dict(color=colors[0],arrowstyle="->"))
axins.annotate('',xy=(0,-0.05),xytext=(0,-0.04),size=8,arrowprops=dict(color=colors[1],arrowstyle="->"))
xlim0,xlim1,ylim0,ylim1=-0.2,0.2,-0.075,0.125
axins.set_xlim(xlim0,xlim1)
axins.set_ylim(ylim0,ylim1)
ax.plot([xlim0,xlim1,xlim1,xlim0,xlim0],[ylim0,ylim0,ylim1,ylim1,ylim0],'k',linewidth=0.5)
# '''

'''110
axins=ax.inset_axes((0.7,0.3,0.27,0.27))
ax_plot(axins)
axins.annotate('',xy=(0,-0.05),xytext=(0,-0.04),size=8,arrowprops=dict(color=colors[0],arrowstyle="->"))
axins.annotate('',xy=(-0.05,scipy.interpolate.interp1d(np.real(mode[1:3,0]),np.imag(mode[1:3,0]))(-0.05)),xytext=(-0.04,scipy.interpolate.interp1d(np.real(mode[1:3,0]),np.imag(mode[1:3,0]))(-0.04)),size=8,arrowprops=dict(color=colors[0],arrowstyle="->"))
axins.annotate('',xy=(0,-0.14),xytext=(0,-0.13),size=8,arrowprops=dict(color=colors[1],arrowstyle="->"))
xlim0,xlim1,ylim0,ylim1=-0.2,0.2,-0.15,0.05
axins.set_xlim(xlim0,xlim1)
axins.set_ylim(ylim0,ylim1)
ax.plot([xlim0,xlim1,xlim1,xlim0,xlim0],[ylim0,ylim0,ylim1,ylim1,ylim0],'k',linewidth=0.5)
'''

plt.xlabel('$\mathrm{Re}(\omega)$',fontsize=12)
plt.ylabel('$\mathrm{Im}(\omega)$',fontsize=12)
patches=[plt.plot([],[],'k',marker=markers[i],markersize=4,label=f'$l={i}$')[0] for i in range(0,l_max+1)]
plt.legend(handles=patches,handlelength=0,loc='lower center')
plt.savefig(file_location+'/2.picture/modes2.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/modes2.pdf',bbox_inches='tight')
plt.show()
plt.close()



plt.axhline(0,color='grey')
for i in [0,1,3,5]:
    plt.plot(range(0,5),np.real(mode[:,i]),'o:',markersize=4,linewidth=1)
plt.locator_params(axis='x',integer=True)
plt.xlabel('$l$',fontsize=12)
plt.ylabel('$\mathrm{Re}(\omega)$',fontsize=12)
plt.savefig(file_location+'/2.picture/real_part.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/real_part.pdf',bbox_inches='tight')
plt.show()
plt.close()

plt.axhline(0,color='grey')
for i in [0,1,3,5]:
    plt.plot(range(0,5),np.imag(mode[:,i]),'o:',markersize=4,linewidth=1)
plt.locator_params(axis='x',integer=True)
plt.xlabel('$l$',fontsize=12)
plt.ylabel('$\mathrm{Im}(\omega)$',fontsize=12)
plt.savefig(file_location+'/2.picture/imag_part.png',bbox_inches='tight')
plt.savefig(file_location+'/2.picture/imag_part.pdf',bbox_inches='tight')
plt.show()
plt.close()