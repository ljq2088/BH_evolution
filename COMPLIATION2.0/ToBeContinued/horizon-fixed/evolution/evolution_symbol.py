import sympy as sp
import sys
sys.path.append('../..')
import my_module.my_module as my

# def cosh(x):
#     return (sp.exp(x)+sp.exp(-x))/2
# def sinh(x):
#     return (sp.exp(x)-sp.exp(-x))/2

def cosh(x):
    return sp.cosh(x)
def sinh(x):
    return sp.sinh(x)

""" Setup
--------------------------------------------
"""
# $\mathcal{L}=R-\frac{1}{2}\nabla_\mu\phi\nabla^\mu\phi-V(\phi)$

'''1.gravity'''
v,z,x,y=sp.symbols(r'v,z,x,y',real=True)
coordinates=sp.Array([v,z,x,y])
V,beta,xi,eta,alpha,Theta,chi=sp.symbols(r'V,\beta,\xi,\eta,\alpha,\Theta,\chi',cls=sp.Function,real=True)
V,beta,xi,eta,alpha,Theta,chi=V(v,z,x),beta(v,z,x),xi(v,z,x),eta(v,z,x),alpha(v,z,x),Theta(v,z,x),chi(v,z,x)
# xi=0
# alpha=0
eta=0
Theta=0
U_u=sp.Array([xi,eta])
h_ll=sp.exp(2*chi)*sp.Array([[sp.exp(alpha)*cosh(Theta),sinh(Theta)],[sinh(Theta),sp.exp(-alpha)*cosh(Theta)]])
metric=1/z**2*sp.Array([[-(sp.exp(2*beta)*V*z**3-my.dot(my.dot(U_u,h_ll),U_u)),-sp.exp(2*beta),-my.dot(h_ll,U_u)[0],-my.dot(h_ll,U_u)[1]],[-sp.exp(2*beta),0,0,0],[-my.dot(h_ll,U_u)[0],0,h_ll[0,0],h_ll[0,1]],[-my.dot(h_ll,U_u)[1],0,h_ll[1,0],h_ll[1,1]]])
inverse_metric=sp.Array(metric.tomatrix().inv())
riemann_R,ricci_R,scalar_R,einstein_G=my.curvature(coordinates,metric)

'''2.matter'''
phi=sp.symbols(r'\phi',cls=sp.Function,real=True)
phi=phi(v,z,x)
# P=sp.symbols(r'P',cls=sp.Function,real=True)
# P=P(phi)
P=-6*sp.cosh(phi/sp.sqrt(3))-phi**4/5
nabla_phi=my.nabla(phi,coordinates,metric)
T_phi=(sp.tensorproduct(nabla_phi,nabla_phi)-(my.dot(nabla_phi,my.dot(inverse_metric,nabla_phi))/2+P)*metric)/2

# T_phi=T_phi.subs(phi,0).doit()

'''3.equations'''
scalar_eq=sp.tensorcontraction(my.dot(inverse_metric,my.nabla(nabla_phi,coordinates,metric)),(0,1))-P.diff(phi)
einstein_eq=einstein_G-T_phi

# G_ll=ricci_R-P.subs(phi,0)*metric/2

""" Equation simplification
--------------------------------------------
"""
'''1.combination'''
# E1=(einstein_eq[1,1]).expand()
# E2=(einstein_eq[1,2]).expand()
# E3=(einstein_eq[0,1]-V*z**3/2*einstein_eq[1,1]+xi*einstein_eq[1,2]).expand()
# E4=(einstein_eq[2,2]/sp.exp(alpha+beta)-einstein_eq[3,3]/sp.exp(-alpha+beta)).expand()
# E5=(scalar_eq/sp.exp(chi)).expand()
# E6=(einstein_eq[0,0]-V*z**3*einstein_eq[0,1]+xi*einstein_eq[0,2]).expand()
# E7=(einstein_eq[2,0]-V*z**3*einstein_eq[2,1]+xi*einstein_eq[2,2]).expand()

E_beta=(sp.tensorcontraction(my.dot(sp.derive_by_array(inverse_metric,V),einstein_eq),(0,1))/sp.derive_by_array(inverse_metric,V)[1,1]).expand()
E_xi=(sp.tensorcontraction(my.dot(sp.derive_by_array(inverse_metric,xi),einstein_eq),(0,1))/sp.derive_by_array(inverse_metric,xi)[1,2]).expand()
E_V=(sp.tensorcontraction(my.dot(sp.derive_by_array(inverse_metric,beta),einstein_eq),(0,1))/sp.derive_by_array(inverse_metric,beta)[0,1]).expand()
E_chi=(sp.tensorcontraction(my.dot(sp.derive_by_array(inverse_metric,chi)/sp.derive_by_array(inverse_metric,chi)[2,2],einstein_eq),(0,1))).expand()
E_alpha=(sp.tensorcontraction(my.dot(sp.derive_by_array(inverse_metric,alpha),einstein_eq),(0,1))/sp.derive_by_array(inverse_metric,alpha)[2,2]).expand()
E_phi=(scalar_eq).expand()
E_V_evolution=((my.dot(inverse_metric[1,:],einstein_eq[:,0])+my.dot(my.dot(inverse_metric[1,:],einstein_eq[:,2:4]),U_u))/inverse_metric[1,0]).expand()
E_xi_evolution=(my.dot(inverse_metric[1,:],einstein_eq[:,2])/inverse_metric[1,0]).expand()

'''2.normalization'''
E1_normalization=(E_beta*z/4).expand()
E2_normalization=(E_xi/2).expand()
E3_normalization=(E_V/2).expand()
E4_normalization=(E_alpha*sp.exp(-alpha+2*beta-2*chi)/2).expand()
E5_normalization=(-E_phi*sp.exp(2*beta)/2/z**2).expand()
E6_normalization=(E_V_evolution).expand()
E7_normalization=(E_xi_evolution).expand()

Xi=sp.exp(4*chi-2*beta+alpha)*xi.diff(z)/z**2
dt_chi=chi.diff(v)-z**2/2*V*(z*chi.diff(z)-1)
dt_alpha=alpha.diff(v)-z**3/2*V*alpha.diff(z)
# dt_Theta=Theta.diff(v)-z**3/2*V*Theta.diff(z)
dt_phi=phi.diff(v)-z**3/2*V*phi.diff(z)

-(E2_normalization-z**2*sp.exp(-2*chi)/2*Xi.diff(z)).expand()
-(E3_normalization-2*sp.exp(-2*chi)*z**2*(sp.exp(2*chi)/z**2*dt_chi).diff(z)).expand()
-(E4_normalization-(sp.exp(-chi)*z*(sp.exp(chi)/z*dt_alpha).diff(z)+alpha.diff(z)*dt_chi)).expand()
-(E5_normalization-(sp.exp(-chi)*z*(sp.exp(chi)/z*dt_phi).diff(z)+phi.diff(z)*dt_chi)).expand()

dt_chi=sp.symbols(r'd_{t}\chi',cls=sp.Function,real=True)
dt_chi=dt_chi(v,z,x)
dt_alpha=sp.symbols(r'd_{t}\alpha',cls=sp.Function,real=True)
dt_alpha=dt_alpha(v,z,x)
dt_phi=sp.symbols(r'd_{t}\phi',cls=sp.Function,real=True)
dt_phi=dt_phi(v,z,x)
# E_VH=E6_normalization.subs(chi.diff(v),z**2/2*V*(z*chi.diff(z)-1)-xi*chi.diff(x)-xi.diff(x)/2).doit().expand().subs(chi.diff(v),z**2/2*V*(z*chi.diff(z)-1)-xi*chi.diff(x)-xi.diff(x)/2).doit().expand().subs([(alpha.diff(v),dt_alpha+z**3/2*V*alpha.diff(z)),(phi.diff(v),dt_phi+z**3/2*V*phi.diff(z))]).doit().expand()
E_VH=E6_normalization.subs(chi.diff(v),z**2/2*V*(z*chi.diff(z)-1)-xi*chi.diff(x)-xi.diff(x)/2).doit().expand().subs(chi.diff(v),dt_chi+z**2/2*V*(z*chi.diff(z)-1)).doit().expand().subs([(alpha.diff(v),dt_alpha+z**3/2*V*alpha.diff(z)),(phi.diff(v),dt_phi+z**3/2*V*phi.diff(z))]).doit().expand()
E_VH=(E_VH-E_VH.coeff(chi.diff(z,z))/E1_normalization.coeff(chi.diff(z,z))*E1_normalization).expand()

'''3.subtraction'''
r'''
chi3=sp.symbols(r'\chi_3',cls=sp.Function,real=True)
chi3=chi3(v,x)
V_tilde,beta_tilde,xi_tilde,alpha_tilde,phi_tilde=sp.symbols(r'\tilde{V},\tilde{\beta},\tilde{\xi},\tilde{\alpha},\tilde{\phi}',cls=sp.Function)
V_tilde,beta_tilde,xi_tilde,alpha_tilde,phi_tilde=V_tilde(v,z,x),beta_tilde(v,z,x),xi_tilde(v,z,x),alpha_tilde(v,z,x),phi_tilde(v,z,x)

V_subt=V
beta_subt=-z**3/2*chi3+z**4*beta_tilde
chi_subt=sp.log(1+2*z**3*chi3)/4
xi_subt=xi
alpha_subt=alpha
phi_subt=0

subtraction_subslist=[(V,V_subt),(beta,beta_subt),(chi,chi_subt),(xi,xi_subt),(alpha,alpha_subt),(phi,phi_subt)]

E1_subtraction=(E1_normalization).subs(subtraction_subslist).doit().expand()
E2_subtraction=(E2_normalization).subs(subtraction_subslist).doit().expand()
E3_subtraction=(E3_normalization).subs(subtraction_subslist).doit().expand()
E4_subtraction=(E4_normalization).subs(subtraction_subslist).doit().expand()
E5_subtraction=(E5_normalization).subs(subtraction_subslist).doit().expand()
E6_subtraction=(E6_normalization).subs(subtraction_subslist).doit().expand()
E7_subtraction=(E7_normalization).subs(subtraction_subslist).doit().expand()

# 边界上方程的形式
E1_subtraction.subs(z,0).doit().expand()
E2_subtraction.subs(z,0).doit().expand()
E3_subtraction.subs(z,0).doit().expand()
E4_subtraction.subs(z,0).doit().expand()
E5_subtraction.subs(z,0).doit().expand()
E6_subtraction.subs(z,0).doit().expand()
E7_subtraction.subs(z,0).doit().expand()
'''

""" Schwarzschild-AdS
--------------------------------------------
"""
'''
V_Sch=1/z**3*(1-z**3)
beta_Sch=0
chi_Sch=0
alpha_Sch=0
xi_Sch=0
phi_Sch=0

Sch_subslist=[(V,V_Sch),(beta,beta_Sch),(chi,chi_Sch),(alpha,alpha_Sch),(xi,xi_Sch),(phi,phi_Sch)]

# 验证它满足以上方程
E1_normalization.subs(Sch_subslist).doit().expand()
E2_normalization.subs(Sch_subslist).doit().expand()
E3_normalization.subs(Sch_subslist).doit().expand()
E4_normalization.subs(Sch_subslist).doit().expand()
E5_normalization.subs(Sch_subslist).doit().expand()
E6_normalization.subs(Sch_subslist).doit().expand()
E7_normalization.subs(Sch_subslist).doit().expand()
'''

""" Asymptotic behavior
--------------------------------------------
"""
# r'''
V0,V1,V2,V3,V4,V5,beta0,beta1,beta2,beta3,beta4,beta5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,phi0,phi1,phi2,phi3,phi4,phi5=sp.symbols(r'V_0,V_1,V_2,V_3,V_4,V_5,\beta_0,\beta_1,\beta_2,\beta_3,\beta_4,\beta_5,\chi_0,\chi_1,\chi_2,\chi_3,\chi_4,\chi_5,\xi_0,\xi_1,\xi_2,\xi_3,\xi_4,\xi_5,\alpha_0,\alpha_1,\alpha_2,\alpha_3,\alpha_4,\alpha_5,\phi_0,\phi_1,\phi_2,\phi_3,\phi_4,\phi_5',cls=sp.Function,real=True)
V0,V1,V2,V3,V4,V5,beta0,beta1,beta2,beta3,beta4,beta5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,phi0,phi1,phi2,phi3,phi4,phi5=V0(v,x),V1(v,x),V2(v,x),V3(v,x),V4(v,x),V5(v,x),beta0(v,x),beta1(v,x),beta2(v,x),beta3(v,x),beta4(v,x),beta5(v,x),chi0(v,x),chi1(v,x),chi2(v,x),chi3(v,x),chi4(v,x),chi5(v,x),xi0(v,x),xi1(v,x),xi2(v,x),xi3(v,x),xi4(v,x),xi5(v,x),alpha0(v,x),alpha1(v,x),alpha2(v,x),alpha3(v,x),alpha4(v,x),alpha5(v,x),phi0(v,x),phi1(v,x),phi2(v,x),phi3(v,x),phi4(v,x),phi5(v,x)

V_asymp=(V0+V1*z+V2*z**2+V3*z**3+sp.O(z**4))/z**3
beta_asymp=beta0+beta1*z+beta2*z**2+beta3*z**3+sp.O(z**4)
# chi_asymp=chi0+chi1*z+chi2*z**2+chi3*z**3+sp.O(z**4)
chi_asymp=sp.log(1+2*z**3*chi3)/4
xi_asymp=xi0+xi1*z+xi2*z**2+xi3*z**3+sp.O(z**4)
alpha_asymp=alpha0+alpha1*z+alpha2*z**2+alpha3*z**3+sp.O(z**4)
phi_asymp=phi0+phi1*z+phi2*z**2+phi3*z**3+sp.O(z**4)

asymptotic_behavior_subslist=[(V,V_asymp),(beta,beta_asymp),(chi,chi_asymp),(xi,xi_asymp),(alpha,alpha_asymp),(phi,phi_asymp)]

equation_asymp_subslist=[] # 由方程给出的渐近行为
boundary_condition_subslist=[] # 在z=0处手动添加的边界条件

equation_asymp_subslist=equation_asymp_subslist+[(beta1,0),(phi0,0),(xi1,-2*sp.exp(-alpha0+2*beta0)*beta0.diff(x)),(V0,sp.exp(2*beta0))]
equation_asymp_subslist=equation_asymp_subslist+[(V1,-xi0.diff(x))]
'''
boundary_condition_subslist=boundary_condition_subslist+[(beta0,0),(xi0,0)]

equation_asymp_subslist=equation_asymp_subslist+[(alpha1,alpha0.diff(v)),(alpha2,0)]
boundary_condition_subslist=boundary_condition_subslist+[(alpha0,0)]

equation_asymp_subslist=equation_asymp_subslist+[(xi2,0),(V2,-2*beta2),(beta2,-phi1**2/16),(beta3,-phi1*phi2/6-chi3/2)]
source=sp.symbols(r'\phi_1',real=True)
boundary_condition_subslist=boundary_condition_subslist+[(phi1,source)]
'''
boundary_condition_subslist=boundary_condition_subslist+[(xi0,0)]

equation_asymp_subslist=equation_asymp_subslist+[(alpha1,alpha0.diff(v)*sp.exp(-2*beta0)),(alpha2,0)]
boundary_condition_subslist=boundary_condition_subslist+[(alpha0,0)]

equation_asymp_subslist=equation_asymp_subslist+[(xi2,0),(V2,-2*beta2*sp.exp(2*beta0)+sp.exp(2*beta0).diff(x,x).expand()),(beta2,-phi1**2/16),(beta3,-phi1*phi2/6-chi3/2)]
source=sp.symbols(r'\phi_1',real=True)
boundary_condition_subslist=boundary_condition_subslist+[(phi1,source)]

# sp.series(E1_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
# sp.series(E2_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,2)
# sp.series(E3_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,2)
# sp.series(E4_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,2)
# sp.series(E5_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,2)
# sp.series(E6_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
# sp.series(E7_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)

V3_eq=sp.series(E6_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3).coeff(z**2)
xi3_eq=sp.series(E7_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3).coeff(z**2)

pv_xi3=(xi3.diff(v)-xi3_eq/xi3_eq.coeff(xi3.diff(v))).expand()
pv_V3=(V3.diff(v)-V3_eq/V3_eq.coeff(V3.diff(v))).expand()



Xi=sp.exp(4*chi-2*beta+alpha)*xi.diff(z)/z**2
dt_chi=chi.diff(v)-z**2/2*V*(z*chi.diff(z)-1)
dt_alpha=alpha.diff(v)-z**3/2*V*alpha.diff(z)
# dt_Theta=Theta.diff(v)-z**3/2*V*Theta.diff(z)
dt_phi=phi.diff(v)-z**3/2*V*phi.diff(z)

sp.series(Xi.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)
sp.series((sp.exp(2*chi)/z**2*dt_chi).expand().subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)
sp.series((sp.exp(chi)/z*dt_alpha).expand().subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,2)
sp.series((sp.exp(chi)/z*dt_phi).expand().subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,2)
sp.series(dt_chi.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
sp.series(dt_alpha.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
sp.series(dt_phi.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
# '''



import re
import pyperclip

def modify_latex(input_str, fun_str_list):
    # Rule 1: Remove function arguments display
    modified_str = re.sub(r'{\\left\((.*?)\\right\)}', '', input_str)

    for fun_str in fun_str_list:
        # Rule 2: Replace v's derivative with \dot
        modified_str = modified_str.replace(r'\frac{\partial}{\partial v} '+fun_str, r'\dot{'+fun_str+'}')

        # Rule 3: Replace z's derivative with '
        modified_str = modified_str.replace(r'\frac{\partial}{\partial z} '+fun_str, fun_str+"'")

        # Rule 4: Replace x's derivative with _x
        modified_str = modified_str.replace(r'\frac{\partial}{\partial x} '+fun_str, fun_str+'_x')

        # Rule 5: Modify second derivatives
        modified_str = modified_str.replace(r'\frac{\partial^{2}}{\partial v^{2}} '+fun_str, r'\ddot{'+fun_str+'}')
        modified_str = modified_str.replace(r'\frac{\partial^{2}}{\partial z^{2}} '+fun_str, fun_str+"''")
        modified_str = modified_str.replace(r'\frac{\partial^{2}}{\partial x^{2}} '+fun_str, fun_str+'_{xx}')
        modified_str = modified_str.replace(r'\frac{\partial^{2}}{\partial v\partial z} '+fun_str, r'\dot{'+fun_str+"}'")
        modified_str = modified_str.replace(r'\frac{\partial^{2}}{\partial z\partial x} '+fun_str, fun_str+"'_x")
        modified_str = modified_str.replace(r'\frac{\partial^{2}}{\partial v\partial x} '+fun_str, r'\dot{'+fun_str+'}_x')

    # Rule 6: Merge exponential terms
    modified_str = re.sub(r'e\^{(.*?)} e\^{(-(.*?))}', r"e^{\1\2}", modified_str)
    modified_str = re.sub(r'e\^{(.*?)} e\^{((.*?))}', r"e^{\1+\2}", modified_str)

    return modified_str

input_str=r'\frac{z^{3} \frac{\partial}{\partial x} V \frac{\partial}{\partial z} \xi}{2} - \frac{z^{3} e^{- \alpha} e^{2 \beta} e^{- 2 \chi} \frac{\partial}{\partial x} V \frac{\partial}{\partial x} \alpha}{2} + z^{3} e^{- \alpha} e^{2 \beta} e^{- 2 \chi} \frac{\partial}{\partial x} V \frac{\partial}{\partial x} \beta + \frac{z^{3} e^{- \alpha} e^{2 \beta} e^{- 2 \chi} \frac{\partial^{2}}{\partial x^{2}} V}{2} - \frac{\xi^{2} \left(\frac{\partial}{\partial x} \alpha\right)^{2}}{2} - \frac{\xi^{2} \left(\frac{\partial}{\partial x} \phi\right)^{2}}{2} - \xi d_{t}\alpha \frac{\partial}{\partial x} \alpha - \xi d_{t}\phi \frac{\partial}{\partial x} \phi - \xi \frac{\partial}{\partial x} \alpha \frac{\partial}{\partial x} \xi - \frac{d_{t}\alpha^{2}}{2} - d_{t}\alpha \frac{\partial}{\partial x} \xi - \frac{d_{t}\phi^{2}}{2} - \frac{\left(\frac{\partial}{\partial x} \xi\right)^{2}}{2}'
fun_str_list=['V',r'\beta',r'\xi',r'\alpha',r'\chi',r'\phi']

pyperclip.copy(modify_latex(input_str,fun_str_list))
pyperclip.copy(modify_latex(sp.latex(_),fun_str_list))