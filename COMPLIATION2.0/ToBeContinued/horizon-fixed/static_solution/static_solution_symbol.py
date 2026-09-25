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
V,beta,xi,eta,alpha,Theta,chi=V(z),beta(z),xi(z),eta(z),alpha(z),Theta(z),chi(z)
f=sp.symbols(r'f',cls=sp.Function,real=True)
f=f(z)
xi=0
alpha=0
eta=0
Theta=0
U_u=sp.Array([xi,eta])
h_ll=sp.exp(2*chi)*sp.Array([[sp.exp(alpha)*cosh(Theta),sinh(Theta)],[sinh(Theta),sp.exp(-alpha)*cosh(Theta)]])
metric=1/z**2*sp.Array([[-(sp.exp(2*beta)*V*z**3-my.dot(my.dot(U_u,h_ll),U_u)),-sp.exp(2*beta),-my.dot(h_ll,U_u)[0],-my.dot(h_ll,U_u)[1]],[-sp.exp(2*beta),0,0,0],[-my.dot(h_ll,U_u)[0],0,h_ll[0,0],h_ll[0,1]],[-my.dot(h_ll,U_u)[1],0,h_ll[1,0],h_ll[1,1]]])
inverse_metric=sp.Array(metric.tomatrix().inv())
riemann_R,ricci_R,scalar_R,einstein_G=my.curvature(coordinates,metric)

'''2.matter'''
phi=sp.symbols(r'\phi',cls=sp.Function,real=True)
phi=phi(z)
P=sp.symbols(r'P',cls=sp.Function,real=True)
P=P(phi)
# P=-6*sp.cosh(phi/sp.sqrt(3))-phi**4/5
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
# E_xi=(sp.tensorcontraction(my.dot(sp.derive_by_array(inverse_metric,xi),einstein_eq),(0,1))/sp.derive_by_array(inverse_metric,xi)[1,2]).expand()
E_V=(sp.tensorcontraction(my.dot(sp.derive_by_array(inverse_metric,beta),einstein_eq),(0,1))/sp.derive_by_array(inverse_metric,beta)[0,1]).expand()
E_chi=(sp.tensorcontraction(my.dot(sp.derive_by_array(inverse_metric,chi)/sp.derive_by_array(inverse_metric,chi)[2,2],einstein_eq),(0,1))).expand()
# E_alpha=(sp.tensorcontraction(my.dot(sp.derive_by_array(inverse_metric,alpha),einstein_eq),(0,1))/sp.derive_by_array(inverse_metric,alpha)[2,2]).expand()
E_phi=(scalar_eq).expand()
# E_V_evolution=((my.dot(inverse_metric[1,:],einstein_eq[:,0])+my.dot(my.dot(inverse_metric[1,:],einstein_eq[:,2:4]),U_u))/inverse_metric[1,0]).expand()
# E_xi_evolution=(my.dot(inverse_metric[1,:],einstein_eq[:,2])/inverse_metric[1,0]).expand()

'''2.normalization'''
E1_normalization=(E_beta*z/4).expand()
# E2_normalization=(E_xi/2).expand()
E3_normalization=(E_V/2).expand().subs(V,f/z**3).doit().expand()
# E4_normalization=(E_alpha*sp.exp(-alpha+2*beta-2*chi)/2).expand()
E5_normalization=(-E_phi*sp.exp(2*beta)/2/z**2).expand()
# E6_normalization=(E_V_evolution).expand()
# E7_normalization=(E_xi_evolution).expand()
E8_normalization=(E_chi/2).expand()

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

""" Variation
--------------------------------------------
"""
equation=sp.Array([E1_normalization,E3_normalization,E5_normalization,E1_normalization])
variation=my.variation(equation,(beta,chi,f,phi),z,2)

""" Functionalization
--------------------------------------------
"""
arg_parameter=[]
arg_variable=[z]
arg_field=my.symbol_arg_field((beta,chi,f,phi),z,2)
arguments=arg_parameter+arg_variable+arg_field

equation_function=sp.lambdify(arguments,equation,'numpy')
variation_function=sp.lambdify(arguments,variation,'numpy')
# error_function=sp.lambdify(arguments,E8_normalization,'numpy')