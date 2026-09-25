import sympy as sp
import sys
sys.path.append('../..')
import my_module.my_module as my

""" Setup
--------------------------------------------
"""
# $\mathcal{L}=R-\frac{1}{2}\nabla_\mu\phi\nabla^\mu\phi-V(\phi)$

'''1.gravity'''
v,z,theta,varphi=sp.symbols(r'v,z,\theta,\varphi',real=True)
coordinates=sp.Array([v,z,theta,varphi])
f,chi=sp.symbols('f,\chi',cls=sp.Function,real=True)
f,chi=f(z),chi(z)
# L=sp.symbols('L',real=True)
L=sp.S(1)
metric=L**2/z**2*sp.Array([[-f*sp.exp(-chi),-sp.exp(-chi),0,0],[-sp.exp(-chi),0,0,0],[0,0,1,0],[0,0,0,sp.sin(theta)**2]])
# metric=L**2/z**2*sp.Array([[-f*sp.exp(-chi),-sp.exp(-chi),0,0],[-sp.exp(-chi),0,0,0],[0,0,1,0],[0,0,0,1]]) # 平面拓扑
inverse_metric=sp.Array(metric.tomatrix().inv())
riemann_R,ricci_R,scalar_R,einstein_G=my.curvature(coordinates,metric)

'''2.matter'''
phi=sp.symbols('\phi',cls=sp.Function,real=True)
phi=phi(z)
# V=sp.symbols('V',cls=sp.Function,real=True)
# V=V(phi)
V=-6*sp.cosh(phi/sp.sqrt(3))-phi**4/5
nabla_phi=my.nabla(phi,coordinates,metric)
T_phi=(sp.tensorproduct(nabla_phi,nabla_phi)-(my.dot(nabla_phi,my.dot(inverse_metric,nabla_phi))/2+V)*metric)/2

'''3.equations'''
scalar_eq=sp.tensorcontraction(my.dot(inverse_metric,my.nabla(nabla_phi,coordinates,metric)),(0,1))-V.diff(phi)
einstein_eq=einstein_G-T_phi

""" Equation simplification
--------------------------------------------
"""
'''1.combination'''
E1=(einstein_eq[1,1]).expand()
E2=(einstein_eq[0,1]-f/2*einstein_eq[1,1]).expand() # 这个等式平面拓扑和球拓扑有区别
E3=(scalar_eq/2*sp.exp(-chi)*L**2/z**2).expand()
E4=(einstein_eq[2,2]).expand()

# 验证方程不独立
# (einstein_eq[0,0]-f*einstein_eq[0,1]).expand()
# (einstein_eq[3,3]-sp.sin(theta)**2*einstein_eq[2,2]).expand()
# (E4+E1.diff(z)*z*f*sp.exp(chi)/4-E2.diff(z)*z*sp.exp(chi)/2+E3*z*sp.exp(chi)*phi.diff(z)/2-E2*z*sp.exp(chi)*chi.diff(z)/2+E1*(z*f.diff(z)*sp.exp(chi)/2-f*sp.exp(chi)/2)).expand()

'''2.normalization'''
E1_normalization=(E1*z).expand()
E2_normalization=(E2*z**2).expand() # 这个等式平面拓扑和球拓扑有区别
E3_normalization=(2*E3*z**2).expand()
E4_normalization=(2*E4*z**2).expand()

""" Schwarzschild-AdS
--------------------------------------------
"""
'''
f_Sch=z**2+1-2*z**3
chi_Sch=0
phi_Sch=0

Sch_subslist=[(f,f_Sch),(chi,chi_Sch),(phi,phi_Sch)]

# 验证它满足以上方程
E1_normalization.subs(Sch_subslist).doit().expand()
E2_normalization.subs(Sch_subslist).doit().expand()
E3_normalization.subs(Sch_subslist).doit().expand()
E4_normalization.subs(Sch_subslist).doit().expand()
'''

""" Asymptotic behavior
--------------------------------------------
"""
'''
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,phi0,phi1,phi2,phi3,phi4,phi5=sp.symbols('f_0,f_1,f_2,f_3,f_4,f_5,\chi_0,\chi_1,\chi_2,\chi_3,\chi_4,\chi_5,\phi_0,\phi_1,\phi_2,\phi_3,\phi_4,\phi_5',real=True)

f_asymp=f0+f1*z+f2*z**2+f3*z**3+sp.O(z**4)
chi_asymp=chi0+chi1*z+chi2*z**2+chi3*z**3+sp.O(z**4)
phi_asymp=phi0+phi1*z+phi2*z**2+phi3*z**3+sp.O(z**4)

asymptotic_behavior_subslist=[(f,f_asymp),(chi,chi_asymp),(phi,phi_asymp)]

equation_asymp_subslist=[] # 由方程给出的渐近行为
boundary_condition_subslist=[] # 在z=0处手动添加的边界条件

equation_asymp_subslist=equation_asymp_subslist+[(chi1,0),(chi2,phi1**2/8),(chi3,phi1*phi2/3),(phi0,0)]
equation_asymp_subslist=equation_asymp_subslist+[(f0,sp.exp(-chi0)),(f1,0),(f2,(phi1**2/8+1)*sp.exp(-chi0))]

# source=sp.symbols('\phi_1',real=True)
# boundary_condition_subslist=boundary_condition_subslist+[(chi0,0),(phi1,source)]

my.series(E1_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0)
my.series(E2_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0)
my.series(E3_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0)
# 下面这个方程没有给出独有的渐近行为
my.series(E4_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0)
'''

""" Equation subtraction
--------------------------------------------
"""
f_tilde,chi_tilde,phi_tilde=sp.symbols(r'\tilde{f},\tilde{\chi},\tilde{\phi}',cls=sp.Function)
f_tilde,chi_tilde,phi_tilde=f_tilde(z),chi_tilde(z),phi_tilde(z)

subtraction_subslist=[(f,1+z**2*f_tilde),(phi,z*phi_tilde)]

E1_subtraction=(E1_normalization).subs(subtraction_subslist).doit().expand()
E2_subtraction=(E2_normalization).subs(subtraction_subslist).doit().expand()
E3_subtraction=(E3_normalization).subs(subtraction_subslist).doit().expand()
E4_subtraction=(E4_normalization).subs(subtraction_subslist).doit().expand()

# 边界上方程的形式
# E1_subtraction.subs(z,0).doit().expand()
# E2_subtraction.subs(z,0).doit().expand()
# E3_subtraction.subs(z,0).doit().expand()
# E4_subtraction.subs(z,0).doit().expand()

""" Variation
--------------------------------------------
"""
equation=sp.Array([E2_subtraction,E1_subtraction,E3_subtraction])
variation=my.variation(equation,(f_tilde,chi,phi_tilde),z,2)

""" Functionalization
--------------------------------------------
"""
arg_parameter=[]
arg_variable=[z]
arg_field=my.symbol_arg_field((f_tilde,chi,phi_tilde),z,2)
arguments=arg_parameter+arg_variable+arg_field

equation_function=sp.lambdify(arguments,equation,'numpy')
variation_function=sp.lambdify(arguments,variation,'numpy')
error_function=sp.lambdify(arguments,E4_subtraction,'numpy')

""" Thermodynamic quantities ($\kappa_4^2=1$)
--------------------------------------------
"""
# BS坐标下，视界半径不再固定是z=1
K_up=sp.Array([1,0,0,0])
K_low=my.dot(metric,K_up)

# $K^\mu \nabla_\mu K_\nu=-\kappa K_\nu$
# left=my.dot(K_up,my.nabla(K_low,coordinates,metric))
# right=K_low
# kappa=(-left[1]/right[1]).expand()

# $\kappa^2=-\frac{1}{2}(\nabla_\mu K_\nu)(\nabla^\mu K^\nu)$
# kappa=sp.sqrt(-sp.tensorcontraction(sp.tensorproduct(my.nabla(K_low,coordinates,metric),my.dot(my.dot(inverse_metric,my.nabla(K_low,coordinates,metric)),inverse_metric)),(0,2),(1,3))/2).expand()

# T=kappa/(2*sp.pi)

# 以下表达式必须在视界处取值
# T=(sp.sqrt((f.diff(z)/2)**2)/(2*sp.pi)).subs(subtraction_subslist).doit().expand()
T=((-f.diff(z)/2)/(2*sp.pi)).expand()
S=(2*sp.pi*L**2/z**2*4*sp.pi).expand()
s=(2*sp.pi*L**2/z**2).expand()

T_function=sp.lambdify(f.diff(z),T,'numpy')
s_function=sp.lambdify(z,s,'numpy')