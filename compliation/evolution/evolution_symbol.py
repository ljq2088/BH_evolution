import sympy as sp
import sys
sys.path.append('..')
import my_module.my_module as my

""" Setup
--------------------------------------------
"""
# $\mathcal{L}=R-\frac{1}{2}\nabla_\mu\phi\nabla^\mu\phi-V(\phi)$

'''1.gravity'''
v,z,theta,varphi=sp.symbols(r'v,z,\theta,\varphi',real=True)
coordinates=sp.Array([v,z,theta,varphi])
f,chi=sp.symbols(r'f,\chi',cls=sp.Function,real=True)
f,chi=f(v,z,theta),chi(v,z,theta)
xi,A=sp.symbols(r'\xi,A',cls=sp.Function,real=True)
xi,A=xi(v,z,theta),A(v,z,theta)
# L=sp.symbols('L',real=True)
L=sp.S(1)
metric=L**2/z**2*sp.Array([[-(f*sp.exp(-chi)-sp.exp(A)*xi**2),-sp.exp(-chi),-xi*sp.exp(A),0],[-sp.exp(-chi),0,0,0],[-xi*sp.exp(A),0,sp.exp(A),0],[0,0,0,sp.exp(-A)*sp.sin(theta)**2]])
inverse_metric=sp.Array(metric.tomatrix().inv())
riemann_R,ricci_R,scalar_R,einstein_G=my.curvature(coordinates,metric)

'''2.matter'''
phi=sp.symbols(r'\phi',cls=sp.Function,real=True)
phi=phi(v,z,theta)
V=sp.symbols(r'V',cls=sp.Function,real=True)
V=V(phi)
# V=-6*sp.cosh(phi/sp.sqrt(3))-phi**4/5
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
E2=(einstein_eq[1,2]).expand() # 结果比平面拓扑多了一项
E3=(einstein_eq[0,1]-f/2*einstein_eq[1,1]+xi*einstein_eq[1,2]).expand() # 结果比平面拓扑多了五项
E4=(einstein_eq[2,2]/sp.exp(A+chi)-einstein_eq[3,3]/(sp.exp(-A+chi)*sp.sin(theta)**2)).expand() # 结果比平面拓扑多了四项
E5=(scalar_eq/sp.exp(chi)).expand() # 结果比平面拓扑多了两项
E6=(einstein_eq[0,0]-f*einstein_eq[0,1]+xi*einstein_eq[0,2]).expand() # 结果比平面拓扑多了若干项
E7=(einstein_eq[2,0]-f*einstein_eq[2,1]+xi*einstein_eq[2,2]).expand() # 结果比平面拓扑多了若干项

'''2.normalization'''
E1_normalization=(-E1*z/2).expand()
E2_normalization=(-2*E2/z**2).expand()
E3_normalization=(-E3/z**2).expand()
E4_normalization=(-E4/2).expand()
E5_normalization=(E5/z**2/2).expand()
E6_normalization=(2*E6).expand()
E7_normalization=(-2*E7).expand()

'''3.subtraction'''
r'''
f_tilde,chi_tilde,xi_tilde,A_tilde,phi_tilde=sp.symbols(r'\tilde{f},\tilde{\chi},\tilde{\xi},\tilde{A},\tilde{\phi}',cls=sp.Function)
f_tilde,chi_tilde,xi_tilde,A_tilde,phi_tilde=f_tilde(v,z,theta),chi_tilde(v,z,theta),xi_tilde(v,z,theta),A_tilde(v,z,theta),phi_tilde(v,z,theta)

subtraction_subslist=[(f,1+z**2*f_tilde)]

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
f_Sch=z**2+1-2*z**3
chi_Sch=0
A_Sch=0
xi_Sch=0
phi_Sch=0

Sch_subslist=[(f,f_Sch),(chi,chi_Sch),(A,A_Sch),(xi,xi_Sch),(phi,phi_Sch)]

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
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,A0,A1,A2,A3,A4,A5,phi0,phi1,phi2,phi3,phi4,phi5=sp.symbols(r'f_0,f_1,f_2,f_3,f_4,f_5,\chi_0,\chi_1,\chi_2,\chi_3,\chi_4,\chi_5,\xi_0,\xi_1,\xi_2,\xi_3,\xi_4,\xi_5,A_0,A_1,A_2,A_3,A_4,A_5,\phi_0,\phi_1,\phi_2,\phi_3,\phi_4,\phi_5',cls=sp.Function,real=True)
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,A0,A1,A2,A3,A4,A5,phi0,phi1,phi2,phi3,phi4,phi5=f0(v,theta),f1(v,theta),f2(v,theta),f3(v,theta),f4(v,theta),f5(v,theta),chi0(v,theta),chi1(v,theta),chi2(v,theta),chi3(v,theta),chi4(v,theta),chi5(v,theta),xi0(v,theta),xi1(v,theta),xi2(v,theta),xi3(v,theta),xi4(v,theta),xi5(v,theta),A0(v,theta),A1(v,theta),A2(v,theta),A3(v,theta),A4(v,theta),A5(v,theta),phi0(v,theta),phi1(v,theta),phi2(v,theta),phi3(v,theta),phi4(v,theta),phi5(v,theta)

f_asymp=f0+f1*z+f2*z**2+f3*z**3+sp.O(z**4)
chi_asymp=chi0+chi1*z+chi2*z**2+chi3*z**3+sp.O(z**4)
xi_asymp=xi0+xi1*z+xi2*z**2+xi3*z**3+sp.O(z**4)
A_asymp=A0+A1*z+A2*z**2+A3*z**3+sp.O(z**4)
phi_asymp=phi0+phi1*z+phi2*z**2+phi3*z**3+sp.O(z**4)

asymptotic_behavior_subslist=[(f,f_asymp),(chi,chi_asymp),(xi,xi_asymp),(A,A_asymp),(phi,phi_asymp)]

equation_asymp_subslist=[] # 由方程给出的渐近行为
boundary_condition_subslist=[] # 在z=0处手动添加的边界条件

equation_asymp_subslist=equation_asymp_subslist+[(chi1,0),(phi0,0),(xi1,sp.exp(-A0-chi0)*chi0.diff(theta)),(f0,sp.exp(-chi0))]
equation_asymp_subslist=equation_asymp_subslist+[(f1,-sp.cot(theta)*xi0-xi0.diff(theta))]
boundary_condition_subslist=boundary_condition_subslist+[(chi0,0),(xi0,0)]

equation_asymp_subslist=equation_asymp_subslist+[(A1,A0.diff(v)),(A2,0)]
boundary_condition_subslist=boundary_condition_subslist+[(A0,0)]

equation_asymp_subslist=equation_asymp_subslist+[(xi2,0),(f2,chi2+1),(chi2,phi1**2/8),(chi3,phi1*phi2/3)]
source=sp.symbols(r'\phi_1',real=True)
boundary_condition_subslist=boundary_condition_subslist+[(phi1,source)]

# sp.series(E1_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
# sp.series(E2_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
# sp.series(E3_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,4)
# sp.series(E4_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
# sp.series(E5_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,4)
# sp.series(E6_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,4)
# sp.series(E7_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,4)

f3_eq=sp.series(E6_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,4).coeff(z**3)
xi3_eq=sp.series(E7_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,4).coeff(z**3)

pv_xi3=(xi3.diff(v)-xi3_eq/xi3_eq.coeff(xi3.diff(v))).expand()
pv_f3=(f3.diff(v)-f3_eq/f3_eq.coeff(f3.diff(v))).expand()
# '''

""" Asymptotic behavior in theta direction
--------------------------------------------
"""
# r'''
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,A0,A1,A2,A3,A4,A5,phi0,phi1,phi2,phi3,phi4,phi5=sp.symbols(r'f_0,f_1,f_2,f_3,f_4,f_5,\chi_0,\chi_1,\chi_2,\chi_3,\chi_4,\chi_5,\xi_0,\xi_1,\xi_2,\xi_3,\xi_4,\xi_5,A_0,A_1,A_2,A_3,A_4,A_5,\phi_0,\phi_1,\phi_2,\phi_3,\phi_4,\phi_5',cls=sp.Function,real=True)
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,A0,A1,A2,A3,A4,A5,phi0,phi1,phi2,phi3,phi4,phi5=f0(v,z),f1(v,z),f2(v,z),f3(v,z),f4(v,z),f5(v,z),chi0(v,z),chi1(v,z),chi2(v,z),chi3(v,z),chi4(v,z),chi5(v,z),xi0(v,z),xi1(v,z),xi2(v,z),xi3(v,z),xi4(v,z),xi5(v,z),A0(v,z),A1(v,z),A2(v,z),A3(v,z),A4(v,z),A5(v,z),phi0(v,z),phi1(v,z),phi2(v,z),phi3(v,z),phi4(v,z),phi5(v,z)

f_asymp=f0+f1*theta+f2*theta**2+f3*theta**3+sp.O(theta**4)
chi_asymp=chi0+chi1*theta+chi2*theta**2+chi3*theta**3+sp.O(theta**4)
xi_asymp=xi0+xi1*theta+xi2*theta**2+xi3*theta**3+sp.O(theta**4)
A_asymp=A0+A1*theta+A2*theta**2+A3*theta**3+sp.O(theta**4)
phi_asymp=phi0+phi1*theta+phi2*theta**2+phi3*theta**3+sp.O(theta**4)

asymptotic_behavior_subslist=[(f,f_asymp),(chi,chi_asymp),(xi,xi_asymp),(A,A_asymp),(phi,phi_asymp)]

equation_asymp_subslist=[] # 由方程给出的渐近行为
boundary_condition_subslist=[] # 在z=0处手动添加的边界条件

equation_asymp_subslist=equation_asymp_subslist+[(A0.diff(z),0)]
equation_asymp_subslist=equation_asymp_subslist+[(A0.diff(v),0)]
boundary_condition_subslist=boundary_condition_subslist+[]

ab1=sp.series(E1_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),theta,0,0).coeff(1/theta)
ab2=sp.series(E2_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),theta,0,0).coeff(1/theta)
ab3=sp.series(E3_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),theta,0,0).coeff(1/theta)
ab4=sp.series(E4_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),theta,0,0).coeff(1/theta)
ab5=sp.series(E5_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),theta,0,0).coeff(1/theta)
ab6=sp.series(E6_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),theta,0,0).coeff(1/theta)
ab7=sp.series(E7_normalization.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),theta,0,0).coeff(1/theta)

ab3_normalization=(2*ab3*z**2).expand()
ab4_normalization=(2*ab4).expand()
ab5_normalization=(2*ab5).expand()
ab6_normalization=(ab6).expand()
ab7_normalization=(ab7).expand()

ab3_elimination=ab3_normalization+ab4_normalization
ab4_elimination=ab4_normalization
ab5_elimination=ab5_normalization
ab6_elimination=(ab6_normalization-ab3_elimination*sp.exp(A0+chi0)*xi0**2/3+ab4_elimination*sp.exp(A0+chi0)*xi0**2).expand()
ab7_elimination=(ab7_normalization-ab3_elimination*sp.exp(A0+chi0)*xi0-ab4_elimination*sp.exp(A0+chi0)*xi0).expand()
# '''