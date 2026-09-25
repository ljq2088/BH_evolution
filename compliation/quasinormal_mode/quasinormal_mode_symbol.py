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
# V=sp.symbols('V',cls=sp.Function,real=True)
# V=V(phi)
# m=0
# m=sp.sqrt(2)*sp.I
# V=-6/L**2+m**2*phi**2/2
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
E2=(einstein_eq[1,2]).expand() # 结果比平面拓扑多了一项
E3=(einstein_eq[0,1]-f/2*einstein_eq[1,1]+xi*einstein_eq[1,2]).expand() # 结果比平面拓扑多了五项
E4=(einstein_eq[2,2]/sp.exp(A+chi)-einstein_eq[3,3]/(sp.exp(-A+chi)*sp.sin(theta)**2)).expand() # 结果比平面拓扑多了四项
E5=(scalar_eq/sp.exp(chi)).expand() # 结果比平面拓扑多了两项
E6=(einstein_eq[0,0]-f*einstein_eq[0,1]+xi*einstein_eq[0,2]).expand() # 结果比平面拓扑多了若干项
E7=(einstein_eq[2,0]-f*einstein_eq[2,1]+xi*einstein_eq[2,2]).expand() # 结果比平面拓扑多了若干项

'''2.normalization'''
E1_normalization=(E1*z).expand()
E2_normalization=(E2*z).expand()
# E3_normalize=(E3*z**2*sp.exp(A+chi)).expand() # 这样的话变分之后会影响\delta A组合成\delta a
E3_normalization=(E3*z**2).expand()
E4_normalization=(E4*z).expand()
E5_normalization=(E5).expand()
E6_normalization=(E6*z).expand()
E7_normalization=(E7*z).expand()

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

""" Perturbation equation
--------------------------------------------
"""
'''1.variation'''
equation0=sp.Array([E1_normalization,E2_normalization,E3_normalization,E4_normalization,E5_normalization,E6_normalization,E7_normalization])
variation0=my.variation(equation0,(f,chi,xi,A,phi),(v,z,theta),2)

delta_f,delta_chi,delta_xi,delta_A,delta_phi=sp.symbols(r'\delta{f},\delta\chi,\delta\xi,\delta{A},\delta\phi',cls=sp.Function)
delta_f,delta_chi,delta_xi,delta_A,delta_phi=delta_f(v,z,theta),delta_chi(v,z,theta),delta_xi(v,z,theta),delta_A(v,z,theta),delta_phi(v,z,theta)

E_perturbation=sp.tensorcontraction(sp.tensorproduct(variation0,my.partial_field((delta_f,delta_chi,delta_xi,delta_A,delta_phi),(v,z,theta),2)),(1,3),(2,4))

'''2.spherical static background'''
spherical_static_subslist=[(f.diff(theta),0),(chi.diff(theta),0),(xi,0),(A,0),(phi.diff(theta),0),(f.diff(v),0),(chi.diff(v),0),(phi.diff(v),0)]

Ep_ssbg=E_perturbation.subs(spherical_static_subslist).doit()
equation_ssbg=equation0.subs(spherical_static_subslist).doit()
equation_bg=sp.Array([equation_ssbg[0],equation_ssbg[2],equation_ssbg[4]])

'''3.laplace'''
delta_Xi,delta_a=sp.symbols(r'\delta\Xi,\delta{a}',cls=sp.Function)
delta_Xi,delta_a=delta_Xi(v,z,theta),delta_a(v,z,theta)

laplace2_delta_f,laplace2_delta_chi,laplace2_delta_Xi,laplace2_delta_a,laplace2_delta_phi=sp.symbols('\Delta_{2}\delta{f},\Delta_{2}\delta\chi,\Delta_{2}\delta\Xi,\Delta_{2}\delta{a},\Delta_{2}\delta\phi',cls=sp.Function)
laplace2_delta_f,laplace2_delta_chi,laplace2_delta_Xi,laplace2_delta_a,laplace2_delta_phi=laplace2_delta_f(v,z,theta),laplace2_delta_chi(v,z,theta),laplace2_delta_Xi(v,z,theta),laplace2_delta_a(v,z,theta),laplace2_delta_phi(v,z,theta)

laplace22_delta_chi=sp.symbols(r'\Delta_{2}^{2}\delta\chi',cls=sp.Function)
laplace22_delta_chi=laplace22_delta_chi(v,z,theta)

p_theta2_subslist=[(delta_f.diff(theta,theta),laplace2_delta_f-sp.cos(theta)/sp.sin(theta)*delta_f.diff(theta)),(delta_chi.diff(theta,theta),laplace2_delta_chi-sp.cos(theta)/sp.sin(theta)*delta_chi.diff(theta)),(delta_xi.diff(theta),delta_Xi-sp.cos(theta)/sp.sin(theta)*delta_xi),(delta_A.diff(theta,theta),delta_a-3*sp.cos(theta)/sp.sin(theta)*delta_A.diff(theta)+2*delta_A),(delta_phi.diff(theta,theta),laplace2_delta_phi-sp.cos(theta)/sp.sin(theta)*delta_phi.diff(theta))]

Ep1=Ep_ssbg[0].expand()
Ep2=((Ep_ssbg[1]*sp.sin(theta)).expand().diff(theta)/sp.sin(theta)).expand().subs(spherical_static_subslist+p_theta2_subslist).doit().expand()
# 如果E3_normalize=(E3*z**2*sp.exp(A+chi)).expand()，则需要：
# Ep3=(Ep_ssbg[2]-E3_normalization*(delta_A+delta_chi)).expand().subs(spherical_static_subslist+p_theta2_subslist).doit().expand()
Ep3=(Ep_ssbg[2]).expand().subs(spherical_static_subslist+p_theta2_subslist).doit().expand()
Ep4=(((Ep_ssbg[3].subs(spherical_static_subslist+p_theta2_subslist).doit().expand()*sp.sin(theta)**2).expand().diff(theta)/sp.sin(theta)).expand().expand().subs(spherical_static_subslist+p_theta2_subslist).doit().expand().diff(theta)/sp.sin(theta)).expand().subs(spherical_static_subslist+p_theta2_subslist+[(laplace2_delta_chi.diff(theta,theta),laplace22_delta_chi-sp.cos(theta)/sp.sin(theta)*laplace2_delta_chi.diff(theta)),(delta_Xi.diff(theta,theta),laplace2_delta_Xi-sp.cos(theta)/sp.sin(theta)*delta_Xi.diff(theta))]).doit().expand()
Ep5=Ep_ssbg[4].expand().subs(spherical_static_subslist+p_theta2_subslist).doit().expand()
Ep6=Ep_ssbg[5].expand().subs(spherical_static_subslist+p_theta2_subslist).doit().expand()
Ep7=((Ep_ssbg[6]*sp.sin(theta)).expand().diff(theta)/sp.sin(theta)).expand().subs(spherical_static_subslist+p_theta2_subslist).doit().expand()

'''
Ep1_simplification=delta_chi.diff(z)-z/2*phi.diff(z)*delta_phi.diff(z)
Ep2_simplification=(sp.exp(chi)*delta_Xi.diff(z)/z**2).diff(z)+1/z**2*(delta_a.diff(z)+laplace2_delta_chi.diff(z)+2*laplace2_delta_chi/z-phi.diff(z)*laplace2_delta_phi)
Ep3_simplification=(delta_f/z**3).diff(z)-(delta_Xi/z**3+L**2/2/z**4*sp.exp(-chi)*(V.diff(phi)*delta_phi-V*delta_chi)-((delta_Xi/2/z**2).diff(z)+sp.exp(-chi)*delta_a/2/z**2+sp.exp(-chi)/2/z**2*(laplace2_delta_chi-2*delta_chi)))
Ep4_simplification=delta_a.diff(v,z)-delta_a.diff(v)/z-(z**2/2*(f*delta_a.diff(z)/z**2).diff(z)-z**2/2*(1/z**2*(laplace2_delta_Xi+2*delta_Xi)).diff(z)-sp.exp(-chi)/2*(laplace22_delta_chi+2*laplace2_delta_chi))
Ep5_simplification=delta_phi.diff(v,z)-delta_phi.diff(v)/z-(z**2/2*((phi.diff(z)*delta_f+f*delta_phi.diff(z))/z**2).diff(z)-phi.diff(z)*delta_Xi/2+sp.exp(-chi)*laplace2_delta_phi/2-L**2/2/z**2*sp.exp(-chi)*(V.diff(phi,phi)*delta_phi-V.diff(phi)*delta_chi))

# laplace2_f=f.diff(theta,theta)+sp.cos(theta)/sp.sin(theta)*f.diff(theta)
# laplace2_chi=chi.diff(theta,theta)+sp.cos(theta)/sp.sin(theta)*chi.diff(theta)
laplace2_f=sp.S(0)
laplace2_chi=sp.S(0)
Ep6_simplification=sp.exp(chi)*delta_Xi.diff(v,z)-(-2/z*(laplace2_delta_f+laplace2_f*delta_chi+f*laplace2_delta_chi)+(laplace2_delta_f.diff(z)-laplace2_chi.diff(z)*delta_f-chi.diff(z)*laplace2_delta_f)-f*(delta_a.diff(z)+laplace2_delta_chi.diff(z))+delta_a.diff(v)+laplace2_delta_chi.diff(v)+f*phi.diff(z)*laplace2_delta_phi+2*delta_Xi)
Ep7_simplification=2/z*(delta_f.diff(v)+f*delta_chi.diff(v))-(sp.exp(-chi)*laplace2_delta_f-(f*delta_Xi).diff(z)+f*chi.diff(z)*delta_Xi-2*(delta_Xi.diff(v)-f*delta_Xi.diff(z))+f*phi.diff(z)*delta_phi.diff(v))

(Ep1-Ep1_simplification*2).expand()
(Ep2-Ep2_simplification*z**3/2).expand()
(Ep3-Ep3_simplification*z**4).expand()
(Ep4-Ep4_simplification*z*2).expand()
(Ep5+Ep5_simplification*z**2*2).expand()
(Ep7+Ep6_simplification*z/2).expand()
(Ep6+Ep7_simplification*z/2).expand()
'''

'''3.laplace eigenvalue'''
l=sp.symbols('l',real=True)
laplace_subslist=[(laplace2_delta_f,-l*(l+1)*delta_f),(laplace2_delta_chi,-l*(l+1)*delta_chi),(laplace2_delta_Xi,-l*(l+1)*delta_Xi),(laplace2_delta_a,-l*(l+1)*delta_a),(laplace2_delta_phi,-l*(l+1)*delta_phi),(laplace22_delta_chi,(-l*(l+1))**2*delta_chi)]

Ep1_substitution=Ep1.subs(laplace_subslist).doit().expand()
Ep2_substitution=Ep2.subs(laplace_subslist).doit().expand()
Ep3_substitution=Ep3.subs(laplace_subslist).doit().expand()
Ep4_substitution=Ep4.subs(laplace_subslist).doit().expand()
Ep5_substitution=Ep5.subs(laplace_subslist).doit().expand()
Ep6_substitution=Ep6.subs(laplace_subslist).doit().expand()
Ep7_substitution=Ep7.subs(laplace_subslist).doit().expand()

# 边界上方程的形式
# Ep1_substitution.subs(z,0).doit().expand()
# Ep2_substitution.subs(z,0).doit().expand()
# Ep3_substitution.subs(z,0).doit().expand()
# Ep4_substitution.subs(z,0).doit().expand()
# Ep5_substitution.subs(z,0).doit().expand()
# Ep6_substitution.subs(z,0).doit().expand()
# Ep7_substitution.subs(z,0).doit().expand()

'''4. coefficient'''
equation=sp.Array([Ep6_substitution,Ep1_substitution,Ep7_substitution,Ep4_substitution,Ep5_substitution])
delta_field=(delta_f,delta_chi,delta_Xi,delta_a,delta_phi)

# # l=0时
# equation=sp.Array([Ep6_substitution,Ep1_substitution,Ep5_substitution])
# delta_field=(delta_f,delta_chi,delta_phi)

variation=my.variation(equation,delta_field,(v,z),2)

error_variation=my.variation((Ep2_substitution,Ep3_substitution),delta_field,(v,z),2)

""" Functionalization
--------------------------------------------
"""
arg_parameter=[l]
arg_variable=[z]
arg_field=my.symbol_arg_field((f,chi,phi),z,2)
arguments=arg_parameter+arg_variable+arg_field

equation_bg_function=sp.lambdify(arguments,equation_bg,'numpy')
variation_function=sp.lambdify(arguments,variation,'numpy')
error_variation_function=sp.lambdify(arguments,error_variation,'numpy')

# 如果只用标量场方程来求纯标量扰动的QNM
variation_phi=my.variation(Ep5_substitution,delta_phi,(v,z),2)
variation_phi_function=sp.lambdify(arguments,variation_phi,'numpy')
error_variation_phi=my.variation(Ep5_substitution,delta_phi,(v,z),2)
error_variation_phi_function=sp.lambdify(arguments,error_variation_phi,'numpy')

""" Asymptotic behavior
--------------------------------------------
"""
r'''
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,A0,A1,A2,A3,A4,A5,phi0,phi1,phi2,phi3,phi4,phi5=sp.symbols(r'f_0,f_1,f_2,f_3,f_4,f_5,\chi_0,\chi_1,\chi_2,\chi_3,\chi_4,\chi_5,\xi_0,\xi_1,\xi_2,\xi_3,\xi_4,\xi_5,A_0,A_1,A_2,A_3,A_4,A_5,\phi_0,\phi_1,\phi_2,\phi_3,\phi_4,\phi_5',real=True)

f_asymp=f0+f1*z+f2*z**2+sp.O(z**3)
chi_asymp=chi0+chi1*z+chi2*z**2+sp.O(z**3)
xi_asymp=xi0+xi1*z+xi2*z**2+sp.O(z**3)
A_asymp=A0+A1*z+A2*z**2+sp.O(z**3)
phi_asymp=phi0+phi1*z+phi2*z**2+sp.O(z**3)

omega=sp.symbols(r'\omega',real=True)

asymptotic_behavior_subslist=[(delta_f.diff(v),-sp.I*omega*f_asymp),(delta_chi.diff(v),-sp.I*omega*chi_asymp),(delta_Xi.diff(v),-sp.I*omega*xi_asymp),(delta_a.diff(v),-sp.I*omega*A_asymp),(delta_phi.diff(v),-sp.I*omega*phi_asymp),(delta_f,f_asymp),(delta_chi,chi_asymp),(delta_Xi,xi_asymp),(delta_a,A_asymp),(delta_phi,phi_asymp)] # diff(v)必须排在前面

my.series(Ep1_substitution.subs(asymptotic_behavior_subslist,simultaneous=True).doit(),z,0).doit()
my.series(Ep2_substitution.subs(asymptotic_behavior_subslist,simultaneous=True).doit(),z,0).doit()
my.series(Ep3_substitution.subs(asymptotic_behavior_subslist,simultaneous=True).doit(),z,0).doit()
my.series(Ep4_substitution.subs(asymptotic_behavior_subslist,simultaneous=True).doit(),z,0).doit()
my.series(Ep5_substitution.subs(asymptotic_behavior_subslist,simultaneous=True).doit(),z,0).doit()
my.series(Ep6_substitution.subs(asymptotic_behavior_subslist,simultaneous=True).doit(),z,0).doit()
my.series(Ep7_substitution.subs(asymptotic_behavior_subslist,simultaneous=True).doit(),z,0).doit()
'''