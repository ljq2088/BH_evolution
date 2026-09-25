import sympy as sp
import sys
sys.path.append('..')
import my_module.my_module as my

def cosh(x):
    return (sp.exp(x)+sp.exp(-x))/2
def sinh(x):
    return (sp.exp(x)-sp.exp(-x))/2

# def cosh(x):
#     return sp.cosh(x)
# def sinh(x):
#     return sp.sinh(x)

def exp_to_hyper(expression,variable):
    expression=expression.expand()

    coeff_p2=expression.coeff(sp.exp(2*variable))
    coeff_n2=expression.coeff(sp.exp(-2*variable))
    coeff_cosh2=(coeff_p2+coeff_n2)*2
    coeff_sinhcosh=(coeff_p2-coeff_n2)*2

    coeff_p=expression.coeff(sp.exp(variable))
    coeff_n=expression.coeff(sp.exp(-variable))
    coeff_cosh=coeff_p+coeff_n
    coeff_sinh=coeff_p-coeff_n

    return (expression-coeff_cosh2*(sp.exp(2*variable)+2+sp.exp(-2*variable))/4-coeff_sinhcosh*(sp.exp(2*variable)-sp.exp(-2*variable))/4-coeff_p*sp.exp(variable)-coeff_n*sp.exp(-variable)+coeff_cosh2*sp.cosh(variable)**2+coeff_sinhcosh*sp.sinh(variable)*sp.cosh(variable)+coeff_cosh*sp.cosh(variable)+coeff_sinh*sp.sinh(variable)).expand()

""" Setup
--------------------------------------------
"""
# $\mathcal{L}=R-\frac{1}{2}\nabla_\mu\phi\nabla^\mu\phi-V(\phi)$

'''1.gravity'''
v,z,theta,varphi=sp.symbols(r'v,z,\theta,\varphi',real=True)
coordinates=sp.Array([v,z,theta,varphi])
f,chi,xi,eta,A,B=sp.symbols(r'f,\chi,\xi,\eta,A,B',cls=sp.Function,real=True)
f,chi,xi,eta,A,B=f(v,z,theta),chi(v,z,theta),xi(v,z,theta),eta(v,z,theta),A(v,z,theta),B(v,z,theta)
xi_u=sp.Array([xi,eta])
h_ll=sp.Array([[sp.exp(A)*cosh(B),sinh(B)*sp.sin(theta)],[sinh(B)*sp.sin(theta),sp.exp(-A)*cosh(B)*sp.sin(theta)**2]])
# L=sp.symbols('L',real=True)
L=sp.S(1)
metric=L**2/z**2*sp.Array([[-(f*sp.exp(-chi)-my.dot(my.dot(xi_u,h_ll),xi_u)),-sp.exp(-chi),-my.dot(xi_u,h_ll)[0],-my.dot(xi_u,h_ll)[1]],[-sp.exp(-chi),0,0,0],[-my.dot(xi_u,h_ll)[0],0,h_ll[0,0],h_ll[0,1]],[-my.dot(xi_u,h_ll)[1],0,h_ll[1,0],h_ll[1,1]]])
inverse_metric=sp.Array(metric.tomatrix().inv())
riemann_R,ricci_R,scalar_R,einstein_G=my.curvature(coordinates,metric)

'''2.matter'''
phi=sp.symbols('\phi',cls=sp.Function,real=True)
phi=phi(v,z,theta)
V=sp.symbols('V',cls=sp.Function,real=True)
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
E1=(einstein_eq[1,1]).expand().factor().expand()
E2=(einstein_eq[1,2]).expand().factor().expand()
E3=(einstein_eq[1,3]).expand().factor().expand()
E4=(einstein_eq[1,0]-f/2*einstein_eq[1,1]+xi*einstein_eq[1,2]+eta*einstein_eq[1,3]).expand().factor().expand()
E5=(einstein_eq[2,2]*sp.exp(-A)-einstein_eq[3,3]*sp.exp(A)/sp.sin(theta)**2).expand().factor().expand()
E6=((einstein_eq[2,2]*sp.exp(-A)+einstein_eq[3,3]*sp.exp(A)/sp.sin(theta)**2)*sinh(B)-2*einstein_eq[2,3]*cosh(B)/sp.sin(theta)).expand().factor().expand()
E7=(scalar_eq/sp.exp(chi)).expand().factor().expand()
# E8=(einstein_eq[2,0]-f*einstein_eq[2,1]+xi*einstein_eq[2,2]+eta*einstein_eq[2,3]).expand().factor().expand()
# E9=(einstein_eq[3,0]-f*einstein_eq[3,1]+xi*einstein_eq[3,2]+eta*einstein_eq[3,3]).expand().factor().expand()
E10=(einstein_eq[0,0]-f*einstein_eq[0,1]+xi*einstein_eq[0,2]+eta*einstein_eq[0,3]).expand().factor().expand()

"""
'''compare with planar topology'''
h_uu=sp.Array(h_ll.tomatrix().inv())
Theta_ll=sp.exp(chi)*h_ll
Theta_uu=sp.Array(Theta_ll.tomatrix().inv())
p_l=my.dot(Theta_ll,xi_u.diff(z))/z**2
q_tilde=(p_l[1]/sp.sin(theta)).expand()
Pi_A=(A.diff(v)/z-f*A.diff(z)/2/z+xi*A.diff(theta)/z)*cosh(B)
Pi_B=B.diff(v)/z-f*B.diff(z)/2/z+xi*B.diff(theta)/z

E1_planar=chi.diff(z)-z/4*(A.diff(z)**2*cosh(B)**2+B.diff(z)**2+phi.diff(z)**2)
E2_planar=p_l[0].diff(z)+1/z**2*((A.diff(z)*cosh(B)**2+chi.diff(z)).diff(theta)+2*chi.diff(theta)/z-(A.diff(z)*A.diff(theta)*cosh(B)**2+B.diff(z)*B.diff(theta)+phi.diff(z)*phi.diff(theta)))
E3_planar=q_tilde.diff(z)+1/z**2*(sp.exp(-A)*(B.diff(z)+A.diff(z)*sinh(B)*cosh(B))).diff(theta)
E4_planar=-(f/z**3).diff(z)+xi.diff(theta)/z**3-(xi.diff(theta)/2/z**2).diff(z)+(xi.diff(z)*p_l[0]+eta.diff(z)*p_l[1])/4+sp.exp(-chi)/2/z**2*((sp.exp(-A)*cosh(B)).diff(theta)-sp.exp(-A)*cosh(B)*chi.diff(theta)).diff(theta)+sp.exp(-A-chi)/4/z**2*cosh(B)*(chi.diff(theta)**2+phi.diff(theta)**2)+L**2/2/z**4*sp.exp(-chi)*V
temp=sp.exp(-A-chi)/4/z*(chi.diff(theta)**2+phi.diff(theta)**2-2*chi.diff(theta,theta))
E5_planar=-Pi_A.diff(z)-A.diff(z)*sinh(B)*Pi_B-f*A.diff(z)*cosh(B)/2/z**2+z**3*sp.exp(-chi)*(p_l[0]**2*sp.exp(-A)-q_tilde**2*sp.exp(A))/4+(xi.diff(z)*A.diff(theta)+xi.diff(theta)*A.diff(z))*cosh(B)/2/z-z*sp.exp(-A)*my.dot((xi_u.diff(theta)/2/z**2).diff(z),h_ll[:,0])-sp.exp(-A)/z*my.dot(h_ll[0,:].diff(z),xi_u.diff(theta))+temp
E6_planar=-Pi_B.diff(z)+A.diff(z)*sinh(B)*Pi_A-f*B.diff(z)/2/z**2+z**3*sp.exp(-chi)*(2*p_l[0]*q_tilde*cosh(B)-sinh(B)*(p_l[0]**2*sp.exp(-A)+q_tilde**2*sp.exp(A)))/4+(xi.diff(z)*B.diff(theta)+xi.diff(theta)*B.diff(z))/2/z-z*sp.exp(-A)*(eta.diff(theta)/2/z**2).diff(z)*sp.sin(theta)-sp.exp(-A)/z*my.dot(my.dot(h_uu[1,:],h_ll.diff(z)),xi_u.diff(theta))*sp.sin(theta)-temp*sinh(B)
E7_planar=-phi.diff(v,z)+phi.diff(v)/z+z**2/2*((f*phi.diff(z)-xi*phi.diff(theta))/z**2).diff(z)+(sp.exp(-A-chi)*cosh(B)*phi.diff(theta)-xi*phi.diff(z)).diff(theta)/2-L**2/2/z**2*sp.exp(-chi)*V.diff(phi)

def dn(fun):
    return fun.diff(v)-f*fun.diff(z)+xi*fun.diff(theta)+eta*fun.diff(varphi)
partial_l_Theta_ul=sp.permutedims(my.dot(Theta_uu,sp.derive_by_array(Theta_ll,coordinates),(1,1)),(1,0,2))
dn_Theta_ul=partial_l_Theta_ul[0]-f*partial_l_Theta_ul[1]+xi*partial_l_Theta_ul[2]+eta*partial_l_Theta_ul[3]
E10_planar=(-(f*xi).diff(z)+xi*my.dot(my.dot(xi_u.diff(z),Theta_ll),xi_u)-my.dot(xi_u,dn_Theta_ul[0,:])+my.dot(Theta_uu[0,:],sp.Array([f.diff(theta),f.diff(varphi)]))-2*dn(xi)+xi*xi.diff(theta)-my.dot(my.dot(xi_u,Theta_ll),xi_u.diff(theta))*Theta_uu[0,0]).diff(theta)+sp.S(1)/sp.S(2)*sp.tensorcontraction(my.dot(h_uu.diff(v),dn(h_ll)),(0,1))-phi.diff(v)*dn(phi)-my.dot(partial_l_Theta_ul[0,0,:],xi_u.diff(theta))+my.dot(xi_u,my.dot(Theta_ll,xi_u.diff(z)).diff(v))-2/z*(f.diff(v)+f*chi.diff(v))

E1_planar=E1_planar.expand()
E2_planar=E2_planar.expand()
E3_planar=E3_planar.expand()
E4_planar=E4_planar.expand()
E5_planar=E5_planar.expand()
E6_planar=E6_planar.expand()
E7_planar=E7_planar.expand()
E10_planar=E10_planar.expand().factor().expand()

'''2.simplification'''
E1=(E1/E1.coeff(chi.diff(z))*E1_planar.coeff(chi.diff(z))).expand().factor().expand()
E2=(E2/E2.coeff(xi.diff(z,z))*E2_planar.coeff(xi.diff(z,z))).expand().factor().expand()
E3=(E3/E3.coeff(eta.diff(z,z))*E3_planar.coeff(eta.diff(z,z))).expand().factor().expand()
E4=(E4/E4.coeff(f.diff(z))*E4_planar.coeff(f.diff(z))).expand().factor().expand()
E5=(E5/E5.coeff(A.diff(v,z))*E5_planar.coeff(A.diff(v,z))).expand().factor().expand()
E6=(E6/E6.coeff(B.diff(v,z))*E6_planar.coeff(B.diff(v,z))).expand().factor().expand()
E7=(E7/E7.coeff(phi.diff(v,z))*E7_planar.coeff(phi.diff(v,z))).expand().factor().expand()
E10=(E10/E10.coeff(f.diff(v))*E10_planar.coeff(f.diff(v))).expand().factor().expand()

exp_to_hyper(E1-E1_planar,B)
exp_to_hyper(E2-E2_planar,B)
exp_to_hyper(E3-E3_planar,B)
exp_to_hyper(E4-E4_planar,B)
exp_to_hyper(E5-E5_planar,B)
exp_to_hyper(E6-E6_planar,B)
exp_to_hyper(E7-E7_planar,B)
exp_to_hyper(E10-E10_planar,B)
"""

'''2.simplification'''
h_uu=sp.Array(h_ll.tomatrix().inv())
Theta_ll=sp.exp(chi)*h_ll
Theta_uu=sp.Array(Theta_ll.tomatrix().inv())
p_l=my.dot(Theta_ll,xi_u.diff(z))/z**2
q_tilde=(p_l[1]/sp.sin(theta)).expand()
Pi_A=(A.diff(v)/z-f*A.diff(z)/2/z+xi*A.diff(theta)/z)*cosh(B)
Pi_B=B.diff(v)/z-f*B.diff(z)/2/z+xi*B.diff(theta)/z

cot_theta=sp.cos(theta)/sp.sin(theta)

E1_simplification=chi.diff(z)-z/4*(A.diff(z)**2*cosh(B)**2+B.diff(z)**2+phi.diff(z)**2)
E2_simplification=p_l[0].diff(z)+1/z**2*((A.diff(z)*cosh(B)**2+chi.diff(z)).diff(theta)+2*chi.diff(theta)/z-(A.diff(z)*A.diff(theta)*cosh(B)**2+B.diff(z)*B.diff(theta)+phi.diff(z)*phi.diff(theta))+2*cot_theta*A.diff(z)*cosh(B)**2)
E3_simplification=q_tilde.diff(z)+1/z**2*((sp.exp(-A)*(B.diff(z)+A.diff(z)*sinh(B)*cosh(B))).diff(theta)+2*sp.exp(-A)*cot_theta*(B.diff(z)+A.diff(z)*sinh(B)*cosh(B)))
E4_simplification=-(f/z**3).diff(z)+xi.diff(theta)/z**3-(xi.diff(theta)/2/z**2).diff(z)+(xi.diff(z)*p_l[0]+eta.diff(z)*p_l[1])/4+sp.exp(-chi)/2/z**2*((sp.exp(-A)*cosh(B)).diff(theta)-sp.exp(-A)*cosh(B)*chi.diff(theta)).diff(theta)+sp.exp(-A-chi)/4/z**2*cosh(B)*(chi.diff(theta)**2+phi.diff(theta)**2)+L**2/2/z**4*sp.exp(-chi)*V-cot_theta/2/z**2*(xi.diff(z)+sp.exp(-A-chi)*(3*A.diff(theta)*cosh(B)+chi.diff(theta)*cosh(B)-3*B.diff(theta)*sinh(B))-4*xi/z)-sp.exp(-A-chi)*cosh(B)/z**2
temp=sp.exp(-A-chi)/4/z*(chi.diff(theta)**2+phi.diff(theta)**2-2*chi.diff(theta,theta))
E5_simplification=-Pi_A.diff(z)-A.diff(z)*sinh(B)*Pi_B-f*A.diff(z)*cosh(B)/2/z**2+z**3*sp.exp(-chi)*(p_l[0]**2*sp.exp(-A)-q_tilde**2*sp.exp(A))/4+(xi.diff(z)*A.diff(theta)+xi.diff(theta)*A.diff(z))*cosh(B)/2/z-z*sp.exp(-A)*my.dot((xi_u.diff(theta)/2/z**2).diff(z),h_ll[:,0])-sp.exp(-A)/z*my.dot(h_ll[0,:].diff(z),xi_u.diff(theta))+temp+cot_theta/2/z*(cosh(B)*(xi.diff(z)-xi*A.diff(z)-2*xi/z)+sp.exp(-A-chi)*chi.diff(theta)+2*xi*B.diff(z)*sinh(B))
E6_simplification=-Pi_B.diff(z)+A.diff(z)*sinh(B)*Pi_A-f*B.diff(z)/2/z**2+z**3*sp.exp(-chi)*(2*p_l[0]*q_tilde*cosh(B)-sinh(B)*(p_l[0]**2*sp.exp(-A)+q_tilde**2*sp.exp(A)))/4+(xi.diff(z)*B.diff(theta)+xi.diff(theta)*B.diff(z))/2/z-z*sp.exp(-A)*(eta.diff(theta)/2/z**2).diff(z)*sp.sin(theta)-sp.exp(-A)/z*my.dot(my.dot(h_uu[1,:],h_ll.diff(z)),xi_u.diff(theta))*sp.sin(theta)-temp*sinh(B)-cot_theta/2/z*(xi*B.diff(z)+2*xi*A.diff(z)*sinh(B)*cosh(B)+sp.exp(-A-chi)*chi.diff(theta)*sinh(B))
E7_simplification=-phi.diff(v,z)+phi.diff(v)/z+z**2/2*((f*phi.diff(z)-xi*phi.diff(theta))/z**2).diff(z)+(sp.exp(-A-chi)*cosh(B)*phi.diff(theta)-xi*phi.diff(z)).diff(theta)/2-L**2/2/z**2*sp.exp(-chi)*V.diff(phi)+cot_theta/2*(sp.exp(-A-chi)*phi.diff(theta)*cosh(B)-xi*phi.diff(z))

E1_simplification=E1_simplification.expand()
E2_simplification=E2_simplification.expand()
E3_simplification=E3_simplification.expand()
E4_simplification=E4_simplification.expand()
E5_simplification=E5_simplification.expand()
E6_simplification=E6_simplification.expand()
E7_simplification=E7_simplification.expand()

E1=(E1/E1.coeff(chi.diff(z))*E1_simplification.coeff(chi.diff(z))).expand().factor().expand()
E2=(E2/E2.coeff(xi.diff(z,z))*E2_simplification.coeff(xi.diff(z,z))).expand().factor().expand()
E3=(E3/E3.coeff(eta.diff(z,z))*E3_simplification.coeff(eta.diff(z,z))).expand().factor().expand()
E4=(E4/E4.coeff(f.diff(z))*E4_simplification.coeff(f.diff(z))).expand().factor().expand()
E5=(E5/E5.coeff(A.diff(v,z))*E5_simplification.coeff(A.diff(v,z))).expand().factor().expand()
E6=(E6/E6.coeff(B.diff(v,z))*E6_simplification.coeff(B.diff(v,z))).expand().factor().expand()
E7=(E7/E7.coeff(phi.diff(v,z))*E7_simplification.coeff(phi.diff(v,z))).expand().factor().expand()

""" Asymptotic behavior
--------------------------------------------
"""
# r'''
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,eta0,eta1,eta2,eta3,eta4,eta5,A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,phi0,phi1,phi2,phi3,phi4,phi5=sp.symbols(r'f_0,f_1,f_2,f_3,f_4,f_5,\chi_0,\chi_1,\chi_2,\chi_3,\chi_4,\chi_5,\xi_0,\xi_1,\xi_2,\xi_3,\xi_4,\xi_5,\eta_0,\eta_1,\eta_2,\eta_3,\eta_4,\eta_5,A_0,A_1,A_2,A_3,A_4,A_5,B_0,B_1,B_2,B_3,B_4,B_5,\phi_0,\phi_1,\phi_2,\phi_3,\phi_4,\phi_5',cls=sp.Function,real=True)
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,eta0,eta1,eta2,eta3,eta4,eta5,A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,phi0,phi1,phi2,phi3,phi4,phi5=f0(v,theta),f1(v,theta),f2(v,theta),f3(v,theta),f4(v,theta),f5(v,theta),chi0(v,theta),chi1(v,theta),chi2(v,theta),chi3(v,theta),chi4(v,theta),chi5(v,theta),xi0(v,theta),xi1(v,theta),xi2(v,theta),xi3(v,theta),xi4(v,theta),xi5(v,theta),eta0(v,theta),eta1(v,theta),eta2(v,theta),eta3(v,theta),eta4(v,theta),eta5(v,theta),A0(v,theta),A1(v,theta),A2(v,theta),A3(v,theta),A4(v,theta),A5(v,theta),B0(v,theta),B1(v,theta),B2(v,theta),B3(v,theta),B4(v,theta),B5(v,theta),phi0(v,theta),phi1(v,theta),phi2(v,theta),phi3(v,theta),phi4(v,theta),phi5(v,theta)

f_asymp=f0+f1*z+f2*z**2+f3*z**3+sp.O(z**4)
chi_asymp=chi0+chi1*z+chi2*z**2+chi3*z**3+sp.O(z**4)
xi_asymp=xi0+xi1*z+xi2*z**2+xi3*z**3+sp.O(z**4)
eta_asymp=eta0+eta1*z+eta2*z**2+eta3*z**3+sp.O(z**4)
A_asymp=A0+A1*z+A2*z**2+A3*z**3+sp.O(z**4)
B_asymp=B0+B1*z+B2*z**2+B3*z**3+sp.O(z**4)
phi_asymp=phi0+phi1*z+phi2*z**2+phi3*z**3+sp.O(z**4)

asymptotic_behavior_subslist=[(f,f_asymp),(chi,chi_asymp),(xi,xi_asymp),(eta,eta_asymp),(A,A_asymp),(B,B_asymp),(phi,phi_asymp)]

equation_asymp_subslist=[] # 由方程给出的渐近行为
boundary_condition_subslist=[] # 在z=0处手动添加的边界条件

boundary_condition_subslist=boundary_condition_subslist+[(chi0,0),(xi0,0),(eta0,0),(A0,0),(B0,0)]
equation_asymp_subslist=equation_asymp_subslist+[(chi1,0),(xi1,0),(eta1,0),(phi0,0),(f0,1),(f1,0),(A1,0),(B1,0),(chi2,phi1**2/8),(chi3,phi1*phi2/3),(xi2,0),(eta2,0),(f2,phi1**2/8+1),(A2,0),(B2,0)]

source=sp.symbols('\phi_1',real=True)
boundary_condition_subslist=boundary_condition_subslist+[(phi1,source)]

sp.series(E1_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
sp.series(E2_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,0)
sp.series(E3_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,0)
sp.series(E4_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,0)
sp.series(E5_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)
sp.series(E6_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)
sp.series(E7_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)

xi3_eq=sp.series(E8.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3).coeff(z**2)
eta3_eq=sp.series(E9.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3).coeff(z**2)
f3_eq=sp.series(E10.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3).coeff(z**2)

pv_xi3=(xi3.diff(v)-xi3_eq/xi3_eq.coeff(xi3.diff(v))).expand()
pv_eta3=(eta3.diff(v)-eta3_eq/eta3_eq.coeff(eta3.diff(v))).expand()
pv_f3=(f3.diff(v)-f3_eq/f3_eq.coeff(f3.diff(v))).expand()

sp.series(p_l[0].subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)
sp.series(q_tilde.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)
# '''