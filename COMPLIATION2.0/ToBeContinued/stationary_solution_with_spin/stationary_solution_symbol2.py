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
chi,xi,eta,f,A,B=sp.symbols(r'\chi,\xi,\eta,f,A,B',cls=sp.Function,real=True)
chi,xi,eta,f,A,B=chi(z,theta),xi(z,theta),eta(z,theta),f(z,theta),A(z,theta),B(z,theta)
xi_u=sp.Array([xi,eta])
h_ll=sp.Array([[sp.exp(A)*cosh(B),sinh(B)*sp.sin(theta)],[sinh(B)*sp.sin(theta),sp.exp(-A)*cosh(B)*sp.sin(theta)**2]])
# L=sp.symbols('L',real=True)
L=sp.S(1)
metric=L**2/z**2*sp.Array([[-(f*sp.exp(-chi)-my.dot(my.dot(xi_u,h_ll),xi_u)),-sp.exp(-chi),-my.dot(xi_u,h_ll)[0],-my.dot(xi_u,h_ll)[1]],[-sp.exp(-chi),0,0,0],[-my.dot(xi_u,h_ll)[0],0,h_ll[0,0],h_ll[0,1]],[-my.dot(xi_u,h_ll)[1],0,h_ll[1,0],h_ll[1,1]]])
inverse_metric=sp.Array(metric.tomatrix().inv())
riemann_R,ricci_R,scalar_R,einstein_G=my.curvature(coordinates,metric)

'''2.matter'''
phi=sp.symbols('\phi',cls=sp.Function,real=True)
phi=phi(z,theta)
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
'''
E1=(einstein_eq[1,1]).expand().factor().expand()
E2=(einstein_eq[1,2]).expand().factor().expand()
E3=(einstein_eq[1,3]).expand().factor().expand()
E4=(einstein_eq[1,0]-f/2*einstein_eq[1,1]+xi*einstein_eq[1,2]+eta*einstein_eq[1,3]).expand().factor().expand()
E5=(einstein_eq[2,2]*sp.exp(-A)-einstein_eq[3,3]*sp.exp(A)/sp.sin(theta)**2).expand().factor().expand()
E6=((einstein_eq[2,2]*sp.exp(-A)+einstein_eq[3,3]*sp.exp(A)/sp.sin(theta)**2)*sinh(B)-2*einstein_eq[2,3]*cosh(B)/sp.sin(theta)).expand().factor().expand()
E7=(scalar_eq/sp.exp(chi)).expand().factor().expand()
E8=(einstein_eq[2,0]-f*einstein_eq[2,1]+xi*einstein_eq[2,2]+eta*einstein_eq[2,3]).expand().factor().expand()
E9=(einstein_eq[3,0]-f*einstein_eq[3,1]+xi*einstein_eq[3,2]+eta*einstein_eq[3,3]).expand().factor().expand()
E10=(einstein_eq[0,0]-f*einstein_eq[0,1]+xi*einstein_eq[0,2]+eta*einstein_eq[0,3]).expand().factor().expand()

with open('equations_combination.txt', 'w') as file:
    file.write(str([E1,E2,E3,E4,E5,E6,E7,E8,E9,E10]).replace('\\','').replace('(z, theta)','').replace('exp','sp.exp').replace('Derivative','sp.Derivative').replace('sqrt','sp.sqrt').replace('sin','sp.sin').replace('cos','sp.cos'))
'''

with open('equations_combination.txt', 'r') as file:
    equations_combination_str = file.read()
    E1,E2,E3,E4,E5,E6,E7,E8,E9,E10 = eval(equations_combination_str)

'''planar topology'''
h_uu=sp.Array(h_ll.tomatrix().inv())
Theta_ll=sp.exp(chi)*h_ll
Theta_uu=sp.Array(Theta_ll.tomatrix().inv())
p_l=my.dot(Theta_ll,xi_u.diff(z))/z**2
Pi_A=(A.diff(v)/z-f*A.diff(z)/2/z+xi*A.diff(theta)/z)*cosh(B)
Pi_B=B.diff(v)/z-f*B.diff(z)/2/z+xi*B.diff(theta)/z

E1_planar=chi.diff(z)-z/4*(A.diff(z)**2*cosh(B)**2+B.diff(z)**2+phi.diff(z)**2)
E2_planar=p_l[0].diff(z)+1/z**2*((A.diff(z)*cosh(B)**2+chi.diff(z)).diff(theta)+2*chi.diff(theta)/z-(A.diff(z)*A.diff(theta)*cosh(B)**2+B.diff(z)*B.diff(theta)+phi.diff(z)*phi.diff(theta)))
E3_planar=p_l[1].diff(z)+1/z**2*(sp.exp(-A)*(B.diff(z)+A.diff(z)*sinh(B)*cosh(B))).diff(theta)
E4_planar=-(f/z**3).diff(z)+xi.diff(theta)/z**3-(xi.diff(theta)/2/z**2).diff(z)+z**2*(sp.exp(chi)/z**4*(sp.exp(A)*xi.diff(z)**2*cosh(B)+2*xi.diff(z)*eta.diff(z)*sinh(B)+sp.exp(-A)*eta.diff(z)**2*cosh(B)))/4+sp.exp(-chi)/2/z**2*((sp.exp(-A)*cosh(B)).diff(theta)-sp.exp(-A)*cosh(B)*chi.diff(theta)).diff(theta)+sp.exp(-A-chi)/4/z**2*cosh(B)*(chi.diff(theta)**2+phi.diff(theta)**2)+L**2/2/z**4*sp.exp(-chi)*V
temp=sp.exp(-A-chi)/4/z*(chi.diff(theta)**2+phi.diff(theta)**2-2*chi.diff(theta,theta))
E5_planar=-Pi_A.diff(z)-A.diff(z)*sinh(B)*Pi_B-f*A.diff(z)*cosh(B)/2/z**2+z**3*sp.exp(-chi)*(p_l[0]**2*sp.exp(-A)-p_l[1]**2*sp.exp(A))/4+(xi.diff(z)*A.diff(theta)+xi.diff(theta)*A.diff(z))*cosh(B)/2/z-z*sp.exp(-A)*my.dot((xi_u.diff(theta)/2/z**2).diff(z),h_ll[:,0])-sp.exp(-A)/z*my.dot(h_ll[0,:].diff(z),xi_u.diff(theta))+temp
E6_planar=-Pi_B.diff(z)+A.diff(z)*sinh(B)*Pi_A-f*B.diff(z)/2/z**2+z**3*sp.exp(-chi)*(2*p_l[0]*p_l[1]*cosh(B)-sinh(B)*(p_l[0]**2*sp.exp(-A)+p_l[1]**2*sp.exp(A)))/4+(xi.diff(z)*B.diff(theta)+xi.diff(theta)*B.diff(z))/2/z-z*sp.exp(-A)*(eta.diff(theta)/2/z**2).diff(z)-sp.exp(-A)/z*my.dot(my.dot(h_uu[1,:],h_ll.diff(z)),xi_u.diff(theta))-temp*sinh(B)
E7_planar=-phi.diff(v,z)+phi.diff(v)/z+z**2/2*((f*phi.diff(z)-xi*phi.diff(theta))/z**2).diff(z)+(sp.exp(-A-chi)*cosh(B)*phi.diff(theta)-xi*phi.diff(z)).diff(theta)/2-L**2/2/z**2*sp.exp(-chi)*V.diff(phi)

E1_planar=E1_planar.expand()
E2_planar=E2_planar.expand()
E3_planar=E3_planar.expand()
E4_planar=E4_planar.expand()
E5_planar=E5_planar.expand()
E6_planar=E6_planar.expand()
E7_planar=E7_planar.expand()

'''2.simplification'''
E1=(E1/E1.coeff(chi.diff(z))*E1_planar.coeff(chi.diff(z))).expand().factor().expand()
E2=(E2/E2.coeff(xi.diff(z,z))*E2_planar.coeff(xi.diff(z,z))).expand().factor().expand()
E3=(E3/E3.coeff(eta.diff(z,z))*E3_planar.coeff(eta.diff(z,z))).expand().factor().expand()
E4=(E4/E4.coeff(f.diff(z))*E4_planar.coeff(f.diff(z))).expand().factor().expand()
E5=(E5/E5.coeff(A.diff(z,z))*E5_planar.coeff(A.diff(z,z))).expand().factor().expand()
E6=(E6/E6.coeff(B.diff(z,z))*E6_planar.coeff(B.diff(z,z))).expand().factor().expand()
E7=(E7/E7.coeff(phi.diff(z,z))*E7_planar.coeff(phi.diff(z,z))).expand().factor().expand()

# 写成cos(theta)/sin(theta)的形式
E1_simplification=exp_to_hyper(E1,B)
E2_simplification=exp_to_hyper(E2,B)
E3_simplification=exp_to_hyper(E3/sp.sin(theta),B)
E4_simplification=exp_to_hyper(E4,B)
E5_simplification=exp_to_hyper(E5,B)
E6_simplification=exp_to_hyper(E6,B)
E7_simplification=exp_to_hyper(E7,B)
E8_simplification=exp_to_hyper(E8,B)
E9_simplification=exp_to_hyper(E9/sp.sin(theta),B)
E10_simplification=exp_to_hyper(E10,B)

# 消掉所有分母中的sin(theta)
# E1_simplification=exp_to_hyper(E1,B)
# E2_simplification=exp_to_hyper(E2*sp.sin(theta),B)
# E3_simplification=exp_to_hyper(E3,B)
# E4_simplification=exp_to_hyper(E4*sp.sin(theta),B)
# E5_simplification=exp_to_hyper(E5*sp.sin(theta),B)
# E6_simplification=exp_to_hyper(E6*sp.sin(theta),B)
# E7_simplification=exp_to_hyper(E7*sp.sin(theta),B)
# E8_simplification=exp_to_hyper(E8*sp.sin(theta),B)
# E9_simplification=exp_to_hyper(E9,B)
# E10_simplification=exp_to_hyper(E10*sp.sin(theta),B)

""" Equation subtraction
--------------------------------------------
"""
'''1.subtraction'''
chi_tilde,xi_tilde,eta_tilde,f_tilde,A_tilde,B_tilde,phi_tilde=sp.symbols(r'\tilde{\chi},\tilde{\xi},\tilde{\eta},\tilde{f},\tilde{A},\tilde{B},\tilde{\phi}',cls=sp.Function)
chi_tilde,xi_tilde,eta_tilde,f_tilde,A_tilde,B_tilde,phi_tilde=chi_tilde(z,theta),xi_tilde(z,theta),eta_tilde(z,theta),f_tilde(z,theta),A_tilde(z,theta),B_tilde(z,theta),phi_tilde(z,theta)

subtraction_subslist=[(xi,z**3*xi_tilde/sp.sin(theta)),(eta,z**3*eta_tilde),(f,1+z**2*f_tilde),(A,z**3*A_tilde),(B,z**3*B_tilde),(phi,z*phi_tilde)]

E1_subtraction=(E1_simplification).subs(subtraction_subslist).doit().expand()
E2_subtraction=(E2_simplification).subs(subtraction_subslist).doit().expand()
E3_subtraction=(E3_simplification).subs(subtraction_subslist).doit().expand()
E4_subtraction=(E4_simplification).subs(subtraction_subslist).doit().expand()
E5_subtraction=(E5_simplification).subs(subtraction_subslist).doit().expand()
E6_subtraction=(E6_simplification).subs(subtraction_subslist).doit().expand()
E7_subtraction=(E7_simplification).subs(subtraction_subslist).doit().expand()
E8_subtraction=(E8_simplification).subs(subtraction_subslist).doit().expand()
E9_subtraction=(E9_simplification).subs(subtraction_subslist).doit().expand()
E10_subtraction=(E10_simplification).subs(subtraction_subslist).doit().expand()

'''2.normalization'''
E1_normalization=(E1_subtraction).expand()
E2_normalization=(E2_subtraction*z**3).expand()
E3_normalization=(E3_subtraction).expand()
E4_normalization=(E4_subtraction*z**4).expand()
E5_normalization=(E5_subtraction*z).expand()
E6_normalization=(E6_subtraction*z).expand()
E7_normalization=(E7_subtraction*z**2).expand()
E8_normalization=(E8_subtraction*z).expand()
E9_normalization=(E9_subtraction*z**-2).expand()
E10_normalization=(E10_subtraction*z**-2).expand()

# 方程在边界上的形式
# E1_normalization.subs(z,0).doit().expand()
# E2_normalization.subs(z,0).doit().expand()
# E3_normalization.subs(z,0).doit().expand()
# E4_normalization.subs(z,0).doit().expand()
# E5_normalization.subs(z,0).doit().expand()
# E6_normalization.subs(z,0).doit().expand()
# E7_normalization.subs(z,0).doit().expand()
# E8_normalization.subs(z,0).doit().expand()
# E9_normalization.subs(z,0).doit().expand()
# E10_normalization.subs(z,0).doit().expand()

""" Asymptotic behavior
--------------------------------------------
"""
r'''
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,eta0,eta1,eta2,eta3,eta4,eta5,A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,phi0,phi1,phi2,phi3,phi4,phi5=sp.symbols(r'f_0,f_1,f_2,f_3,f_4,f_5,\chi_0,\chi_1,\chi_2,\chi_3,\chi_4,\chi_5,\xi_0,\xi_1,\xi_2,\xi_3,\xi_4,\xi_5,\eta_0,\eta_1,\eta_2,\eta_3,\eta_4,\eta_5,A_0,A_1,A_2,A_3,A_4,A_5,B_0,B_1,B_2,B_3,B_4,B_5,\phi_0,\phi_1,\phi_2,\phi_3,\phi_4,\phi_5',cls=sp.Function,real=True)
f0,f1,f2,f3,f4,f5,chi0,chi1,chi2,chi3,chi4,chi5,xi0,xi1,xi2,xi3,xi4,xi5,eta0,eta1,eta2,eta3,eta4,eta5,A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,phi0,phi1,phi2,phi3,phi4,phi5=f0(theta),f1(theta),f2(theta),f3(theta),f4(theta),f5(theta),chi0(theta),chi1(theta),chi2(theta),chi3(theta),chi4(theta),chi5(theta),xi0(theta),xi1(theta),xi2(theta),xi3(theta),xi4(theta),xi5(theta),eta0(theta),eta1(theta),eta2(theta),eta3(theta),eta4(theta),eta5(theta),A0(theta),A1(theta),A2(theta),A3(theta),A4(theta),A5(theta),B0(theta),B1(theta),B2(theta),B3(theta),B4(theta),B5(theta),phi0(theta),phi1(theta),phi2(theta),phi3(theta),phi4(theta),phi5(theta)

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
equation_asymp_subslist=equation_asymp_subslist+[(chi1,0),(xi1,0),(eta1,0),(phi0,0),(f0,1),(A1,0),(B1,0),(chi2,phi1**2/8),(xi2,0),(eta2,0),(f1,0),(A2,0),(B2,0),(f2,phi1**2/8+1),(chi3,phi1*phi2/3)]

source=sp.symbols('\phi_1',real=True)
boundary_condition_subslist=boundary_condition_subslist+[(phi1,source)]

sp.series(E1_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
sp.series(E2_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,0)
sp.series(E3_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,0)
sp.series(E4_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,0)
sp.series(E5_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)
sp.series(E6_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)
sp.series(E7_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,1)

sp.series(E8_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
sp.series(E9_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)
sp.series(E10_simplification.subs(asymptotic_behavior_subslist,simultaneous=True).doit().subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3)

xi3_eq=sp.series(E8.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3).coeff(z**2)
eta3_eq=sp.series(E9.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3).coeff(z**2)
f3_eq=sp.series(E10.subs(asymptotic_behavior_subslist,simultaneous=True).subs(equation_asymp_subslist+boundary_condition_subslist).doit(),z,0,3).coeff(z**2)
'''

""" Variation
--------------------------------------------
"""
equations=sp.Array([E1_normalization,E2_normalization,E3_normalization,E4_normalization,E5_normalization,E6_normalization,E7_normalization])
fields=[chi,xi_tilde,eta_tilde,f_tilde,A_tilde,B_tilde,phi_tilde]
variation=my.variation(equations,fields,(z,theta),2)

""" Functionalization
--------------------------------------------
"""
arg_parameter=[]
arg_variable=[z,theta]
arg_field=my.symbol_arg_field(fields,(z,theta),2)
arguments=arg_parameter+arg_variable+arg_field

equations_function=sp.lambdify(arguments,equations,'numpy')
variation_function=sp.lambdify(arguments,variation,'numpy')
error_function=sp.lambdify(arguments,E9_normalization,'numpy')

""" Boundary condition
--------------------------------------------
"""
# \partial_v\xi_3=0
boundary_condition_f=(f_tilde.diff(z)/3-A_tilde).diff(theta)-2*phi_tilde*phi_tilde.diff(z,theta)/9-2*sp.cos(theta)/sp.sin(theta)*A_tilde
boundary_condition_f_variation=my.variation(boundary_condition_f,fields,(z,theta),2)
boundary_condition_f_function=sp.lambdify(arguments,boundary_condition_f,'numpy')
boundary_condition_f_variation_function=sp.lambdify(arguments,boundary_condition_f_variation,'numpy')

# \partial_v\eta_3=0
boundary_condition_B=-B_tilde.diff(theta)-2*sp.cos(theta)/sp.sin(theta)*B_tilde
boundary_condition_B_variation=my.variation(boundary_condition_B,fields,(z,theta),2)
boundary_condition_B_function=sp.lambdify(arguments,boundary_condition_B,'numpy')
boundary_condition_B_variation_function=sp.lambdify(arguments,boundary_condition_B_variation,'numpy')

""" Thermodynamic quantities ($\kappa_4^2=1$)
--------------------------------------------
"""
'''
# BS坐标下，视界半径不再固定是z=1
Omega=sp.symbols('\Omega',real=True)
K_u=sp.Array([1,0,0,Omega])
K_l=my.dot(metric,K_u)

# $K^\mu \nabla_\mu K_\nu=-\kappa K_\nu$
left=my.dot(K_u,my.nabla(K_l,coordinates,metric))
right=K_l

Omega_solution=sp.solve(exp_to_hyper(right[3],B),Omega)[0].expand()
# inner_product=exp_to_hyper(my.dot(K_u,K_l),B)
# inner_product.subs(Omega,Omega_solution).expand()
left_1=exp_to_hyper(left[1].expand().factor().expand().subs(Omega,Omega_solution).expand(),B).trigsimp().expand()
kappa=(-left_1/right[1]).expand()
T=(kappa/(2*sp.pi)).expand()

# 以下表达式必须在视界处取值
T_function=sp.lambdify(arg_parameter+arg_variable+[chi,chi.diff(z),xi,xi.diff(z),f,f.diff(z),A,A.diff(z),B,B.diff(z)],T,'numpy')
'''