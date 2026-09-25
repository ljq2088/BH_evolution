import sympy as sp
import sys
sys.path.append('../..')
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
v,z,x,y=sp.symbols(r'v,z,x,y',real=True)
coordinates=sp.Array([v,z,x,y])
V,beta,xi,eta,alpha,Theta,chi=sp.symbols(r'V,\beta,\xi,\eta,\alpha,\Theta,\chi',cls=sp.Function,real=True)
V,beta,xi,eta,alpha,Theta,chi=V(v,z,x),beta(v,z,x),xi(v,z,x),eta(v,z,x),alpha(v,z,x),Theta(v,z,x),chi(v,z,x)
eta=0
Theta=0
U_u=sp.Array([xi,eta])
h_ll=sp.exp(2*chi)*sp.Array([[sp.exp(alpha)*cosh(Theta),sinh(Theta)],[sinh(Theta),sp.exp(-alpha)*cosh(Theta)]])
metric=1/z**2*sp.Array([[-(sp.exp(2*beta)*V*z**3-my.dot(my.dot(U_u,h_ll),U_u)),-sp.exp(2*beta),-my.dot(h_ll,U_u)[0],-my.dot(h_ll,U_u)[1]],[-sp.exp(2*beta),0,0,0],[-my.dot(h_ll,U_u)[0],0,h_ll[0,0],h_ll[0,1]],[-my.dot(h_ll,U_u)[1],0,h_ll[1,0],h_ll[1,1]]])
inverse_metric=sp.Array(metric.tomatrix().inv())

""" Method 1
--------------------------------------------
"""
lambda_=sp.symbols(r'\lambda',cls=sp.Function,real=True)
lambda_=lambda_(v,z,x)
h=sp.symbols(r'h',cls=sp.Function,real=True)
h=h(x)
Phi=z-1
k0=sp.symbols('k_0',real=True)
k_l=lambda_*sp.Array([k0,1,0,0])
k_u=my.dot(inverse_metric,k_l)
k0_null=sp.solve(my.dot(k_u,k_l).expand(),k0)[0]
k_l=k_l.subs(k0,k0_null)
k_u=k_u.subs(k0,k0_null)

geodesic=my.dot(k_u,my.nabla(k_l,coordinates,metric))
E0=(geodesic[0]/z**2/lambda_).expand().factor().expand()
E1=(geodesic[1]/z**2/lambda_).expand().factor().expand()
E2=(geodesic[2]/z**2/lambda_).expand().factor().expand()

coeff2=(E0.coeff(h.diff(x,x))/E2.coeff(h.diff(x,x))).factor().expand()
coeff1=((E0-coeff2*E2).expand()/E1).factor().expand()
(E0-coeff1*E1-coeff2*E2).expand() # v分量由z和x分量组合而来
# ((E1*h.diff(x)+E2)/lambda_).expand() # x分量给出测地线条件

pv_lambda_=sp.solve(E1,lambda_.diff(v))[0].expand() # z分量给出pv_lambda_
E2=(E2.subs(lambda_.diff(v),pv_lambda_)/lambda_).expand()

expansion=sp.tensorcontraction(my.dot(my.nabla(k_l,coordinates,metric),inverse_metric),(0,1)).expand()
E3=(expansion/z**2).expand().factor().expand()
E3=(E3.subs(lambda_.diff(v),pv_lambda_)/lambda_*sp.exp(2*beta)).expand() # 膨胀为0即为表观视界条件
exp_to_hyper(E3,Theta)
# (-E3*h.diff(x)+2*E2).expand() # 可以用表观视界条件化简测地线条件

# Check
h_uu=sp.Array(h_ll.tomatrix().inv())
sp.tensorcontraction(my.dot(my.nabla(my.dot(h_ll,U_u),sp.Array([x,y]),h_ll),h_uu),(0,1)).expand()




def pv_prime(fun):
    return(fun.diff(v)+h.diff(v)*fun.diff(z))

def px_prime(fun):
    return(fun.diff(x)+h.diff(x)*fun.diff(z))

h_eq=(px_prime(xi+h.diff(x)*sp.exp(-A-chi)*sp.cosh(B))+1/h*(f-h.diff(x)**2*sp.exp(-A-chi)*sp.cosh(B))).expand() # 平面拓扑
h_eq=(px_prime(xi+h.diff(x)*sp.exp(-A-chi)*sp.cosh(B))+sp.cot(x)*(xi+h.diff(x)*sp.exp(-A-chi)*sp.cosh(B))+1/h*(f-h.diff(x)**2*sp.exp(-A-chi)*sp.cosh(B))).expand() # 球面拓扑

""" Method 2
--------------------------------------------
"""
k_l=k_l.subs(lambda_,1)
k_u=k_u.subs(lambda_,1)

geodesic=my.dot(k_u,my.nabla(k_l,coordinates,metric))-sp.tensorcontraction(my.dot(my.nabla(k_l,coordinates,metric),inverse_metric),(0,1))*k_l

E0=(geodesic[0]/z**2).expand().factor().expand()
E1=(geodesic[1]/z**2).expand().factor().expand()
E2=(geodesic[2]/z**2).expand().factor().expand()

(E0-coeff1*E1-coeff2*E2).expand() # v分量由z和x分量组合而来
# ((E1*h.diff(x)+E2)/lambda_).expand() # x分量给出测地线条件
exp_to_hyper(E1*sp.exp(-chi),B) # z分量给出表观视界条件