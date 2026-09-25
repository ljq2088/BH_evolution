import sympy as sp
import sys
sys.path.append('..')
import my_module.my_module as my

""" Setup
--------------------------------------------
"""
v,z,x,y=sp.symbols(r'v,z,x,y',real=True)
coordinates=sp.Array([v,z,x,y])
f,chi=sp.symbols(r'f,\chi',cls=sp.Function,real=True)
f,chi=f(v,z,x),chi(v,z,x)
xi,A=sp.symbols(r'\xi,A',cls=sp.Function,real=True)
xi,A=xi(v,z,x),A(v,z,x)
# L=sp.symbols(r'L',real=True)
L=sp.S(1)
metric=L**2/z**2*sp.Array([[-(f*sp.exp(-chi)-sp.exp(A)*xi**2),-sp.exp(-chi),-xi*sp.exp(A),0],[-sp.exp(-chi),0,0,0],[-xi*sp.exp(A),0,sp.exp(A),0],[0,0,0,sp.exp(-A)*sp.sin(x)**2]])
# metric=L**2/z**2*sp.Array([[-(f*sp.exp(-chi)-sp.exp(A)*xi**2),-sp.exp(-chi),-xi*sp.exp(A),0],[-sp.exp(-chi),0,0,0],[-xi*sp.exp(A),0,sp.exp(A),0],[0,0,0,sp.exp(-A)]]) # 平面拓扑
inverse_metric=sp.Array(metric.tomatrix().inv())

""" Method 1
--------------------------------------------
"""
# 错误的理解
# lambda_=sp.symbols(r'\lambda',cls=sp.Function,real=True)
# lambda_=lambda_(v,z,x)
# h=sp.symbols(r'h',cls=sp.Function,real=True)
# h=h(v,x)
# Phi=z-h
# k_low=lambda_*sp.derive_by_array(Phi,coordinates)
# k_up=my.dot(inverse_metric,k_low)
# pv_h=sp.solve(my.dot(k_up,k_low).expand(),h.diff(v))[0]
# k_low=k_low.subs(h.diff(v),pv_h)
# k_up=k_up.subs(h.diff(v),pv_h)

lambda_=sp.symbols(r'\lambda',cls=sp.Function,real=True)
lambda_=lambda_(v,z,x)
h=sp.symbols(r'h',cls=sp.Function,real=True)
h=h(x)
Phi=z-h
k0=sp.symbols('k_0',real=True)
k_low=lambda_*sp.Array([k0,1,-h.diff(x),0])
k_up=my.dot(inverse_metric,k_low)
k0_null=sp.solve(my.dot(k_up,k_low).expand(),k0)[0]
k_low=k_low.subs(k0,k0_null)
k_up=k_up.subs(k0,k0_null)

geodesic=my.dot(k_up,my.nabla(k_low,coordinates,metric))
E0=(geodesic[0]/z**2*sp.exp(A)/lambda_).expand()
E1=(geodesic[1]/z**2*sp.exp(-chi)/lambda_).expand()
E2=(geodesic[2]/z**2*sp.exp(-chi)/lambda_).expand()

(E0+E1*(-f*sp.exp(A+chi)/2+h.diff(x)**2/2)+E2*(xi*sp.exp(A+chi)+h.diff(x))).expand()
((E1*h.diff(x)+E2)/lambda_).expand()

pv_lambda_=sp.solve(E1,lambda_.diff(v))[0].expand()
E2=(E2.subs(lambda_.diff(v),pv_lambda_)/lambda_).expand()

expansion=sp.tensorcontraction(my.dot(my.nabla(k_low,coordinates,metric),inverse_metric),(0,1)).expand()
E3=(expansion/z**2/sp.exp(chi)).expand()
E3=-(E3.subs(lambda_.diff(v),pv_lambda_)/lambda_).expand()
(-E3*h.diff(x)+2*E2).expand()
# (E2.expand().subs(h.diff(v),pv_h).doit().expand()*2/h.diff(x)).expand() # 错误的理解



def pv_prime(fun):
    return(fun.diff(v)+h.diff(v)*fun.diff(z))

def px_prime(fun):
    return(fun.diff(x)+h.diff(x)*fun.diff(z))

h_eq=(px_prime(xi+h.diff(x)*sp.exp(-A-chi))+sp.cot(x)*(xi+h.diff(x)*sp.exp(-A-chi))+1/h*(f-h.diff(x)**2*sp.exp(-A-chi))).expand()
# h_eq=(px_prime(xi+h.diff(x)*sp.exp(-A-chi))+1/h*(f-h.diff(x)**2*sp.exp(-A-chi))).expand() # 平面拓扑

""" Method 2
--------------------------------------------
"""
k_low=k_low.subs(lambda_,1)
k_up=k_up.subs(lambda_,1)

geodesic=my.dot(k_up,my.nabla(k_low,coordinates,metric))-sp.tensorcontraction(my.dot(my.nabla(k_low,coordinates,metric),inverse_metric),(0,1))*k_low

E0=(geodesic[0]/z**2*sp.exp(A)).expand()
E1=(geodesic[1]/z**2*sp.exp(-chi)).expand()
E2=(geodesic[2]/z**2*sp.exp(-chi)).expand()

(E0+E1*(-f*sp.exp(A+chi)/2+h.diff(x)**2/2)+E2*(xi*sp.exp(A+chi)+h.diff(x))).expand()
(E1*h.diff(x)+2*E2).expand()
E1