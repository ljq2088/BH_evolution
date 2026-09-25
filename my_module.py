import sympy as sp
import numpy as np
import scipy
import itertools

""" sympy
--------------------------------------------
"""
def dot(tensor1,tensor2,index=[-1,0]):
    """
    Function
    ----------
    Dot product of adjacent indexes or specified indexes of two tensors.

    Parameters
    ----------
    tensor1 : sympy.Array
    tensor2 : sympy.Array
    index : array_like, optional, the default is [-1,0]
        [-1,0] : 
            content : Einstein summation of the last index of tensor1 and the first index of tensor2. For vector or matrix, it is just the dot product between vector and vector, matrix and vector or matrix and matrix.
        array_like : 
            request : dimension(the index[0]-th index of the tensor1) = dimension(the index[1]-th index of the tensor2).
            !!! Note : the indexes just belong to the tensor1 or tensor2, not the tensor product of them.
            !!! Note : the indexes can be negative.
            content : Einstein summation of corresponding indexes.

    Returns
    -------
    result : sympy.Array
        $\sum T1_{...i}T2_{i...}$
        or
        $\sum T1_{...i...}T2_{...i...}$
    
    Raises
    ------
    IndexError
        If the index is out of range of tensor1 or tensor2.
    """
    index=list(index)
    if index[0] >= tensor1.rank() or index[0] < -tensor1.rank():
        raise IndexError('tensor1 index out of range')
    elif index[1] >= tensor2.rank() or index[1] < -tensor2.rank():
        raise IndexError('tensor2 index out of range')
    else:
        if index[0] < 0:
            index[0]=index[0]+tensor1.rank()
        if index[1] < 0:
            index[1]=index[1]+tensor2.rank()
        result=sp.tensorcontraction(sp.tensorproduct(tensor1,tensor2),(index[0],index[1]+tensor1.rank()))
        return result

def expand(tensor):
    """
    Function
    ----------
    Expand each tensor component

    Parameters
    ----------
    tensor : sympy.Array or sympy.symbols
        if tensor.rank() > 1, the function is needed

    Returns
    -------
    result : sympy.Array

    """
    if hasattr(tensor, '__iter__'):
        tensor=sp.MutableDenseNDimArray(tensor)
        for index in itertools.product(*[range(i) for i in tensor.shape]):
            tensor[index]=tensor[index].expand()
        return sp.Array(tensor)
    else:
        return tensor.expand()

def christoffel(coordinate,metric):
    """
    Function
    ----------
    Giving the Christoffel connection of the metric.

    Parameters
    ----------
    coordinate : one-dimension sympy.Array
        spacetime coordinate
    metric : two-dimension sympy.Array
        spacetime metric

    Returns
    -------
    result : three-dimension sympy.Array
        $\Gamma^{\rho}_{\mu\nu}=\frac{1}{2}g^{\rho\sigma}(\partial_{\mu}g_{\sigma\nu}+\partial_{\nu}g_{\mu\sigma}-\partial_{\sigma}g_{\mu\nu})$

    """
    if coordinate.rank() != 1 or metric.rank() != 2:
        raise NameError('symbolcoordinatepackage_christoffel')
    inverse_metric=sp.Array(metric.tomatrix().inv())
    partial_metric=sp.derive_by_array(metric,coordinate)
    Gamma=sp.S(1)/sp.S(2)*dot(inverse_metric,sp.permutedims(partial_metric,(1,0,2))+sp.permutedims(partial_metric,(2,1,0))-partial_metric)
    return Gamma

def curvature(coordinate,metric,connection=None):
    """
    Function
    ----------
    Curvature.

    Parameters
    ----------
    coordinate : one-dimension sympy.Array
        spacetime coordinate
    metric : two-dimension sympy.Array
        spacetime metric
    connection : three-dimension sympy.Array, optional, the default is None
        None : 
            using the Christoffel connection of the metric
        rank-3 sympy.Array : 
            connection

    Returns
    -------
    result1 : four-dimension sympy.Array
        Riemann curvature : $R^{i}_{jkl}=\partial_{k}\Gamma^{i}_{jl}-\partial_{l}\Gamma^{i}_{jk}+\Gamma^{i}_{mk}\Gamma^{m}_{jl}-\Gamma^{i}_{ml}\Gamma^{m}_{jk}$
    result2 : two-dimension sympy.Array
        Ricci curvarure : $R_{mn}=R^{k}_{mkn}$
    result3 : sympy.symbols
        scalar curvature : $R=g^{mn}R_{mn}$
    result4 : two-dimension sympy.Array
        Einstein tensor : $G_{\mu\nu}=R_{\mu\nu}-\frac{1}{2}Rg_{\mu\nu}$

    """
    if connection is None:
        connection=christoffel(coordinate,metric)
    if coordinate.rank() != 1 or metric.rank() != 2 or connection.rank() != 3:
        raise NameError('symbolcoordinatepackage_curvature')
    inverse_metric=sp.Array(metric.tomatrix().inv())
    partial_connection=sp.derive_by_array(connection,coordinate)
    connection2=dot(connection,connection,(1,0))
    RiemannR=sp.permutedims(partial_connection-sp.permutedims(partial_connection,(3,1,2,0)),(1,2,0,3))+sp.permutedims(connection2-sp.permutedims(connection2,(0,3,2,1)),(0,2,1,3))
    RicciR=sp.tensorcontraction(RiemannR,(0,2))
    ScalarR=sp.tensorcontraction(dot(inverse_metric,RicciR),(0,1))
    EinsteinG=RicciR-sp.S(1)/sp.S(2)*ScalarR*metric
    return RiemannR,RicciR,ScalarR,EinsteinG

def nabla(tensor,coordinate,metric,connection=None):
    """
    Function
    ----------
    Covariant derivative.

    Parameters
    ----------
    tensor : sympy.Array or sympy.symbols
        tensor or scalar
        !!! Note : Lower index tensor
    coordinate : one-dimension sympy.Array
        spacetime coordinate
    metric : two-dimension sympy.Array
        spacetime metric
    connection : three-dimension sympy.Array, optional, the default is None
        None : 
            using the Christoffel connection of the metric
        three-dimension sympy.Array : 
            connection

    Returns
    -------
    result : (tensor.rank()+1) dimension sympy.Array
        $\nabla_{\mu}T_{i_{1}...i_{p}}=\partial_{\mu}T_{i_{1}...i_{p}}-\Gamma^{k}_{\mu i_{1}}T_{k,i_{2}...i_{p}}-...\Gamma^{k}_{\mu i_{p}}T_{i_{1}...i_{p-1},k}$

    """
    if connection is None:
        connection=christoffel(coordinate,metric)
    if coordinate.rank() != 1 or metric.rank() != 2 or connection.rank() != 3:
        raise NameError('symbolcoordinatepackage_nabla')
    if tensor.is_Function == True or tensor.is_Symbol == True:
        return sp.derive_by_array(tensor,coordinate)
    elif tensor.rank() >= 1:
        if tensor.shape != (tensor.shape[0],)*tensor.rank():
            raise NameError('symbolcoordinatepackage_nabla')
        result=sp.derive_by_array(tensor,coordinate);
        for i in range(0,tensor.rank()):
            perm=list(range(0,tensor.rank()+1))
            perm.remove(1)
            perm.insert(i+1,1)
            perm=tuple(perm)
            result=result-sp.permutedims(dot(connection,tensor,(0,i)),perm)
        return result
    else:
        raise NameError('symbolcoordinatepackage_nabla')

def series(expr,x=None,x0=0,n=None):
    """
    sympy.series()函数的扩展，尽可能展到最高阶（适用于expr里有O(x**m)的情况），这里设置了最高展到10阶
    !!!Note : 关于expr，如果有嵌套函数，内部函数的自变量要先expand()，比如sp.exp(phi_asymp**2)，要写成sp.exp((phi_asymp**2).expand())
    """
    if n is None:
        n=0
        while n >= 0:
            try:
                result=sp.series(expr,x,x0,n)
                print(n)
            except:
                return result
            else:
                if n == 10:
                    return result
                elif n < 10:
                    n=n+1
    elif type(n) == int:
        result=sp.series(expr,x,x0,n)
        print(n)
        return result

def partial_field(fields,variables,max_derive_order=2):
    fields=sp.Array(fields)
    variables=sp.Array(variables)
    if fields.rank() == 0:
        fields=sp.Array([fields])[0]
    if max_derive_order < 0:
        raise ValueError('`max_derive_order` must be greater than or equal to zero')
    else:
        if variables.rank() == 0:
            partial_array=[fields]
            for i in range(0,max_derive_order):
                partial_array.append(sp.derive_by_array(partial_array[i],variables))
            partial_array=sp.Array(partial_array).transpose()
            return partial_array
        elif variables.rank() == 1:
            if max_derive_order > 2:
                raise NotImplementedError('for multiple variables, derivatives of more than two orders are not supported')
            else:
                variables_number=len(variables)
                # zeroth derivative
                partial_array=[fields]
                if max_derive_order >= 1:
                    # first derivative
                    for i in range(0,variables_number):
                        partial_array.append(sp.derive_by_array(fields,variables[i]))
                    if max_derive_order >= 2:
                        # second derivative
                        for i in range(0,variables_number):
                            for j in range(i,variables_number):
                                partial_array.append(sp.derive_by_array(sp.derive_by_array(fields,variables[i]),variables[j]))
                partial_array=sp.Array(partial_array).transpose()
                return partial_array
        else:
            raise ValueError('`variables` must be symbol/Function or 1-d array_like of symbol/Function')

def variation(equations,fields,variables,max_derive_order=2):
    """
    Function
    ----------
    Variation of equations for each field, and field can be a function of multiple variables. 
    This function is mainly used to iteratively solve nonlinear ordinary differential (partial differential) equations through the fixed point method.
    !!! Note: 目前只支持一个变量

    Parameters
    ----------
    equations : tuple or sympy.symbols
        equations
    fields : tuple or sympy.symbols
        fields
        !!! Note : the number of equations = the number of fields
    variables : tuple or sympy.symbols
        the independent variable(s) of the field function, which determines whether they are ordinary differential or partial differential
        !!! Note : 目前只支持一个变量
    max_derive_order : int
        the order of the highest derivative contained in the system of equations

    Returns
    -------
    result : sympy.Array
        $result_{ijmn}=\delta E_{i}/\delta \partial^{n}_{x^{m}}f_{j}$
        !!! Note : There are the linear terms of the field ($\delta E_{i}/\delta f_{j}/$) with the same number of variables, but in fact only one is needed
        !!! 目前只支持一个变量，所以没有m指标，输出的是three-dimension sympy.Array
        !!! 支持多个变量了，不过这个时候max_derive_order必须等于2，此时n指标不再表示几阶导，而是表示所有可能的导数（按variable的顺序排序，比如variable=(t,z)，那么顺序是(0阶导,pt,pz,pt2,ptpz,pz2)）

    """
    equations=sp.Array(equations)
    if equations.rank() == 0:
        equations=sp.Array([equations])[0]
    return sp.permutedims(sp.derive_by_array(equations,partial_field(fields,variables,max_derive_order)),(2,0,1))

def symbol_arg_field(fields,variables,max_derive_order=2):
    partial_array=partial_field(fields,variables,max_derive_order)
    # 对于sympy.Array，len(sympy.Array)给出的其实是它的总长度（即压平之后的长度，对应numpy.array的size）
    return list(partial_array.reshape(len(partial_array)))

def exact_symbol_arg_field(expr,partial_array):
    expr=sp.Array(expr)
    if expr.rank() == 0:
        expr=sp.Array([expr])[0]
    else:
        expr=expr.reshape(np.array(expr.shape).prod())
    expr_derive=sp.derive_by_array(expr,partial_array)
    result=[]
    index=[]
    for i in range(0,partial_array.shape[0]):
        for j in range(0,partial_array.shape[1]):
            if expr_derive[i,j,:] != sp.Array([sp.S(0)]*len(expr)):
                result.append(partial_array[i,j])
                index.append((i,j))
    return result,index

def lambdify_ode(equation,field,variable,arg_parameter,partial_array,max_derive_order=2):
    equation_variation=variation(equation,field,variable,max_derive_order)
    equation_constterm=equation.subs(field,0).doit().expand()

    equation_variation_args=list(arg_parameter)+[variable]+exact_symbol_arg_field(equation_variation,partial_array)[0]
    equation_constterm_args=list(arg_parameter)+[variable]+exact_symbol_arg_field(equation_constterm,partial_array)[0]
    equation_variation_indexs=exact_symbol_arg_field(equation_variation,partial_array)[1]
    equation_constterm_indexs=exact_symbol_arg_field(equation_constterm,partial_array)[1]

    equation_variation_function=sp.lambdify(equation_variation_args,equation_variation,'numpy')
    equation_constterm_function=sp.lambdify(equation_constterm_args,equation_constterm,'numpy')

    return equation_variation,equation_constterm,equation_variation_args,equation_constterm_args,equation_variation_indexs,equation_constterm_indexs,equation_variation_function,equation_constterm_function

""" numpy
--------------------------------------------
"""
def chebyshev_spectrum(N,xmin,xmax):
    """
    Function
    ----------
    Chebyshev pseudospectral method.

    Parameters
    ----------
    N : int
        the number of Chebyshev points.
        !!! Note : the index starts at 0 and end at N-1: j=0,1,...,N-1
    xmin : float
        the minimum value of the domain.
    xmax : float
        the maximum value of the domain.

    Returns
    -------
    result1 : numpy.array
        $x_j=\cos(j\pi/(N-1)),\quad j=0,1,...,N-1$, mapping to the domain.
        !!! Note : there are N Chebyshev points
    result2 : numpy.array
        Chebyshev first-order differentiation matrix.
    result3 : numpy.array
        Chebyshev second-order differentiation matrix.

    """
    D1=np.zeros((N,N))
    D2=np.zeros((N,N))
    # 计算[-1,1]的情形
    x=np.cos(np.arange(N)*np.pi/(N-1))
    c=np.ones(N);c[0]=2;c[-1]=2
    for i in range(N):
        for j in range(N):
            if i != j:
                D1[i,j]=c[i]/c[j]*(-1)**(i+j)/(x[i]-x[j])
    D1=D1-np.diag(np.sum(D1,axis=1))
    D2=D1.dot(D1)
    return (xmax+xmin)/2+(xmax-xmin)/2*x,2/(xmax-xmin)*D1,(2/(xmax-xmin))**2*D2

def fourier_spectrum(N,xmin,length):
    """
    Function
    ----------
    Fourier pseudospectral method

    Parameters
    ----------
    N : int
        the number of Fourier points.
        !!! Note : the index starts at 0 and end at N-1: j=0,1,...,N-1
    xmin : float
        the minimum value of the domain.
    length : float
        the length of the domain, in the other word, the period of the function.

    Returns
    -------
    result1 : numpy.array
        $x_j=j\times length/N,\quad j=0,...,N-1$, mapping to the domain.
        !!! Note : there are N Fourier points, and include starting point of the domain, no end point of the domain
    result2 : numpy.array
        Fourier first-order differentiation matrix.
    result3 : numpy.array
        Fourier second-order differentiation matrix.
    result4 : numpy.array
        Fourier third-order differentiation matrix.

    """
    D1=np.zeros((N,N))
    D2=np.zeros((N,N))
    D3=np.zeros((N,N))
    # 计算[0,2\pi)的情形
    h=2*np.pi/N
    if N%2 == 0:
        for i in range(N):
            for j in range(i):
                    D1[i,j]=(-1)**(i-j)/(2*np.tan((i-j)*h/2))
                    D2[i,j]=-(-1)**(i-j)/(2*(np.sin((i-j)*h/2))**2)
                    D3[i,j]=(-1)**(i-j)*(3/(4*(np.sin((i-j)*h/2))**2)-N**2/8)/np.tan((i-j)*h/2)
        D1=D1-D1.T
        D2=D2+D2.T-((N**2+2)/12)*np.eye(N)
        D3=D3-D3.T
    else:
        for i in range(N):
            for j in range(i):
                D1[i,j]=(-1)**(i-j)/(2*np.sin((i-j)*h/2))
        D1=D1-D1.T
        D2=D1.dot(D1)
        D3=D2.dot(D1)
    # 计算[xmin,xmin+length)的情形
    x=xmin+np.arange(0,N)*length/N
    return x,(2*np.pi/length)*D1,(2*np.pi/length)**2*D2,(2*np.pi/length)**3*D3

def numeric_arg_field(fields,D1,D2):
    field_number=len(fields)
    result=[]
    for i in range(0,field_number):
        result=result+[fields[i],D1.dot(fields[i]),D2.dot(fields[i])]
    return np.array(result)

def chebyshev_interp(z_interp,z,f=None,Warn=True):
    if not hasattr(z_interp, '__iter__'):
        z_interp=[z_interp]
    z_interp=np.array(z_interp)
    if z_interp.ndim != 1:
        raise ValueError('`z_interp` is not a number or a 1-d array-like')
    if z.ndim != 1:
        raise ValueError('`z` is not 1-d')
    if Warn:
        if z_interp.min() < z.min() or z_interp.max() > z.max():
            try:
                raise UserWarning(f'some interpolation points in [{z_interp.min()}, {z_interp.max()}] are out of range [{z.min()}, {z.max()}] and may not be accurate')
            except UserWarning as warning:
                print(warning)

    N=len(z)
    weight=np.array([(-1)**0/2]+[(-1)**i for i in range(1,N-1)]+[(-1)**(N-1)/2])
    temp=1/(z_interp[:,None]-z)
    interp=weight*temp / (weight.dot(temp.T))[:,None]

    if f is None:
        return np.nan_to_num(interp,nan=1)
    elif f.shape != z.shape:
        raise ValueError('`f` does not mathch to `z`')
    else:
        return np.nan_to_num(interp,nan=1).dot(f)

def fourier_interp(x_interp,x,f):
    if not hasattr(x_interp, '__iter__'):
        x_interp=[x_interp]
    x_interp=np.array(x_interp)
    if x_interp.ndim != 1:
        raise ValueError('`x_interp` is not a number or a 1-d array-like')
    if x.ndim != 1:
        raise ValueError('`x` is not 1-d')

    N=len(x)
    h=x[1]-x[0]
    length=h*N
    # np.fft.fftfreq(N,h)是真实的频率，而k是np.fft.ifft计算中用到的求和指标
    k=np.fft.fftfreq(N,h)*length
    fourier_coefficients=np.fft.fft(f)
    # 平移并缩放到0到2\pi之间
    x_interp=(x_interp-x[0])*(2*np.pi/length)
    f_interp=1/N*np.sum(fourier_coefficients*np.exp(1j*k*x_interp[:,None]),axis=1).real

    return f_interp

def newton_iteration(variation_function,equation_function,parameters,variable,fields_initial,bc,D1,D2,precision=1e-10,Print=False):
    """
    Function
    ----------
    Solve nonlinear second-order ordinary differential equations by fixed point iteration method.

    Parameters
    ----------
    variation_function : function
        Jacobian matrix variational function
    equation_function : function
        nonlinear second-order ordinary differential equations
        !!! Note : Not the opposite
    parameters : tuple or list
        parameters in the equations
    variable : numpy.array
        the spectral points of the independent variable of the unit function
    fields_initial : two-dimension numpy.array
        the initial value of each field function
        !!! Note : The order is consistent with the variational order
    bc : two-dimension tuple
        bc[n] is a tuple whose size is 5, corresponding a boundary condition:
        bc[n]=(i-th equation,j-th field,location:0/-1,type:1/2/3,boundary value)
        !!! Note : Boundary value is not the opposite
        !!! Example: 第一类：f(0)=a
                            \delta f=-(f(0)-a)
                    第二类：f'(0)=b
                            \delta f'=-(f'(0)-b)
                    第三类：f''(0)=c
                            \delta f''=-(f''(0)-c)
                    第四类：允许有多个场，允许有两阶及以下的导数，此时bc[n][4]表示的boundary value需要输入一个表示边界条件并且等于0的sympy表达式：eq=a f''+b f'+c f+d g+e在lambdify之后的函数，而bc[n][1]不再表示第几个场，而是输入eq的variation在lambdify之后的函数。
    D1 : numpy.array
        first-order differentiation matrix
    D2 : numpy.array
        second-order differentiation matrix
    precision : float
        Accuracy reached by iteration
        the default is 1e-10
        !!! Note : The sum of the accuracy of each component
        !!! 现在用的是max

    Returns
    -------
    result : two-dimension numpy.array
        the value of each field function
        !!! Note : The order is consistent with the initialvalue

    """
    fields=fields_initial.copy()
    point_number=len(variable)
    field_number=len(fields)
    boundary_number=len(bc)
    precision_old=precision
    ones=np.ones(point_number)

    delta_f=np.ones(field_number*point_number)
    n=1
    while np.abs(delta_f).max() > precision:
        arguments=list(parameters)+[variable]+list(numeric_arg_field(fields,D1,D2))
        variation_value=variation_function(*arguments)
        equation=equation_function(*arguments)

        M=np.zeros((field_number,field_number,point_number,point_number))
        for i in range(0,field_number):
            for j in range(0,field_number):
                M[i,j]=np.diag(variation_value[i][j][0]*ones)+np.diag(variation_value[i][j][1]*ones).dot(D1)+np.diag(variation_value[i][j][2]*ones).dot(D2)
        E=np.array(equation).flatten()
        if Print:
            print('max(equation):',np.abs(E).max())
        A=np.vstack(tuple([np.hstack(tuple(M[i,:])) for i in range(0,field_number)]))

        for i in range(0,boundary_number):
            A[bc[i][0]*point_number-bc[i][2]*(point_number-1),:]=0
            if bc[i][3] == 1:
                A[bc[i][0]*point_number-bc[i][2]*(point_number-1),bc[i][1]*point_number:bc[i][1]*point_number+point_number]=np.eye(point_number)[bc[i][2]]
                E[bc[i][0]*point_number-bc[i][2]*(point_number-1)]=fields[bc[i][1]][bc[i][2]]-bc[i][4]
            elif bc[i][3] == 2:
                A[bc[i][0]*point_number-bc[i][2]*(point_number-1),bc[i][1]*point_number:bc[i][1]*point_number+point_number]=D1[bc[i][2]]
                E[bc[i][0]*point_number-bc[i][2]*(point_number-1)]=D1.dot(fields[bc[i][1]])[bc[i][2]]-bc[i][4]
            elif bc[i][3] == 3:
                A[bc[i][0]*point_number-bc[i][2]*(point_number-1),bc[i][1]*point_number:bc[i][1]*point_number+point_number]=D2[bc[i][2]]
                E[bc[i][0]*point_number-bc[i][2]*(point_number-1)]=D2.dot(fields[bc[i][1]])[bc[i][2]]-bc[i][4]
            elif bc[i][3] == 4:
                variation_bc=bc[i][1](*arguments)
                equation_bc=bc[i][4](*arguments)
                M_bc=np.zeros((1,field_number,point_number,point_number))
                for j in range(0,1):
                        for k in range(0,field_number):
                            M_bc[j,k]=np.diag(variation_bc[j][k][0]*ones)+np.diag(variation_bc[j][k][1]*ones).dot(D1)+np.diag(variation_bc[j][k][2]*ones).dot(D2)
                A_bc=np.vstack(tuple([np.hstack(tuple(M_bc[j,:])) for j in range(0,1)]))

                A[bc[i][0]*point_number-bc[i][2]*(point_number-1),:]=A_bc[bc[i][2]]
                E[bc[i][0]*point_number-bc[i][2]*(point_number-1)]=equation_bc[bc[i][2]]
            else:
                raise ValueError(f'there are no {bc[i][3]}th kind of boundary conditions')
        delta_f=np.linalg.solve(A,-E)
        fields=fields+delta_f.reshape(field_number,point_number)
        if n >= 500:
            Print=True
            precision=precision+0.1*precision_old
        if Print:
            print(n)
            print('max(delta_f):',np.abs(delta_f).max())
        n=n+1
    
    arguments=list(parameters)+[variable]+list(numeric_arg_field(fields,D1,D2))
    equation=equation_function(*arguments)
    E=np.array(equation).flatten()
    print('max(equation):',np.abs(E).max())
    if n > 500:
        print('precision:',precision)
    return fields

def get_modes(background_fields,background_equation_function,variation_function,error_variation_function,parameters,variable,D1,D2,bc,mode_abs_min,second_time_derivative=False):
    arguments=list(parameters)+[variable]+list(numeric_arg_field(background_fields,D1,D2))
    background_error=background_equation_function(*arguments)
    print('background_error:',np.abs(background_error).max())
    variation_value=variation_function(*arguments)

    field_number=len(variation_value)
    point_number=len(variable)
    boundary_number=len(bc)
    ones=np.ones(point_number)

    M0_temp=np.zeros((field_number,field_number,point_number,point_number),dtype=complex)
    M1_temp=np.zeros((field_number,field_number,point_number,point_number),dtype=complex)
    M2_temp=np.zeros((field_number,field_number,point_number,point_number),dtype=complex)
    for i in range(0,field_number):
        for j in range(0,field_number):
            M0_temp[i,j]=np.diag(variation_value[i][j][0]*ones)+np.diag(variation_value[i][j][2]*ones).dot(D1)+np.diag(variation_value[i][j][5]*ones).dot(D2)
            M1_temp[i,j]=np.diag(variation_value[i][j][1]*ones)+np.diag(variation_value[i][j][4]*ones).dot(D1)
            M2_temp[i,j]=np.diag(variation_value[i][j][3]*ones)

    M0=np.vstack(tuple([np.hstack(tuple(M0_temp[i,:])) for i in range(0,field_number)]))
    M1=-1j*np.vstack(tuple([np.hstack(tuple(M1_temp[i,:])) for i in range(0,field_number)]))
    M2=(-1j)**2*np.vstack(tuple([np.hstack(tuple(M2_temp[i,:])) for i in range(0,field_number)]))

    for i in range(0,boundary_number):
        M0[bc[i][0]*point_number-bc[i][2]*(point_number-1),:]=0
        M1[bc[i][0]*point_number-bc[i][2]*(point_number-1),:]=0
        M1[bc[i][0]*point_number-bc[i][2]*(point_number-1),:]=0
        if bc[i][3] == 1:
            M0[bc[i][0]*point_number-bc[i][2]*(point_number-1),bc[i][1]*point_number:bc[i][1]*point_number+point_number]=np.eye(point_number)[bc[i][2]]
        elif bc[i][3] == 2:
            M0[bc[i][0]*point_number-bc[i][2]*(point_number-1),bc[i][1]*point_number:bc[i][1]*point_number+point_number]=D1[bc[i][2]]
        elif bc[i][3] == 3:
            M0[bc[i][0]*point_number-bc[i][2]*(point_number-1),bc[i][1]*point_number:bc[i][1]*point_number+point_number]=D2[bc[i][2]]
        else:
            raise ValueError(f'there are no {bc[i][3]}th kind of boundary conditions')

    # (M0+w M1+w2 M2) v=0
    # (A+w B) v=0
    if second_time_derivative:
        A=np.vstack((np.hstack((M0,M1)),np.hstack((0*np.eye(field_number*point_number),np.eye(field_number*point_number)))))
        B=np.vstack((np.hstack((0*np.eye(field_number*point_number),M2)),np.hstack((-np.eye(field_number*point_number),0*np.eye(field_number*point_number)))))

        w,v0=scipy.linalg.eig(A,-B)
        v=v0.transpose().reshape(2*field_number*point_number,2,field_number,point_number)[:,0,:,:]
    else:
        w,v0=scipy.linalg.eig(M0,-M1)
        v=v0.transpose().reshape(field_number*point_number,field_number,point_number)

    error_variation=error_variation_function(*arguments)
    error_number=len(error_variation)
    error=np.full((error_number,field_number*point_number),np.nan)
    for i in range(0,error_number):
        for n in np.where((np.abs(w)>mode_abs_min) & (np.abs(w)<1e2))[0]:
            error_temp=np.zeros(point_number)
            for j in range(0,field_number):
                error_temp=error_temp+error_variation[i][j][0]*v[n][j]+(-1j*w[n])*error_variation[i][j][1]*v[n][j]+error_variation[i][j][2]*D1.dot(v[n][j])+(-1j*w[n])**2*error_variation[i][j][3]*v[n][j]+(-1j)*w[n]*error_variation[i][j][4]*D1.dot(v[n][j])+error_variation[i][j][5]*D2.dot(v[n][j])
            error[i,n]=np.abs(error_temp).max()
    print('max_error:',np.nan_to_num(error,nan=0).max())
    print('min_error:',np.nan_to_num(error,nan=1).min())
    return w,v,error

def get_accurate_mode_index(precision_list,mode_number,mode_base,*args):
    if not hasattr(precision_list, '__iter__'):
        precision_list=[precision_list]
    distance=[]
    for i in range(0,len(mode_base)):
        distance_temp=[]
        for j in range(0,len(args)):
            distance_temp.append(np.nan_to_num(np.abs(mode_base[i]-args[j]),nan=1).min())
        distance.append(np.array(distance_temp).max())
    for precision in precision_list:
        index=[]
        for i in range(0,len(mode_base)):
            if distance[i] < precision:
                index.append(i)
        if len(index) >= mode_number:
            break
    print('precision:',precision)
    return index

def solve_ode(variation_function,constterm_function,parameters,variable,fields,bc,D1,D2):
    """
    Function
    ----------
    Solve a second-order linear ordinary differential equation by spectral method.

    Parameters
    ----------
    variation_function : function
        要解的ODE的变分的lambdify
    constterm_function : function
        要解的ODE的常数项（即把要解的场替换成0）的lambdify
    bc : two-dimension tuple
        bc[n] is a tuple whose size is 5, corresponding a boundary condition:
        bc[n]=(i-th equation,j-th field,location:0/-1,type:1/2/3,boundary value)
        !!! Note : Boundary value is not the opposite
        !!! Example: 第一类：f(0)=a
                            \delta f=-(f(0)-a)
                    第二类：f'(0)=b
                            \delta f'=-(f'(0)-b)
                    第三类：f'(0)+k f(0)=c（可以到二阶导）
                            \delta f'+k \delta f=-(f'(0)+k f(0)-c)
                    第四类：允许有多个场，允许有导数，此时bc[n][1]失效
        若是第三类和第四类，boundary value需要输入,一个等于0的sympy表达式：eq=f'(0)+k f(0)-c的变分及其本身lambdify之后的函数组成的元组
    D1 : numpy.array
        first-order differentiation matrix
    D2 : numpy.array
        second-order differentiation matrix

    Returns
    -------
    result : one-dimension numpy.array
        the solution

    """
    point_number=variable.size
    field_number=fields.shape[0]
    boundary_number=len(bc)

    arguments=list(parameters)+[variable]
    for i in range(0,field_number):
        arguments=arguments+[fields[i],D1.dot(fields[i]),D2.dot(fields[i])]

    variation_value=variation_function(*arguments)
    A=np.diag(variation_value[0][0][0]*np.ones(point_number))+np.diag(variation_value[0][0][1]*np.ones(point_number)).dot(D1)+np.diag(variation_value[0][0][2]*np.ones(point_number)).dot(D2)
    b=-constterm_function(*arguments)*np.ones(point_number)

    # 边界条件
    for i in range(0,boundary_number):
        row=bc[i][2]
        A[row,:]=0
        if bc[i][3] == 1:
            A[row,bc[i][2]]=1
            b[row]=bc[i][4]
        elif bc[i][3] == 2:
            A[row,:]=D1[bc[i][2]]
            b[row]=bc[i][4]
        else:
            raise NameError('numericalprocessingpackage_solve_ode_BoundaryCondition')
    return np.linalg.inv(A).dot(b)

def get_ode_matrix_all(variation_function,constterm_function,parameters,variable,fields,bc,D1,D2,flag):
    """
    Function
    ----------
    Solve a second-order linear ordinary differential equation by spectral method.

    Parameters
    ----------
    variation_function : function
        要解的ODE的变分的lambdify
    constterm_function : function
        要解的ODE的常数项（即把要解的场替换成0）的lambdify
    bc : two-dimension tuple
        bc[n] is a tuple whose size is 5, corresponding a boundary condition:
        bc[n]=(i-th equation,j-th field,location:0/-1,type:1/2/3,boundary value)
        !!! but for here, bc[n][0]表示替换矩阵的第几行（如果bc[n][0]==None，则替换第bc[n][2]行），而bc[n][1]不起作用
        !!! Note : Boundary value is not the opposite
        !!! Example: 第一类：f(0)=a
                            \delta f=-(f(0)-a)
                    第二类：f'(0)=b
                            \delta f'=-(f'(0)-b)
                    第三类：f'(0)+k f(0)=c（可以到二阶导）
                            \delta f'+k \delta f=-(f'(0)+k f(0)-c)
                    第四类：允许有多个场，允许有导数，此时bc[n][1]失效
        若是第三类和第四类，boundary value需要输入,一个等于0的sympy表达式：eq=f'(0)+k f(0)-c的变分及其本身lambdify之后的函数组成的元组
    D1 : numpy.array
        first-order differentiation matrix
    D2 : numpy.array
        second-order differentiation matrix

    Returns
    -------
    result1 : two-dimension numpy.array
        A: the derivative operator
        !!! Note : not the inverse
    result1 : one-dimension numpy.array
        b: the opposite of the constterm
        !!! Note: the solution of the ode is np.linalg.inv(A).dot(b)

    """
    point_number=variable.size
    field_number=fields.shape[0]
    boundary_number=len(bc)

    arguments=list(parameters)+[variable]
    for i in range(0,field_number):
        arguments=arguments+[fields[i],D1.dot(fields[i]),D2.dot(fields[i])]
    
    if flag == 0:
        variation_value=variation_function(*arguments)
        A=np.diag(variation_value[0][0][0]*np.ones(point_number))+np.diag(variation_value[0][0][1]*np.ones(point_number)).dot(D1)+np.diag(variation_value[0][0][2]*np.ones(point_number)).dot(D2)
        for i in range(0,boundary_number):
            row=bc[i][2]
            A[row,:]=0
            if bc[i][3] == 1:
                A[row,bc[i][2]]=1
            elif bc[i][3] == 2:
                A[row,:]=D1[bc[i][2]]
            else:
                raise NameError('numericalprocessingpackage_solve_ode_BoundaryCondition')
        return A
    elif flag == 1:
        b=-constterm_function(*arguments)*np.ones(point_number)
        # 边界条件
        for i in range(0,boundary_number):
            row=bc[i][2]
            if bc[i][3] == 1:
                b[row]=bc[i][4]
            elif bc[i][3] == 2:
                b[row]=bc[i][4]
            else:
                raise NameError('numericalprocessingpackage_solve_ode_BoundaryCondition')
        return b
    elif flag == 3:
        variation_value=variation_function(*arguments)
        A=np.diag(variation_value[0][0][0]*np.ones(point_number))+np.diag(variation_value[0][0][1]*np.ones(point_number)).dot(D1)+np.diag(variation_value[0][0][2]*np.ones(point_number)).dot(D2)
        b=-constterm_function(*arguments)*np.ones(point_number)

        # 边界条件
        for i in range(0,boundary_number):
            row=bc[i][2]
            A[row,:]=0
            if bc[i][3] == 1:
                A[row,bc[i][2]]=1
                b[row]=bc[i][4]
            elif bc[i][3] == 2:
                A[row,:]=D1[bc[i][2]]
                b[row]=bc[i][4]
            else:
                raise NameError('numericalprocessingpackage_solve_ode_BoundaryCondition')
        return A,b

def get_ode_matrix(flag,variation_function,constterm_function,bc,D1,D2,*args):
    """
    Function
    ----------
    Solve a second-order linear ordinary differential equation by spectral method.

    Parameters
    ----------
    variation_function : function
        要解的ODE的变分的lambdify
    constterm_function : function
        要解的ODE的常数项（即把要解的场替换成0）的lambdify
    bc : two-dimension tuple
        bc[n] is a tuple whose size is 5, corresponding a boundary condition:
        bc[n]=(i-th equation,j-th field,location:0/-1,type:1/2/3,boundary value)
        !!! but for here, bc[n][0]表示替换矩阵的第几行（如果bc[n][0]==None，则替换第bc[n][2]行），而bc[n][1]不起作用
        !!! Note : Boundary value is not the opposite
        !!! Example: 第一类：f(0)=a
                            \delta f=-(f(0)-a)
                    第二类：f'(0)=b
                            \delta f'=-(f'(0)-b)
                    第三类：f'(0)+k f(0)=c（可以到二阶导）
                            \delta f'+k \delta f=-(f'(0)+k f(0)-c)
                    第四类：允许有多个场，允许有导数，此时bc[n][1]失效
        若是第三类和第四类，boundary value需要输入,一个等于0的sympy表达式：eq=f'(0)+k f(0)-c的变分及其本身lambdify之后的函数组成的元组
    D1 : numpy.array
        first-order differentiation matrix
    D2 : numpy.array
        second-order differentiation matrix

    Returns
    -------
    result1 : two-dimension numpy.array
        A: the derivative operator
        !!! Note : not the inverse
    result1 : one-dimension numpy.array
        b: the opposite of the constterm
        !!! Note: the solution of the ode is np.linalg.inv(A).dot(b)

    """
    point_number=D1.shape[0]
    boundary_number=len(bc)
    
    if flag == 0:
        variation_value=variation_function(*args)
        A=np.diag(variation_value[0][0][0]*np.ones(point_number))+np.diag(variation_value[0][0][1]*np.ones(point_number)).dot(D1)+np.diag(variation_value[0][0][2]*np.ones(point_number)).dot(D2)
        for i in range(0,boundary_number):
            row=bc[i][2]
            A[row,:]=0
            if bc[i][3] == 1:
                A[row,bc[i][2]]=1
            elif bc[i][3] == 2:
                A[row,:]=D1[bc[i][2]]
            else:
                raise NameError('numericalprocessingpackage_linear_ode_BoundaryCondition')
        return A
    elif flag == 1:
        b=-constterm_function(*args)*np.ones(point_number)
        # 边界条件
        for i in range(0,boundary_number):
            row=bc[i][2]
            if bc[i][3] == 1:
                b[row]=bc[i][4]
            elif bc[i][3] == 2:
                b[row]=bc[i][4]
            else:
                raise NameError('numericalprocessingpackage_linear_ode_BoundaryCondition')
        return b

def get_ode_operator(variation_function,bc,D1,D2,*args):
    point_number=D1.shape[0]
    boundary_number=len(bc)

    variation_value=variation_function(*args)
    A=np.diag(variation_value[0][0][0]*np.ones(point_number))+np.diag(variation_value[0][0][1]*np.ones(point_number)).dot(D1)+np.diag(variation_value[0][0][2]*np.ones(point_number)).dot(D2)
    for i in range(0,boundary_number):
        row=bc[i][2]
        A[row,:]=0
        if bc[i][3] == 1:
            A[row,bc[i][2]]=1
        elif bc[i][3] == 2:
            A[row,:]=D1[bc[i][2]]
        elif bc[i][3] == 3:
            A[row,:]=D2[bc[i][2]]
        elif bc[i][3] == 4:
            A[row,:]=D1[bc[i][2]].dot(D2)
        else:
            raise NameError('numericalprocessingpackage_linear_ode_BoundaryCondition')
    return A