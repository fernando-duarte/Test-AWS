import Pkg
Pkg.add("NLsolve")
Pkg.add("JuMP")
Pkg.add("Ipopt")
Pkg.add("Complementarity")

using LinearAlgebra, Random, Distributions, NLsolve
using Test
using NLopt
using JuMP, Ipopt
using Complementarity


y=0.0;
p_bar0 = [55.0+y, 55.0+y, 140.0, 55.0+y, 55.0+y]; # total liabilities
c0 = [50.0, 50.0, 150.0, 50.0, 50.0]; # outside assetsp_bar
w0 = [5.0, 5.0, 10.0, 5.0, 5.0]; # net worth
A0 = [0 y/p_bar0[1] 0 0 0; 0 0 0 y/p_bar0[2] 0; 10.0/p_bar0[3] 10.0/p_bar0[3] 0 10.0/p_bar0[3] 10.0/p_bar0[3]; 0 0 0 0 y/p_bar0[4]; y/p_bar0[5] 0 0 0 0]; # matrix of relative liabilities
g0 = 0.0; # bankruptcy costs
δ0 = [0.01, 0.1, 0.3, 0.2, 0.03]; # probability of default
b0 = [55.0, 55.0, 100.0, 55.0, 55]; # outside liabilities
a0 = w0 .+ p_bar0; # total assets
d0=  a0 .- c0;# inside assets
f0 = p_bar0 .- b0;# inside liabilities

N = length(c0); # number of nodes


m = Model(Ipopt.Optimizer) # settings for the solver

p0 = p_bar0

x0=[0.0, 0.0, 0.0, 0.0, 0.0]

@variable(m, 0<=p[i=1:N]<=p0[i], start = p0[i]) 
@variable(m, 0<=c[i=1:N]) 
#@variable(m, 0<=x[i=1:N]<=c0[i], start = x0[i])  
@variable(m, 0<=A[i=1:N, j=1:N]<=1,start=A0[i,j]/2)  # start=A0[i,j]

@constraint(m, sum(A,dims=2).*p_bar0 .== [f0...]) # payments to other nodes add up to inside liabilities f
@constraint(m, A' * p_bar0 .== [d0...]) # payments from other nodes add up to inside assets d

# liabilities are net liabilities: A[i,i]=0 and A[i,j]A[j,i]=0
@constraint(m, [i = 1:N], A[i,i]==0)
for i=1:N
    j=1
    while j < i
        @complements(m, 0 <= A[i,j],  A[j,i] >= 0)
        j += 1
    end
end
#@constraint(m, A[1,2] ⟂ A[2,1])
#@constraint(model, xx1[t=1:T], x[t] => {next(t, 1) + next(t, 2) == 0})

maxfun(n1, n2) = max(n1, n2)
JuMP.register(m, :maxfun, 2, maxfun, autodiff=true)
minfun(n1, n2) = min(n1, n2)
JuMP.register(m, :minfun, 2, minfun, autodiff=true)

# clearing vector
myexpr = (1+g0)*(A'*p .+ c .- x0) .- g0.*p_bar0
@variable(m, aux[i=1:N])
@constraint(m, aux .== myexpr )
for i = 1:N
    @NLconstraint(m,  minfun(p_bar0[i], maxfun(aux[i],0)) == p[i] ) 
    #@NLconstraint(m,  min(p_bar0[i], max(aux[i],0)) == p[i] ) 
end

@NLobjective(m, Min , sum( x0[i]+p_bar0[i]-p[i] for i=1:N) ) #*sum(x[i]+p_bar[i]-p[i] for i=1:N) 

JuMP.optimize!(m)
termination_status(m)
objective_value(m)

AA = JuMP.value.(A)
pp = JuMP.value.(p)
cc = JuMP.value.(c)


tol = 1e-6
@test norm(sum(AA,dims=2).* p_bar0 .- [f0...]) < tol
@test norm(AA' * p_bar0 .- [d0...]) < tol
@test norm(pp .- min.(p_bar0, max.((1+g0)*(AA'*pp .+ cc .- x0) .-g0.*p_bar0,0)) ) < tol
@test norm(diag(AA)) < tol
@test norm([AA[i,j]*AA[j,i] for i=1:N , j=1:N]) < tol
@test all(0 .<=pp)
@test all(0 .<=AA.<=1)














# using Knet

# # Define convolutional layer:
# struct Conv; w; b; f; end
# (c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
# Conv(w1,w2,cx,cy,f=relu) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f)

# # Define dense layer:
# struct Dense; w; b; f; end
# (d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)
# Dense(i::Int,o::Int,f=relu) = Dense(param(o,i), param0(o), f)

# # Define a chain of layers and a loss function:
# struct Chain; layers; end
# (c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
# (c::Chain)(x,y) = nll(c(x),y)

# # Load MNIST data:
# include(Knet.dir("data","mnist.jl"))
# dtrn, dtst = mnistdata()

# # Define, train and test LeNet (about 30 secs on a gpu to reach 99% accuracy)
# LeNet = Chain((Conv(5,5,1,20), Conv(5,5,20,50), Dense(800,500), Dense(500,10,identity)))
# adam!(LeNet, repeat(dtrn,10))
# accuracy(LeNet, dtst)