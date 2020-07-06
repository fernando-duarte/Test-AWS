## load data

using Pkg
Pkg.add("XLSX")
Pkg.add("DataFrameMeta")
Pkg.add("Missings")
using DataFrames, XLSX, DataFrameMeta
using Missings

xf = XLSX.readxlsx("node_stats_for_simulation.xlsx") #xf["BHCs"]["A1:I1"]
data = vcat( [(XLSX.eachtablerow(xf[s]) |> DataFrames.DataFrame) for s in XLSX.sheetnames(xf)]... )
data = data[data.w .> 0, :]
N = size(data,1) # number of nodes
sort!(data, :assets, rev = true)
# rescale units
units = 1e6;
data[:,[:w, :c, :assets, :p_bar, :b]] .= data[!,[:w, :c, :assets, :p_bar, :b]]./units

# keep track of missing variables
col_with_miss = names(data)[[any(ismissing.(col)) for col = eachcol(data)]]
data_nm = dropmissing(data, disallowmissing=true)
          
# take a look
names(data) # column names
describe(data)


colwise(x -> any(ismissing.(x)), data)


# network primitives from the data
p_bar0 = data[!,:p_bar] # total liabilities
c0 = data[!,:c] # outside assets
a0 = data[!,:assets] # total assets
w0 = data[!,:w] # net worth
b0 = data[!,:b] # outside liabilities

# other primitives
d0 =  a0 .- c0 # inside assets
f0 = p_bar0 .- b0;# inside liabilities


# parameters
g0 = 0.0 # bankruptcy cost

## Optimization

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


m = Model(Ipopt.Optimizer) # settings for the solver

p0 = p_bar0
x0 = 0.0*similar(p0)

rng = MersenneTwister(1234);
A0 = rand(rng,N,N);[A0[i,i]=0.0 for i=1:N];A0=LowerTriangular(A0);

@variable(m, 0<=p[i=1:N]<=p0[i], start = p0[i]) 
@variable(m, 0<=c[i=1:N]) 
@variable(m, 0<=b[i=1:N]) 

fix(v::VariableRef, value::Number; force::Bool = false)

#@variable(m, 0<=x[i=1:N]<=c0[i], start = x0[i])  
@variable(m, 0<=A[i=1:N, j=1:N]<=1,start=A0[i,j])  # start=A0[i,j]

@constraint(m, sum(A,dims=2).*p_bar0 .== f0) # payments to other nodes add up to inside liabilities f
@constraint(m, A' * p_bar0 .== d0) # payments from other nodes add up to inside assets d

delete(model, con)

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
myexpr = (1+g0)*(A'*p .+ c0 .- x0) .- g0.*p_bar0
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

tol = 1e-6
@test norm(sum(AA,dims=2).* p_bar0 .- f0) < tol
@test norm(AA' * p_bar0 .- d0) < tol
@test norm(pp .- min.(p_bar0, max.((1+g0)*(AA'*pp .+ c0 .- x0) .-g0.*p_bar0,0)) ) < tol
@test norm(diag(AA)) < tol
@test norm([AA[i,j]*AA[j,i] for i=1:N , j=1:N]) < tol
@test all(0 .<=pp)
@test all(0 .<=AA.<=1)









using Flux, Zygote, Optim, FluxOptTools, Statistics
m      = Chain(Dense(1,3,tanh) , Dense(3,1))
x      = LinRange(-pi,pi,100)'
y      = sin.(x)
loss() = mean(abs2, m(x) .- y)
Zygote.refresh()
pars   = Flux.params(m)
lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))