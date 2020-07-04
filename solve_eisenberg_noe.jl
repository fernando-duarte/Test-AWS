using LinearAlgebra, Random, Distributions, NLsolve
using Test


function clearing_vec!(F, p, x, A)
    F .= p.-min.(p_bar,max.((1+g)*(A'*p .+c .- x) .-g.*p_bar,0))
end

function shock()
    # draw shocks that produce the probability of default δ 
    a = 1
    b = log.(δ)./(log.(1 .-w./c))
    dist = Beta.(a,[b...])
    draws = rand.(dist, 1)
    vcat(draws...).*c
    #rand(Float64,N) # normal
end

function loss(x,A)
     # total loss in actual network for shock x
    p = nlsolve((F, p)->clearing_vec!(F, p, x, A),[p_bar...], autodiff = :forward)
    @test converged(p)
    sum(x.+p_bar.-p.zero) 
end

function loss_d(x)
    # total loss in disconnected network for shock x
    #b_d = b .+ max.(0, c.-b.-w ) # add fictitious outside liabilities 
    #_d = c .+ max.(0, w.-c.+b ) # add fictitious outside assets
    sum(x+max.(0,x.-w))
end

## Glasserman and Young

y=0.0;
p_bar = (55.0+y, 55.0+y, 140.0, 55.0+y, 55.0+y); # total liabilities
c = (50.0, 50.0, 150.0, 50.0, 50.0); # outside assets
w = (5.0, 5.0, 10.0, 5.0, 5.0); # net worth
A0 = [0 y/p_bar[1] 0 0 0; 0 0 0 y/p_bar[2] 0; 10.0/p_bar[3] 10.0/p_bar[3] 0 10.0/p_bar[3] 10.0/p_bar[3]; 0 0 0 0 y/p_bar[4]; y/p_bar[5] 0 0 0 0]; # matrix of relative liabilities
g = 0.0; # bankruptcy costs
δ = (0.01, 0.1, 0.3, 0.2, 0.03); # probability of default
b = (55.0, 55.0, 100.0, 55.0, 55); # outside liabilities

a = w .+ p_bar; # total assets
d=  a .- c;# inside assets
f = p_bar .- b;# inside liabilities
β = (p_bar.-b)./p_bar # financial connectivity: proportion of liabilities inside the network
β⁺ = maximum(β)
N = length(c); # number of nodes

# example from Fernando's presentation and Glasserman-Young
x1 = [0.0, 0.0, 94.0, 0.0, 0.0]
@test loss(x1,A0)==182.0
@test loss_d(x1)==178.0

ratio = loss(x1,A0)/loss_d(x1)
bound = 1 + sum(δ.*c)/((1-β⁺)*sum(c))

@show ratio
@show bound

# draw shock from Beta distribution
x1 = shock()
ratio = loss(x1,A0)/loss_d(x1);
bound = 1 + sum(δ.*c)/((1-β⁺)*sum(c));
@show ratio
@show bound

#= example in https://github.com/siebenbrunner/NetworkValuation 
c = (10.0,10.0,10.0); # net worth
matL = [0.0 20.0 10.0; 5.0 0.0 24.0; 10.0 0.0 0.0]
p_bar = [sum(matL,dims=2)...]
A0 = matL ./ repeat(p_bar, outer=[1,length(c)])
vecEquity = c .+ A0'*p_bar .- p_bar

clearing_vec!([0,0,0], p_bar, [0,0,0])
x0=[0.0,0.0,0.0]
p = nlsolve((F, p)->clearing_vec!(F, p, [x0...]),[p_bar...], autodiff = :forward)

true_p=[24.545454545454547,26.363636363636363,10.0]
@test norm(true_p-p.zero)<1e-10
=#

# solve
# min (1-x)^2 + 100(y-x^2)^2)
# st x + y >= 10

# using Pkg
# Pkg.add("JuMP")
# Pkg.add("Ipopt")
# Pkg.add("Complementarity")

using NLopt
using JuMP, Ipopt
using Complementarity

#m = Model(Ipopt.Optimizer) # settings for the solver
m = Model(solver=NLoptSolver(algorithm=:LD_MMA))

#set_optimizer_attributes(m, "tol" => 1e-12,"dual_inf_tol" => 1e-12,"constr_viol_tol" => 1e-12,"compl_inf_tol" => 1e-12,"acceptable_tol"=> 1e-12)

@variable(m, 0<=A[i=1:N, j=1:N]<=1,start=A0[i,j])  # start=A0[i,j]
@constraint(m, sum(A,dims=2).*p_bar .== [f...]) # payments to other nodes add up to inside liabilities f
@constraint(m, A' * [p_bar...] .== [d...]) # payments from other nodes add up to inside assets d

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

function testLoss(x...)
     x[3]
end
JuMP.register(m, :NLobj1, N^2, testLoss, autodiff=true)

function lossA(matA...)
#     matA = reshape([matA...],N,N)
#     x1 = [0.0, 0.0, 94.0, 0.0, 0.0] 
    #loss(x1,matA) 
    matA[3]+matA[8]
end

JuMP.register(m, :NLobj3, N^2, lossA, autodiff=true)


@NLobjective(m, Min , NLobj1(A...))

JuMP.optimize!(m)
termination_status(m)

AA = JuMP.value.(A)

JuMP.value.(A[3])



norm(sum(AA,dims=2).* [p_bar...] .- [f...])
norm(AA' * [p_bar...] .- [d...])









