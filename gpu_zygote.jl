
#using Pkg
# Pkg.build("SpecialFunctions")
#Pkg.build("FFTW")

using Zygote
using Flux
using Flux: @epochs
using Flux: @functor

using DifferentialEquations,DiffEqSensitivity, DifferentialEquations.EnsembleAnalysis
using Test, Statistics

using CuArrays



function drift(du,u,p,t)
  y0, Z = p
  du[1] = -(u[1]+Z+1.0f0)
end

function stoch(du,u,p,t)
    y0, Z = p
    du[1] = Z
end

tspan = (0.0f0,0.1f0)
num_paths = 1000

function yT(p)
    y0, Z = p

    prob = SDEProblem(drift,stoch,[y0],tspan)
    #[solve(prob,SOSRI(),p=p,sensealg=ForwardDiffSensitivity(),saveat=tspan[2])[1,end] for i=1:num_paths]


    #([mean(sol)],[var(sol)])
    #solve(prob,SOSRI(),p=p,sensealg=ForwardDiffSensitivity())[1,end]
     ensembleprob = EnsembleProblem(prob)
     sol = solve(ensembleprob,SOSRI(),EnsembleThreads(),trajectories=num_paths,SOSRI(),p=p,sensealg=ForwardDiffSensitivity(),saveat=tspan[2])
    reduce(hcat,[predict_ODE_solve()[(i-1)*statesize+1:i*statesize,:] for i in 1:nbatches])
    
#     timepoint_meanvar(sol,tspan[2])
end

struct consFun
  theta1
  theta2
end

consFun(in::Integer, out::Integer) = consFun([50.0f0],[0.0f0]) #consFun([rand()],[rand()]) #consFun([0.0f0],[1.0f0])
(m::consFun)(x) = [m.theta1,m.theta2]
nobs = 1
a = consFun(nobs,nobs)
Flux.@functor consFun (theta1,theta2)

model = a #|> gpu
ps = Flux.params(model)
delete!(ps, model.theta2)

function loss(x) 
    p = model(x)
    q = yT(vcat(p...))
    #msol, vsol = yT(vcat(p...))
    #(msol[1]-m[1]).^2 .+ (vsol[1]-v[1]).^2
    
    #(mean(yT(vcat(p...)))-m[1]).^2 + (var(yT(vcat(p...)))-v[1]).^2
    p[1][1]^2
end

#m = rand(Float32,nobs) # |> gpu
#v = rand(Float32,nobs) #|> gpu

m = repeat([0.0f0],nobs,1) #|> gpu
v = repeat([0.1f0],nobs,1) #|> gpu
xData = [m...,v...];

param_init .= model(xData)
loss_init = loss(xData)
@show param_init

evalcb() = @show loss(xData), model(xData)
opt=Descent(10)
@epochs 2 Flux.train!(loss, ps, xData, opt, cb = evalcb)






















opt=ADAM(0.1)
@epochs 20 Flux.train!(loss, ps, xData, opt, cb = evalcb)

loss_end .= loss(xData)
param_end .= model(xData)

@show loss_init, loss_end
@show param_init, param_end

true_y0 = exp(tspan[2])*m[1] + (exp(tspan[2])-1)*( sqrt(  2*v[1]/(1-exp(-2*tspan[2])) )+1 )
true_Z = sqrt(  2*v[1]/(1-exp(-2*tspan[2])) )

NN_y0, NN_Z = model(xData)

tol = 1e-3
@test  loss(xData)<tol
@test  abs(true_y0-NN_y0[1])<tol
@test  abs(true_Z-NN_Z[1])<tol

mean(yT(vcat(model(xData)...)))
m[1]
var(yT(vcat(model(xData)...)))
v[1]











using DifferentialEquations, Flux

pa = [1.0]

function model1(input) 
  prob = ODEProblem((u, p, t) -> 1.01u * pa[1], 0.5, (0.0, 1.0))
  
  function prob_func(prob, i, repeat)
    remake(prob, u0 = rand() * prob.u0)
  end
  
  ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
  sim = solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 100)
end

Input_time_series = zeros(5, 100)
# loss function
loss(x, y) = Flux.mse(model1(x), y)

data = Iterators.repeated((Input_time_series, 0), 1)

cb = function () # callback function to observe training
  println("Tracked Parameters: ", params(pa))
end


opt = ADAM(0.1)
println("Starting to train")
Flux.@epochs 10 Flux.train!(loss, params(pa), data, opt; cb = cb)