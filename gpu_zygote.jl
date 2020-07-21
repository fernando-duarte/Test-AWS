
using Pkg
# Pkg.build("SpecialFunctions")
# Pkg.build("FFTW")
Pkg.add("NeuralNetDiffEq")
Pkg.add("Tracker")

using DiffEqFlux
using Tracker
using Zygote
using Flux
using Flux: @epochs
using Flux: @functor
using Flux.Tracker

using Flux, Zygote, StochasticDiffEq
using LinearAlgebra, Statistics
using Test, NeuralNetDiffEq

using DifferentialEquations,DiffEqSensitivity, DifferentialEquations.EnsembleAnalysis
using Test, Statistics
using CuArrays
CuArrays.allowscalar(false)
#using CuArrays

## set up and solve forward stochastic differential equation
function drift(u::Tracker.TrackedArray,p,t)
#   y0, Z = p
#  -(u[1]+Z+1.0f0)
    -(u.data +p[2].+1.0f0)
end

function stoch(u::Tracker.TrackedArray,p,t)
#     y0, Z = p
#     Z
    p[2]
end

T = 0.1f0
tspan = (0.0f0,T)
num_paths = 10

g(X) = sum(X.^2)   # terminal condition

# compute y(T)
function yT(p)
    #y0, Z = p

    prob = SDEProblem{false}(drift,stoch,0.0f0,tspan)
    #[convert(Array,solve(prob,SOSRI(),p=[y0[1],Z[1]],u0=y0[1],saveat=0.0:0.01:T,sensealg=TrackerAdjoint()))[end] for i=1:num_paths]
    #drift(p[1][1],[p[1][1],p[2][1]],0).^2+stoch(p[1][1],[p[1][1],p[2][1]],0).^2
    sol = solve(prob,alg=EM(), p=[1.0f0,2.0f0],save_start=true,save_end=true, save_everystep=false,dt=0.01,sensealg=TrackerAdjoint()) #sensealg=ForwardDiffSensitivity())
    Array(sol)[:,end]

    #SensitivityADPassThrough2
end

## create custom layer for the NN
struct consFun
  theta1 # will be used to learn y0
  theta2 # will be used to learn Z
end

# overload consFun 
theta1_init = [10.2f0] # initial value of theta1 before training, use [rand()] for random initi
theta2_init = [0.0f0]
consFun(num_inputs::Integer, num_outputs::Integer) = consFun(theta1_init,theta2_init) 
(cc::consFun)(x) = [cc.theta1,cc.theta2] 

# instantiate layer
num_in = 1
num_out = 2
a = consFun(num_in,num_out) 
Flux.@functor consFun (theta1,theta2) # tell Flux to track theta1, theta2 as parameters

## create NN and training data
model = Chain(a) #|> gpu
ps = Flux.params(model) #|> gpu
#delete!(ps, model[1].theta2) 

m = repeat([1.0f0],num_in,1) #|> gpu
v = repeat([0.1f0],num_in,1) #|> gpu
xData = [m...,v...]; #vcat(m,v) # 

# minimize distance between mean and var of y(T) and [m,v]

function loss(x) 
    p = model(x)
    q = yT(p)
    (mean(q) - m[1])^2 #+(var(q).-v[1]).^2
end

# find y0, Z
evalcb() = @show loss(xData), model(xData)
opt=ADAM(0.05f0)
@epochs 10 Flux.train!(loss, ps, xData, opt, cb = evalcb)

evalcb()

true_y0 = exp(T).*m[1] +(exp(T)-1)*(sqrt(2*v[1]/(1-exp(-2*T)))+1)
true_Z = sqrt(2*v[1]/(1-exp(-2*T)))


@test model(xData)[1]==true_y0
@test model(xData)[2]==true_Z



function delay_lotka_volterra(du,u,h,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)*h(p,t-0.1)[1]
  du[2] = dy = (δ*x - γ)*y
end
h(p,t) = ones(eltype(p),2)
u0 = [1.0,1.0]
prob = DDEProblem(delay_lotka_volterra,u0,h,(0.0,10.0),constant_lags=[0.1])

p = [2.2, 1.0, 2.0, 0.4]
function predict_dde(p)
  Array(concrete_solve(prob,MethodOfSteps(Tsit5()),u0,p,saveat=0.1,sensealg=TrackerAdjoint())
end
loss_dde(p) = sum(abs2,x-1 for x in predict_dde(p))
loss_dde(p)
    
function lotka_volterra_noise(du,u,p,t)
  du[1] = 0.1u[1]
  du[2] = 0.1u[2]
end
u0 = [1.0,1.0]
prob = SDEProblem(lotka_volterra,lotka_volterra_noise,u0,(0.0,10.0))

p = [2.2, 1.0, 2.0, 0.4]
function predict_sde(p)
  Array(concrete_solve(prob,SOSRI(),u0,p,sensealg=ForwardDiffSensitivity(),saveat=0.1))
end
loss_sde(p) = sum(abs2,x-1 for x in predict_sde(p))
loss_sde(p)










x0 = Float32[11.] # initial points
tspan = (0.0f0,5.0f0)
dt = 0.5 # time step
d = 1 # number of dimensions
m = 10 # number of trajectories (batch size)
g(X) = sum(X.^2)   # terminal condition
f(X,u,σᵀ∇u,p,t) = Float32(0.0)
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(ones(Float32,d)) #Matrix d x d

prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)



hls = 10 + d #hidden layer size
opt = Flux.ADAM(0.005)  #optimizer

#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))

# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))

pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)
ans = solve(prob, pdealg, verbose=true, maxiters=200, trajectories=m,
                            alg=EM(), dt=dt, pabstol = 1f-6)

u_analytical(x,t) = sum(x.^2) .+ d*t

analytical_ans = u_analytical(x0, tspan[end])
error_l2 = sqrt((ans-analytical_ans)^2/ans^2)




 X0 = x0  


g
f
μ
σ_f

   opt = pdealg.opt
p=Nothing
    p1,_re1 = Flux.destructure(u0)
    p2,_re2 = Flux.destructure(σᵀ∇u)
    p3 = [p1;p2;p]

    ps = Flux.params(p3)

    re1 = p -> _re1(p[1:length(p1)])
    re2 = p -> _re2(p[(length(p1)+1):(length(p1)+length(p2))])
    re3 = p -> p[(length(p1)+length(p2)+1):end]

    data = Iterators.repeated((), maxiters)
    #hidden layer

    function F(h, p, t)
        u =  h[end]
        X =  h[1:end-1]
        _σᵀ∇u = re2(p)([X;t])
        _p = re3(p)
        _f = -f(X, u, _σᵀ∇u, _p, t)
        vcat(μ(X,_p,t),[_f])
    end

    function G(h, p, t)
        X = h[1:end-1]
        _p = re3(p)
        _σᵀ∇u = re2(p)([X;t])'
        vcat(σ(X,_p,t),_σᵀ∇u)
    end

    function F(h::Tracker.TrackedArray, p, t)
        u =  h[end]
        X =  h[1:end-1].data
        _σᵀ∇u = σᵀ∇u([X;t])
        _f = -f(X, u, _σᵀ∇u, p, t)
        Tracker.collect(vcat(μ(X,p,t),[_f]))
    end



    function G(h::Tracker.TrackedArray, p, t)
        X = h[1:end-1].data
        _σᵀ∇u = σᵀ∇u([X;t])'
        Tracker.collect(vcat(σ(X,p,t),_σᵀ∇u))
    end



    noise = zeros(Float32,d+1,d)

    prob = SDEProblem{false}(F, G, [X0;0f0], tspan, p3, noise_rate_prototype=noise)



    function neural_sde(init_cond)

        map(1:trajectories) do j #TODO add Ensemble Simulation

            predict_ans = Array(solve(prob, alg;

                                         dt = dt,

                                         u0 = init_cond,

                                         p = p3,

                                         save_everystep=false,

                                         sensealg=TrackerAdjoint(),kwargs...))[:,end]

            (X,u) = (predict_ans[1:(end-1)], predict_ans[end])

        end

    end



    function predict_n_sde()

        _u0 = re1(p3)(X0)

        init_cond = [X0;_u0]

        neural_sde(init_cond)

    end



    function loss_n_sde()

        mean(sum(abs2, g(X) - u) for (X,u) in predict_n_sde())

    end



    iters = eltype(X0)[]



    cb = function ()

        save_everystep && push!(iters, u0(X0)[1])

        l = loss_n_sde()

        verbose && println("Current loss is: $l")

        l < pabstol && Flux.stop()

    end



    Flux.train!(loss_n_sde, ps, data, opt; cb = cb)