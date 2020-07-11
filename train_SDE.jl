using Random, Statistics
using DifferentialEquations, DiffEqSensitivity
using Flux
using Flux: @epochs

# Define Loss Function
function loss(x,y)

    # Get Predicted y
    ŷ = model(xData)

    # Extract Y0 and ZT from the Predicted "Y"
    Z = ŷ[1,:]
    y0 = ŷ[2,:]

     #  Compute Loss
     YTout = YT.(y0,Z);
     (mean(YTout)-m)^2 + (std(YTout).^2-v)^2

end

function YT(y0,Z)
# Compute yT for each item in sample using inbuilt SDE Solver
    f(u,p,t) = -(u+Z+1)
    g(u,p,t) = Z
    # Define Other ODE Solver Params
    tspan = (0.0,1.0)
    dt =0.1
    prob = SDEProblem(f,g,y0,tspan)
    sol = solve(prob,EM(),dt=dt,sensealg=TrackerAdjoint())
    return sol[end]
end



# generate training data from the true function
m = 0.0
v = 1.0
xData = repeat([m,v]',5,1)'
y = xData # Doesn't really matter what our y data is, this is just filler.
d = [(xData ,y)]

# create the neural network
model = Chain(
            Dense(2,10),
            Dense(10,10),
            Dense(10,2,relu)
            )
# display initial value of the loss function
@show loss(xData,y)

# extract the parameters of the neural network
ps = params(model)

# train the neural network
evalcb() = @show(loss(xData,y)) # function evaluated in each epoch of training
opt = ADAM(0.01) # pick optimizer
@epochs 3 Flux.train!(loss, ps, d, opt, cb=evalcb)
@show(loss(xData,y))
