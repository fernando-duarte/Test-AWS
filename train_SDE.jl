using Flux
using Flux: @epochs
using DifferentialEquations
using Random
using Statistics

# Define Loss Function
function loss(x,y)

    # Get Predicted y     
    ŷ = model(xData)
    
    # Extract Y0 and ZT from the Predicted "Y"     
    Z = ŷ[1,:]   
    y0 = ŷ[2,:]

    # Define Other ODE Solver Params   
    tspan = (0.0,1.0)
    dt = 1//(2^10)
    yT = zeros(length(y0))
    
    # Compute yT for each item in sample using inbuilt SDE Solver
    for i = 1:length(ŷ)
        f(t,u) = -(u+Z[i]+1)
        g(t,u) = Z[i]
        prob = SDEProblem(f,g,y0,(0.0,1.0))
        sol = solve(prob,EM(),dt=dt)
        yT[i] = sol[end]
    end

     #  Compute Loss
     (mean(yT)-m)^2 + (std(yT).^2-v)^2  

end


# generate training data from the true function
m = 0
v = 1
xData = repeat([m,v]',500,1)'
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
@epochs 100 Flux.train!(loss, ps, d, opt, cb=evalcb)
@show(loss(xData,y))