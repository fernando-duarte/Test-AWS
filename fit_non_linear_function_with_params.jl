using Flux
using Flux: @epochs
using Random

# This code trains the neural network called `model` to fit the function y(x) = exp(T+x) / (1+exp(T+x))

# define the loss function
function loss(x,y)
 ŷ = model(x)
 sum((y .- ŷ).^2)
end

# generate training data from the true function
xData = randn(MersenneTwister(1234),Float32,(1,500) )
T=Float32[0.5];
y = exp.(T.+xData) ./ ( 1 .+exp.(T.+xData) )
d = [(xData ,y)]

# create the neural network
model = Chain(
            Dense(1,10),
            Dense(10,10),
            Dense(10,1,sigmoid)
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

# compare function we are trying to fit with the trained neural network
using Plots
plot(sort([xData...]),sort([y...]), label = "True function")
plot!(sort([xData...]),sort([model(xData)...]), label = "Trained neural network", legend=:bottomright)
