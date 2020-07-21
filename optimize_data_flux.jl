## load packages
using Pkg
Pkg.add("DataFrames")
Pkg.add("XLSX")
Pkg.add("Missings")
Pkg.add("NLsolve")
Pkg.add("Complementarity")
Pkg.add("NLopt")
Pkg.add("JuMP")
Pkg.add("Ipopt")
Pkg.add("SpecialFunctions")
Pkg.add("ForwardDiff")
Pkg.add("AmplNLWriter")
Pkg.add("Cbc")

Pkg.add("FFTW")
Pkg.build("FFTW")
ENV["GRDIR"]="" ; Pkg.build("GR")
Pkg.add("PATHSolver")
Pkg.build("PATHSolver")
Pkg.add("SLEEFPirates")
Pkg.build("SLEEFPirates")
Pkg.add("IterTools")
Pkg.add("Tracker")
Pkg.add("Zygote")
Pkg.add("SpecialFunctions")

Pkg.add("DataFrames")
Pkg.add("XLSX")
Pkg.add("Missings")
Pkg.add("NLsolve")
Pkg.add("Complementarity")
Pkg.add("NLopt")
Pkg.add("JuMP")
Pkg.add("Ipopt")

Pkg.add("ForwardDiff")
Pkg.add("GraphRecipes")
Pkg.add("Gadfly")

Pkg.add("LightGraphs")
Pkg.add("SimpleWeightedGraphs")
Pkg.add("CUDA")
Pkg.add("CuArrays")
Pkg.add("Distributed")
Pkg.add("Surrogates")
Pkg.add("QuadGK")
Pkg.add("DistributionsAD")
Pkg.add("StatsFuns")

using Distributions, DistributionsAD
using GraphRecipes
using Plots
using Gadfly

using LightGraphs
using SimpleWeightedGraphs

using DataFrames, XLSX
using Missings
using ForwardDiff
using LinearAlgebra, Random, Distributions, NLsolve, Complementarity
using Test
using NLopt
using JuMP
using Ipopt
using Zygote
using CUDA
using Flux
using StatsFuns
using Flux: @epochs
using IterTools: ncycle


#Pkg.add(PackageSpec(url="https://github.com/FluxML/Flux.jl", rev="master"))

#= network primitives from the data
p_bar = total liabilities
c = outside assets
assets = total assets
w = net worth
b = outside liabilities
=#

## load data
xf = XLSX.readxlsx("node_stats_forsimulation_all.xlsx") #xf["BHCs"]["A1:I1"]
data = vcat( [(XLSX.eachtablerow(xf[s]) |> DataFrames.DataFrame) for s in XLSX.sheetnames(xf)]... )
unique!(data) # delete duplicate rows
#data[(isequal.(data.tkr,"JPM") .| isequal.(data.tkr,"AIG")) .& isequal.(data.qt_dt,192),:]


data = data[isless.(0,data.w), :]
data = data[isequal.(data.qt_dt,192), :] # keep quarter == 195 = 2008q4
sort!(data, :assets, rev = true)
data = data[1:3,:] # keep small number of nodes, for testing
N = size(data,1) # number of nodes

# rescale units
units = 1e6;
data[:,[:w, :c, :assets, :p_bar, :b]] .= data[!,[:w, :c, :assets, :p_bar, :b]]./units

# fake missing data to make problem feasible
data.b[:] .= missing
data.c[:] .= missing

# create network variables
data.f = data.p_bar .- data.b # inside liabilities
data.d =  data.assets .- data.c # inside assets

# keep track of missing variables
col_with_miss = names(data)[[any(ismissing.(col)) for col = eachcol(data)]] # columns with at least one missing
#data_nm = dropmissing(data, disallowmissing=true) # drop missing
data_nm = coalesce.(data, data.w .+ 0.01) # replace missing by a value
nm_c = findall(x->x==0,ismissing.(data.c))
nm_b = findall(x->x==0,ismissing.(data.b))

# take a look
names(data) # column names
describe(data)

show(data, true)

# parameters
g0 = 0.0 # bankruptcy cost

# function shock(c)
#     # draw shocks that produce the probability of default δ
#     a = 1
#     b = log.(data.delta)./(log.(1 .-data.w./c))
#     dist = Beta.(a,[b...])
#     draws = rand.(dist, 1)
#     vcat(draws...).*c
#     #rand(Float64,N) # normal
# end


## Optimization
rng =  Random.seed!(123)
dist = Beta(2,2)
x =  rand.(dist, 10)
σA = rand(rng,N,N); σA[diagind(σA)] .= 0
σb = fill(0.0,N)
σc = fill(0.0,N)

## fixed point
function contraction(p,x)
   A = (NNlib.hardtanh.(σA).+1)./2
   b = NNlib.sigmoid.(σb).*data.p_bar
   c = NNlib.sigmoid.(σc).*data.assets
   min.(data_nm.p_bar, max.((1+g0)*(A'*p .+ c .- x.*c) .- g0.*data_nm.p_bar,0))
end

contraction_iter(x, n::Integer) = n <= 0 ? data_nm.p_bar  : contraction(contraction_iter(x,n-1),x)

loss_x(x::Real) = -sum(contraction_iter(x,10)).*exp.(logpdf.(dist,x))

# Δx = 0.01
# gridx = 0:Δx:1
# function loss_int(σA,σb,σc)
#     A = (NNlib.hardtanh.(σA).+1)./2
#     b = NNlib.sigmoid.(σb).*data.p_bar
#     c = NNlib.sigmoid.(σc).*data.assets
#     sum(loss_x.(gridx,Ref(σA),Ref(σb),Ref(σc)).*exp.(logpdf.(dist,gridx))*Δx )
# end

function penalty(x)
   A = (NNlib.hardtanh.(σA).+1)./2
   b = NNlib.sigmoid.(σb).*data.p_bar
   c = NNlib.sigmoid.(σc).*data.assets
   con1 = sum(A,dims=2).*data.p_bar .- data.p_bar .+ b
   con2 = A' * data.p_bar .- data.assets .+ c
   con3 = sum(LowerTriangular(A).*LowerTriangular(transpose(A)))
   sum(abs2,con1)+sum(abs2,con2)+abs(con3)
end

function lossFlux(x)
   loss_x(x) + penalty()
end

ps =Flux.params(σA,σb,σc)
opt = Flux.ADAM(0.1)
cbP = function ()
   l = penalty(x)
   println("Current loss is: $l")
end
cbP()
@epochs 1 Flux.train!(penalty, ps, x, opt, cb = cbP)

Asol = (NNlib.hardtanh.(σA).+1)./2
bsol = NNlib.sigmoid.(σb).*data.p_bar
csol = NNlib.sigmoid.(σc).*data.assets
tol = 1e-6
@testset "check solution" begin
   @test norm( sum(Asol,dims=2).* data.p_bar .- (data.p_bar .- bsol)) < tol
   @test norm( Asol' * data.p_bar .- (data.assets .- csol)) < tol
   @test norm(diag(Asol)) < tol
   @test norm([Asol[i,j]*Asol[j,i] for i=1:N , j=1:N]) < tol
   @test all(0 .<=Asol.<=1)
   @test all(0 .<=bsol.<=data.p_bar)
   @test all(0 .<=csol.<=data.assets)
end

Gadfly.spy(Asol)

# Pkg.add("JLD")
# using JLD
# save("/home/ec2-user/SageMaker/Test-AWS/net_opt.jld", "Asol", Asol,"data",data)

## plot
Aplot = deepcopy(Asol)
Aplot[Aplot.<1e-3] .=0

# attributes here: https://docs.juliaplots.org/latest/generated/graph_attributes/
#method `:spectral`, `:sfdp`, `:circular`, `:shell`, `:stress`, `:spring`, `:tree`, `:buchheim`, `:arcdiagram` or `:chorddiagram`.


graphplot(LightGraphs.DiGraph(Aplot),
         nodeshape=:circle,
         markersize = 0.05,
         node_weights = data.assets,
         markercolor = range(colorant"yellow", stop=colorant"red", length=N),
         names = data.nm_short,
         fontsize = 8,
         linecolor = :darkgrey,
         edgewidth = (s,d,w)->5*Asol[s,d],
         arrow=true,
         method= :circular, #:chorddiagram,:circular,:shell
         )


## some tests
tol = 1e-6
x_test = fill(0.0,N,1)
pinf = nlsolve(p->contraction(p, x_test)-p, data_nm.p_bar, autodiff = :forward)
@test norm(contraction_iter(x_test,100)-pinf.zero)<tol

Zygote.gradient(c -> -sum(contraction(data_nm.p_bar,x_test)),data_nm.c)
ForwardDiff.gradient(c -> -sum(contraction(data_nm.p_bar,x_test)),data_nm.c)

Zygote.gradient(c -> -sum(contraction_iter(x_test,20)),data_nm.c)
ForwardDiff.gradient(c -> -sum(contraction_iter(x_test,20)),data_nm.c)

Zygote.gradient(loss_x(x_test),data_nm.c)
Zygote.gradient(c -> -sum(contraction_iter(x0,3,c).*exp.(logpdf.(dist,0.0))),data_nm.c)

loss(data_nm.c)
zy=Zygote.gradient(loss, data_nm.c)
fd=ForwardDiff.gradient(loss, data_nm.c)
