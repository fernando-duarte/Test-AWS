using Pkg
Pkg.add("XLSX")

import DataFrames, XLSX
xf = XLSX.readxlsx("node_stats_for_simulation.xlsx") #xf["BHCs"]["A1:I1"]
bhc = XLSX.eachtablerow(xf["BHCs"]) |> DataFrames.DataFrame
describe(bhc)
names(bhc)

# rescale units
scale = 1e6;
bhc[:,[:w, :c, :assets, :p_bar, :b]] .= bhc[!,[:w, :c, :assets, :p_bar, :b]]./scale



#=

# network primitives
delta is probability of default
beta is connectedness
w is net worth
c is outside assets
a is total assets
p_bar is total liabilities
b is outside liabilities

# other network variables
a = w + p_bar; # total assets
p_bar = a  - w; # total liabilities
d = a -c; # inside assets
f = p_bar -b; # inside liabilities
N_tot = length(c); # number of nodes
type = repmat(i,N_tot ,1); # type of firm

=#

y=0;
p_bar = [55+y 55+y 140 55+y 55+y]; # tot liab
A0 = [0 y/p_bar(1) 0 0 0; 0 0 0 0 y/p_bar(1); 1/14 1/14 0 1/14 1/14; y/p_bar(4) 0 0 0 0; 0 0 0 y/p_bar(5) 0];
c = [50 50 150 50 50]; # outside assets
w = [5 5 10 5 5]; # net worth
g = 0.0;

b = [55 55 100 55 55]; # outside liabilities
a = w+p_bar; # total assets
d=  a-c;# inside assets
f = p_bar-b;# inside liabilities
N = length(c); # number of nodes

struct Params
         data::Dict{String, Any}
       end

Base.getproperty(p::Params, name::Symbol) = getfield(p, :data)[String(name)]

p = Params(Dict("foo" => 1, "bar" => [1,2,3]))
Params(Dict{String,Any}("bar"=>[1, 2, 3],"foo"=>1))

julia> p.foo
1

julia> p.bar
3-element Array{I


for i=1:nsheets
        [num ,txt ,raw ] = xlsread('node_stats_for_simulation.xlsx',sheets );
        headers  = raw (1,:);
        names  = raw (2:end,1);

        # Network primitives
#       p_bar  = vertcat(raw {2:end,strcmpi('p_bar',headers )}); # total liabilities
        c  = vertcat(raw {2:end,strcmpi('c',headers )}) ; # outside assets
        b  = vertcat(raw {2:end,strcmpi('b',headers )}); # outside liabilities
        w  =vertcat(raw {2:end,strcmpi('w',headers )}); # net worth
        delta  =vertcat(raw {2:end,strcmpi('delta',headers )}); # probability of default
        a  = vertcat(raw {2:end,strcmpi('assets',headers )});  # total assets

        # Other network variables
#         a  = w +p_bar ; # total assets
        p_bar  =a  - w ; # total liabilities
        d =  a -c ;# inside assets
        f  = p_bar -b ;# inside liabilities
        N_tot  = length(c ); # number of nodes
        type  = repmat(i,N_tot ,1); # type of firm
        save('network_data')
    end
