using ModelingToolkit, DiffEqOperators, DifferentialEquations, DomainSets
#using ModelingToolkit: Differential
#import ModelingToolkit: Interval, infimum, supremum
using SpecialFunctions, QuadGK, Cubature


using DrWatson

@parameters t, x, y
@variables w(..)
Dx = Differential(x)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)

heaviside(x) = 0.5 * (sign(x) + 1)
v(α) = 2 * cot(2π * α)
δ(x, ϵ = 1e-2) = exp(-x^2 / (2ϵ^2)) / sqrt(2π * ϵ^2)

eq(α) = Dt(w(t, x, y)) ~ Dxx(w(t, x, y)) + Dyy(w(t, x, y)) + heaviside(w(t,x,y) - α)* δ(y) - w(t, x, y) - v(α) * Dx(w(t, x, y)) 

sigmoid(x) = 1/(1+exp(-x))

function integrand(x, y, ξ, α)
       return (1/2π)*besselk(0, sqrt((x+ξ)^2 + y^2)*sqrt(1+v(α)^2/4))*exp(-v(α)*(x+ξ)/2)
end


function exact_sol(x,y ; α = .4)
       integral, error = quadgk(ξ -> integrand(x, y, ξ, α), 0, 100, rtol = 1e-4)
       return integral
end

function solver(p; α = .4)
       # Initial and boundary conditions
       L = p[:L]
       bcs = [w(t, L, y) ~ 0.,# for all t > 0
              w(t, -L, y) ~ 0.,# for all t > 0
              w(t, x, L) ~ 0.,# for all t > 0
              w(t, x, -L) ~ 0.,# for all t > 0
              w(0, x, y) ~ sigmoid(-x)
              ]

       # Space and time domains
       T = p[:T]
       domains = [t ∈ Interval(0.0, T),
              x ∈ Interval(-L, L),
              y ∈ Interval(-L, L)]

       # Method of lines discretization
       dx = dy = p[:dx]
       discretization = MOLFiniteDifference([x => dx, y => dy], t)

       @named pde_system = PDESystem(eq(α), bcs, domains, [t,x,y], [w(t, x, y)])
       prob = discretize(pde_system, discretization)

       # Solve ODE
       sol = solve(prob, ABDF2())
       
       xs, ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains[2:end]]
       return (x = xs, y = ys, w = map(s -> transpose(reshape(s, (length(xs)-2, length(ys)-2))), sol.u), t = sol.t)
end