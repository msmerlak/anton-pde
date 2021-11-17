using DrWatson
@quickactivate

include(srcdir("anton-pde.jl"))
include(srcdir("plotting.jl"))


dx = 0.05
dy = 0.05
Lx = 5
Ly = 5
X, Y = (-Lx:dx:Lx, -Ly:dx:Ly)

trange = (0., 10.)

jac_prototype = L(X,Y)
p = P(X, Y, .1)

u0 = vec([initial_condition(x, y) for x in X, y in Y])

prob = ODEProblem(
    ODEFunction(f!, jac_prototype = L(X,Y)), 
    u0, 
    trange, p
    )

sol = solve(prob, BS3())

@gif for i in 1:40:length(sol.t)
    heatmap(X, Y, transpose(sol.u[i]), title = "t = $(sol.t[i])", clims = (0, 1))
end


plot_sol(sol, X, Y)
