using DrWatson
using Plots
include(srcdir("anton-pde.jl"))

p = Dict(
    :L => 10.,
    :T => 100.,
    :dx => 1.
    ) 

sol = solver(p)


ex = plot(sol.x, sol.y, (x, y) -> exact_sol(x, y), linetype = :contourf, title = "exact")

num = @animate for i = 1:length(sol.w)
    t = sol.t[i]
    plot(ex,
    contourf(sol.w[i], aspect_ratio = 1, title = "numerical, t = $t")
    )
end

contourf(sol.w[i], aspect_ratio = 1, title = "numerical, t = $t")
gif(num, plotsdir("traveling-wave.gif"))


contours(sol.x, sol.y, sol.w[1])

