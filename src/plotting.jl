using Plots
function plot_sol(sol, X, Y)


    # plot exact solution
    ex = plot(X, Y, (x, y) -> exact_sol(x, y), 
    linetype = :heatmap, clims = (0, 0.5), title = "exact")

    # animate numerical solution
    num = @animate for i = 1:50:length(sol.t)
        t = sol.t[i]
        s = reshape(sol.u[i], length(X), length(Y))
        plot(
            heatmap(X, Y, transpose(s),
        clims = (0, 0.5),
        title = "numerical"),
        ex
        )
    end
    gif(num, plotsdir("traveling-wave.gif"))
end