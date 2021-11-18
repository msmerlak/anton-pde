using DifferentialEquations
using DiffEqOperators
using Plots
using QuadGK, SpecialFunctions
using LinearAlgebra, SparseArrays


v(α) = 2 * cot(2π * α)
δ(x, X) = x == 0. ? 1/X.step.hi : 0.
θ(x; β = 20) = 1/(1+exp(-β*x))


## exact solution

function integrand(x, y, ξ, α)
    return (1 / 2π) *
           besselk(0, sqrt((x + ξ)^2 + y^2) * sqrt(1 + v(α)^2 / 4)) *
           exp(-v(α) * (x + ξ) / 2)
end

function exact_sol(x, y; α = 0.2)
    integral, error = quadgk(ξ -> integrand(x, y, ξ, α), 0, 500)
    return integral
end

initial_condition(x, y) = 0.5*heaviside(-x)*exp(-abs(y))

function Δ(X, Y)

	D2x = CenteredDifference(2, 4, X.step.hi, X.len)
	D2y = CenteredDifference(2, 4, Y.step.hi, Y.len)
	Qx = Neumann0BC(X.step.hi)
	Qy = Neumann0BC(Y.step.hi)
	
	laplacian = kron(sparse(I, X.len, X.len), sparse(D2y * Qy)[1]) + kron(sparse(D2x * Qx)[1], sparse(I, Y.len, Y.len))

	return laplacian
end

function ∂x(X, Y)

	Dx = CenteredDifference(1, 2, X.step.hi, X.len)
	Qx = Neumann0BC(X.step.hi)
	
	Dx = kron(sparse(I, Y.len, Y.len), sparse(Dx * Qx)[1])
	return Dx
end


function L(X, Y, α = 0.1)
    return Δ(X, Y) - I + v(α) .* ∂x(X, Y)
end

function S(X, Y)
    return vec(transpose(reduce(hcat, fill(map(y->δ(y, Y), Y), length(X)))))
end

P(X, Y, α, β) = Dict(
    :α => α, # threshold
    :β => β, # steepness parameter
    :L => L(X, Y, α), # linear operator
    :S => S(X,Y) # sources
)

function f!(du, u, p, t)
    @unpack α, β, L, S = p
    mul!(du, L, u)
    @. du += θ(u - α; β = β)*S
end

function f(u, p)
    du = similar(u)
    f!(du, u, p, 0.)
    return du
end
