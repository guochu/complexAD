push!(LOAD_PATH, "src")

import Base.*
using Zygote
using Zygote: @adjoint

import Util: collect_variables_impl!, set_parameters_impl!

using Util: simple_gradient, check_gradient, collect_variables 
using LinearAlgebra: Diagonal, norm
using Random: randperm
using FFTW


times_diag(input::AbstractMatrix, v::AbstractVector) = Diagonal(exp.(-im*v)) * input

# times_diag(input::AbstractMatrix, v::AbstractVector) = _times_diag(input, v)
# @adjoint times_diag(input::AbstractMatrix, v::AbstractVector) = begin
# 	r, f = Zygote.pullback(_times_diag, input, v)
# 	return r, z -> begin
# 	    a, b = f(z)
# 	    return a, real.(b)
# 	end
# end

vec_permutation(input::AbstractMatrix, perm::Vector{Int}) = input[perm, :]

# function times_reflection(input::AbstractMatrix, reflection::AbstractVector)
# 	L = length(reflection)
# 	n = norm(reflection)
# 	mat = reshape(kron(conj(reflection), reflection), L, L)
# 	return input - (2/(n^2)) * (mat * input)
# end

function times_reflection(input::AbstractMatrix, reflection::AbstractVector)
	n = norm(reflection)
	return input - (2/(n^2)) * reshape(kron(reflection, transpose(input) * conj(reflection)), size(input)...)
end

function apply_w_matrix(input, D1, D2, D3, R1, R2, perm)
	step1 = times_diag(input, D1)
	step2 = fft(step1, (1,))
	step3 = times_reflection(step2, R1)
	step4 = vec_permutation(step3, perm)
	step5 = times_diag(step4, D2)
	step6 = ifft(step5, (1,))
	step7 = times_reflection(step6, R2)
	step8 = times_diag(step7, D3)
	return step8
end


struct WMatrix 
	D1::Vector{Float64}
	D2::Vector{Float64}
	D3::Vector{Float64}
	R1::Vector{Complex{Float64}}
	R2::Vector{Complex{Float64}}
	perm::Vector{Int}
end

collect_variables_impl!(a::Vector, b::WMatrix) = begin
    collect_variables_impl!(a, b.D1)
    collect_variables_impl!(a, b.D2)
    collect_variables_impl!(a, b.D3)
    collect_variables_impl!(a, b.R1)
    collect_variables_impl!(a, b.R2)
end

function set_parameters_impl!(m::WMatrix, b::Vector, pos::Int) 
	pos = set_parameters_impl!(m.D1, b, pos)
	pos = set_parameters_impl!(m.D2, b, pos)
	pos = set_parameters_impl!(m.D3, b, pos)
	pos = set_parameters_impl!(m.R1, b, pos)
	return set_parameters_impl!(m.R2, b, pos)
end

get_variables(m::WMatrix) = m.D1, m.D2, m.D3, m.R1, m.R2

@adjoint get_variables(m) = begin
    r = get_variables(m)
    return r, z -> begin
        D1, D2, D3, R1, R2 = z
        return (WMatrix(real(D1), real(D2), real(D3), R1, R2, m.perm), )
    end 
end

get_perm(m::WMatrix) = m.perm
@adjoint get_perm(m::WMatrix) = get_perm(m), z->(nothing,)

function WMatrix(L::Int) 
	D1 = randn(L)
	D2 = randn(L)
	D3 = randn(L)	
	R1 = randn(Complex{Float64}, L)
	R2 = randn(Complex{Float64}, L)
	perm = randperm(L)
	WMatrix(D1,D2,D3,R1,R2,perm)
end

*(m::WMatrix, input::AbstractMatrix) = begin
	D1, D2, D3, R1, R2 = get_variables(m)
    step1 = times_diag(input, D1)
 	step2 = fft(step1, (1,))
	step3 = times_reflection(step2, R1)
	step4 = vec_permutation(step3, get_perm(m))
	step5 = times_diag(step4, D2)
	step6 = ifft(step5, (1,))
	step7 = times_reflection(step6, R2)
	step8 = times_diag(step7, D3)   
	return step8
end



N = 10
L = 5

input = randn(L, N)

m = WMatrix(L)

loss(x::WMatrix) = abs(sum(x * input))

grad = gradient(loss, m)

println(check_gradient(loss, m, dt=1.0e-8, atol=1.0e-4; verbose=1))



