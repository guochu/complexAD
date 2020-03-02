
push!(LOAD_PATH, "src")

using Zygote
using Zygote: @adjoint

using LinearAlgebra: dot
using Util

# # correct gradient for dot
# @adjoint dot(a::AbstractArray, b::AbstractArray) = dot(a, b), z -> (conj(z) .* b, z .* a)

random_hermitian(L::Int) = begin
    m = randn(Complex{Float64}, L, L)
    return m + m'
end

m = random_hermitian(5)

loss(x::AbstractArray) = real(dot(x, m * x) / dot(x, x))

x = randn(Complex{Float64}, 5)

println(check_gradient(loss, x, verbose=1))