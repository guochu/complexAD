push!(LOAD_PATH, "src")

using Zygote
using Zygote: @adjoint

# import VQC: collect_variables_impl!, set_parameters_impl!

using Util: check_gradient 
using LinearAlgebra: norm, I



isunitary(x::AbstractMatrix) = isapprox(x*x', I)

random_hermitian(L::Int) = begin
    m = randn(Complex{Float64}, L, L)
    return m + m'
end
random_unitary(L::Int) = exp(im * random_hermitian(L))


λ = 0.01
L = 4

W = random_unitary(L)

m1 = randn(Complex{Float64}, L, L)
m2 = randn(Complex{Float64}, L, L)

loss(x::AbstractMatrix) = norm((x + m1) * m2)

println("gradient is corrent? $(check_gradient(loss, W, verbose=1)).")

println("W is unitary? $(isunitary(W)).")

grad = gradient(loss, W)[1]

println(isunitary(grad))

A = W * grad' - W' * grad

W1 = (I + (λ/2) * A) \ (I - (λ/2) * A) * W

println(W1' * W1)




