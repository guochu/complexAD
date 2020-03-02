
push!(LOAD_PATH, "src")

using Zygote
using Zygote: @adjoint
using FFTW
using LinearAlgebra: dot
using Util

# Examples of adjoint functions from some common complex functions
# For most of the following functions, Zygote default adjoint functions 
# are correct, but for non-holomorphic ones, such as dot, Zygote may return incorrect 
# gradients currectly.
@adjoint sin(z::Number) = sin(z), v -> (v * cos(conj(z)),)
@adjoint exp(z::Number) = exp(z), v -> (v * exp(conj(z)),)
@adjoint log(z::Number) = log(z), v -> (v / conj(z),)
@adjoint +(z::Number, w::Number) = z + w, v -> (v, v)
@adjoint *(z::Number, w::Number) = z * w, v -> (v*conj(w), v*conj(z))
@adjoint /(z::Number, w::Number) = z / w, v -> (v/conj(w), -v*conj(z)/conj(w)^2)
@adjoint *(z::Number, w::AbstractVector) = z * w, v -> (dot(w, v), v .* conj(z))
@adjoint read(z::Number) = real(z), v -> (real(v),)
@adjoint imag(z::Number) = imag(z), v -> (im*real(v),)
@adjoint abs(z::Number) = abs(z), v -> (real(v)*z/abs(z),)
@adjoint dot(z::AbstractArray, w::AbstractArray) = dot(z, w), v -> (conj(v) .* w, v .* z)
outer(z::AbstractVector, w::AbstractVector) = reshape(kron(w, z), length(z), length(w))
@adjoint outer(z::AbstractVector, w::AbstractVector) = outer(z, w), v -> (v*conj(w), z'*v)
@adjoint *(z::AbstractMatrix, w::AbstractMatrix) = z*w, v -> (v * w', z' * v)
@adjoint fft(z::AbstractVector) = fft(z), v -> (ifft(v) .* length(z),)
@adjoint ifft(z::AbstractVector) = ifft(z), v -> (fft(v) ./ length(z),)


random_hermitian(L::Int) = begin
    m = randn(Complex{Float64}, L, L)
    return m + m'
end

function check_dot()
	m = random_hermitian(5)
	loss(x::AbstractArray) = real(dot(x, m * x) / dot(x, x))
	return check_gradient(loss, randn(Complex{Float64}, 5))
end

function check_outer()
	m = randn(Complex{Float64}, 5)
	loss(x::AbstractVector) = real(sum(outer(x, m)))
	return check_gradient(loss, randn(Complex{Float64}, 5))
end


println(check_dot())
println(check_outer())

