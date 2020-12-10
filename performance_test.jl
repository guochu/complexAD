
using Zygote
using Zygote: @adjoint
using BenchmarkTools
import Base.+


multiply(z::Complex, w::Complex) = z * w

# our approach
adjoint_complex_ad(z::Complex, w::Complex, nu::Complex) = (nu * conj(w), nu * conj(z))

# second approach: treating complex numbers as tuples
adjoint_tuple(az::Real, bz::Real, aw::Real, bw::Real, u::Real, v::Real) = ((u * aw + v * bw, 
	v*aw - u * bw), (u * az + v * bz, v*aw - u*bw))


# third approach: defined a struct for complex numbers, but the complex arithmatic 
# is internally done by reverting to real numbers
struct complex_number
	r::Float64
	i::Float64
end

real_part(x::complex_number) = x.r
imag_part(x::complex_number) = x.i

+(a::complex_number, b::complex_number) = complex_number(real_part(a) + real_part(b), imag_part(a) + imag_part(b))

@adjoint real_part(x::complex_number) = real_part(x), z -> (complex_number(z, 0.), )
@adjoint imag_part(x::complex_number) = imag_part(x), z -> (complex_number(0., z), )
@adjoint complex_number(r::Float64, i::Float64) = complex_number(r, i), z -> (real_part(z), imag_part(z))

multiply_struct(z::complex_number, w::complex_number) = complex_number(real_part(z) * real_part(w) - imag_part(z) * imag_part(w),
real_part(z) * imag_part(w) + imag_part(z) * real_part(w))

function benchmark_test()
	az = 1.3
	bz = 2.0
	aw = 1.7
	bw = 2.3
	z = az + bz*im
	w = aw + bw*im

	u = 0.3
	v = 1.7
	nu = u + v*im

	z2 = complex_number(az, bz)
	w2 = complex_number(aw, bw)
	nu2 = complex_number(u, v)

	# v, f1 = Zygote.pullback(multiply, z, w)

	@btime adjoint_complex_ad($z, $w, $nu)
	@btime adjoint_tuple($az, $bz, $aw, $bw, $u, $v)

	r, back = Zygote.pullback(multiply_struct, z2, w2)

	@btime $back($nu2)
end

benchmark_test()
