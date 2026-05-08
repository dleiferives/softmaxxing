using Pkg
Pkg.add(["SymbolicRegression"])

using SymbolicRegression
using Random
using Serialization
using Dates
using Printf
using Statistics


function gen_chunk1(n::Int)
    x = Vector{Float32}(undef, n)
    y = Vector{Float32}(undef, n)
    for i in 1:n
        x[i]     = randn(Float32)
	y[i]     = 1 / sqrt(x[i])
    end
    return x, y
end


function IADD(x::Float32, y::Float32)
	xi = reinterpret(Int32, x)
	yi = reinterpret(Int32, y)
	zi = xi + yi
	zf = reinterpret(Float32, zi)
	return zf
end


function gen_chunk(n::Int)
    x = Float32[]
    y = Float32[]

    for _ in 1:n
        xi = randn(Float32)
        if isfinite(xi) && xi > 0f0
            push!(x, xi)
            push!(y, 1f0 / sqrt(xi))
        end
    end

    return x, y
end


function gen_chunk_IADD(n::Int)
    x = Float32[]
    y = Float32[]

    for _ in 1:n
        xi = rand(Int32)
	xf = reinterpret(Float32, xi)
	yi = IADD(xf,xf)
	push!(x, xf)
	push!(y, yi)
    end

    return x, y
end

function main()

    options = Options(
        binary_operators = [IADD],
        # unary_operators  = [abs],
        maxsize          = 40,
        batching         = true,
        batch_size       = 1000,
        verbosity        = 1,
        progress         = true,
    )

    x, y   = gen_chunk_IADD(10000)
    X = hcat(x)'

    result = equation_search(
	X, y;
	niterations    = 40,
	options        = options,
	return_state   = true,
	variable_names = ["x"],
    )

end

main()
