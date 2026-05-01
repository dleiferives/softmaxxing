using Pkg
Pkg.add("SymbolicRegression")
using SymbolicRegression
using Random

const DATASET_SIZE  = 10_000
const REFRESH_EVERY = 10
const REFRESH_FRAC  = 0.20
const N_OUTER       = 100_000

function gen_chunk(n::Int)
    m, s = 0.0, 0.0
    X = Matrix{Float32}(undef, 3, n)
    y = Vector{Float32}(undef, n)
    for i in 1:n
        x = randn(Float32)
        m_new = max(m, x)
        if m_new > m
            s *= exp(m - m_new)
            m = m_new
        end
        s_new = s + exp(x - m)
        X[:, i] = [m, s, x]
        y[i] = s_new
        s = s_new
    end
    return X, y
end

function refresh(X, y)
    n_new = round(Int, DATASET_SIZE * REFRESH_FRAC)
    X_new, y_new = gen_chunk(n_new)
    return hcat(X[:, (n_new+1):end], X_new),
           vcat(y[(n_new+1):end],    y_new)
end

function main()
    options = Options(
        binary_operators = [+, -, *],
        unary_operators  = [abs],
        maxsize          = 20,
        elementwise_loss = (pred, target) -> abs(pred - target),
        batching         = true,
        batch_size       = 1000,
        verbosity        = 0,
    )

    X, y = gen_chunk(DATASET_SIZE)
    println("✓ Initial dataset: $DATASET_SIZE rows")

    state = nothing

    for i in 1:N_OUTER
        hof, state = equation_search(
            X, y;
            niterations = 1,
            options     = options,
            saved_state = state,
            stateReturn = true,
            varMap      = ["m", "s", "x"],
        )

        if i % REFRESH_EVERY == 0
            X, y = refresh(X, y)
            best = hof.members[end]
            println("[iter $i] refreshed | loss=$(round(best.score, digits=6)) | $(string_tree(best.tree, options))")
        end
    end
end

main()
