using Pkg
Pkg.add(["SymbolicRegression"])

using SymbolicRegression
using Random
using Serialization
using Dates
using Printf
using Statistics

# ── Config ────────────────────────────────────────────────────────────────────
const DATASET_SIZE    = 100_000
const REFRESH_EVERY   = 1
const REFRESH_FRAC    = 0.15
const N_OUTER         = 100_000
const SAVE_EVERY      = 30 

# Loss penalty weights — add more here as new terms are developed
const NESTING_PENALTY = 1f-4   # per unit of tree depth
# ─────────────────────────────────────────────────────────────────────────────

function tree_depth(node)
    node.degree == 0 && return 0
    node.degree == 1 && return 1 + tree_depth(node.l)
    return 1 + max(tree_depth(node.l), tree_depth(node.r))
end

function custom_loss(tree, dataset, options)
    prediction, is_valid = eval_tree_array(tree, dataset.X, options)
    !is_valid && return Inf32
    mae   = mean(abs.(prediction .- dataset.y))
    depth = Float32(tree_depth(tree))
    return mae + NESTING_PENALTY * depth
end

function make_run_dir()
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir   = joinpath("outputs", stamp)
    mkpath(dir)
    return dir
end

function gen_chunk(n::Int)
    m, s = 0f0, 0f0
    X = Matrix{Float32}(undef, 3, n)
    y = Vector{Float32}(undef, n)
    for i in 1:n
        x     = randn(Float32)
        m_new = max(m, x)
        if m_new > m
            s *= exp(m - m_new)
            m  = m_new
        end
        s_new  = s + exp(x - m)
        X[:,i] = [m, s, x]
        y[i]   = s_new
        s      = s_new
    end
    return X, y
end

function refresh(X::Matrix{Float32}, y::Vector{Float32})
    n_drop = round(Int, DATASET_SIZE * REFRESH_FRAC)
    X_new, y_new = gen_chunk(n_drop)
    return hcat(X[:, (n_drop+1):end], X_new),
           vcat(y[(n_drop+1):end],    y_new)
end

function save_state(result, run_dir::String, iter::Int)
    tag  = iter < 0 ? "final" : lpad(iter, 6, '0')
    path = joinpath(run_dir, "model_$(tag).jls")
    serialize(path, result)
    return path
end

function save_hof(result, run_dir::String, iter::Int, options)
    hof        = result[2]
    dominating = calculate_pareto_frontier(hof)
    isempty(dominating) && return nothing
    tag  = iter < 0 ? "final" : lpad(iter, 6, '0')
    path = joinpath(run_dir, "hof_$(tag).csv")
    open(path, "w") do f
        println(f, "complexity,loss,equation")
        for m in dominating
            c = compute_complexity(m, options)   # ← was m.complexity
            println(f, "$(c),$(m.loss),\"$(m.tree)\"")
        end
    end
    return path
end

# ANSI
const CLR   = "\e[2K\r"
const BOLD  = "\e[1m"
const CYAN  = "\e[36m"
const GREEN = "\e[32m"
const YEL   = "\e[33m"
const RST   = "\e[0m"

function print_header(run_dir::String)
    println(BOLD * "━"^64 * RST)
    println(BOLD * "  SymbolicRegression  —  softmax incremental" * RST)
    println("  run dir : " * CYAN * run_dir * RST)
    println("  dataset : $(DATASET_SIZE) rows | refresh $(Int(REFRESH_FRAC*100))% every $(REFRESH_EVERY) iters")
    println("  saving  : every $(SAVE_EVERY) iters + on Ctrl-C")
    println(BOLD * "━"^64 * RST)
end

function print_status(i::Int, result, refreshed::Bool, saved::Bool)
    hof        = result[2]
    dominating = calculate_pareto_frontier(hof)

    tags = ""
    refreshed && (tags *= YEL * " [↺data]" * RST)
    saved     && (tags *= GREEN * " [💾saved]" * RST)

    if isempty(dominating)
        print(CLR * BOLD * @sprintf("iter %6d", i) * RST * "  …warming up$(tags)")
        return
    end

    best   = last(dominating)
    eq_str = string(best.tree)
    print(CLR *
          BOLD * @sprintf("iter %6d", i) * RST * "  " *
          @sprintf("loss=%.4g", best.loss) * "  " *
          CYAN * eq_str * RST *
          tags)
end

function print_final(result, run_dir::String, options)
    hof        = result[2]
    dominating = calculate_pareto_frontier(hof)
    println("\n" * BOLD * "━"^64 * RST)
    println(BOLD * "  FINAL PARETO FRONT" * RST)
    println(BOLD * "━"^64 * RST)
    if isempty(dominating)
        println("  (no equations found)")
    else
        for m in dominating
            c = compute_complexity(m, options)   # ← was m.complexity
            @printf("  [%2d]  loss=%-12.5g  %s\n", c, m.loss, m.tree)
        end
    end
    println(BOLD * "━"^64 * RST)
    println("  outputs → " * CYAN * run_dir * RST * "\n")
end

function main()
    run_dir = make_run_dir()
    print_header(run_dir)

    options = Options(
        binary_operators = [+, -, *, >],
        unary_operators  = [abs],
        maxsize          = 40,
        loss_function    = custom_loss,
        batching         = true,
        batch_size       = 1000,
        verbosity        = 0,
        progress         = false,
    )

    X, y   = gen_chunk(DATASET_SIZE)
    result = nothing

    Base.exit_on_sigint(false)

    try
        for i in 1:N_OUTER
            result = equation_search(
                X, y;
                niterations    = 20,
                options        = options,
                saved_state    = result,
                return_state   = true,
                variable_names = ["m", "s", "x"],
            )

            refreshed = saved = false

            if i % REFRESH_EVERY == 0
                X, y      = refresh(X, y)
                refreshed = true
            end

            if i % SAVE_EVERY == 0
                save_state(result, run_dir, i)
                save_hof(result, run_dir, i, options)
                saved = true
            end

            print_status(i, result, refreshed, saved)
        end

    catch e
        e isa InterruptException || rethrow(e)
        println("\n" * YEL * "  ⚡ Ctrl-C — saving final state..." * RST)
    end

    if result !== nothing
        save_state(result, run_dir, -1)
        save_hof(result, run_dir, -1, options)
        print_final(result, run_dir, options)
    end
end

main()
