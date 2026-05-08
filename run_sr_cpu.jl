using Pkg
Pkg.add(["SymbolicRegression"])

using SymbolicRegression
using Random
using Serialization
using Dates
using Printf
using Statistics

# ── Config ────────────────────────────────────────────────────────────────────
const DATASET_SIZE  = 100_000
const REFRESH_EVERY = 1
const REFRESH_FRAC  = 0.15
const N_OUTER       = 100_000
const SAVE_EVERY    = 30

# CPU speed penalty weight.
# Stronger than GPU version (2e-4) because CPU throughput is the goal and we
# have solid empirical evidence that tree size dominates CPU speed.
# Weight is ~10% of typical good-equation MAE (~0.001) so accuracy still leads.
const CPU_SPEED_WEIGHT = 1f-4
# ─────────────────────────────────────────────────────────────────────────────

# ── tree helpers ──────────────────────────────────────────────────────────────
# Variable indices: 1=m, 2=s, 3=x  (matches variable_names = ["m","s","x"])
# Operator indices: 1=+, 2=-, 3=*, 4=>  (binary_operators = [+,-,*,>])

function _count_nodes(node)::Int
    node.degree == 0 && return 1
    node.degree == 1 && return 1 + _count_nodes(node.l)
    return 1 + _count_nodes(node.l) + _count_nodes(node.r)
end

function _count_op!(node, op_idx::Int, acc::Ref{Int})
    node.degree == 0 && return
    node.degree >= 1 && (node.op == op_idx && (acc[] += 1); _count_op!(node.l, op_idx, acc))
    node.degree == 2 && _count_op!(node.r, op_idx, acc)
end

function _collect_feat_depths!(node, feat::Int, depth::Int, depths::Vector{Float32})
    if node.degree == 0
        (!node.constant && node.feature == feat) && push!(depths, Float32(depth))
        return
    end
    _collect_feat_depths!(node.l, feat, depth + 1, depths)
    node.degree == 2 && _collect_feat_depths!(node.r, feat, depth + 1, depths)
end

function _collect_var_depths!(node, depth::Int, depths::Vector{Float32})
    if node.degree == 0
        node.constant || push!(depths, Float32(depth))
        return
    end
    _collect_var_depths!(node.l, depth + 1, depths)
    node.degree == 2 && _collect_var_depths!(node.r, depth + 1, depths)
end

function _collect_ops!(node, ops::Set{Int})
    node.degree == 0 && return
    push!(ops, node.op)
    _collect_ops!(node.l, ops)
    node.degree == 2 && _collect_ops!(node.r, ops)
end

function _count_x_products(node, feat::Int = 3)
    node.degree == 0 && return 0
    node.degree == 1 && return _count_x_products(node.l, feat)
    c = _count_x_products(node.l, feat) + _count_x_products(node.r, feat)
    if node.degree == 2 && node.op == 3
        l_has = _subtree_has_feat(node.l, feat)
        r_has = _subtree_has_feat(node.r, feat)
        l_has && r_has && (c += 1)
    end
    return c
end

function _subtree_has_feat(node, feat::Int)
    node.degree == 0 && return !node.constant && node.feature == feat
    _subtree_has_feat(node.l, feat) ||
        (node.degree == 2 && _subtree_has_feat(node.r, feat))
end

# ── CPU speed penalty ─────────────────────────────────────────────────────────
# Returns score in [0, 1]. Higher = predicted slower on CPU.
#
# Empirical basis (cpu/analyze_cpu_direct.py, correlations with f32_gcps):
#   tree_height        r = -0.191  deep trees -> long dep chains -> less ILP
#   all_var_depth_std  r = -0.210  unbalanced variable layout -> irregular SIMD
#   x_avg_depth        r = -0.186  x buried deep -> late values -> stalls
#   n_nodes: light penalty (theoretical: more ops = more cycles)
#   x^2 bonus: polynomial structure with x*x -> better ILP

function cpu_speed_penalty(tree_or_expr)::Float32
    # Newer SR wraps Node in Expression; unwrap to get the raw Node tree.
    tree = hasproperty(tree_or_expr, :degree) ? tree_or_expr : tree_or_expr.tree

    function _height(n)::Int
        n.degree == 0 && return 0
        n.degree == 1 && return 1 + _height(n.l)
        return 1 + max(_height(n.l), _height(n.r))
    end
    h = Float32(_height(tree))

    x_depths = Float32[]
    _collect_feat_depths!(tree, 3, 0, x_depths)
    x_avg_depth = isempty(x_depths) ? 0f0 : mean(x_depths)

    all_depths = Float32[]
    _collect_var_depths!(tree, 0, all_depths)
    depth_std = length(all_depths) > 1 ? std(all_depths) : 0f0

    n    = Float32(_count_nodes(tree))
    x_sq = Float32(_count_x_products(tree))

    score = 0.30f0 * min(h           / 12f0, 1f0)   # height (r = -0.191)
          + 0.30f0 * min(depth_std   /  4f0, 1f0)   # imbalance (r = -0.210)
          + 0.25f0 * min(x_avg_depth /  8f0, 1f0)   # x depth (r = -0.186)
          + 0.15f0 * min(n           / 40f0, 1f0)   # size (theoretical)
          - 0.05f0 * min(x_sq        /  2f0, 1f0)   # x^2 bonus

    return max(score, 0f0)
end

# ── loss ──────────────────────────────────────────────────────────────────────

function custom_loss(tree, dataset, options)
    prediction, is_valid = eval_tree_array(tree, dataset.X, options)
    !is_valid && return Inf32
    mae = mean(abs.(prediction .- dataset.y))
    return mae + CPU_SPEED_WEIGHT * cpu_speed_penalty(tree)
end

# ── boilerplate ───────────────────────────────────────────────────────────────

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
    tag = iter < 0 ? "final" : lpad(iter, 6, '0')
    serialize(joinpath(run_dir, "model_$(tag).jls"), result)
end

function save_hof(result, run_dir::String, iter::Int, options)
    hof        = result[2]
    dominating = calculate_pareto_frontier(hof)
    isempty(dominating) && return nothing
    tag = iter < 0 ? "final" : lpad(iter, 6, '0')
    open(joinpath(run_dir, "hof_$(tag).csv"), "w") do f
        println(f, "complexity,loss,equation")
        for m in dominating
            c = compute_complexity(m, options)
            println(f, "$(c),$(m.loss),\"$(m.tree)\"")
        end
    end
end

const CLR   = "\e[2K\r"
const BOLD  = "\e[1m"
const CYAN  = "\e[36m"
const GREEN = "\e[32m"
const YEL   = "\e[33m"
const RST   = "\e[0m"

function print_header(run_dir::String)
    println(BOLD * "━"^64 * RST)
    println(BOLD * "  SymbolicRegression  —  CPU-throughput-guided search" * RST)
    println("  run dir    : " * CYAN * run_dir * RST)
    println("  dataset    : $(DATASET_SIZE) rows | refresh $(Int(REFRESH_FRAC*100))% every $(REFRESH_EVERY) iters")
    println("  cpu_speed_weight: $(CPU_SPEED_WEIGHT)  (size×0.50, greater×0.25, x_depth×0.15, ops×0.10)")
    println("  saving     : every $(SAVE_EVERY) iters + on Ctrl-C")
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
    best = last(dominating)
    print(CLR *
          BOLD * @sprintf("iter %6d", i) * RST * "  " *
          @sprintf("loss=%.4g  spd=%.3f", best.loss, cpu_speed_penalty(best.tree)) * "  " *
          CYAN * string(best.tree) * RST *
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
            c = compute_complexity(m, options)
            @printf("  [%2d]  loss=%-12.5g  cpu_spd=%.3f  %s\n",
                    c, m.loss, cpu_speed_penalty(m.tree), m.tree)
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
        maxsize          = 35,           # tighter than GPU (40) → smaller trees → faster CPU
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
