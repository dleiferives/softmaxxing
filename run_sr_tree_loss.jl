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

# Small speed nudge — only differentiates at the accuracy frontier.
# Keeps MAE as the dominant signal; this just breaks ties toward faster forms.
# Features (from feature/throughput correlation analysis at ±30% loss threshold):
#   penalise: deep x occurrences, high variance in var depths, const-heavy leaves
#   penalise: many distinct operator types
# Weight chosen so the penalty is <5% of a typical good-equation MAE (~0.003).
const SPEED_WEIGHT = 2f-4
# ─────────────────────────────────────────────────────────────────────────────

# ── tree feature helpers ──────────────────────────────────────────────────────
# Variable indices: 1=m, 2=s, 3=x

# Collect depths of all variable leaves (any feature) into `depths`.
function _collect_var_depths!(node, depth::Int, depths::Vector{Float32})
    if node.degree == 0
        node.constant || push!(depths, Float32(depth))
        return
    end
    _collect_var_depths!(node.l, depth + 1, depths)
    node.degree == 2 && _collect_var_depths!(node.r, depth + 1, depths)
end

# Collect depths of occurrences of a specific feature index.
function _collect_feat_depths!(node, feat::Int, depth::Int, depths::Vector{Float32})
    if node.degree == 0
        (!node.constant && node.feature == feat) && push!(depths, Float32(depth))
        return
    end
    _collect_feat_depths!(node.l, feat, depth + 1, depths)
    node.degree == 2 && _collect_feat_depths!(node.r, feat, depth + 1, depths)
end

# Count (n_var_leaves, n_const_leaves).
function _count_leaf_types(node)
    node.degree == 0 && return node.constant ? (0, 1) : (1, 0)
    lv, lc = _count_leaf_types(node.l)
    node.degree == 1 && return (lv, lc)
    rv, rc = _count_leaf_types(node.r)
    return (lv + rv, lc + rc)
end

# Count distinct operator indices used anywhere in the tree.
function _collect_ops!(node, ops::Set{Int})
    node.degree == 0 && return
    push!(ops, node.op)
    _collect_ops!(node.l, ops)
    node.degree == 2 && _collect_ops!(node.r, ops)
end

# Count x*x-style products (x appears on both sides of a multiply).
# Proxy for polynomial degree in x — higher degree correlates with faster kernels.
function _count_x_products(node, feat::Int = 3)
    node.degree == 0 && return 0
    node.degree == 1 && return _count_x_products(node.l, feat)
    c = _count_x_products(node.l, feat) + _count_x_products(node.r, feat)
    if node.degree == 2 && node.op == 3   # op index 3 = * (binary_operators = [+,-,*,>])
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


# ── speed penalty ─────────────────────────────────────────────────────────────
# Returns a score in roughly [0, 1]. Higher = predicted slower.
# Each sub-term is normalised so it saturates near 1 for "very bad" values.

function speed_penalty(tree)::Float32
    # ── Feature 1: average depth of x occurrences (r = -0.59) ────────────────
    x_depths = Float32[]
    _collect_feat_depths!(tree, 3, 0, x_depths)
    x_avg_depth = isempty(x_depths) ? 0f0 : mean(x_depths)

    # ── Feature 2: std of all variable depths (r = -0.47) ────────────────────
    all_depths = Float32[]
    _collect_var_depths!(tree, 0, all_depths)
    var_depth_std = length(all_depths) > 1 ? std(all_depths) : 0f0

    # ── Feature 3: fraction of leaves that are constants (r = -0.56) ─────────
    n_var, n_const = _count_leaf_types(tree)
    n_leaves = n_var + n_const
    frac_const = n_leaves > 0 ? Float32(n_const / n_leaves) : 0.5f0

    # ── Feature 4: number of distinct operator types (r = -0.44) ─────────────
    ops = Set{Int}()
    _collect_ops!(tree, ops)
    n_ops = Float32(length(ops))

    # ── Feature 5: x polynomial degree proxy (r = +0.40 → reward) ────────────
    x_sq = Float32(_count_x_products(tree))
    x_poly_bonus = min(x_sq / 2f0, 1f0)   # saturates at 2 x-products

    # ── Weighted combination ──────────────────────────────────────────────────
    score = 0.28f0 * min(x_avg_depth  / 8f0, 1f0)   # x depth penalty
          + 0.20f0 * min(var_depth_std / 4f0, 1f0)   # depth variance penalty
          + 0.30f0 * frac_const                        # const-leaf penalty
          + 0.14f0 * min(n_ops        / 5f0, 1f0)    # op-variety penalty
          - 0.08f0 * x_poly_bonus                     # x² reward

    return max(score, 0f0)
end

# ── loss ──────────────────────────────────────────────────────────────────────

function custom_loss(tree, dataset, options)
    prediction, is_valid = eval_tree_array(tree, dataset.X, options)
    !is_valid && return Inf32
    mae = mean(abs.(prediction .- dataset.y))
    return mae + SPEED_WEIGHT * speed_penalty(tree)
end

# ── boilerplate (unchanged from run_sr.jl) ────────────────────────────────────

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
    serialize(joinpath(run_dir, "model_$(tag).jls"), result)
end

function save_hof(result, run_dir::String, iter::Int, options)
    hof        = result[2]
    dominating = calculate_pareto_frontier(hof)
    isempty(dominating) && return nothing
    tag  = iter < 0 ? "final" : lpad(iter, 6, '0')
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
    println(BOLD * "  SymbolicRegression  —  softmax + speed-aware loss" * RST)
    println("  run dir    : " * CYAN * run_dir * RST)
    println("  dataset    : $(DATASET_SIZE) rows | refresh $(Int(REFRESH_FRAC*100))% every $(REFRESH_EVERY) iters")
    println("  speed_weight: $(SPEED_WEIGHT)  (x_depth, var_depth_std, frac_const, n_ops, -x_poly)")
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
          @sprintf("loss=%.4g", best.loss) * "  " *
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
            @printf("  [%2d]  loss=%-12.5g  spd=%.3f  %s\n",
                    c, m.loss, speed_penalty(m.tree), m.tree)
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
