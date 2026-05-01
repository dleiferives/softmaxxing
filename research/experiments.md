 - Manual parameter loop unrolling
   - In SR train it like with N wide loop unrolling (by passing in next N) -> see performance tradeoff
 - Train ai model / SR to predict performance
 - Reduced instructions & depth (possibly give defaults like x\*x or x - m)
   1) We pass in things that we expect it to use / behaviours we expect such as `x*x` and `x-m`
   2) We reduce the total depth allows 1 -> N -> see performance tradeoff
   3) We change allowed instructions -> see performance tradoff
   4) We change maximum equation size -> see performance tradeoff
 - Do other forms of search such as monte carlo &| grad decent
   - come up with different eq forms -> grad decent to optimze them? for some set of functions $s = f(x,m,s,a1,a2...)$
     1) $f = (a1 * x) + (a2 * (x - m)) + (a3 * x * x) + s
     2) $f = (a1 * (x - m)) + (a2 * x * x) + (a3 * s)
