So I'm now going to explain what I want to go towards:
  #1 constants are not mutated -> we just do the mcmc thing to solve for their best value. Or like maybe that is a mutation like solve all
  constants. and then it tries to solve all the constants as best it can. If a constant is added then we will do the mcmc solving immediately on it
  and on all constants.
  #2 Instead of randomly choosing a register or even having the ideas of registers really we want each instruction to instead refer to data source.
  So like ssa. -> this will greatly reduce the space that we're searching through. As two programs that use one differnet register name would have
  been different previously.
  #3 We want to decouple the generation of things to evaluate and their evaluation. This will let us optimize the jit.
  #4 We will optimize the jit by making huge long programs of all the / many of the programs that need to be evaluated. Where we basically
  concatinate them all. They read in from memory (with some offset with a register we don't use or something or like stack or smthn.) the value
  they are to compute -> they run -> they store into an array their result. At the bottom it loops back to the top and goes to the next entry in
  the array of things to eval across all them -> goes all the way back down with them storing into the array again -> untill they all have evaled.
  right? way faster.
  #5 By decoupling eval from generation we are going to run all on some test set right -> the best performing ones and a few random ones are ran on
  a larger test set just to verify their quality. This will also let us do things like in #1 some member of the population that evolves a constant
  generating two children to be evaled one where only the new constant is mcmcd and another where all it constants are

    - The novelty/frontier system in population.cpp - novelty_seen, frontier_queue, frontier_hashes, frontier_boost_on, total_mutations,
    finalize_child, enqueue_boost — massive complexity that was coupled to GSTAG_ENABLE_CACHE
  Is the only thing that is even somewhat enabling this project to work at the moment so you will not remove.


- [ ] We can use the SR libraires to find / create the inverse of our generated output. Like we can use them to find the equations that we're somewhat simulating right. And by so doing we will be able to get the error and how close we are. With like a huge space of representabable problems -> which allows us to create rewrite rules per these approximated forms. And we can see if they are consistent.
