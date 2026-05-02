#pragma once
#include "program.hpp"
#include <cstddef>

using JitFn = float (*)(float);

// Owns the mmap'd executable memory for one compiled program.
struct JitProgram {
    void*  mem  = nullptr;
    size_t size = 0;
    JitFn  fn   = nullptr;

    JitProgram() = default;
    JitProgram(void* m, size_t s, JitFn f) : mem(m), size(s), fn(f) {}
    ~JitProgram();
    JitProgram(const JitProgram&)            = delete;
    JitProgram& operator=(const JitProgram&) = delete;
    JitProgram(JitProgram&&) noexcept;
    JitProgram& operator=(JitProgram&&) noexcept;
};

JitProgram jit_compile(const Program& prog);
