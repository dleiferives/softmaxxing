#include "jit.hpp"
#include <sys/mman.h>

JitProgram::~JitProgram() {
    if (mem && size) munmap(mem, size);
}

JitProgram::JitProgram(JitProgram&& o) noexcept
    : mem(o.mem), size(o.size), fn(o.fn)
{
    o.mem = nullptr; o.size = 0; o.fn = nullptr;
}

JitProgram& JitProgram::operator=(JitProgram&& o) noexcept {
    if (this != &o) {
        if (mem && size) munmap(mem, size);
        mem = o.mem; size = o.size; fn = o.fn;
        o.mem = nullptr; o.size = 0; o.fn = nullptr;
    }
    return *this;
}

// Phase 2: DAG JIT not yet implemented; fitness.cpp falls back to interpreter.
JitProgram jit_compile(const Program& /*prog*/) {
    return JitProgram{};
}
