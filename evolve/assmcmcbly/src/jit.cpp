#include "jit.hpp"
#include "dag.hpp"
#include <sys/mman.h>
#include <cstring>
#include <cstdint>
#include <stdexcept>

// ── x86-64 byte emitter ────────────────────────────────────────────────────────
//
// VM register i lives at [rsp + i*4].  All 16 registers are 32-bit values.
// The RegType tag is not stored — it is only set, never read, during execution.
//
// Calling convention: SystemV AMD64
//   float fn(float input)  — arg in xmm0, return in xmm0
//   Clobbers: eax, ecx, xmm0, xmm1  (all caller-saved, so no save/restore)

struct E {
    uint8_t* p;

    void b(uint8_t x)    { *p++ = x; }
    void w32(uint32_t x) { memcpy(p, &x, 4); p += 4; }

    // ModRM + SIB for [rsp + disp8].  disp must fit in uint8_t (0–127).
    // Mod=01, Reg=reg, RM=4(SIB); SIB: SS=0, Index=4(none), Base=4(rsp)
    void mem_rsp(uint8_t reg, uint8_t d) {
        b(uint8_t(0x40 | (reg << 3) | 4));  // ModRM
        b(0x24);                              // SIB
        b(d);
    }

    void load_eax(uint8_t d)    { b(0x8B); mem_rsp(0, d); }   // mov eax, [rsp+d]
    void load_ecx(uint8_t d)    { b(0x8B); mem_rsp(1, d); }   // mov ecx, [rsp+d]
    void store_eax(uint8_t d)   { b(0x89); mem_rsp(0, d); }   // mov [rsp+d], eax
    void load_xmm0(uint8_t d)   { b(0xF3); b(0x0F); b(0x10); mem_rsp(0, d); } // movss xmm0,[rsp+d]
    void load_xmm1(uint8_t d)   { b(0xF3); b(0x0F); b(0x10); mem_rsp(1, d); } // movss xmm1,[rsp+d]
    void store_xmm0(uint8_t d)  { b(0xF3); b(0x0F); b(0x11); mem_rsp(0, d); } // movss [rsp+d],xmm0

    // SSE binary: op xmm0, [rsp+s2]
    void fop_mem(uint8_t opc, uint8_t s2) {
        b(0xF3); b(0x0F); b(opc); mem_rsp(0, s2);
    }

    // Set-cc on al then movzx eax,al
    void setcc_eax(uint8_t cc) {
        b(0x0F); b(cc); b(0xC0);             // setcc al
        b(0x0F); b(0xB6); b(0xC0);           // movzx eax, al
    }

    void emit_instr(const Instr& ins) {
        const uint8_t d   = uint8_t((ins.dst  & 0xF) * 4);
        const uint8_t s1  = uint8_t((ins.src1 & 0xF) * 4);
        const uint8_t s2  = uint8_t((ins.src2 & 0xF) * 4); // reg 15 if IMM_SRC — only used by non-imm path
        const bool    imm = (ins.src2 == Program::IMM_SRC);
        uint32_t imm32 = 0;
        if (imm) memcpy(&imm32, &ins.lit.i, 4);

        switch (ins.op) {

        // ── integer binary ────────────────────────────────────────────────
        case Op::IADD:
            load_eax(s1);
            if (imm) { b(0x05); w32(imm32); }           // add eax, imm32
            else     { b(0x03); mem_rsp(0, s2); }        // add eax, [rsp+s2]
            store_eax(d);
            break;
        case Op::ISUB:
            load_eax(s1);
            if (imm) { b(0x2D); w32(imm32); }           // sub eax, imm32
            else     { b(0x2B); mem_rsp(0, s2); }
            store_eax(d);
            break;
        case Op::IMUL:
            load_eax(s1);
            if (imm) { b(0x69); b(0xC0); w32(imm32); }  // imul eax, eax, imm32
            else     { load_ecx(s2); b(0x0F); b(0xAF); b(0xC1); } // imul eax, ecx
            store_eax(d);
            break;

        // ── float binary (no immediate form) ─────────────────────────────
        case Op::FADD:
            load_xmm0(s1);
            fop_mem(0x58, s2);         // addss xmm0, [rsp+s2]
            store_xmm0(d);
            break;
        case Op::FSUB:
            load_xmm0(s1);
            fop_mem(0x5C, s2);         // subss xmm0, [rsp+s2]
            store_xmm0(d);
            break;
        case Op::FMUL:
            load_xmm0(s1);
            fop_mem(0x59, s2);         // mulss xmm0, [rsp+s2]
            store_xmm0(d);
            break;

        // ── bitwise ───────────────────────────────────────────────────────
        case Op::BAND:
            load_eax(s1);
            if (imm) { b(0x25); w32(imm32); }           // and eax, imm32
            else     { b(0x23); mem_rsp(0, s2); }
            store_eax(d);
            break;
        case Op::BOR:
            load_eax(s1);
            if (imm) { b(0x0D); w32(imm32); }           // or eax, imm32
            else     { b(0x0B); mem_rsp(0, s2); }
            store_eax(d);
            break;
        case Op::BXOR:
            load_eax(s1);
            if (imm) { b(0x35); w32(imm32); }           // xor eax, imm32
            else     { b(0x33); mem_rsp(0, s2); }
            store_eax(d);
            break;

        // ── shifts (imm8 form; x86 masks cl/imm by 31 for 32-bit) ────────
        case Op::LSHL:
        case Op::ASHL:
            load_eax(s1);
            if (imm) { b(0xC1); b(0xE0); b(uint8_t(ins.lit.i & 31)); } // shl eax, imm8
            else     { load_ecx(s2); b(0xD3); b(0xE0); }                // shl eax, cl
            store_eax(d);
            break;
        case Op::LSHR:
            load_eax(s1);
            if (imm) { b(0xC1); b(0xE8); b(uint8_t(ins.lit.i & 31)); } // shr eax, imm8
            else     { load_ecx(s2); b(0xD3); b(0xE8); }
            store_eax(d);
            break;
        case Op::ASHR:
            load_eax(s1);
            if (imm) { b(0xC1); b(0xF8); b(uint8_t(ins.lit.i & 31)); } // sar eax, imm8
            else     { load_ecx(s2); b(0xD3); b(0xF8); }
            store_eax(d);
            break;

        // ── integer compares → 0/1 ────────────────────────────────────────
        case Op::ILT:
            load_eax(s1);
            if (imm) { b(0x3D); w32(imm32); }           // cmp eax, imm32
            else     { b(0x3B); mem_rsp(0, s2); }
            setcc_eax(0x9C);           // setl al
            store_eax(d);
            break;
        case Op::IEQ:
            load_eax(s1);
            if (imm) { b(0x3D); w32(imm32); }
            else     { b(0x3B); mem_rsp(0, s2); }
            setcc_eax(0x94);           // sete al
            store_eax(d);
            break;
        case Op::ULT:
            load_eax(s1);
            if (imm) { b(0x3D); w32(imm32); }
            else     { b(0x3B); mem_rsp(0, s2); }
            setcc_eax(0x92);           // setb al
            store_eax(d);
            break;
        case Op::UEQ:
            load_eax(s1);
            if (imm) { b(0x3D); w32(imm32); }
            else     { b(0x3B); mem_rsp(0, s2); }
            setcc_eax(0x94);           // sete al
            store_eax(d);
            break;

        // ── float compares → 0/1 ─────────────────────────────────────────
        case Op::FLT:
            // ucomiss: CF=1 if less OR unordered(NaN); PF=1 if unordered.
            // setb AND setnp: result=1 only when less AND ordered (NaN < x == false).
            // No xor before ucomiss — xor overwrites flags before setcc can read them.
            load_xmm0(s1);
            load_xmm1(s2);
            b(0x0F); b(0x2E); b(0xC1); // ucomiss xmm0, xmm1
            b(0x0F); b(0x92); b(0xC0); // setb  al  (CF=1: less or unordered)
            b(0x0F); b(0x9B); b(0xC1); // setnp cl  (PF=0: ordered, not NaN)
            b(0x22); b(0xC1);           // and al, cl
            b(0x0F); b(0xB6); b(0xC0); // movzx eax, al
            store_eax(d);
            break;
        case Op::FEQ:
            // ucomiss: ZF=1 if equal OR unordered(NaN); PF=1 if unordered.
            // sete AND setnp: result=1 only when equal AND ordered (NaN == x == false).
            load_xmm0(s1);
            load_xmm1(s2);
            b(0x0F); b(0x2E); b(0xC1); // ucomiss xmm0, xmm1
            b(0x0F); b(0x94); b(0xC0); // sete  al  (ZF=1: equal or unordered)
            b(0x0F); b(0x9B); b(0xC1); // setnp cl  (PF=0: ordered, not NaN)
            b(0x22); b(0xC1);           // and al, cl
            b(0x0F); b(0xB6); b(0xC0); // movzx eax, al
            store_eax(d);
            break;

        // ── unary ─────────────────────────────────────────────────────────
        case Op::BNOT:
            load_eax(s1);
            b(0xF7); b(0xD0);          // not eax
            store_eax(d);
            break;
        case Op::LNOT:
            load_eax(s1);
            b(0x85); b(0xC0);          // test eax, eax
            setcc_eax(0x94);           // sete al  (1 if eax==0)
            store_eax(d);
            break;
        case Op::INEG:
            load_eax(s1);
            b(0xF7); b(0xD8);          // neg eax
            store_eax(d);
            break;
        case Op::FNEG:
            load_xmm0(s1);
            b(0xB8); w32(0x80000000u); // mov eax, 0x80000000  (sign mask)
            b(0x66); b(0x0F); b(0x6E); b(0xC8); // movd xmm1, eax
            b(0x0F); b(0x57); b(0xC1); // xorps xmm0, xmm1
            store_xmm0(d);
            break;

        // ── bit-reinterpret (type tag only changes; bits unchanged) ───────
        case Op::ITF:
        case Op::FTI:
            load_eax(s1);
            store_eax(d);
            break;

        // ── loads ─────────────────────────────────────────────────────────
        case Op::LOADI: {
            uint32_t bits;
            memcpy(&bits, &ins.lit.i, 4);
            b(0xB8); w32(bits);        // mov eax, imm32
            store_eax(d);
            break;
        }
        case Op::LOADF: {
            uint32_t bits;
            memcpy(&bits, &ins.lit.f, 4);
            b(0xB8); w32(bits);        // mov eax, float_bits
            store_eax(d);
            break;
        }
        case Op::MOV:
            load_eax(s1);
            store_eax(d);
            break;

        default: break;
        }
    }
};

// ── Static arena ─────────────────────────────────────────────────────────────
//
// One mmap'd RWX region, allocated once and reused every compile.
// Safe because fitness() creates a JitProgram, runs it 100×, destroys it,
// then creates the next one — there is never concurrent write+execute.
// max per-program size: prologue(48) + 32*512 instrs + epilogue(12) = ~16.5 KB

static constexpr size_t ARENA_SIZE = 64 * 1024;  // 64 KB

static uint8_t* get_arena() {
    static uint8_t* arena = []() -> uint8_t* {
        void* p = mmap(nullptr, ARENA_SIZE,
                       PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
            throw std::runtime_error("jit arena: mmap failed");
        return static_cast<uint8_t*>(p);
    }();
    return arena;
}

// ── JitProgram lifetime ───────────────────────────────────────────────────────
// mem is nullptr when the arena is used (no per-instance munmap needed).

JitProgram::~JitProgram() {
    if (mem) munmap(mem, size);
}

JitProgram::JitProgram(JitProgram&& o) noexcept
    : mem(o.mem), size(o.size), fn(o.fn)
{
    o.mem = nullptr;
}

JitProgram& JitProgram::operator=(JitProgram&& o) noexcept {
    if (this != &o) {
        if (mem) munmap(mem, size);
        mem = o.mem; size = o.size; fn = o.fn;
        o.mem = nullptr;
    }
    return *this;
}

// ── Compiler ──────────────────────────────────────────────────────────────────

JitProgram jit_compile(const Program& prog) {
    uint8_t* buf = get_arena();
    E e{ buf };

    // ── Prologue ──────────────────────────────────────────────────────────
    // sub rsp, 64        ; 48 83 EC 40  — allocate register file
    e.b(0x48); e.b(0x83); e.b(0xEC); e.b(0x40);

    // Zero all 16 VM registers (64 bytes) with xmm1:
    //   xorps xmm1, xmm1   ; 0F 57 C9
    //   movups [rsp+N], xmm1
    e.b(0x0F); e.b(0x57); e.b(0xC9);
    for (uint8_t off = 0; off < 64; off += 16) {
        e.b(0x0F); e.b(0x11); e.mem_rsp(1, off); // movups [rsp+off], xmm1
    }

    // reg[0] = input (xmm0 holds the float arg on entry, still intact)
    //   movss [rsp+0], xmm0  ; F3 0F 11 44 24 00
    e.store_xmm0(0);

    // ── Instructions (dead instructions skipped) ─────────────────────────
    bool live[Program::MAX_INSTRS];
    compute_dag(prog, live);
    for (int i = 0; i < prog.num_instrs; ++i)
        if (live[i]) e.emit_instr(prog.instrs[i]);

    // ── Epilogue ──────────────────────────────────────────────────────────
    // movss xmm0, [rsp+0]   ; return reg[0] as float
    e.load_xmm0(0);
    // add rsp, 64           ; 48 83 C4 40
    e.b(0x48); e.b(0x83); e.b(0xC4); e.b(0x40);
    // ret                   ; C3
    e.b(0xC3);

    JitFn fn = reinterpret_cast<JitFn>(buf);
    return JitProgram{ nullptr, 0, fn };  // arena-owned, no per-instance munmap
}
