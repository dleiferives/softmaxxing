#include "codegen.hpp"
#include "dag.hpp"
#include <fstream>
#include <iomanip>
#include <cstring>

// ── Helpers ───────────────────────────────────────────────────────────────────

static const char* op_name(Op op) {
    switch (op) {
    case Op::IADD:  return "IADD";  case Op::ISUB:  return "ISUB";
    case Op::IMUL:  return "IMUL";  case Op::FADD:  return "FADD";
    case Op::FSUB:  return "FSUB";  case Op::FMUL:  return "FMUL";
    case Op::BAND:  return "BAND";  case Op::BOR:   return "BOR";
    case Op::BXOR:  return "BXOR";  case Op::LSHL:  return "LSHL";
    case Op::LSHR:  return "LSHR";  case Op::ASHL:  return "ASHL";
    case Op::ASHR:  return "ASHR";  case Op::ILT:   return "ILT";
    case Op::IEQ:   return "IEQ";   case Op::ULT:   return "ULT";
    case Op::UEQ:   return "UEQ";   case Op::FLT:   return "FLT";
    case Op::FEQ:   return "FEQ";   case Op::BNOT:  return "BNOT";
    case Op::LNOT:  return "LNOT";  case Op::INEG:  return "INEG";
    case Op::FNEG:  return "FNEG";  case Op::ITF:   return "ITF";
    case Op::FTI:   return "FTI";   case Op::LOADI: return "LOADI";
    case Op::LOADF: return "LOADF"; case Op::MOV:   return "MOV";
    default: return "???";
    }
}

// Write a human-readable assembly comment for one instruction.
static void write_asm_comment(std::ostream& o, const Instr& ins) {
    o << "/* " << op_name(ins.op) << " r" << int(ins.dst % Program::NUM_REGS);
    switch (ins.op) {
    case Op::LOADI:
        o << " 0x" << std::hex << uint32_t(ins.lit.i) << std::dec;
        break;
    case Op::LOADF:
        o << " " << ins.lit.f;
        break;
    case Op::BNOT: case Op::LNOT: case Op::INEG: case Op::FNEG:
    case Op::ITF:  case Op::FTI:  case Op::MOV:
        o << " r" << int(ins.src1 % Program::NUM_REGS);
        break;
    default:
        o << " r" << int(ins.src1 % Program::NUM_REGS);
        if (ins.src2 == Program::IMM_SRC)
            o << " #0x" << std::hex << uint32_t(ins.lit.i) << std::dec;
        else
            o << " r" << int(ins.src2 % Program::NUM_REGS);
        break;
    }
    o << " */";
}

// ── jit_eval emitter ─────────────────────────────────────────────────────────
//
// Emits one C statement per live instruction, directly accessing r[N].i/u/f.

static void emit_jit_instr(std::ostream& o, const Instr& ins) {
    const int    d   = ins.dst  % Program::NUM_REGS;
    const int    s1  = ins.src1 % Program::NUM_REGS;
    const bool   imm = (ins.src2 == Program::IMM_SRC);
    const int    s2r = ins.src2 % Program::NUM_REGS;
    const uint32_t litu = uint32_t(ins.lit.i);

    // Write src2 operand as C expression (int / uint / float flavour).
    auto bi = [&](std::ostream& s) -> std::ostream& {  // signed int
        if (imm) return s << "(int32_t)0x" << std::hex << litu << "U" << std::dec;
        return s << "r[" << s2r << "].i";
    };
    auto bu = [&](std::ostream& s) -> std::ostream& {  // unsigned
        if (imm) return s << "0x" << std::hex << litu << "U" << std::dec;
        return s << "r[" << s2r << "].u";
    };
    auto bf = [&](std::ostream& s) -> std::ostream& {  // float (no immediate)
        return s << "r[" << s2r << "].f";
    };

    o << "    ";
    switch (ins.op) {
    case Op::IADD: o<<"r["<<d<<"].i = r["<<s1<<"].i + "; bi(o); o<<";\n"; break;
    case Op::ISUB: o<<"r["<<d<<"].i = r["<<s1<<"].i - "; bi(o); o<<";\n"; break;
    case Op::IMUL: o<<"r["<<d<<"].i = r["<<s1<<"].i * "; bi(o); o<<";\n"; break;
    case Op::FADD: o<<"r["<<d<<"].f = r["<<s1<<"].f + "; bf(o); o<<";\n"; break;
    case Op::FSUB: o<<"r["<<d<<"].f = r["<<s1<<"].f - "; bf(o); o<<";\n"; break;
    case Op::FMUL: o<<"r["<<d<<"].f = r["<<s1<<"].f * "; bf(o); o<<";\n"; break;
    case Op::BAND: o<<"r["<<d<<"].u = r["<<s1<<"].u & "; bu(o); o<<";\n"; break;
    case Op::BOR:  o<<"r["<<d<<"].u = r["<<s1<<"].u | "; bu(o); o<<";\n"; break;
    case Op::BXOR: o<<"r["<<d<<"].u = r["<<s1<<"].u ^ "; bu(o); o<<";\n"; break;
    case Op::LSHL: o<<"r["<<d<<"].u = r["<<s1<<"].u << ("; bu(o); o<<" & 31u);\n"; break;
    case Op::LSHR: o<<"r["<<d<<"].u = r["<<s1<<"].u >> ("; bu(o); o<<" & 31u);\n"; break;
    case Op::ASHL: o<<"r["<<d<<"].i = r["<<s1<<"].i << (int)("; bu(o); o<<" & 31u);\n"; break;
    case Op::ASHR: o<<"r["<<d<<"].i = r["<<s1<<"].i >> (int)("; bu(o); o<<" & 31u);\n"; break;
    case Op::ILT:  o<<"r["<<d<<"].i = r["<<s1<<"].i <  "; bi(o); o<<" ? 1 : 0;\n"; break;
    case Op::IEQ:  o<<"r["<<d<<"].i = r["<<s1<<"].i == "; bi(o); o<<" ? 1 : 0;\n"; break;
    case Op::ULT:  o<<"r["<<d<<"].i = r["<<s1<<"].u <  "; bu(o); o<<" ? 1 : 0;\n"; break;
    case Op::UEQ:  o<<"r["<<d<<"].i = r["<<s1<<"].u == "; bu(o); o<<" ? 1 : 0;\n"; break;
    case Op::FLT:  o<<"r["<<d<<"].i = r["<<s1<<"].f <  "; bf(o); o<<" ? 1 : 0;\n"; break;
    case Op::FEQ:  o<<"r["<<d<<"].i = r["<<s1<<"].f == "; bf(o); o<<" ? 1 : 0;\n"; break;
    case Op::BNOT: o<<"r["<<d<<"].u = ~r["<<s1<<"].u;\n"; break;
    case Op::LNOT: o<<"r["<<d<<"].i = r["<<s1<<"].i == 0 ? 1 : 0;\n"; break;
    case Op::INEG: o<<"r["<<d<<"].i = -r["<<s1<<"].i;\n"; break;
    case Op::FNEG: o<<"r["<<d<<"].f = -r["<<s1<<"].f;\n"; break;
    case Op::ITF:  o<<"r["<<d<<"].u = r["<<s1<<"].u; /* reinterpret int→float */\n"; break;
    case Op::FTI:  o<<"r["<<d<<"].u = r["<<s1<<"].u; /* reinterpret float→int */\n"; break;
    case Op::MOV:  o<<"r["<<d<<"].u = r["<<s1<<"].u;\n"; break;
    case Op::LOADI: {
        o << "r[" << d << "].u = 0x" << std::hex << litu << "U;\n" << std::dec;
        break;
    }
    case Op::LOADF: {
        uint32_t bits; memcpy(&bits, &ins.lit.f, 4);
        o << "r[" << d << "].u = 0x" << std::hex << bits << "U; "
          << "/* " << ins.lit.f << "f */\n" << std::dec;
        break;
    }
    default:
        o << "/* unsupported op " << int(ins.op) << " */\n";
        break;
    }
}

// ── Top-level emitter ─────────────────────────────────────────────────────────

void emit_solution(const Program& prog, int gen, double fit,
                   const std::string& run_dir) {
    std::string path = run_dir + "/" + std::to_string(gen) + ".cpp";
    std::ofstream f(path);
    if (!f) return;

    bool live[Program::MAX_INSTRS];
    compute_dag(prog, live);

    // Collect live instructions in order.
    int live_idx[Program::MAX_INSTRS];
    int nlive = 0;
    for (int i = 0; i < prog.num_instrs; i++)
        if (live[i]) live_idx[nlive++] = i;

    // ── File header ──────────────────────────────────────────────────────────
    f << "// Generated: gen=" << gen << "  fit=" << std::setprecision(8) << fit << "\n"
      << "// Compile:   g++ -O2 -o eval_" << gen << " runner.cpp " << gen << ".cpp\n"
      << "//\n"
      << "// jit_eval — straight-line C, one statement per instruction\n"
      << "// vm_eval  — register-machine interpreter loop\n"
      << "//\n"
      << "// Program listing (" << nlive << " live / " << prog.num_instrs << " total):\n";

    // Emit program listing as block comment.
    int chrom_cursor = 0;
    int ci = 0;
    for (int i = 0; i < prog.num_instrs; i++) {
        if (i == chrom_cursor) {
            f << "//   [chrom " << ci << "]\n";
            chrom_cursor += prog.chrom_lens[ci];
            ci++;
        }
        f << "//     " << (live[i] ? " " : "~");  // ~ marks dead
        const Instr& ins = prog.instrs[i];
        f << op_name(ins.op) << "  r" << int(ins.dst % Program::NUM_REGS);
        switch (ins.op) {
        case Op::LOADI:
            f << "  0x" << std::hex << uint32_t(ins.lit.i) << std::dec; break;
        case Op::LOADF:
            f << "  " << ins.lit.f; break;
        case Op::BNOT: case Op::LNOT: case Op::INEG: case Op::FNEG:
        case Op::ITF:  case Op::FTI:  case Op::MOV:
            f << "  r" << int(ins.src1 % Program::NUM_REGS); break;
        default:
            f << "  r" << int(ins.src1 % Program::NUM_REGS);
            if (ins.src2 == Program::IMM_SRC)
                f << "  #0x" << std::hex << uint32_t(ins.lit.i) << std::dec;
            else
                f << "  r" << int(ins.src2 % Program::NUM_REGS);
            break;
        }
        f << "\n";
    }
    f << "\n#include <stdint.h>\n#include <string.h>\n\n"
      << "typedef union { uint32_t u; int32_t i; float f; } R;\n\n";

    // ── jit_eval ─────────────────────────────────────────────────────────────
    f << "float jit_eval(float x) {\n"
      << "    R r[16];\n"
      << "    for (int _i=0;_i<16;_i++) r[_i].u=0;\n"
      << "    r[0].f = x;\n";
    for (int k = 0; k < nlive; k++) {
        const Instr& ins = prog.instrs[live_idx[k]];
        f << "    "; write_asm_comment(f, ins); f << "\n";
        emit_jit_instr(f, ins);
    }
    f << "    return r[0].f;\n}\n\n";

    // ── vm_eval ───────────────────────────────────────────────────────────────
    f << "float vm_eval(float x) {\n"
      << "    typedef struct { unsigned char op,dst,src1,src2; unsigned int lit; } I;\n"
      << "    static const I p[] = {\n";
    for (int k = 0; k < nlive; k++) {
        const Instr& ins = prog.instrs[live_idx[k]];
        uint32_t litu; memcpy(&litu, &ins.lit, 4);
        f << "        { "
          << int(uint8_t(ins.op))          << ","
          << int(ins.dst % Program::NUM_REGS)  << ","
          << int(ins.src1 % Program::NUM_REGS) << ","
          << (ins.src2 == Program::IMM_SRC ? 255 : int(ins.src2 % Program::NUM_REGS)) << ","
          << "0x" << std::hex << litu << "U" << std::dec
          << " }, ";
        write_asm_comment(f, ins);
        f << "\n";
    }
    f << "    };\n"
      << "    R r[16];\n"
      << "    for (int _i=0;_i<16;_i++) r[_i].u=0;\n"
      << "    r[0].f = x;\n"
      << "    for (int _i=0; _i<(int)(sizeof(p)/sizeof(p[0])); _i++) {\n"
      << "        unsigned char op=p[_i].op, d=p[_i].dst&15, s1=p[_i].src1&15;\n"
      << "        R b; b.u = (p[_i].src2==255) ? p[_i].lit : r[p[_i].src2&15].u;\n"
      << "        R a = r[s1]; R* D = &r[d];\n"
      << "        switch(op) {\n"
      << "        case 0:  D->i=a.i+b.i; break;\n"        // IADD
      << "        case 1:  D->i=a.i-b.i; break;\n"        // ISUB
      << "        case 2:  D->i=a.i*b.i; break;\n"        // IMUL
      << "        case 3:  D->f=a.f+b.f; break;\n"        // FADD
      << "        case 4:  D->f=a.f-b.f; break;\n"        // FSUB
      << "        case 5:  D->f=a.f*b.f; break;\n"        // FMUL
      << "        case 6:  D->u=a.u&b.u; break;\n"        // BAND
      << "        case 7:  D->u=a.u|b.u; break;\n"        // BOR
      << "        case 8:  D->u=a.u^b.u; break;\n"        // BXOR
      << "        case 9:  D->u=a.u<<(b.u&31); break;\n"  // LSHL
      << "        case 10: D->u=a.u>>(b.u&31); break;\n"  // LSHR
      << "        case 11: D->i=a.i<<(int)(b.u&31); break;\n" // ASHL
      << "        case 12: D->i=a.i>>(int)(b.u&31); break;\n" // ASHR
      << "        case 13: D->i=a.i< b.i?1:0; break;\n"   // ILT
      << "        case 14: D->i=a.i==b.i?1:0; break;\n"   // IEQ
      << "        case 15: D->i=a.u< b.u?1:0; break;\n"   // ULT
      << "        case 16: D->i=a.u==b.u?1:0; break;\n"   // UEQ
      << "        case 17: D->i=a.f< b.f?1:0; break;\n"   // FLT
      << "        case 18: D->i=a.f==b.f?1:0; break;\n"   // FEQ
      << "        case 19: D->u=~a.u; break;\n"            // BNOT
      << "        case 20: D->i=a.i==0?1:0; break;\n"     // LNOT
      << "        case 21: D->i=-a.i; break;\n"            // INEG
      << "        case 22: D->f=-a.f; break;\n"            // FNEG
      << "        case 23: D->u=a.u; break;\n"             // ITF
      << "        case 24: D->u=a.u; break;\n"             // FTI
      << "        case 25: D->u=p[_i].lit; break;\n"       // LOADI
      << "        case 26: D->u=p[_i].lit; break;\n"       // LOADF
      << "        case 27: D->u=a.u; break;\n"             // MOV
      << "        }\n"
      << "    }\n"
      << "    return r[0].f;\n"
      << "}\n";
}
