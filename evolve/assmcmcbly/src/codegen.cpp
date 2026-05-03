#include "codegen.hpp"
#include "dag.hpp"
#include <fstream>
#include <iomanip>
#include <cstring>

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

// Emit one straight-line C assignment for node i.
static void emit_node_c(std::ostream& o, const Program& prog, int i) {
    const Node& nd = prog.nodes[i];
    const int   s1 = nd.src1;
    bool        imm = (nd.src2 == Program::IMM_SRC);
    const int   s2 = imm ? 0 : int(nd.src2);
    uint32_t    litu; std::memcpy(&litu, &nd.lit, 4);

    o << "    /* n" << i << " " << op_name(nd.op) << " */\n"
      << "    v[" << i << "].";

    switch (nd.op) {
    case Op::IADD:  o<<"i=v["<<s1<<"].i+v["<<s2<<"].i;\n"; break;
    case Op::ISUB:  o<<"i=v["<<s1<<"].i-v["<<s2<<"].i;\n"; break;
    case Op::IMUL:  o<<"i=v["<<s1<<"].i*v["<<s2<<"].i;\n"; break;
    case Op::FADD:  o<<"f=v["<<s1<<"].f+v["<<s2<<"].f;\n"; break;
    case Op::FSUB:  o<<"f=v["<<s1<<"].f-v["<<s2<<"].f;\n"; break;
    case Op::FMUL:  o<<"f=v["<<s1<<"].f*v["<<s2<<"].f;\n"; break;
    case Op::BAND:
        if (imm) o<<"u=v["<<s1<<"].u&0x"<<std::hex<<litu<<"U;\n"<<std::dec;
        else     o<<"u=v["<<s1<<"].u&v["<<s2<<"].u;\n";
        break;
    case Op::BOR:
        if (imm) o<<"u=v["<<s1<<"].u|0x"<<std::hex<<litu<<"U;\n"<<std::dec;
        else     o<<"u=v["<<s1<<"].u|v["<<s2<<"].u;\n";
        break;
    case Op::BXOR:
        if (imm) o<<"u=v["<<s1<<"].u^0x"<<std::hex<<litu<<"U;\n"<<std::dec;
        else     o<<"u=v["<<s1<<"].u^v["<<s2<<"].u;\n";
        break;
    case Op::LSHL:
        if (imm) o<<"u=v["<<s1<<"].u<<("<<(litu&31u)<<"u);\n";
        else     o<<"u=v["<<s1<<"].u<<(v["<<s2<<"].u&31u);\n";
        break;
    case Op::LSHR:
        if (imm) o<<"u=v["<<s1<<"].u>>("<<(litu&31u)<<"u);\n";
        else     o<<"u=v["<<s1<<"].u>>(v["<<s2<<"].u&31u);\n";
        break;
    case Op::ASHL:
        if (imm) o<<"i=v["<<s1<<"].i<<(int)("<<(litu&31u)<<"u);\n";
        else     o<<"i=v["<<s1<<"].i<<(int)(v["<<s2<<"].u&31u);\n";
        break;
    case Op::ASHR:
        if (imm) o<<"i=v["<<s1<<"].i>>(int)("<<(litu&31u)<<"u);\n";
        else     o<<"i=v["<<s1<<"].i>>(int)(v["<<s2<<"].u&31u);\n";
        break;
    case Op::ILT:
        if (imm) o<<"i=v["<<s1<<"].i<(int32_t)0x"<<std::hex<<litu<<"?1:0;\n"<<std::dec;
        else     o<<"i=v["<<s1<<"].i<v["<<s2<<"].i?1:0;\n";
        break;
    case Op::IEQ:
        if (imm) o<<"i=v["<<s1<<"].i==(int32_t)0x"<<std::hex<<litu<<"?1:0;\n"<<std::dec;
        else     o<<"i=v["<<s1<<"].i==v["<<s2<<"].i?1:0;\n";
        break;
    case Op::ULT:
        if (imm) o<<"i=v["<<s1<<"].u<0x"<<std::hex<<litu<<"U?1:0;\n"<<std::dec;
        else     o<<"i=v["<<s1<<"].u<v["<<s2<<"].u?1:0;\n";
        break;
    case Op::UEQ:
        if (imm) o<<"i=v["<<s1<<"].u==0x"<<std::hex<<litu<<"U?1:0;\n"<<std::dec;
        else     o<<"i=v["<<s1<<"].u==v["<<s2<<"].u?1:0;\n";
        break;
    case Op::FLT:  o<<"i=v["<<s1<<"].f<v["<<s2<<"].f?1:0;\n";  break;
    case Op::FEQ:  o<<"i=v["<<s1<<"].f==v["<<s2<<"].f?1:0;\n"; break;
    case Op::BNOT: o<<"u=~v["<<s1<<"].u;\n";                    break;
    case Op::LNOT: o<<"i=v["<<s1<<"].i==0?1:0;\n";              break;
    case Op::INEG: o<<"i=-v["<<s1<<"].i;\n";                    break;
    case Op::FNEG: o<<"f=-v["<<s1<<"].f;\n";                    break;
    case Op::ITF:  o<<"u=v["<<s1<<"].u;\n";                     break;
    case Op::FTI:  o<<"u=v["<<s1<<"].u;\n";                     break;
    case Op::MOV:  o<<"u=v["<<s1<<"].u;\n";                     break;
    case Op::LOADI: o<<"u=0x"<<std::hex<<litu<<"U;\n"<<std::dec; break;
    case Op::LOADF: {
        float fv = nd.lit.f; uint32_t bits; std::memcpy(&bits, &fv, 4);
        o<<"u=0x"<<std::hex<<bits<<"U; /* "<<fv<<"f */\n"<<std::dec; break;
    }
    default: o<<"u=0; /* unsupported op "<<int(nd.op)<<" */\n"; break;
    }
}

void emit_solution(const Program& prog, int gen, double fit,
                   const std::string& run_dir) {
    std::string path = run_dir + "/" + std::to_string(gen) + ".cpp";
    std::ofstream f(path);
    if (!f) return;

    bool live[Program::MAX_NODES];
    int  nlive = compute_live(prog, live);

    // Live function node indices in topological order (index order is valid).
    int live_idx[Program::MAX_NODES];
    int n_lidx = 0;
    for (int i = prog.n_inputs; i < prog.n_nodes; i++)
        if (live[i]) live_idx[n_lidx++] = i;

    // ── Header / listing ─────────────────────────────────────────────────────
    f << "// Generated: gen=" << gen << "  fit=" << std::setprecision(8) << fit << "\n"
      << "// Compile:   g++ -O2 -o eval_" << gen << " runner.cpp " << gen << ".cpp\n"
      << "//\n"
      << "// DAG: " << nlive << " live / " << prog.n_nodes << " total nodes"
      << "  (n_inputs=" << prog.n_inputs << "  output=n" << prog.output_node << ")\n"
      << "//\n";

    for (int i = 0; i < prog.n_nodes; i++) {
        f << "//  " << (live[i] ? " " : "~") << "n" << i << " = ";
        if (i < prog.n_inputs) {
            f << "INPUT[" << i << "]";
        } else {
            const Node& nd = prog.nodes[i];
            f << op_name(nd.op);
            switch (nd.op) {
            case Op::LOADI:
                f<<"(0x"<<std::hex<<uint32_t(nd.lit.i)<<std::dec<<")"; break;
            case Op::LOADF:
                f<<"("<<nd.lit.f<<")"; break;
            case Op::BNOT: case Op::LNOT: case Op::INEG: case Op::FNEG:
            case Op::ITF:  case Op::FTI:  case Op::MOV:
                f<<"(n"<<nd.src1<<")"; break;
            default:
                f<<"(n"<<nd.src1<<", ";
                if (nd.src2 == Program::IMM_SRC)
                    f<<"#0x"<<std::hex<<uint32_t(nd.lit.i)<<std::dec;
                else
                    f<<"n"<<nd.src2;
                f<<")"; break;
            }
        }
        if (i == prog.output_node) f << "  <-- output";
        f << "\n";
    }

    f << "\n#include <stdint.h>\n#include <string.h>\n\n"
      << "typedef union { uint32_t u; int32_t i; float f; } V;\n\n";

    // ── jit_eval: one C statement per live node ──────────────────────────────
    f << "float jit_eval(float x) {\n"
      << "    V v[" << prog.n_nodes << "];\n"
      << "    for (int _i=0;_i<" << prog.n_nodes << ";_i++) v[_i].u=0;\n"
      << "    v[0].f = x;\n";
    for (int k = 0; k < n_lidx; k++)
        emit_node_c(f, prog, live_idx[k]);
    f << "    return v[" << prog.output_node << "].f;\n}\n\n";

    // ── vm_eval: compact table (dst,op,s1,s2,lit) + interpreter loop ─────────
    f << "float vm_eval(float x) {\n"
      << "    typedef struct { unsigned short dst,op,s1,s2; unsigned int lit; } N;\n"
      << "    static const N p[] = {\n";
    for (int k = 0; k < n_lidx; k++) {
        int i = live_idx[k];
        const Node& nd = prog.nodes[i];
        uint32_t litu; std::memcpy(&litu, &nd.lit, 4);
        f << "        { " << i << ","
          << int(uint8_t(nd.op)) << ","
          << nd.src1 << ","
          << (nd.src2 == Program::IMM_SRC ? 0xFFFF : int(nd.src2)) << ","
          << "0x"<<std::hex<<litu<<"U"<<std::dec
          << " }, /* "<<op_name(nd.op)<<" */\n";
    }
    f << "    };\n"
      << "    V v[" << prog.n_nodes << "];\n"
      << "    for (int _i=0;_i<" << prog.n_nodes << ";_i++) v[_i].u=0;\n"
      << "    v[0].f = x;\n"
      << "    for (int _i=0;_i<(int)(sizeof(p)/sizeof(p[0]));_i++) {\n"
      << "        unsigned d=p[_i].dst,op=p[_i].op,s1=p[_i].s1,s2=p[_i].s2;\n"
      << "        V a=v[s1], b=(s2==0xFFFFu)?((V){.u=p[_i].lit}):v[s2];\n"
      << "        V *D=&v[d];\n"
      << "        switch(op){\n"
      << "        case  0: D->i=a.i+b.i; break;\n"
      << "        case  1: D->i=a.i-b.i; break;\n"
      << "        case  2: D->i=a.i*b.i; break;\n"
      << "        case  3: D->f=a.f+b.f; break;\n"
      << "        case  4: D->f=a.f-b.f; break;\n"
      << "        case  5: D->f=a.f*b.f; break;\n"
      << "        case  6: D->u=a.u&b.u; break;\n"
      << "        case  7: D->u=a.u|b.u; break;\n"
      << "        case  8: D->u=a.u^b.u; break;\n"
      << "        case  9: D->u=a.u<<(b.u&31); break;\n"
      << "        case 10: D->u=a.u>>(b.u&31); break;\n"
      << "        case 11: D->i=a.i<<(int)(b.u&31); break;\n"
      << "        case 12: D->i=a.i>>(int)(b.u&31); break;\n"
      << "        case 13: D->i=a.i< b.i?1:0; break;\n"
      << "        case 14: D->i=a.i==b.i?1:0; break;\n"
      << "        case 15: D->i=a.u< b.u?1:0; break;\n"
      << "        case 16: D->i=a.u==b.u?1:0; break;\n"
      << "        case 17: D->i=a.f< b.f?1:0; break;\n"
      << "        case 18: D->i=a.f==b.f?1:0; break;\n"
      << "        case 19: D->u=~a.u; break;\n"
      << "        case 20: D->i=a.i==0?1:0; break;\n"
      << "        case 21: D->i=-a.i; break;\n"
      << "        case 22: D->f=-a.f; break;\n"
      << "        case 23: D->u=a.u; break;\n"
      << "        case 24: D->u=a.u; break;\n"
      << "        case 25: D->u=p[_i].lit; break;\n"
      << "        case 26: D->u=p[_i].lit; break;\n"
      << "        case 27: D->u=a.u; break;\n"
      << "        }\n    }\n"
      << "    return v[" << prog.output_node << "].f;\n}\n";
}
