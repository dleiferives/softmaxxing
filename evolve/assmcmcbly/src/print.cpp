#include "print.hpp"
#include "dag.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>

const char* op_str(Op op) {
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

void print_program(const Program& prog) {
    bool live[Program::MAX_NODES];
    compute_live(prog, live);

    std::cout << "  n_inputs=" << prog.n_inputs
              << "  n_nodes="  << prog.n_nodes
              << "  output=n"  << prog.output_node << "\n";

    for (int i = 0; i < prog.n_nodes; i++) {
        bool is_live = live[i];
        std::cout << (is_live ? "   " : "  ~");
        std::cout << "n" << i << " = ";

        if (i < prog.n_inputs) {
            std::cout << "INPUT[" << i << "]";
        } else {
            const Node& nd = prog.nodes[i];
            std::cout << op_str(nd.op);
            switch (nd.op) {
            case Op::LOADI:
                std::cout << "(0x" << std::hex << uint32_t(nd.lit.i) << std::dec << ")";
                break;
            case Op::LOADF:
                std::cout << "(" << nd.lit.f << ")";
                break;
            case Op::BNOT: case Op::LNOT: case Op::INEG: case Op::FNEG:
            case Op::ITF:  case Op::FTI:  case Op::MOV:
                std::cout << "(n" << nd.src1 << ")";
                break;
            default:
                std::cout << "(n" << nd.src1 << ", ";
                if (nd.src2 == Program::IMM_SRC)
                    std::cout << "#0x" << std::hex << uint32_t(nd.lit.i) << std::dec;
                else
                    std::cout << "n" << nd.src2;
                std::cout << ")";
                break;
            }
        }
        if (i == prog.output_node) std::cout << "  <-- output";
        std::cout << "\n";
    }
}
