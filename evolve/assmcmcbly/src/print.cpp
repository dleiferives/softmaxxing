#include "print.hpp"
#include <iostream>

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
    for (int ci = 0; ci < prog.num_chroms; ci++) {
        std::cout << "  [chrom " << ci << "]\n";
        int cs = prog.chrom_start(ci);
        for (int ii = 0; ii < prog.chrom_lens[ci]; ii++) {
            const Instr& ins = prog.instrs[cs + ii];
            std::cout << "    " << op_str(ins.op) << "  r" << int(ins.dst);
            switch (ins.op) {
            case Op::LOADI:
                std::cout << "  0x" << std::hex << uint32_t(ins.lit.i) << std::dec;
                break;
            case Op::LOADF:
                std::cout << "  " << ins.lit.f;
                break;
            case Op::BNOT: case Op::LNOT: case Op::INEG: case Op::FNEG:
            case Op::ITF:  case Op::FTI:  case Op::MOV:
                std::cout << "  r" << int(ins.src1);
                break;
            default: // binary
                std::cout << "  r" << int(ins.src1);
                if (ins.src2 == Program::IMM_SRC)
                    std::cout << "  #0x" << std::hex << uint32_t(ins.lit.i) << std::dec;
                else
                    std::cout << "  r" << int(ins.src2);
                break;
            }
            std::cout << "\n";
        }
    }
}
