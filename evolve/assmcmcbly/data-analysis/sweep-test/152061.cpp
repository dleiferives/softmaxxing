// Generated: gen=152061  fit=3.8640537e-05
// Compile:   g++ -O2 -o eval_152061 runner.cpp 152061.cpp
//
// jit_eval — straight-line C, one statement per instruction
// vm_eval  — register-machine interpreter loop
//
// Program listing (7 live / 7 total):
//    LOADF  r12  8.3015026e-23
//    ASHL  r3  r12  #0x1f584b30
//    FMUL  r0  r3  r0
//    LOADI  r11  0xbee002cf
//    LSHR  r8  r0  #0x6ee234c1
//    FMUL  r10  r8  r11
//    ISUB  r0  r12  r10

#include <stdint.h>
#include <string.h>

typedef union { uint32_t u; int32_t i; float f; } R;

float jit_eval(float x) {
    R r[16];
    for (int _i=0;_i<16;_i++) r[_i].u=0;
    r[0].f = x;
    /* LOADF r12 8.3015026e-23 */
    r[12].u = 0x1ac8b7d0U; /* 8.3015026e-23f */
    /* ASHL r3 r12 #0x1f584b30 */
    r[3].i = r[12].i << (int)(0x1f584b30U & 31u);
    /* FMUL r0 r3 r0 */
    r[0].f = r[3].f * r[0].f;
    /* LOADI r11 0xbee002cf */
    r[11].u = 0xbee002cfU;
    /* LSHR r8 r0 #0x6ee234c1 */
    r[8].u = r[0].u >> (0x6ee234c1U & 31u);
    /* FMUL r10 r8 r11 */
    r[10].f = r[8].f * r[11].f;
    /* ISUB r0 r12 r10 */
    r[0].i = r[12].i - r[10].i;
    return r[0].f;
}

float vm_eval(float x) {
    typedef struct { unsigned char op,dst,src1,src2; unsigned int lit; } I;
    static const I p[] = {
        { 26,12,0,0,0x1ac8b7d0U }, /* LOADF r12 8.3015026e-23 */
        { 11,3,12,255,0x1f584b30U }, /* ASHL r3 r12 #0x1f584b30 */
        { 5,0,3,0,0x65639b14U }, /* FMUL r0 r3 r0 */
        { 25,11,12,2,0xbee002cfU }, /* LOADI r11 0xbee002cf */
        { 10,8,0,255,0x6ee234c1U }, /* LSHR r8 r0 #0x6ee234c1 */
        { 5,10,8,11,0x127a7c40U }, /* FMUL r10 r8 r11 */
        { 1,0,12,10,0xf52ec557U }, /* ISUB r0 r12 r10 */
    };
    R r[16];
    for (int _i=0;_i<16;_i++) r[_i].u=0;
    r[0].f = x;
    for (int _i=0; _i<(int)(sizeof(p)/sizeof(p[0])); _i++) {
        unsigned char op=p[_i].op, d=p[_i].dst&15, s1=p[_i].src1&15;
        R b; b.u = (p[_i].src2==255) ? p[_i].lit : r[p[_i].src2&15].u;
        R a = r[s1]; R* D = &r[d];
        switch(op) {
        case 0:  D->i=a.i+b.i; break;
        case 1:  D->i=a.i-b.i; break;
        case 2:  D->i=a.i*b.i; break;
        case 3:  D->f=a.f+b.f; break;
        case 4:  D->f=a.f-b.f; break;
        case 5:  D->f=a.f*b.f; break;
        case 6:  D->u=a.u&b.u; break;
        case 7:  D->u=a.u|b.u; break;
        case 8:  D->u=a.u^b.u; break;
        case 9:  D->u=a.u<<(b.u&31); break;
        case 10: D->u=a.u>>(b.u&31); break;
        case 11: D->i=a.i<<(int)(b.u&31); break;
        case 12: D->i=a.i>>(int)(b.u&31); break;
        case 13: D->i=a.i< b.i?1:0; break;
        case 14: D->i=a.i==b.i?1:0; break;
        case 15: D->i=a.u< b.u?1:0; break;
        case 16: D->i=a.u==b.u?1:0; break;
        case 17: D->i=a.f< b.f?1:0; break;
        case 18: D->i=a.f==b.f?1:0; break;
        case 19: D->u=~a.u; break;
        case 20: D->i=a.i==0?1:0; break;
        case 21: D->i=-a.i; break;
        case 22: D->f=-a.f; break;
        case 23: D->u=a.u; break;
        case 24: D->u=a.u; break;
        case 25: D->u=p[_i].lit; break;
        case 26: D->u=p[_i].lit; break;
        case 27: D->u=a.u; break;
        }
    }
    return r[0].f;
}
