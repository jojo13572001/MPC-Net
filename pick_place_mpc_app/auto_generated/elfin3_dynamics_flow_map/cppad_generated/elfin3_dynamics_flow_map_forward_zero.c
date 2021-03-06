#include <math.h>
#include <stdio.h>

typedef struct Array {
    void* data;
    unsigned long size;
    int sparse;
    const unsigned long* idx;
    unsigned long nnz;
} Array;

struct LangCAtomicFun {
    void* libModel;
    int (*forward)(void* libModel,
                   int atomicIndex,
                   int q,
                   int p,
                   const Array tx[],
                   Array* ty);
    int (*reverse)(void* libModel,
                   int atomicIndex,
                   int p,
                   const Array tx[],
                   Array* px,
                   const Array py[]);
};

void elfin3_dynamics_flow_map_forward_zero(double const *const * in,
                                           double*const * out,
                                           struct LangCAtomicFun atomicFun) {
   //independent variables
   const double* x = in[0];

   //dependent variables
   double* y = out[0];

   // auxiliary variables
   double v[151];

   v[0] = cos(x[2]);
   v[1] = sin(x[2]);
   v[2] = -0.06881 * v[1];
   v[3] = v[2] * x[7];
   v[4] = v[3] * x[8];
   v[5] = 0 - v[4];
   v[6] = v[0] * x[7];
   v[7] = v[6] * x[8];
   v[8] = x[8] * x[8];
   v[9] = v[1] * x[7];
   v[10] = v[9] * v[9];
   v[11] = cos(x[3]);
   v[12] = 0 - v[11];
   v[13] = sin(x[3]);
   v[14] = 0 - v[13];
   v[15] = v[14] * v[9] + v[12] * v[6];
   v[16] = -0.266 * v[9];
   v[17] = -1 * x[8] + x[9];
   v[18] = 0 - v[11];
   v[19] = 0 - v[13];
   v[20] = 0.06881 * v[0];
   v[21] = v[20] * x[7];
   v[22] = 0.266 * v[13];
   v[23] = v[18] * v[3] + v[19] * v[21] + v[22] * x[8];
   v[24] = v[15] * v[16] - v[17] * v[23];
   v[25] = v[15] * v[17];
   v[26] = v[17] * v[17];
   v[27] = v[11] * v[9] + v[14] * v[6];
   v[28] = v[27] * v[27];
   v[29] = sin(x[4]);
   v[30] = -0.06881 * v[29];
   v[31] = cos(x[4]);
   v[32] = v[31] * v[27] + v[29] * v[17];
   v[33] = 0.06881 * v[27] + -1 * v[23];
   v[34] = v[32] * v[33];
   v[35] = -1 * v[15] + x[10];
   v[36] = 0.10523 * v[31];
   v[37] = 0.06881 * v[31];
   v[38] = -0.10523 * v[29];
   v[13] = 0 - v[13];
   v[39] = -0.266 * v[11];
   v[40] = v[13] * v[3] + v[11] * v[21] + v[39] * x[8];
   v[41] = v[36] * v[17] + v[37] * v[15] + v[38] * v[27] + v[29] * v[16] + v[31] * v[40];
   v[42] = v[35] * v[41];
   v[43] = v[34] - v[42];
   v[44] = 0 - v[29];
   v[45] = v[44] * v[27] + v[31] * v[17];
   v[46] = v[32] * v[45];
   v[47] = v[45] * v[35];
   v[48] = cos(x[5]);
   v[49] = sin(x[5]);
   v[50] = 0 - v[49];
   v[51] = v[48] * v[32] + v[50] * v[35];
   v[52] = 0 - v[48];
   v[53] = 0 - v[49];
   v[54] = -0.21877 * v[49];
   v[55] = v[52] * v[33] + v[53] * v[41] + v[54] * v[45];
   v[56] = 0 - v[48];
   v[57] = v[50] * v[32] + v[56] * v[35];
   v[49] = 0 - v[49];
   v[58] = 0.21877 * v[48];
   v[59] = v[49] * v[33] + v[48] * v[41] + v[58] * v[45];
   v[60] = v[51] * v[55] - v[57] * v[59];
   v[61] = v[45] + x[11];
   v[62] = v[51] * v[61];
   v[63] = v[57] * v[61];
   v[64] = cos(x[6]);
   v[65] = sin(x[6]);
   v[66] = v[64] * v[51] + v[65] * v[61];
   v[67] = -1 * v[55];
   v[68] = v[66] * v[67];
   v[69] = -1 * v[57] + x[12];
   v[70] = 0.1065 * v[64];
   v[71] = -0.1065 * v[65];
   v[72] = -0.10523 * v[29];
   v[73] = -0.10523 * v[31];
   v[74] = 0 - v[29];
   v[75] = v[72] * v[17] + v[30] * v[15] + v[73] * v[27] + v[31] * v[16] + v[74] * v[40];
   v[76] = -0.21877 * v[32] + v[75];
   v[77] = v[70] * v[61] + v[71] * v[51] + v[65] * v[76] + v[64] * v[59];
   v[78] = v[69] * v[77];
   v[79] = v[68] - v[78];
   v[80] = 0 - v[65];
   v[81] = v[80] * v[51] + v[64] * v[61];
   v[82] = v[66] * v[81];
   v[83] = v[81] * v[69];
   v[84] = (0 - v[66]) * x[12];
   v[85] = v[81] * x[12];
   v[86] = (0 - v[77]) * x[12];
   v[87] = -0.1065 * v[64];
   v[88] = 0 - v[65];
   v[89] = v[71] * v[61] + v[87] * v[51] + v[64] * v[76] + v[88] * v[59];
   v[90] = v[89] * x[12];
   v[91] = v[66] * v[69];
   v[67] = v[81] * v[67] - v[69] * v[89];
   v[92] = v[81] * v[81];
   v[93] = v[66] * v[66];
   v[78] = -0.00015326385347 * (v[78] - v[68]) + -3.08690000000011e-06 * v[82] + -1.3623974e-05 * v[91] - 0.00042046279017 * v[67] - 6.388487e-06 * v[83] - 9.071669e-06 * (v[92] - v[93]);
   v[68] = x[18] - v[78];
   v[69] = v[69] * v[69];
   v[94] = -0.4551941 * v[79] + -0.00015326385347 * v[82] + 0.0194326913231 * v[83] + -3.44659905309698e-06 * v[84] + -0.0194310751607934 * v[85] + 0.455155327241608 * v[86] + -0.000106368865241701 * v[90] + (-0.00015326385347 * v[68]) / 0.0006058328 - 0.00042046279017 * (v[69] + v[93]);
   v[95] = 0.4551941 * v[67] + 0.0194326913231 * v[91] + 0.00042046279017 * v[82] + 0.019423235951715 * v[84] + 4.43376632791221e-06 * v[85] + -0.000106368865241701 * v[86] + 0.454902288533342 * v[90] + (-0.00042046279017 * v[68]) / 0.0006058328 - -0.00015326385347 * (v[69] + v[92]);
   v[96] = -0.00015326385347 * v[64] + 0.00042046279017 * v[80];
   v[97] = v[64] * v[87] + v[65] * v[71];
   v[98] = -0.000106368865241701 * v[97];
   v[99] = 4.43376632791221e-06 + v[98];
   v[100] = 0 - 0.454902288533342 * v[97];
   v[101] = 0.019423235951715 + v[100];
   v[102] = v[65] * v[99] + v[64] * v[101];
   v[103] = 0.455155327241608 * v[97];
   v[104] = -0.0194310751607934 + v[103];
   v[105] = -3.44659905309698e-06 - v[98];
   v[106] = v[65] * v[104] + v[64] * v[105];
   v[107] = v[102] * v[65] + v[106] * v[64];
   v[103] = 0.00143234843361418 + -0.0388621503215868 * v[97] + v[103] * v[97];
   v[98] = 9.21533335885832e-06 + -3.44659905309698e-06 * v[97] - 4.43376632791221e-06 * v[97] - v[98] * v[97];
   v[108] = v[65] * v[103] + v[64] * v[98];
   v[100] = 0.0014290225239379 - 0.0388464719034299 * v[97] - v[100] * v[97];
   v[98] = v[65] * v[98] + v[64] * v[100];
   v[97] = v[108] * v[65] + v[98] * v[64];
   v[109] = 0.009312066 + v[97];
   v[110] = v[107] / v[109];
   v[111] = 1.232863523889e-06 + -0.00015326385347 * v[64] + -0.00042046279017 * v[65] - v[96] - 1.1342228e-05 * v[110];
   v[112] = (0 - v[51]) * x[11];
   v[105] = v[64] * v[104] + v[80] * v[105];
   v[101] = v[64] * v[99] + v[80] * v[101];
   v[98] = -8.1296676e-08 + v[98] * v[80] + v[108] * v[64];
   v[108] = -0.09107395141674 + v[105] * v[64] + v[101] * v[65] - v[98] * v[110];
   v[99] = v[57] * x[11];
   v[96] = -1.232863523889e-06 + v[96];
   v[104] = 0 - v[96] * v[110];
   v[113] = (0 - v[59]) * x[11];
   v[114] = -0.000106368865241701 * v[65] + -3.87727583922026e-05 * v[64];
   v[115] = -0.000291811466658021 * v[65] + -0.000106368865241701 * v[64];
   v[106] = 0.09107395141674 + v[106] * v[80] + v[102] * v[64];
   v[102] = v[114] * v[80] + v[115] * v[64] - v[106] * v[110];
   v[116] = v[55] * x[11];
   v[59] = v[61] * v[59];
   v[117] = v[51] * v[76];
   v[118] = v[51] * v[57];
   v[89] = v[66] * v[89] - v[81] * v[77];
   v[67] = 0.0194326913231 * v[67] + 9.071669e-06 * v[83] + 6.388487e-06 * (v[69] - v[93]) + 0.0014290225239379 * v[84] + 9.21533335885832e-06 * v[85] + -3.44659905309698e-06 * v[86] + 0.019423235951715 * v[90] + (-1.3623974e-05 * v[68]) / 0.0006058328 - -0.00015326385347 * v[89] - -0.000826583 * v[91] - -1.3623974e-05 * v[82];
   v[69] = 0.00042046279017 * v[89] + 0.0194326913231 * v[79] + -0.0008234961 * v[83] + 6.388487e-06 * v[82] + 9.21533335885832e-06 * v[84] + 0.00143234843361418 * v[85] + -0.0194310751607934 * v[86] + 4.43376632791221e-06 * v[90] + (6.388487e-06 * v[68]) / 0.0006058328 - 9.071669e-06 * v[91] - -1.3623974e-05 * (v[69] - v[92]);
   v[76] = v[57] * v[76] - v[61] * v[55];
   v[57] = v[57] * v[57];
   v[51] = v[51] * v[51];
   v[55] = -1.232863523889e-06 * (v[59] - v[117]) + -0.0074296073 * v[118] + 1.1342228e-05 * v[62] + v[64] * v[67] + v[65] * v[69] + v[71] * v[94] + v[70] * v[95] - -0.09107395141674 * v[76] - -8.1296676e-08 * v[63] - 2.9481043e-07 * (v[57] - v[51]);
   v[82] = x[17] - v[55];
   v[79] = 2.493714 * v[60] + -1.232863523889e-06 * v[62] + -0.09107395141674 * v[63] + v[64] * v[94] + v[65] * v[95] + v[111] * v[112] + v[108] * v[99] + v[104] * v[113] + v[102] * v[116] + (v[107] * v[82]) / v[109] - -0.039577832148846 * (v[57] + v[51]);
   v[81] = v[58] * v[102];
   v[77] = -0.21877 * v[50];
   v[66] = v[77] * v[104];
   v[119] = v[81] - v[66];
   v[120] = v[106] / v[109];
   v[101] = v[105] * v[80] + v[101] * v[64] - v[98] * v[120];
   v[81] = 0 - v[81];
   v[105] = -0.00015326385347 * v[65] + 0.00042046279017 * v[64];
   v[121] = v[96] / v[109];
   v[122] = 0.039577832148846 - v[105] - v[98] * v[121];
   v[123] = 0 - v[58] * v[104];
   v[124] = v[101] * v[58] + v[81] * v[58] - v[122] * v[77] - v[123] * v[77];
   v[105] = -0.039577832148846 + -0.00015326385347 * v[80] + -0.00042046279017 * v[64] + v[105] - 1.1342228e-05 * v[120];
   v[125] = v[77] * v[102];
   v[126] = 0 - 1.1342228e-05 * v[121];
   v[127] = v[105] * v[58] + v[125] * v[58] - v[126] * v[77] - v[66] * v[77];
   v[128] = 0 - v[127];
   v[129] = 0.009210478 + v[50] * v[124] + v[48] * v[128];
   v[115] = v[115] * v[65] + v[114] * v[64];
   v[110] = 2.9489081 + v[115] - v[107] * v[110];
   v[114] = v[77] * v[110];
   v[130] = v[114] + v[111];
   v[131] = 0 - v[130];
   v[132] = -0 - v[58] * v[110];
   v[133] = v[132] + v[108];
   v[134] = v[48] * v[131] + v[50] * v[133];
   v[135] = -3.9495582524087e-06 + v[134];
   v[115] = 2.94857751577495 - v[115] - v[106] * v[120];
   v[120] = 0 - v[106] * v[121];
   v[136] = v[58] * v[115] - v[77] * v[120];
   v[121] = 2.9489081 - v[96] * v[121];
   v[137] = v[58] * v[120] - v[77] * v[121];
   v[138] = v[136] * v[58] - v[137] * v[77];
   v[97] = 0.01154503 + v[100] + v[103] - v[97] - v[98] * v[98] / v[109];
   v[100] = v[97] - 2. * v[108] * v[58] - v[132] * v[58] - v[138];
   v[103] = 1.1342228e-05 / v[109];
   v[139] = 2.9481043e-07 - v[98] * v[103];
   v[132] = v[139] + v[108] * v[77] + v[132] * v[77] - v[111] * v[58];
   v[108] = v[50] * v[100] + v[56] * v[132];
   v[103] = 0.0041154227 - 1.1342228e-05 * v[103];
   v[114] = 2. * v[111] * v[77] + v[103] + v[114] * v[77] - v[138];
   v[132] = v[50] * v[132] + v[56] * v[114];
   v[77] = v[108] * v[50] + v[132] * v[56];
   v[111] = 0.0068190927 + v[138] + v[77];
   v[140] = v[135] / v[111];
   v[141] = v[119] - v[129] * v[140];
   v[142] = (0 - v[32]) * x[10];
   v[131] = v[50] * v[131] + v[56] * v[133];
   v[132] = -5.9386804e-07 + v[132] * v[50] + v[108] * v[48];
   v[108] = -0.114437878273289 - v[131] - v[132] * v[140];
   v[143] = v[45] * x[10];
   v[144] = 1.1441233 + v[110] - v[135] * v[140];
   v[145] = (0 - v[41]) * x[10];
   v[146] = 0 - v[104];
   v[123] = v[123] + v[122];
   v[66] = v[126] + v[66] - v[119];
   v[147] = v[50] * v[123] + v[56] * v[66];
   v[81] = v[101] + v[81] - v[119];
   v[125] = v[125] + v[105];
   v[148] = v[50] * v[81] + v[56] * v[125];
   v[149] = 0.066456836647354 + v[147] * v[50] + v[148] * v[48];
   v[140] = 2. * v[104] * v[50] + 2. * v[102] * v[48] + v[56] * v[102] + v[50] * v[146] - v[149] * v[140];
   v[104] = v[75] * x[10];
   v[150] = v[32] * v[35];
   v[61] = v[61] * v[61];
   v[103] = -0.039577832148846 * v[76] + 2.9481043e-07 * v[63] + -8.1296676e-08 * (v[61] - v[51]) + -1 * (v[78] + (0.0006058328 * v[68]) / 0.0006058328) + v[103] * v[112] + v[139] * v[99] + v[126] * v[113] + v[105] * v[116] + (1.1342228e-05 * v[82]) / v[109] - -1.232863523889e-06 * v[60] - -0.002232964 * v[62] - 1.1342228e-05 * v[118];
   v[117] = v[117] - v[59];
   v[139] = -0.09107395141674 * v[60] + -0.039577832148846 * v[117] + 0.0051966433 * v[63] + -8.1296676e-08 * v[118] + v[80] * v[67] + v[64] * v[69] + v[87] * v[94] + v[71] * v[95] + v[139] * v[112] + v[97] * v[99] + v[122] * v[113] + v[101] * v[116] + (v[98] * v[82]) / v[109] - 2.9481043e-07 * v[62] - 1.1342228e-05 * (v[61] - v[57]);
   v[33] = v[45] * v[33] - v[35] * v[75];
   v[97] = v[45] * v[45];
   v[69] = v[32] * v[32];
   v[42] = -3.9495582524087e-06 * (v[42] - v[34]) + -0.005308822 * v[46] + 0.009210478 * v[150] + v[56] * v[103] + v[50] * v[139] - -0.066456836647354 * v[33] - -5.9386804e-07 * v[47] - 1.1055176e-07 * (v[97] - v[69]);
   v[34] = x[16] - v[42];
   v[35] = v[35] * v[35];
   v[67] = -1.1441233 * v[43] + -3.9495582524087e-06 * v[46] + 0.114437878273289 * v[47] + v[79] + v[141] * v[142] + v[108] * v[143] + v[144] * v[145] + v[140] * v[104] + (v[135] * v[34]) / v[111] - -0.066456836647354 * (v[35] + v[69]);
   v[117] = -2.493714 * v[117] + -1.232863523889e-06 * v[118] + -0.039577832148846 * v[63] + -1 * (0.4551941 * v[89] + -0.00015326385347 * v[91] + 0.00042046279017 * v[83] + 0.00015326385347 * v[84] + 0.00042046279017 * v[85] - 0.0194326913231 * (v[92] + v[93])) + v[126] * v[112] + v[122] * v[99] + v[121] * v[113] + v[120] * v[116] + (v[96] * v[82]) / v[109] - -0.09107395141674 * (v[61] + v[51]);
   v[61] = 2.493714 * v[76] + -0.039577832148846 * v[62] + -0.09107395141674 * v[118] + v[88] * v[94] + v[64] * v[95] + v[105] * v[112] + v[101] * v[99] + v[120] * v[113] + v[115] * v[116] + (v[106] * v[82]) / v[109] - -1.232863523889e-06 * (v[61] + v[57]);
   v[137] = v[137] + v[130];
   v[136] = v[136] + v[133];
   v[133] = v[149] / v[111];
   v[131] = 0.114437878273289 + v[137] * v[50] + v[136] * v[48] + v[131] - v[129] * v[133];
   v[66] = v[48] * v[123] + v[50] * v[66];
   v[125] = v[48] * v[81] + v[50] * v[125];
   v[81] = v[66] * v[50] + v[125] * v[48] + v[119] - v[132] * v[133];
   v[121] = v[121] - v[110];
   v[115] = v[115] - v[110];
   v[123] = v[50] * v[115] + v[56] * v[120];
   v[120] = v[50] * v[120] + v[56] * v[121];
   v[130] = v[123] * v[50] + v[120] * v[56];
   v[115] = 1.1441233 + v[121] + v[115] + v[110] - v[130] - v[149] * v[133];
   v[121] = 1.1441233 * v[33] + 0.114437878273289 * v[150] + -0.066456836647354 * v[46] + v[53] * v[117] + v[48] * v[61] + v[131] * v[142] + v[81] * v[143] + v[140] * v[145] + v[115] * v[104] + (v[149] * v[34]) / v[111] - -3.9495582524087e-06 * (v[35] + v[97]);
   v[133] = -1 * v[37];
   v[105] = 0.06881 * v[44];
   v[101] = v[133] * v[115] - v[105] * v[140];
   v[57] = v[133] * v[140] - v[105] * v[144];
   v[76] = v[101] * v[133] - v[57] * v[105];
   v[118] = v[31] * v[73] + v[29] * v[72];
   v[95] = v[118] * v[140];
   v[148] = v[147] * v[56] + v[148] * v[50] + v[119];
   v[147] = v[148] / v[111];
   v[120] = v[120] * v[50] + v[123] * v[48] - v[149] * v[147];
   v[123] = v[133] * v[120];
   v[119] = v[95] - v[123];
   v[146] = v[50] * v[102] + v[48] * v[146] - v[135] * v[147];
   v[102] = v[118] * v[144] - v[133] * v[146];
   v[94] = v[81] * v[133] + v[119] * v[133] - v[108] * v[105] - v[102] * v[105];
   v[62] = v[105] * v[120] - v[118] * v[115];
   v[126] = v[105] * v[146];
   v[95] = v[126] - v[95];
   v[122] = v[131] * v[133] + v[62] * v[133] - v[141] * v[105] - v[95] * v[105];
   v[51] = 0 - v[122];
   v[89] = 0.008665591 + v[44] * v[94] + v[31] * v[51];
   v[77] = 0.024117988 + v[114] + v[100] + v[138] - v[77] - v[132] * v[132] / v[111];
   v[125] = -0.066456836647354 + v[66] * v[56] + v[125] * v[50] - v[132] * v[147];
   v[130] = 1.1441233 + v[110] + v[130] - v[148] * v[147];
   v[110] = v[118] * v[146] - v[133] * v[130];
   v[66] = 2. * v[108] * v[118] + v[77] + v[102] * v[118] - 2. * v[125] * v[133] - v[110] * v[133] - v[76];
   v[114] = v[129] / v[111];
   v[128] = 1.1055176e-07 + 2. * v[127] * v[50] + 2. * v[124] * v[48] + v[56] * v[124] + v[50] * v[128] - v[132] * v[114];
   v[147] = 3.9495582524087e-06 + v[137] * v[56] + v[136] * v[50] - v[134] - v[129] * v[147];
   v[133] = v[141] * v[118] + v[128] + v[125] * v[105] + v[110] * v[105] - v[81] * v[118] - v[147] * v[133] - v[119] * v[118];
   v[136] = v[29] * v[66] + v[31] * v[133];
   v[114] = 0.018809166 + v[138] - v[129] * v[114];
   v[138] = v[105] * v[130] - v[118] * v[120];
   v[118] = 2. * v[147] * v[105] + v[114] + v[138] * v[105] - 2. * v[131] * v[118] - v[62] * v[118] - v[76];
   v[133] = v[29] * v[133] + v[31] * v[118];
   v[105] = v[136] * v[29] + v[133] * v[31];
   v[137] = 0.0138978325 + v[76] + v[105];
   v[134] = v[89] / v[137];
   v[127] = 0.0143962465 + v[76] - v[89] * v[134];
   v[124] = (0 - v[27]) * x[9];
   v[133] = -4.669831e-06 + v[133] * v[44] + v[136] * v[31];
   v[134] = 2.9172152e-06 + -1 * 2. * v[122] * v[44] + -1 * 2. * v[94] * v[31] + v[31] * v[94] + v[29] * v[51] - v[133] * v[134];
   v[51] = v[15] * x[9];
   v[126] = v[123] - v[126];
   v[138] = v[138] + v[147];
   v[123] = 0 - v[138];
   v[110] = v[110] + v[125];
   v[122] = v[31] * v[123] + v[44] * v[110];
   v[94] = 4.70038291209208e-05 + v[122];
   v[136] = v[94] / v[137];
   v[100] = v[126] - v[89] * v[136];
   v[93] = (0 - v[40]) * x[9];
   v[138] = -1 * (v[57] + v[138]);
   v[101] = -1 * (v[101] + v[110]);
   v[110] = v[29] * v[123] + v[31] * v[110];
   v[102] = v[102] + v[108];
   v[95] = v[141] + v[95] - v[126];
   v[123] = v[29] * v[102] + v[31] * v[95];
   v[119] = v[81] + v[119] - v[126];
   v[62] = v[62] + v[131];
   v[57] = v[29] * v[119] + v[31] * v[62];
   v[92] = 0.125386070707496 + v[123] * v[44] + v[57] * v[31];
   v[91] = v[92] / v[137];
   v[83] = 0.17063653859 + v[138] * v[44] + v[101] * v[31] + v[110] - v[89] * v[91];
   v[63] = v[23] * x[9];
   v[60] = v[17] * v[40];
   v[16] = v[27] * v[16];
   v[59] = v[27] * v[15];
   v[17] = v[27] * v[17];
   v[75] = v[32] * v[75] - v[45] * v[41];
   v[114] = 0.114437878273289 * v[33] + 1.1055176e-07 * v[47] + -5.9386804e-07 * (v[35] - v[69]) + (v[109] * v[82]) / v[109] + v[55] + v[54] * v[117] + v[58] * v[61] + v[114] * v[142] + v[128] * v[143] + v[141] * v[145] + v[131] * v[104] + (v[129] * v[34]) / v[111] - -3.9495582524087e-06 * v[75] - -0.0172988953 * v[150] - 0.009210478 * v[46];
   v[128] = -0.066456836647354 * v[75] + 0.114437878273289 * v[43] + -0.0119900733 * v[47] + -5.9386804e-07 * v[46] + v[50] * v[103] + v[48] * v[139] + -0.21877 * v[79] + v[128] * v[142] + v[77] * v[143] + v[108] * v[145] + v[81] * v[104] + (v[132] * v[34]) / v[111] - 1.1055176e-07 * v[150] - 0.009210478 * (v[35] - v[97]);
   v[77] = v[15] * v[15];
   v[81] = 4.70038291209208e-05 * (v[60] - v[16]) + -0.0106215965 * v[59] + 0.008665591 * v[17] + v[31] * v[114] + v[29] * v[128] + v[72] * v[67] + v[36] * v[121] - -0.125386070707496 * v[24] - -4.669831e-06 * v[25] - 2.9172152e-06 * (v[77] - v[28]);
   v[35] = x[15] - v[81];
   v[40] = v[27] * v[23] - v[15] * v[40];
   v[42] = 0.17063653859 * v[24] + 2.9172152e-06 * v[25] + -4.669831e-06 * (v[26] - v[28]) + v[30] * v[67] + v[37] * v[121] + -1 * (v[42] + (v[111] * v[34]) / v[111]) + v[127] * v[124] + v[134] * v[51] + v[100] * v[93] + v[83] * v[63] + (v[89] * v[35]) / v[137] - 4.70038291209208e-05 * v[40] - -0.0111200105 * v[17] - 0.008665591 * v[59];
   v[16] = v[16] - v[60];
   v[75] = 1.1441233 * v[75] + -3.9495582524087e-06 * v[150] + -0.066456836647354 * v[47] + v[52] * v[117] + v[49] * v[61] + v[147] * v[142] + v[125] * v[143] + v[146] * v[145] + v[120] * v[104] + (v[148] * v[34]) / v[111] - 0.114437878273289 * (v[97] + v[69]);
   v[105] = 0.025017843 + v[118] + v[66] + v[76] - v[105] - v[133] * v[133] / v[137];
   v[110] = -0.17063653859 - v[110] - v[133] * v[136];
   v[95] = v[31] * v[102] + v[44] * v[95];
   v[62] = v[31] * v[119] + v[44] * v[62];
   v[119] = v[95] * v[44] + v[62] * v[31] + v[126] - v[133] * v[91];
   v[128] = -0.125386070707496 * v[40] + 0.17063653859 * v[16] + -0.000498413999999999 * v[25] + -4.669831e-06 * v[59] + v[44] * v[114] + v[31] * v[128] + 0.06881 * v[75] + v[73] * v[67] + v[38] * v[121] + v[134] * v[124] + v[105] * v[51] + v[110] * v[93] + v[119] * v[63] + (v[133] * v[35]) / v[137] - 2.9172152e-06 * v[17] - 0.008665591 * (v[26] - v[77]);
   v[114] = -1 * v[39];
   v[144] = v[144] - v[130];
   v[115] = v[115] - v[130];
   v[102] = v[29] * v[115] + v[31] * v[140];
   v[140] = v[29] * v[140] + v[31] * v[144];
   v[118] = v[102] * v[29] + v[140] * v[31];
   v[115] = 3.3222008 + v[144] + v[115] + v[130] - v[118] - v[92] * v[91];
   v[144] = -0.266 * v[14];
   v[91] = 0 - v[146];
   v[146] = -1 * 2. * v[146] * v[44] + -1 * 2. * v[120] * v[31] + v[31] * v[120] + v[29] * v[91] - v[92] * v[136];
   v[66] = v[114] * v[115] - v[144] * v[146];
   v[136] = 3.3222008 + v[130] - v[94] * v[136];
   v[76] = v[114] * v[146] - v[144] * v[136];
   v[147] = v[66] * v[114] - v[76] * v[144];
   v[57] = v[123] * v[31] + v[57] * v[29] + v[126];
   v[123] = v[57] / v[137];
   v[62] = -0.125386070707496 + v[95] * v[31] + v[62] * v[29] - v[133] * v[123];
   v[118] = 3.3222008 + v[130] + v[118] - v[57] * v[123];
   v[130] = -0 - v[114] * v[118];
   v[105] = v[105] - 2. * v[62] * v[114] - v[130] * v[114] - v[147];
   v[101] = -4.70038291209208e-05 + v[138] * v[31] + v[101] * v[29] - v[122] - v[89] * v[123];
   v[134] = v[134] + v[62] * v[144] + v[130] * v[144] - v[101] * v[114];
   v[138] = v[14] * v[105] + v[12] * v[134];
   v[122] = v[144] * v[118];
   v[127] = 2. * v[101] * v[144] + v[127] + v[122] * v[144] - v[147];
   v[134] = v[14] * v[134] + v[12] * v[127];
   v[95] = v[138] * v[14] + v[134] * v[12];
   v[140] = v[140] * v[44] + v[102] * v[31] - v[92] * v[123];
   v[102] = v[144] * v[140];
   v[123] = v[44] * v[120] + v[31] * v[91] - v[94] * v[123];
   v[91] = v[144] * v[123];
   v[120] = v[83] * v[114] + v[102] * v[114] - v[100] * v[144] - v[91] * v[144];
   v[126] = -1 * 2. * v[120];
   v[125] = v[114] * v[140];
   v[61] = -0 - v[125];
   v[117] = -0 - v[114] * v[123];
   v[144] = v[119] * v[114] + v[61] * v[114] - v[110] * v[144] - v[117] * v[144];
   v[114] = -1 * 2. * v[144];
   v[120] = 0 - v[120];
   v[69] = -0.0062872856 + v[126] * v[12] + v[114] * v[14] + v[14] * v[144] + v[11] * v[120];
   v[97] = 0.041265123 + v[147];
   v[150] = v[69] / v[97];
   v[47] = 0.0037761838 + v[147] + v[95] - v[69] * v[150];
   v[60] = (0 - v[9]) * x[8];
   v[120] = 2.2408528e-06 + v[126] * v[14] + v[114] * v[11] - v[12] * v[144] - v[14] * v[120];
   v[150] = 1.1495946e-05 + v[134] * v[14] + v[138] * v[11] - v[120] * v[150];
   v[117] = v[117] + v[110];
   v[125] = v[125] - v[91];
   v[91] = v[100] + v[91] - v[125];
   v[134] = v[14] * v[117] + v[12] * v[91];
   v[61] = v[119] + v[61] - v[125];
   v[102] = v[102] + v[83];
   v[138] = v[14] * v[61] + v[12] * v[102];
   v[122] = v[122] + v[101];
   v[76] = -1 * (v[76] + v[122]);
   v[130] = v[130] + v[62];
   v[66] = -1 * (v[66] + v[130]);
   v[122] = 0 - v[122];
   v[114] = v[11] * v[122] + v[14] * v[130];
   v[144] = -6.1669634999453e-05 + v[76] * v[12] + v[66] * v[14] + v[114];
   v[126] = v[144] / v[97];
   v[27] = v[134] * v[12] + v[138] * v[14] + v[125] - v[69] * v[126];
   v[23] = (0 - v[21]) * x[8];
   v[122] = v[14] * v[122] + v[12] * v[130];
   v[66] = -0.195544491531532 + v[76] * v[14] + v[66] * v[11] - v[122];
   v[76] = v[66] / v[97];
   v[138] = 0.0583478584286286 + v[134] * v[14] + v[138] * v[11] - v[69] * v[76];
   v[134] = v[6] * v[6];
   v[130] = x[8] * v[21];
   v[15] = v[9] * v[6];
   v[139] = v[9] * x[8];
   v[110] = -3.3222008 * v[16] + 4.70038291209208e-05 * v[59] + 0.17063653859 * v[25] + -1 * v[75] + v[100] * v[124] + v[110] * v[51] + v[136] * v[93] + v[146] * v[63] + (v[94] * v[35]) / v[137] - -0.125386070707496 * (v[26] + v[28]);
   v[119] = 3.3222008 * v[24] + 0.17063653859 * v[17] + -0.125386070707496 * v[59] + v[74] * v[67] + v[31] * v[121] + v[83] * v[124] + v[119] * v[51] + v[146] * v[93] + v[115] * v[63] + (v[92] * v[35]) / v[137] - 4.70038291209208e-05 * (v[26] + v[77]);
   v[81] = 0.195544491531532 * v[5] + 2.2408528e-06 * v[7] + 1.1495946e-05 * (v[134] - v[10]) + x[14] - -6.1669634999453e-05 * v[130] - -0.0390159362 * v[15] - -0.0062872856 * v[139] - v[22] * v[110] - v[39] * v[119] - -1 * (v[81] + (v[137] * v[35]) / v[137]);
   v[21] = v[9] * v[3] - v[6] * v[21];
   v[130] = 0 - v[130];
   v[95] = 0.04279212 + v[127] + v[105] + v[147] - v[95] - v[120] * v[120] / v[97];
   v[91] = v[11] * v[117] + v[14] * v[91];
   v[102] = v[11] * v[61] + v[14] * v[102];
   v[61] = -0.0583478584286286 + v[91] * v[12] + v[102] * v[14] - v[120] * v[126];
   v[102] = v[91] * v[14] + v[102] * v[11] + v[125] - v[120] * v[76];
   v[115] = v[115] - v[118];
   v[91] = v[14] * v[115] + v[12] * v[146];
   v[136] = v[136] - v[118];
   v[146] = v[14] * v[146] + v[12] * v[136];
   v[117] = v[91] * v[14] + v[146] * v[12];
   v[127] = 1.6057873 + v[118] + v[117] - v[144] * v[126];
   v[146] = v[146] * v[14] + v[91] * v[11] - v[66] * v[126];
   v[117] = 1.6057873 + v[136] + v[115] + v[118] - v[117] - v[66] * v[76];
   v[136] = v[0] * -0.06881 * v[0] + v[1] * v[2];
   v[115] = -1 * 0.1615 * v[0];
   v[76] = -1 * 2. * v[123];
   v[91] = -1 * 2. * v[140];
   v[126] = 0 - v[123];
   v[105] = v[125] / v[97];
   v[147] = v[76] * v[12] + v[91] * v[14] + v[14] * v[140] + v[11] * v[126] - v[144] * v[105];
   v[9] = v[136] * v[127] - v[115] * v[147];
   v[6] = 0.1615 * (0 - v[1]);
   v[3] = v[6] * v[147];
   v[59] = v[136] * v[146];
   v[126] = v[76] * v[14] + v[91] * v[11] - v[12] * v[140] - v[14] * v[126] - v[66] * v[105];
   v[91] = v[115] * v[126];
   v[76] = v[91] - v[3];
   v[91] = v[59] - v[91];
   v[83] = v[6] * v[126] - v[136] * v[117];
   v[122] = 0.195544491531532 + v[122] - v[120] * v[105];
   v[125] = 1.6057873 + v[118] - v[125] * v[105];
   v[147] = v[136] * v[147] - v[115] * v[125];
   v[105] = 6.1669634999453e-05 - v[114] - v[69] * v[105];
   v[114] = v[27] * v[136] + v[150] + v[122] * v[6] + v[147] * v[6] - v[102] * v[136] - v[105] * v[115] - v[91] * v[136];
   v[118] = (v[115] * v[117] - v[6] * v[146]) * v[115] - (v[115] * v[146] - v[6] * v[127]) * v[6];
   y[6] = (x[13] - v[0] * (0.0583478584286286 * v[5] + 1.1495946e-05 * v[7] + 2.2408528e-06 * (v[8] - v[10]) + v[12] * v[42] + v[14] * v[128] + v[47] * v[60] + v[150] * v[7] + v[27] * v[23] + v[138] * v[4] + (v[69] * v[81]) / v[97] - -6.1669634999453e-05 * v[21] - -0.001526997 * v[139] - -0.0062872856 * v[15]) - v[1] * (0.195544491531532 * v[21] + 0.0583478584286286 * v[130] + 0.0374889392 * v[7] + 2.2408528e-06 * v[15] + v[14] * v[42] + v[11] * v[128] + -0.266 * (3.3222008 * v[40] + 4.70038291209208e-05 * v[17] + -0.125386070707496 * v[25] + v[31] * v[67] + v[29] * v[121] + v[101] * v[124] + v[62] * v[51] + v[123] * v[93] + v[140] * v[63] + (v[57] * v[35]) / v[137] - 0.17063653859 * (v[77] + v[28])) + v[150] * v[60] + v[95] * v[7] + v[61] * v[23] + v[102] * v[4] + (v[120] * v[81]) / v[97] - 1.1495946e-05 * v[139] - -0.0062872856 * (v[8] - v[134])) - v[2] * (-1.6057873 * v[130] + -6.1669634999453e-05 * v[15] + 0.0583478584286286 * v[7] + v[18] * v[110] + v[13] * v[119] + v[27] * v[60] + v[61] * v[7] + v[127] * v[23] + v[146] * v[4] + (v[144] * v[81]) / v[97] - 0.195544491531532 * (v[8] + v[10])) - v[20] * (1.6057873 * v[5] + 0.0583478584286286 * v[139] + 0.195544491531532 * v[15] + v[19] * v[110] + v[11] * v[119] + v[138] * v[60] + v[102] * v[7] + v[146] * v[23] + v[117] * v[4] + (v[66] * v[81]) / v[97] - -6.1669634999453e-05 * (v[8] + v[134])) - 9.81 * ((v[1] * (v[9] + v[61]) + v[0] * (v[3] + v[27] - v[59] - v[76])) * v[0] + (v[1] * (v[102] + v[91] - v[76]) + v[0] * (v[83] + v[138])) * v[1] + v[76])) / (0.011487184 + (v[1] * v[114] + v[0] * (2. * v[105] * v[6] + v[47] + (v[6] * v[125] - v[136] * v[126]) * v[6] - 2. * v[138] * v[136] - v[83] * v[136] - v[118])) * v[0] + (v[1] * (2. * v[61] * v[136] + v[95] + v[9] * v[136] - 2. * v[122] * v[115] - v[147] * v[115] - v[118]) + v[0] * v[114]) * v[1] + v[118]);
   v[60] = v[0] * y[6] + v[60];
   v[7] = v[1] * y[6] + v[7];
   v[23] = 9.81 * v[0] + v[2] * y[6] + v[23];
   v[20] = 9.81 * v[1] + v[20] * y[6] + v[4];
   y[7] = (v[81] - v[69] * v[60] - v[120] * v[7] - v[144] * v[23] - v[66] * v[20]) / v[97];
   v[81] = -1 * y[7];
   v[124] = v[12] * v[60] + v[14] * v[7] + v[124];
   v[60] = v[14] * v[60] + v[11] * v[7] + v[51];
   v[7] = -0.266 * v[7];
   v[93] = v[18] * v[23] + v[19] * v[20] + v[22] * y[7] + v[93];
   v[20] = v[13] * v[23] + v[11] * v[20] + v[39] * y[7] + v[63];
   y[8] = (v[35] - v[137] * v[81] - v[89] * v[124] - v[133] * v[60] - v[57] * v[7] - v[94] * v[93] - v[92] * v[20]) / v[137];
   v[57] = -1 * v[124];
   v[81] = v[81] + y[8];
   v[142] = v[31] * v[81] + v[44] * v[60] + v[142];
   v[143] = v[29] * v[81] + v[31] * v[60] + v[143];
   v[93] = 0.06881 * v[60] + -1 * v[93];
   v[145] = v[72] * v[81] + v[30] * v[124] + v[73] * v[60] + v[31] * v[7] + v[74] * v[20] + v[145];
   v[81] = v[36] * v[81] + v[37] * v[124] + v[38] * v[60] + v[29] * v[7] + v[31] * v[20] + v[104];
   y[9] = (v[34] - v[111] * v[57] - v[129] * v[142] - v[132] * v[143] - v[148] * v[93] - v[135] * v[145] - v[149] * v[81]) / v[111];
   v[57] = v[57] + y[9];
   v[112] = v[56] * v[57] + v[50] * v[143] + v[112];
   v[57] = v[50] * v[57] + v[48] * v[143] + v[99];
   v[145] = -0.21877 * v[143] + v[145];
   v[116] = v[49] * v[93] + v[48] * v[81] + v[58] * v[142] + v[116];
   y[10] = (v[82] - v[109] * v[142] - 1.1342228e-05 * v[112] - v[98] * v[57] - v[107] * v[145] - v[96] * (v[52] * v[93] + v[53] * v[81] + v[54] * v[142] + v[113]) - v[106] * v[116]) / v[109];
   v[142] = v[142] + y[10];
   y[11] = (v[68] - 0.0006058328 * -1 * v[112] - -1.3623974e-05 * (v[64] * v[142] + v[80] * v[57] + v[84]) - 6.388487e-06 * (v[65] * v[142] + v[64] * v[57] + v[85]) - -0.00015326385347 * (v[71] * v[142] + v[87] * v[57] + v[64] * v[145] + v[88] * v[116] + v[86]) - -0.00042046279017 * (v[70] * v[142] + v[71] * v[57] + v[65] * v[145] + v[64] * v[116] + v[90])) / 0.0006058328;
   // dependent variables without operations
   y[0] = x[7];
   y[1] = x[8];
   y[2] = x[9];
   y[3] = x[10];
   y[4] = x[11];
   y[5] = x[12];
}

