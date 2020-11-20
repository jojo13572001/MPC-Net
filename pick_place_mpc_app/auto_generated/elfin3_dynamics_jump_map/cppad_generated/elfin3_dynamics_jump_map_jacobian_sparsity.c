void elfin3_dynamics_jump_map_jacobian_sparsity(unsigned long const** row,
                                                unsigned long const** col,
                                                unsigned long* nnz) {
   static unsigned long const rows[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
   static unsigned long const cols[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
   *row = rows;
   *col = cols;
   *nnz = 12;
}
