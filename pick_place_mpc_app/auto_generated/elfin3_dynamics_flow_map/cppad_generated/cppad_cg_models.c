void cppad_cg_models(char const *const** names,
                     int* count) {
   static const char* const models[] = {
      "elfin3_dynamics_flow_map"};
   *names = models;
   *count = 1;
}
