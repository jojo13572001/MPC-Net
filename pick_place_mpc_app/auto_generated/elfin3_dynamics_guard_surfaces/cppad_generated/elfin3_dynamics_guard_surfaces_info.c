void elfin3_dynamics_guard_surfaces_info(const char** baseName,
                                         unsigned long* m,
                                         unsigned long* n,
                                         unsigned int* indCount,
                                         unsigned int* depCount) {
   *baseName = "double  d";
   *m = 1;
   *n = 13;
   *depCount = 1; // number of dependent array variables
   *indCount = 1; // number of independent array variables
}

