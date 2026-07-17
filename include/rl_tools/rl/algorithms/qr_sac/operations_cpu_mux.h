// the qr_sac operations are backend-agnostic (the gemm-heavy work happens inside the network
// forward/backward, dispatched via the nn operations mux) — all backends use the cpu operations
#include "../../../rl/algorithms/qr_sac/operations_cpu.h"
