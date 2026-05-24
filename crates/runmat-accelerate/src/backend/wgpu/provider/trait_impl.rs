use super::*;

include!("trait_impl_methods/context_constructors_random_poly.rs");
include!("trait_impl_methods/elementwise_tensor_signal.rs");
include!("trait_impl_methods/linalg_reduction_core.rs");
include!("trait_impl_methods/linalg_advanced_pagefun.rs");
include!("trait_impl_methods/indexing_io_telemetry.rs");

impl AccelProvider for WgpuProvider {
    context_constructors_random_poly_methods!();
    elementwise_tensor_signal_methods!();
    linalg_reduction_core_methods!();
    linalg_advanced_pagefun_methods!();
    indexing_io_telemetry_methods!();
}
