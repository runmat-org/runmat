use super::*;

impl AccelProvider for WgpuProvider {
    include!("trait_impl_methods/context_constructors_random_poly.rs");
    include!("trait_impl_methods/elementwise_tensor_signal.rs");
    include!("trait_impl_methods/linalg_reduction_core.rs");
    include!("trait_impl_methods/linalg_advanced_pagefun.rs");
    include!("trait_impl_methods/indexing_io_telemetry.rs");
}
