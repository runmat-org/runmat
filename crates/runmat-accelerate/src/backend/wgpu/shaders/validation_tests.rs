//! GPU-free validation of production WGSL against the Naga paired with WGPU.

mod static_cases;
mod double_literals;
mod dynamic_numeric;
mod deep_learning;
mod dynamic_comms_index;
mod static_interp_window_symmetry;
mod elementwise_complex_binary;
mod elementwise_complex_unary;
mod elementwise_real_unary;
mod elementwise_round_digits;
mod static_filter_gradient;
mod static_numeric;
mod dynamic_signal;
mod static_creation_indexing;
mod static_fft_linalg;
mod static_logic_transforms;
mod integer_builders;
mod reduction_scan_gaps;
mod basic_elementwise_pipelines;
mod solve_stochastic;
mod image_normalize_templates;

pub(crate) use crate::wgsl::validation_tests::{validate_generated, RequiredCaps};
