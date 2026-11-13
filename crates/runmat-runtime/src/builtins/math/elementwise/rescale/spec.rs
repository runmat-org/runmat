use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType,
};

use super::BUILTIN_NAME;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::rescale")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("range-scale"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction { name: "reduce_min" },
        ProviderHook::Reduction { name: "reduce_max" },
        ProviderHook::Custom("rescale"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat computes the complete range operation on the host and restores the result to the exact provider owner when any operand is resident.",
};
