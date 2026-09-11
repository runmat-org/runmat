use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::setfield")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec { name: "setfield", op_kind: GpuOpKind::Custom("setfield"), supported_precisions: &[], broadcast: BroadcastSemantics::None, provider_hooks: &[], constant_strategy: ConstantStrategy::InlineLiteral, residency: ResidencyPolicy::InheritInputs, nan_mode: ReductionNaN::Include, two_pass_threshold: None, workgroup_size: None, accepts_nan_mode: false, notes: "Direct field replacement preserves resident handles. Indexed resident mutation gathers only the selected target for host assignment." };

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::setfield")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec { name: "setfield", shape: ShapeRequirements::Any, constant_strategy: ConstantStrategy::InlineLiteral, elementwise: None, reduction: None, emits_nan: false, notes: "Field assignment is a fusion boundary; only indexed mutation of a resident target gathers data." };
