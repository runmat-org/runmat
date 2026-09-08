use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::structfun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec { name: "structfun", op_kind: GpuOpKind::Custom("host-struct-map"), supported_precisions: &[], broadcast: BroadcastSemantics::None, provider_hooks: &[], constant_strategy: ConstantStrategy::InlineLiteral, residency: ResidencyPolicy::GatherImmediately, nan_mode: ReductionNaN::Include, two_pass_threshold: None, workgroup_size: None, accepts_nan_mode: false, notes: "Executes on the host and gathers provider-resident field values through their owning provider before callback invocation." };

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::structfun")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "structfun",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Host callback execution makes structfun a fusion boundary.",
};
