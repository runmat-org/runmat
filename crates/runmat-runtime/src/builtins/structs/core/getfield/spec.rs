use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::getfield")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec { name: "getfield", op_kind: GpuOpKind::Custom("getfield"), supported_precisions: &[], broadcast: BroadcastSemantics::None, provider_hooks: &[], constant_strategy: ConstantStrategy::InlineLiteral, residency: ResidencyPolicy::InheritInputs, nan_mode: ReductionNaN::Include, two_pass_threshold: None, workgroup_size: None, accepts_nan_mode: false, notes: "Direct field retrieval preserves resident handles. Indexed resident access gathers authoritative storage through the owning provider." };

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::getfield")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec { name: "getfield", shape: ShapeRequirements::Any, constant_strategy: ConstantStrategy::InlineLiteral, elementwise: None, reduction: None, emits_nan: false, notes: "Field-path access is a fusion boundary because it inspects host metadata and may gather an indexed resident value." };
