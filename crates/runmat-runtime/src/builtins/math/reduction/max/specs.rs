use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::max")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "max",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_max_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_max",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes:
        "Providers should implement reduce_max_dim / reduce_max. Requests that require omitnan, comparisonmethod overrides, or complex inputs fall back to the host implementation.",
};

use super::fusion_contract;
use super::fusion_pairwise;

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::max", semantics = fusion_pairwise::SEMANTICS, epilogue = crate::builtins::common::spec::FusionEpilogueOperation::Maximum, reduction = fusion_contract::DECLARATION)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "max",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            writer.write_formatted(format_args!("accumulator = max(accumulator, {input});"))
        }),
    }),
    emits_nan: true,
    notes: "Fusion planner emits canonical reduction kernels; providers may substitute custom WGSL via reduce_max_dim hooks.",
};
