use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::floating_conversions::single"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "single",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_single",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Casts tensors to float32. Providers may implement `unary_single`; otherwise the runtime gathers, converts, and re-uploads to keep gpuArray results resident.",
};

const FUSION_SEMANTICS: crate::builtins::common::spec::FusionSemanticDeclaration =
    crate::builtins::common::spec::FusionSemanticDeclaration {
        binding: runmat_builtins::BuiltinBindingIdentity {
            builtin: runmat_builtins::SINGLE_CATALOG_ENTRY.identity,
            variant: "default",
        },
        revision: 2,
        admit: None,
        admit_static: None,
        required_real_unary: None,
        required_arithmetic: None,
        helpers: &[],
        scalar_operation: Some(runmat_types::ScalarOperation::Cast(
            runmat_types::NumericClass::Single,
        )),
    };

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::floating_conversions::single",
    semantics = FUSION_SEMANTICS,
    host_resources = crate::builtins::math::host_resources::real_cast
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "single",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            crate::builtins::common::spec::write_real_cast_expression(
                runmat_types::NumericClass::Single,
                ctx,
                writer,
            )
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a cast via `f32` then widens back to the WGSL scalar type when necessary.",
};
