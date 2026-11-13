use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy,
    FusionSemanticDeclaration, FusionStructuralOperation, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::NAME;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::ops::matrix_arithmetic::mtimes"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::MatMul,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Binary {
        name: "matmul",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Calls the provider `matmul` hook for supported floating matrix products; integer scalar products gather through exact typed storage, apply saturating class-preserving multiplication, and restore GPU residency when an input was resident.",
};

const FUSION_SEMANTICS: FusionSemanticDeclaration = FusionSemanticDeclaration {
    binding: runmat_builtins::BuiltinBindingIdentity {
        builtin: runmat_builtins::MTIMES_CATALOG_ENTRY.identity,
        variant: "default",
    },
    revision: 2,
    admit: None,
    admit_static: None,
    required_real_unary: None,
    required_arithmetic: None,
    helpers: &[],
    scalar_operation: Some(runmat_types::ScalarOperation::Operator(
        runmat_types::OperatorKind::MatrixMultiply,
    )),
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::ops::matrix_arithmetic::mtimes",
    semantics = FUSION_SEMANTICS,
    structural = FusionStructuralOperation::MatrixMultiply
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Exact two-input, one-output calls declare a matrix-product graph operation. Existing matmul execution retains class, shape, layout and provider admission; unsupported operands use the ordinary runtime path.",
};
