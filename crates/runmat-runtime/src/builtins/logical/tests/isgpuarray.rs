//! MATLAB-compatible `isgpuarray` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::BuiltinResult;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isgpuarray")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isgpuarray",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Reports whether the value is a gpuArray without gathering device buffers.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isgpuarray")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isgpuarray",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that executes outside of fusion pipelines.",
};

#[runtime_builtin(
    name = "isgpuarray",
    category = "logical/tests",
    summary = "Return true when a value is stored as a gpuArray handle.",
    keywords = "isgpuarray,gpuarray,gpu,type,logical",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    builtin_path = "crate::builtins::logical::tests::isgpuarray"
)]
async fn isgpuarray_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(value, Value::GpuTensor(_))))
}

fn bool_scalar_type(_: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{Tensor, Value};

    fn run_isgpuarray(value: Value) -> BuiltinResult<Value> {
        block_on(super::isgpuarray_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_gpu_values_report_false() {
        assert_eq!(run_isgpuarray(Value::Num(1.0)).unwrap(), Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_handles_report_true() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = run_isgpuarray(Value::GpuTensor(handle.clone())).expect("isgpuarray");
            assert_eq!(result, Value::Bool(true));
            provider.free(&handle).ok();
        });
    }
}
