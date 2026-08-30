//! MATLAB-compatible `uint32` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, UINT32_ERROR_INTERNAL as ERROR_INTERNAL,
    UINT32_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    UINT32_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::elementwise::integer_cast::{cast_value, CastError, IntegerTarget};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "uint32";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::uint32")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "uint32",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cast_to_integer")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Real gpuArray inputs use the provider resident integer-cast hook. Paired-complex inputs use exact owner-resolved fallback, and both return native uint32 gpuArray storage.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::uint32")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "uint32",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Resident integer cast uses provider-native integer storage; fusion can target the provider hook when integer buffers are supported.",
};

#[runtime_builtin(
    name = "uint32",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::uint32"
)]
async fn uint32_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(error_with_detail(
            &ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    cast_value(value, IntegerTarget::U32)
        .await
        .map_err(|cause| match cause {
            CastError::Unsupported(type_name) => conversion_error(&type_name),
            CastError::Internal(detail) => error_with_detail(&ERROR_INTERNAL, detail),
        })
}

fn conversion_error(type_name: &str) -> RuntimeError {
    error_with_detail(
        &ERROR_INVALID_INPUT,
        format!("conversion to uint32 from {type_name} is not possible"),
    )
}

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let message = format!("{}: {}", error.message, detail);
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataOwned, HostTensorView, IntegerElementType};
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage, Tensor};

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(uint32_builtin(value, Vec::new()))
    }

    #[test]
    fn uint32_scalar_saturates_and_rounds() {
        assert_eq!(
            call(Value::Num(3.5)).expect("uint32"),
            Value::Int(IntValue::U32(4))
        );
        assert_eq!(
            call(Value::Num(-1.0)).expect("uint32"),
            Value::Int(IntValue::U32(0))
        );
        assert_eq!(
            call(Value::Num(f64::INFINITY)).expect("uint32"),
            Value::Int(IntValue::U32(u32::MAX))
        );
        assert_eq!(
            call(Value::Num(f64::NAN)).expect("uint32"),
            Value::Int(IntValue::U32(0))
        );
    }

    #[test]
    fn uint32_tensor_preserves_shape_and_class() {
        let tensor =
            Tensor::new(vec![-2.0, 2.49, 2.5, (u32::MAX as f64) + 99.0], vec![2, 2]).unwrap();
        let result = call(Value::Tensor(tensor)).expect("uint32");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.materialize_f64(), vec![0.0, 2.0, 3.0, u32::MAX as f64]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U32(vec![0, 2, 3, u32::MAX]))
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn uint32_preserves_complex_integer_input() {
        let result = call(Value::Complex(1.0, 0.0)).expect("complex integer conversion");
        let Value::ComplexTensor(tensor) = result else {
            panic!("expected typed complex integer result");
        };
        assert_eq!(
            tensor.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U32(vec![1]),
                    IntegerStorage::U32(vec![0]),
                )
                .expect("matching components")
            )
        );
    }

    #[test]
    fn uint32_gpu_roundtrip_stays_resident() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 4.4, 5.6, (u32::MAX as f64) + 99.0], vec![2, 2])
                .expect("source");
            let handle = provider
                .upload(&HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload");
            let result = call(Value::GpuTensor(handle)).expect("uint32");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpuArray result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U32)
            );
            assert_eq!(handle.shape, vec![2, 2]);
            assert_eq!(
                block_on(provider.download_integer(&handle))
                    .expect("download uint32 cast")
                    .data,
                HostIntegerDataOwned::U32(vec![0, 4, 6, u32::MAX])
            );
        });
    }
}
