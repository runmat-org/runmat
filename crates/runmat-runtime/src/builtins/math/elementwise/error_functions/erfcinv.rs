//! Real `erfcinv` execution for RunMat.
//!
//! `erfcinv` computes the inverse complementary error function for real inputs.
//! The implementation uses a monotonic bisection over `erfc` to keep tails stable
//! without depending on a platform-specific inverse special function.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, ERFCINV_ERROR_INTERNAL, ERFCINV_ERROR_INVALID_ARGUMENT,
    ERFCINV_ERROR_INVALID_INPUT, ERFCINV_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericScalar, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use crate::builtins::math::resident_real_unary as real_unary;

const BUILTIN_NAME: &str = "erfcinv";
const MAX_POSITIVE_RESULT: f64 = 32.0;
const BISECTION_STEPS: usize = 110;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::error_functions::erfcinv"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_erfcinv",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may evaluate erfcinv directly on real device buffers; unsupported providers fall back to host evaluation and re-upload when possible.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::error_functions::erfcinv"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes:
        "Fusion planner currently falls back to provider or host elementwise erfcinv evaluation.",
};

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn internal_error(detail: impl AsRef<str>) -> RuntimeError {
    error_with_detail(&ERFCINV_ERROR_INTERNAL, detail)
}

#[runtime_builtin(
    name = "erfcinv",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::error_functions::erfcinv"
)]
async fn erfcinv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    reject_excess_outputs()?;
    if !rest.is_empty() {
        return Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_ARGUMENT,
            "erfcinv accepts exactly one input",
        ));
    }
    match value {
        Value::GpuTensor(handle) => erfcinv_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "complex inputs are not supported",
        )),
        Value::String(_) | Value::StringArray(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "expected real numeric input, got string",
        )),
        Value::SparseTensor(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "sparse inputs are not supported",
        )),
        Value::Bool(_) | Value::LogicalArray(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "logical inputs are not supported",
        )),
        Value::Int(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "integer-class inputs are not supported",
        )),
        Value::CharArray(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "char inputs are not supported",
        )),
        other => erfcinv_real(other),
    }
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(error_with_detail(
            &ERFCINV_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    Ok(())
}

async fn erfcinv_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Err(reason) = real_unary::validate_input(&handle) {
        return Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            reason.detail(),
        ));
    }

    let provider = real_unary::exact_owner(&handle)
        .ok_or_else(|| internal_error("GPU provider unavailable for input"))?;
    match provider.unary_erfcinv(&handle).await {
        Ok(mut out) if real_unary::output_matches(&out, &handle, provider) => {
            real_unary::preserve_residency_intent(&mut out, &handle);
            return Ok(gpu_helpers::resident_gpu_value(out));
        }
        Ok(out) => {
            real_unary::reject_output(&out, &handle, provider);
            return Err(internal_error(
                "provider unary_erfcinv returned malformed output",
            ));
        }
        Err(err) if real_unary::hook_is_unsupported(&err) => {}
        Err(err) => {
            return Err(internal_error(format!(
                "provider unary_erfcinv failed: {err}"
            )))
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let result = erfcinv_tensor(tensor)?;

    let out = real_unary::restore_fallback(&result, &handle, provider).map_err(|err| {
        internal_error(format!(
            "failed to restore fallback result to input provider: {err}"
        ))
    })?;
    Ok(gpu_helpers::resident_gpu_value(out))
}

fn erfcinv_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|detail| error_with_detail(&ERFCINV_ERROR_INVALID_INPUT, detail))?;
    erfcinv_tensor(tensor).map(erfcinv_tensor_into_value)
}

fn erfcinv_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = match tensor.into_numeric_storage().map_err(internal_error)? {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(erfcinv_scalar).collect::<Vec<_>>())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| erfcinv_scalar(f64::from(value)) as f32)
                .collect::<Vec<_>>(),
        ),
        NumericStorage::I8(_)
        | NumericStorage::I16(_)
        | NumericStorage::I32(_)
        | NumericStorage::I64(_)
        | NumericStorage::U8(_)
        | NumericStorage::U16(_)
        | NumericStorage::U32(_)
        | NumericStorage::U64(_) => {
            return Err(error_with_detail(
                &ERFCINV_ERROR_INVALID_INPUT,
                "integer-class tensors are not supported",
            ))
        }
    };
    Tensor::from_numeric_storage(storage, shape).map_err(internal_error)
}

fn erfcinv_tensor_into_value(tensor: Tensor) -> Value {
    if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64 {
        let Some(NumericScalar::F64(value)) = tensor.numeric_value_at(0) else {
            unreachable!("scalar double erfcinv result has F64 storage")
        };
        Value::Num(value)
    } else {
        Value::Tensor(tensor)
    }
}

pub(crate) fn erfcinv_scalar(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=2.0).contains(&value) {
        return f64::NAN;
    }
    if value == 0.0 {
        return f64::INFINITY;
    }
    if value == 2.0 {
        return f64::NEG_INFINITY;
    }
    if value == 1.0 {
        return 0.0;
    }
    if value > 1.0 {
        return -erfcinv_positive_tail(2.0 - value);
    }
    erfcinv_positive_tail(value)
}

fn erfcinv_positive_tail(target: f64) -> f64 {
    debug_assert!(target > 0.0 && target < 1.0);
    let mut lo = 0.0;
    let mut hi = 1.0;
    while hi < MAX_POSITIVE_RESULT && libm::erfc(hi) > target {
        lo = hi;
        hi *= 2.0;
    }
    if libm::erfc(hi) > target {
        return hi;
    }

    for _ in 0..BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        if libm::erfc(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{GpuHandleProvenance, HostTensorView};
    use runmat_value::{CharArray, ComplexTensor, IntValue, LogicalArray, SparseTensor};

    fn erfcinv_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::erfcinv_builtin(value, Vec::new()))
    }

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        if actual.is_nan() && expected.is_nan() {
            return;
        }
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn scalar_values_match_reference_points() {
        match erfcinv_builtin(Value::Num(1.0)).unwrap() {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar, got {other:?}"),
        }
        let cases = [
            (0.3, 0.732_869_077_959_216_6, 2e-14),
            (0.5, 0.476_936_276_204_469_8, 2e-14),
            (1.5, -0.476_936_276_204_469_8, 2e-14),
            (0.999_999_999_999, 8.862_073_205_887_489e-13, 2e-16),
            (1.000_000_000_001, -8.863_057_115_425_171e-13, 2e-16),
            (1e-100, 15.065_574_702_592_645, 5e-13),
            (f64::MIN_POSITIVE, 26.543_258_454_250_98, 5e-13),
        ];
        for (input, expected, tol) in cases {
            assert_close(erfcinv_scalar(input), expected, tol);
        }
    }

    #[test]
    fn tiny_tail_inputs_remain_ordered_and_finite() {
        let realmin = erfcinv_scalar(f64::MIN_POSITIVE);
        let subnormal = erfcinv_scalar(f64::from_bits(1));
        assert!(realmin.is_finite());
        assert!(subnormal.is_finite());
        assert!(subnormal > realmin);
        assert!(subnormal < MAX_POSITIVE_RESULT);
    }

    #[test]
    fn endpoints_and_out_of_domain_match_matlab_shape() {
        assert!(erfcinv_scalar(0.0).is_infinite() && erfcinv_scalar(0.0).is_sign_positive());
        assert!(erfcinv_scalar(2.0).is_infinite() && erfcinv_scalar(2.0).is_sign_negative());
        assert!(erfcinv_scalar(-0.1).is_nan());
        assert!(erfcinv_scalar(2.1).is_nan());
        assert!(erfcinv_scalar(f64::NAN).is_nan());
    }

    #[test]
    fn tensor_preserves_shape_and_single_dtype() {
        let tensor =
            Tensor::new_with_dtype(vec![0.5, 1.0, 1.5, 2.5], vec![2, 2], NumericDType::F32)
                .unwrap();
        match erfcinv_builtin(Value::Tensor(tensor)).unwrap() {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.numeric_dtype(), NumericDType::F32);
                let NumericStorage::F32(values) =
                    out.into_numeric_storage().expect("single storage")
                else {
                    panic!("expected native single storage");
                };
                assert_close(libm::erfc(f64::from(values[0])), 0.5, 1e-6);
                assert_eq!(values[1], 0.0);
                assert_close(libm::erfc(f64::from(values[2])), 1.5, 1e-6);
                assert!(values[3].is_nan());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn rejects_logical_integer_and_char_inputs() {
        assert!(erfcinv_builtin(Value::Bool(true)).is_err());
        assert!(erfcinv_builtin(Value::Int(IntValue::U8(2))).is_err());
        let chars = CharArray::new(vec!['\0', '\u{1}'], 1, 2).unwrap();
        assert!(erfcinv_builtin(Value::CharArray(chars)).is_err());
    }

    #[test]
    fn rejects_complex_string_and_sparse_inputs() {
        assert!(erfcinv_builtin(Value::Complex(1.0, 1.0)).is_err());
        let complex = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
        assert!(erfcinv_builtin(Value::ComplexTensor(complex)).is_err());
        assert!(erfcinv_builtin(Value::String("1".to_string())).is_err());
        let sparse = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
        assert!(erfcinv_builtin(Value::SparseTensor(sparse)).is_err());
    }

    #[test]
    fn rejects_logical_and_integer_class_arrays() {
        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        assert!(erfcinv_builtin(Value::LogicalArray(logical)).is_err());
        let ints = Tensor::new_with_dtype(vec![0.0, 1.0], vec![1, 2], NumericDType::U8).unwrap();
        assert!(erfcinv_builtin(Value::Tensor(ints)).is_err());
    }

    #[test]
    fn gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let host = Tensor::new(vec![0.25, 0.75, 1.25, 1.75], vec![2, 2]).unwrap();
            let cpu = match erfcinv_builtin(Value::Tensor(host.clone())).unwrap() {
                Value::Tensor(tensor) => tensor,
                other => panic!("expected cpu tensor, got {other:?}"),
            };
            let view = HostTensorView {
                data: host.as_f64_slice().expect("double host"),
                shape: &host.shape,
            };
            let mut handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_provenance(
                &mut handle,
                GpuHandleProvenance::Explicit,
            );
            let resident = erfcinv_builtin(Value::GpuTensor(handle)).unwrap();
            let Value::GpuTensor(output) = &resident else {
                panic!("expected gpu tensor, got {resident:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_provenance(output),
                Some(GpuHandleProvenance::Explicit)
            );
            let gpu = test_support::gather(resident).expect("gather");
            assert_eq!(gpu.shape, cpu.shape);
            for (actual, expected) in gpu
                .as_f64_slice()
                .expect("double gpu result")
                .iter()
                .zip(cpu.as_f64_slice().expect("double cpu result"))
            {
                assert_close(*actual, *expected, 1e-12);
            }
        });
    }

    #[test]
    fn gpu_rejects_integer_and_logical_storage_before_provider_dispatch() {
        test_support::with_test_provider(|provider| {
            let integer =
                Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![0, 1]), vec![1, 2])
                    .unwrap();
            let integer_handle =
                gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
            let integer_error = erfcinv_builtin(Value::GpuTensor(integer_handle)).unwrap_err();
            assert_eq!(
                integer_error.identifier(),
                ERFCINV_ERROR_INVALID_INPUT.identifier
            );

            let logical_source = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
            let logical_handle = provider
                .upload(&HostTensorView {
                    data: logical_source
                        .as_f64_slice()
                        .expect("double logical source"),
                    shape: &logical_source.shape,
                })
                .expect("logical upload");
            runmat_accelerate_api::set_handle_logical(&logical_handle, true);
            let logical_error = erfcinv_builtin(Value::GpuTensor(logical_handle)).unwrap_err();
            assert_eq!(
                logical_error.identifier(),
                ERFCINV_ERROR_INVALID_INPUT.identifier
            );
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_provider_keeps_erfcinv_resident() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let tensor = Tensor::new(vec![0.25, 0.5, 1.0, 1.5, 1.75], vec![1, 5]).unwrap();
        let cpu = erfcinv_tensor(tensor.clone()).expect("cpu erfcinv");
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let view = HostTensorView {
            data: tensor.as_f64_slice().expect("double tensor"),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(super::erfcinv_gpu(handle)).expect("gpu erfcinv");
        assert!(
            matches!(gpu_value, Value::GpuTensor(_)),
            "erfcinv should keep WGPU provider results resident"
        );
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-8,
            runmat_accelerate_api::ProviderPrecision::F32 => 2e-4,
        };
        for (actual, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu.materialize_f64().iter())
        {
            assert_close(*actual, *expected, tol);
        }
    }
}
