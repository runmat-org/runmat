//! MATLAB-compatible `realsqrt` builtin.
//!
//! `realsqrt` is the real-domain square-root helper: nonnegative real inputs
//! are evaluated element-wise, while complex inputs and negative real values
//! raise errors instead of promoting to complex results.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, SparseTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::dispatcher::download_handle_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "realsqrt";

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real elementwise square-root result.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric, logical, char, sparse, or gpuArray input.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = realsqrt(X)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REALSQRT.INVALID_INPUT",
    identifier: Some("RunMat:realsqrt:InvalidInput"),
    when: "Input cannot be interpreted as real numeric, logical, char, sparse, or gpuArray data.",
    message: "realsqrt: invalid input",
};

const ERROR_DOMAIN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REALSQRT.DOMAIN",
    identifier: Some("RunMat:realsqrt:ComplexResult"),
    when: "At least one real input value is negative and would require a complex result.",
    message: "realsqrt: input must be nonnegative",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REALSQRT.INTERNAL",
    identifier: Some("RunMat:realsqrt:Internal"),
    when: "Internal tensor construction or provider interaction failed.",
    message: "realsqrt: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INVALID_INPUT, ERROR_DOMAIN, ERROR_INTERNAL];

pub const REALSQRT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::realsqrt")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "realsqrt",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction { name: "reduce_min" },
        ProviderHook::Unary { name: "unary_sqrt" },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat uses provider sqrt kernels only after proving gpuArray inputs are nonnegative; otherwise it gathers to enforce real-domain errors.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::realsqrt"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "realsqrt",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion sink because negative inputs must raise a domain error instead of producing NaN.",
};

#[runtime_builtin(
    name = "realsqrt",
    category = "math/elementwise",
    summary = "Compute real square roots and reject values that require complex results.",
    keywords = "realsqrt,square root,real,elementwise,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::realsqrt::REALSQRT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::realsqrt"
)]
async fn realsqrt_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => realsqrt_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "complex input is not supported",
        )),
        Value::CharArray(chars) => realsqrt_char_array(chars),
        Value::SparseTensor(sparse) => realsqrt_sparse(sparse),
        Value::String(_) | Value::StringArray(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "expected real numeric input",
        )),
        other => realsqrt_real(other),
    }
}

async fn realsqrt_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "complex gpuArray input is not supported",
        ));
    }

    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match gpu_has_negative_input(provider, &handle).await {
            Ok(true) => {
                return Err(error_with_detail(
                    &ERROR_DOMAIN,
                    "gpuArray contains negative values",
                ));
            }
            Ok(false) => {
                if let Ok(out) = provider.unary_sqrt(&handle).await {
                    return Ok(gpu_helpers::resident_gpu_value(out));
                }
            }
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(err);
                }
                // Fall back to host evaluation when reduction proof is unavailable.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    realsqrt_tensor(tensor)
}

async fn gpu_has_negative_input(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .await
        .map_err(|e| internal_error(format!("realsqrt: reduce_min failed: {e}")))?;
    let download = download_handle_async(provider, &min_handle)
        .await
        .map_err(|e| internal_error(format!("realsqrt: reduce_min download failed: {e}")));
    let _ = provider.free(&min_handle);
    let host = download?;
    if host.data.iter().any(|value| value.is_nan()) {
        return Err(internal_error(
            "realsqrt: reduce_min result contained NaN; host validation required",
        ));
    }
    Ok(host.data.iter().any(|&value| value < 0.0))
}

fn realsqrt_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))?;
    realsqrt_tensor(tensor)
}

fn realsqrt_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let input_dtype = tensor.dtype;
    let output_dtype = if matches!(input_dtype, runmat_builtins::NumericDType::F32) {
        runmat_builtins::NumericDType::F32
    } else {
        runmat_builtins::NumericDType::F64
    };
    let shape = tensor.shape.clone();
    let data = tensor::tensor_into_values_f64(tensor);
    ensure_nonnegative(&data)?;
    let data: Vec<f64> = data
        .into_iter()
        .map(|value| canonical_zero(value.sqrt()))
        .collect();
    let out = Tensor::new_with_dtype(data, shape, output_dtype)
        .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))?;
    Ok(tensor_into_realsqrt_value(tensor::coerce_tensor_dtype(
        out,
        output_dtype,
    )))
}

fn realsqrt_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = chars
        .data
        .iter()
        .map(|&ch| canonical_zero((ch as u32 as f64).sqrt()))
        .collect();
    let out = Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))?;
    Ok(tensor::tensor_into_value(out))
}

fn realsqrt_sparse(sparse: SparseTensor) -> BuiltinResult<Value> {
    let values = sparse
        .integer_storage()
        .map(|storage| {
            storage
                .exact_values()
                .into_iter()
                .map(|value| value.to_f64())
                .collect()
        })
        .unwrap_or_else(|| sparse.values.clone());
    ensure_nonnegative(&values)?;
    let values: Vec<f64> = values
        .into_iter()
        .map(|value| canonical_zero(value.sqrt()))
        .collect();
    SparseTensor::new(
        sparse.rows,
        sparse.cols,
        sparse.col_ptrs,
        sparse.row_indices,
        values,
    )
    .map(Value::SparseTensor)
    .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))
}

fn tensor_into_realsqrt_value(tensor: Tensor) -> Value {
    if tensor.dtype == runmat_builtins::NumericDType::F64 {
        tensor::tensor_into_value(tensor)
    } else {
        Value::Tensor(tensor)
    }
}

fn ensure_nonnegative(values: &[f64]) -> BuiltinResult<()> {
    if values.iter().any(|&value| value < 0.0) {
        return Err(error_with_detail(
            &ERROR_DOMAIN,
            "input contains negative values",
        ));
    }
    Ok(())
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn internal_error(detail: impl std::fmt::Display) -> RuntimeError {
    error_with_detail(&ERROR_INTERNAL, detail)
}

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType,
        ResolveContext, SparseTensor, Type,
    };

    use crate::builtins::common::test_support;

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(realsqrt_builtin(value))
    }

    #[test]
    fn descriptor_covers_core_form() {
        assert_eq!(REALSQRT_DESCRIPTOR.signatures[0].label, "Y = realsqrt(X)");
    }

    #[test]
    fn type_resolver_preserves_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn scalar_nonnegative_values_return_real_roots() {
        match call(Value::Num(9.0)).unwrap() {
            Value::Num(value) => assert!((value - 3.0).abs() < 1.0e-12),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
        match call(Value::Int(IntValue::I32(16))).unwrap() {
            Value::Num(value) => assert!((value - 4.0).abs() < 1.0e-12),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
        match call(Value::Bool(true)).unwrap() {
            Value::Num(value) => assert!((value - 1.0).abs() < 1.0e-12),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }

    #[test]
    fn scalar_negative_errors_instead_of_promoting_complex() {
        let err = call(Value::Num(-1.0)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
    }

    #[test]
    fn nan_and_infinity_follow_ieee_real_sqrt() {
        match call(Value::Tensor(
            Tensor::new(vec![f64::NAN, f64::INFINITY], vec![1, 2]).unwrap(),
        ))
        .unwrap()
        {
            Value::Tensor(tensor) => {
                assert!(tensor.data[0].is_nan());
                assert!(tensor.data[1].is_infinite());
                assert!(tensor.data[1].is_sign_positive());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn dense_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![2, 2]).unwrap();
        match call(Value::Tensor(tensor)).unwrap() {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn dense_typed_integer_tensor_reads_storage_exactly_and_returns_double() {
        let mut tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 4, 9]), vec![2, 2]).unwrap();
        tensor.data.fill(f64::NAN);

        match call(Value::Tensor(tensor)).unwrap() {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.dtype, NumericDType::F64);
                assert!(out.integer_storage().is_none());
                assert_eq!(out.data, vec![0.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn dense_typed_integer_negative_errors_from_exact_storage() {
        let mut tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, -4]), vec![1, 2]).unwrap();
        tensor.data.fill(4.0);

        let err = call(Value::Tensor(tensor)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
    }

    #[test]
    fn single_tensor_preserves_dtype() {
        let tensor = Tensor::new_with_dtype(vec![2.0, 9.0], vec![1, 2], NumericDType::F32).unwrap();
        match call(Value::Tensor(tensor)).unwrap() {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.dtype, NumericDType::F32);
                assert_eq!(out.data[0], (2.0f32.sqrt()) as f64);
                assert_eq!(out.data[1], 3.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn dense_tensor_with_negative_errors() {
        let tensor = Tensor::new(vec![1.0, -4.0], vec![1, 2]).unwrap();
        let err = call(Value::Tensor(tensor)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
    }

    #[test]
    fn logical_and_char_inputs_are_supported() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        match call(Value::LogicalArray(logical)).unwrap() {
            Value::Tensor(out) => assert_eq!(out.data, vec![1.0, 0.0, 1.0, 0.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        match call(Value::CharArray(chars)).unwrap() {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert!((out.data[0] - (65.0f64).sqrt()).abs() < 1.0e-12);
                assert!((out.data[1] - (90.0f64).sqrt()).abs() < 1.0e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn complex_inputs_error() {
        let err = call(Value::Complex(1.0, 0.0)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);

        let tensor = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
        let err = call(Value::ComplexTensor(tensor)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn string_inputs_error() {
        let err = call(Value::String("9".to_string())).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn sparse_inputs_preserve_sparsity() {
        let sparse =
            SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![4.0, 9.0, 16.0]).unwrap();
        match call(Value::SparseTensor(sparse)).unwrap() {
            Value::SparseTensor(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 2);
                assert_eq!(out.col_ptrs, vec![0, 2, 3]);
                assert_eq!(out.row_indices, vec![0, 2, 1]);
                assert_eq!(out.values, vec![2.0, 3.0, 4.0]);
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[test]
    fn sparse_typed_integer_values_read_storage_exactly_and_return_double() {
        let mut sparse = SparseTensor::new_integer(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            IntegerStorage::U16(vec![4, 9, 16]),
        )
        .unwrap();
        sparse.values.fill(f64::NAN);

        match call(Value::SparseTensor(sparse)).unwrap() {
            Value::SparseTensor(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 2);
                assert_eq!(out.col_ptrs, vec![0, 2, 3]);
                assert_eq!(out.row_indices, vec![0, 2, 1]);
                assert_eq!(out.values, vec![2.0, 3.0, 4.0]);
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[test]
    fn sparse_typed_integer_negative_errors_from_exact_storage() {
        let mut sparse =
            SparseTensor::new_integer(2, 1, vec![0, 1], vec![1], IntegerStorage::I16(vec![-4]))
                .unwrap();
        sparse.values.fill(4.0);

        let err = call(Value::SparseTensor(sparse)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
    }

    #[test]
    fn sparse_negative_value_errors() {
        let sparse = SparseTensor::new(2, 1, vec![0, 1], vec![1], vec![-4.0]).unwrap();
        let err = call(Value::SparseTensor(sparse)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
    }

    #[test]
    fn gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                })
                .unwrap();
            let result = call(Value::GpuTensor(handle)).unwrap();
            let gathered = test_support::gather(result).unwrap();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![0.0, 1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn gpu_negative_value_errors() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -4.0], vec![1, 2]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                })
                .unwrap();
            let err = call(Value::GpuTensor(handle)).unwrap_err();
            assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
        });
    }

    #[test]
    fn complex_gpu_input_errors_before_provider_sqrt() {
        test_support::with_test_provider(|provider| {
            let tensor = ComplexTensor::new(vec![(1.0, 0.0), (4.0, 2.0)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).unwrap();
            let err = call(Value::GpuTensor(handle)).unwrap_err();
            assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
        });
    }
}
