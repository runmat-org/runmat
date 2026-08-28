//! MATLAB-compatible `realsqrt` builtin.
//!
//! `realsqrt` is the real-domain square-root helper: nonnegative real inputs
//! are evaluated element-wise, while complex inputs and negative real values
//! raise errors instead of promoting to complex results.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericStorage, SparseTensor, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "realsqrt";

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The public input classes are single and double; host, sparse, and resident integer values reject by class before domain evaluation or provider sqrt dispatch.",
}];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = realsqrt(X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "realsqrt has no integer overload; the empty accepted-class mask is intentional and prevents generic numeric coercion from admitting integers.",
    }];

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
    description: "Real single or double numeric input.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = realsqrt(X)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REALSQRT.INVALID_INPUT",
    identifier: Some("RunMat:realsqrt:InvalidInput"),
    when: "Input is not real single or double numeric data.",
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
    type_resolver(realsqrt_type),
    descriptor(crate::builtins::math::elementwise::realsqrt::REALSQRT_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::elementwise::realsqrt::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::realsqrt"
)]
async fn realsqrt_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => realsqrt_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "complex input is not supported",
        )),
        Value::SparseTensor(sparse) => realsqrt_sparse(sparse),
        Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "expected real single or double input",
        )),
        other => realsqrt_real(other),
    }
}

fn realsqrt_type(args: &[Type], context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Int | Type::Bool | Type::Logical { .. }) => Type::Unknown,
        _ => numeric_unary_type(args, context),
    }
}

async fn realsqrt_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "integer gpuArray input is not supported",
        ));
    }
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
    let download = gpu_helpers::download_native_values_async(provider, &min_handle)
        .await
        .map_err(|e| internal_error(format!("realsqrt: reduce_min download failed: {e}")));
    let _ = provider.free(&min_handle);
    let host = download?;
    if host.data.iter().any(|value| value.is_nan()) {
        return Err(internal_error(
            "realsqrt: reduce_min result contained NaN; host validation required",
        ));
    }
    Ok(host.data.iter().any(|value| value.is_negative()))
}

fn realsqrt_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))?;
    realsqrt_tensor(tensor)
}

fn realsqrt_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            ensure_nonnegative(&values)?;
            NumericStorage::F64(
                values
                    .into_iter()
                    .map(|value| canonical_zero(value.sqrt()))
                    .collect(),
            )
        }
        NumericStorage::F32(values) => {
            ensure_nonnegative_f32(&values)?;
            NumericStorage::F32(
                values
                    .into_iter()
                    .map(|value| canonical_zero_f32(value.sqrt()))
                    .collect(),
            )
        }
        _ => {
            return Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                "expected real single or double input",
            ))
        }
    };
    let out = Tensor::from_numeric_storage(output, shape)
        .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))?;
    Ok(tensor_into_realsqrt_value(out))
}

fn realsqrt_sparse(sparse: SparseTensor) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() || sparse.is_logical() {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "expected real single or double input",
        ));
    }
    let dtype = sparse.numeric_dtype();
    let values = sparse.materialize_f64();
    ensure_nonnegative(&values)?;
    let values: Vec<f64> = values
        .into_iter()
        .map(|value| canonical_zero(value.sqrt()))
        .collect();
    let output = match dtype {
        Some(NumericDType::F32) => SparseTensor::new_f32(
            sparse.rows,
            sparse.cols,
            sparse.col_ptrs,
            sparse.row_indices,
            values.into_iter().map(|value| value as f32).collect(),
        ),
        Some(NumericDType::F64) => SparseTensor::new(
            sparse.rows,
            sparse.cols,
            sparse.col_ptrs,
            sparse.row_indices,
            values,
        ),
        Some(_) | None => unreachable!("integer and logical sparse input rejected above"),
    };
    output
        .map(Value::SparseTensor)
        .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))
}

fn tensor_into_realsqrt_value(tensor: Tensor) -> Value {
    if tensor.numeric_dtype() == NumericDType::F64 {
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

fn ensure_nonnegative_f32(values: &[f32]) -> BuiltinResult<()> {
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

fn canonical_zero_f32(value: f32) -> f32 {
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
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::{
        CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType,
        SparseTensor,
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
        let out = realsqrt_type(
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
    fn type_resolver_rejects_known_integer_and_logical_inputs() {
        assert_eq!(
            realsqrt_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
        assert_eq!(
            realsqrt_type(
                &[Type::Logical {
                    shape: Some(vec![Some(1), Some(2)]),
                }],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Unknown
        );
    }

    #[test]
    fn double_scalar_nonnegative_value_returns_real_root() {
        match call(Value::Num(9.0)).unwrap() {
            Value::Num(value) => assert!((value - 3.0).abs() < 1.0e-12),
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
                assert!(tensor.materialize_f64()[0].is_nan());
                assert!(tensor.materialize_f64()[1].is_infinite());
                assert!(tensor.materialize_f64()[1].is_sign_positive());
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
                assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn dense_typed_integer_tensor_is_rejected_by_class() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 4, 9]), vec![2, 2]).unwrap();

        let err = call(Value::Tensor(tensor)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn dense_negative_typed_integer_is_rejected_by_class_before_domain() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, -4]), vec![1, 2]).unwrap();

        let err = call(Value::Tensor(tensor)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn single_tensor_preserves_dtype() {
        let tensor = Tensor::from_f32(vec![2.0, 9.0], vec![1, 2]).unwrap();
        match call(Value::Tensor(tensor)).unwrap() {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.numeric_dtype(), NumericDType::F32);
                assert_eq!(
                    out.into_numeric_storage().unwrap(),
                    NumericStorage::F32(vec![2.0f32.sqrt(), 3.0])
                );
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
    fn integer_logical_and_char_inputs_are_rejected() {
        let integer = call(Value::Int(IntValue::I32(16))).unwrap_err();
        assert_eq!(integer.identifier(), ERROR_INVALID_INPUT.identifier);
        let boolean = call(Value::Bool(true)).unwrap_err();
        assert_eq!(boolean.identifier(), ERROR_INVALID_INPUT.identifier);

        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let logical = call(Value::LogicalArray(logical)).unwrap_err();
        assert_eq!(logical.identifier(), ERROR_INVALID_INPUT.identifier);

        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let chars = call(Value::CharArray(chars)).unwrap_err();
        assert_eq!(chars.identifier(), ERROR_INVALID_INPUT.identifier);
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
                assert_eq!(out.materialize_f64(), vec![2.0, 3.0, 4.0]);
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[test]
    fn sparse_single_inputs_preserve_native_single_storage() {
        let sparse =
            SparseTensor::new_f32(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![4.0, 9.0, 16.0])
                .unwrap();
        let Value::SparseTensor(output) = call(Value::SparseTensor(sparse)).unwrap() else {
            panic!("expected sparse tensor");
        };
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(output.as_f32_slice(), Some(&[2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn sparse_typed_integer_values_are_rejected_by_class() {
        let sparse = SparseTensor::new_integer(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            IntegerStorage::U16(vec![4, 9, 16]),
        )
        .unwrap();

        let err = call(Value::SparseTensor(sparse)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn sparse_negative_typed_integer_is_rejected_by_class_before_domain() {
        let sparse =
            SparseTensor::new_integer(2, 1, vec![0, 1], vec![1], IntegerStorage::I16(vec![-4]))
                .unwrap();

        let err = call(Value::SparseTensor(sparse)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
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
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .unwrap();
            let result = call(Value::GpuTensor(handle)).unwrap();
            let gathered = test_support::gather(result).unwrap();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn gpu_negative_value_errors() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -4.0], vec![1, 2]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
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

    #[test]
    fn integer_gpu_input_errors_before_provider_sqrt() {
        test_support::with_test_provider(|provider| {
            let values = [4u64, u64::MAX];
            let shape = [1usize, 2usize];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &shape,
                })
                .expect("upload integer gpu tensor");
            let err = call(Value::GpuTensor(handle)).unwrap_err();
            assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
        });
    }
}
