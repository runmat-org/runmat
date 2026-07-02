//! MATLAB-compatible `erfcinv` builtin.
//!
//! `erfcinv` computes the inverse complementary error function for real inputs.
//! The implementation uses a monotonic bisection over `erfc` to keep tails stable
//! without depending on a platform-specific inverse special function.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, HostTensorView};
use runmat_builtins::{
    shape_rules::element_count_if_known, BuiltinCompletionPolicy, BuiltinDescriptor,
    BuiltinErrorDescriptor, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinSignatureDescriptor, NumericDType, ResolveContext, Tensor, Type,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "erfcinv";
const MAX_POSITIVE_RESULT: f64 = 32.0;
const BISECTION_STEPS: usize = 110;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise inverse complementary error-function result.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real single or double input in the interval [0, 2].",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = erfcinv(X)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERFCINV.INVALID_INPUT",
    identifier: Some("RunMat:erfcinv:InvalidInput"),
    when: "Input cannot be interpreted as a real, nonsparse numeric array.",
    message: "erfcinv: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERFCINV.INTERNAL",
    identifier: Some("RunMat:erfcinv:Internal"),
    when: "Internal tensor construction, gathering, or upload failed.",
    message: "erfcinv: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const ERFCINV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::erfcinv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "GPU-aware host fallback only: RunMat gathers GPU tensors, computes erfcinv on the host, and uploads the result when a provider is active.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::erfcinv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes:
        "erfcinv currently executes through the runtime path because no provider hook exists yet.",
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
    error_with_detail(&ERROR_INTERNAL, detail)
}

fn erfcinv_type(args: &[Type], _context: &ResolveContext) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Num => Type::Num,
        Type::Tensor { shape: Some(dims) } if element_count_if_known(dims) == Some(1) => Type::Num,
        Type::Tensor { shape: Some(dims) } => Type::Tensor {
            shape: Some(dims.clone()),
        },
        Type::Tensor { shape: None } => Type::tensor(),
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runtime_builtin(
    name = "erfcinv",
    category = "math/elementwise",
    summary = "Compute inverse complementary error-function values.",
    keywords = "erfcinv,inverse complementary error function,special,elementwise,gpu",
    accel = "unary",
    type_resolver(erfcinv_type),
    descriptor(crate::builtins::math::elementwise::erfcinv::ERFCINV_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::erfcinv"
)]
async fn erfcinv_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => erfcinv_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "complex inputs are not supported",
        )),
        Value::String(_) | Value::StringArray(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "expected real numeric input, got string",
        )),
        Value::SparseTensor(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "sparse inputs are not supported",
        )),
        Value::Bool(_) | Value::LogicalArray(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "logical inputs are not supported",
        )),
        Value::Int(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "integer-class inputs are not supported",
        )),
        Value::CharArray(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "char inputs are not supported",
        )),
        other => erfcinv_real(other),
    }
}

async fn erfcinv_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "complex gpuArray inputs are not supported",
        ));
    }

    let provider = runmat_accelerate_api::provider_for_handle(&handle);
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let result = erfcinv_tensor(tensor)?;

    if let Some(provider) = provider {
        let view = HostTensorView {
            data: &result.data,
            shape: &result.shape,
        };
        match provider.upload(&view) {
            Ok(handle) => {
                runmat_accelerate_api::mark_residency(&handle);
                return Ok(Value::GpuTensor(handle));
            }
            Err(err) if err.to_string() == "interaction pending..." => {
                return Err(build_runtime_error("interaction pending...")
                    .with_builtin(BUILTIN_NAME)
                    .build())
            }
            Err(err) => {
                return Err(internal_error(format!(
                    "failed to upload erfcinv result to GPU provider: {err}"
                )))
            }
        }
    }

    Ok(erfcinv_tensor_into_value(result))
}

fn erfcinv_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))?;
    erfcinv_tensor(tensor).map(erfcinv_tensor_into_value)
}

fn erfcinv_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let dtype = match tensor.dtype {
        NumericDType::F32 => NumericDType::F32,
        NumericDType::F64 => NumericDType::F64,
        _ => {
            return Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                "integer-class tensors are not supported",
            ))
        }
    };
    let data = tensor
        .data
        .iter()
        .map(|&value| cast_output(erfcinv_scalar(value), dtype))
        .collect::<Vec<_>>();
    Tensor::new_with_dtype(data, tensor.shape.clone(), dtype)
        .map_err(|detail| internal_error(&detail))
}

fn erfcinv_tensor_into_value(tensor: Tensor) -> Value {
    if tensor.data.len() == 1 && tensor.dtype == NumericDType::F64 {
        Value::Num(tensor.data[0])
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

fn cast_output(value: f64, dtype: NumericDType) -> f64 {
    if dtype == NumericDType::F32 {
        value as f32 as f64
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, SparseTensor, Type};

    fn erfcinv_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::erfcinv_builtin(value))
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
    fn descriptor_covers_core_form() {
        let labels = ERFCINV_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"Y = erfcinv(X)"));
    }

    #[test]
    fn type_resolver_reports_numeric_shape() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        };
        let out = erfcinv_type(&[ty], &ResolveContext::new(Vec::new()));
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
        assert_eq!(
            erfcinv_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
        assert_eq!(
            erfcinv_type(
                &[Type::Logical {
                    shape: Some(vec![Some(1), Some(2)])
                }],
                &ResolveContext::new(Vec::new())
            ),
            Type::Unknown
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
                assert_eq!(out.dtype, NumericDType::F32);
                assert_close(libm::erfc(out.data[0]), 0.5, 1e-6);
                assert_eq!(out.data[1], 0.0);
                assert_close(libm::erfc(out.data[2]), 1.5, 1e-6);
                assert!(out.data[3].is_nan());
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
                data: &host.data,
                shape: &host.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = match erfcinv_builtin(Value::GpuTensor(handle)).unwrap() {
                value @ Value::GpuTensor(_) => test_support::gather(value).expect("gather"),
                other => panic!("expected gpu tensor, got {other:?}"),
            };
            assert_eq!(gpu.shape, cpu.shape);
            for (actual, expected) in gpu.data.iter().zip(cpu.data.iter()) {
                assert_close(*actual, *expected, 1e-12);
            }
        });
    }
}
