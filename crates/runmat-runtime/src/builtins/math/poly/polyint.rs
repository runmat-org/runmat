//! MATLAB-compatible `polyint` builtin with GPU-aware semantics for RunMat.

use log::{trace, warn};
use num_complex::Complex64;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::poly::type_resolvers::polyint_type;
use crate::dispatcher;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const EPS: f64 = 1.0e-12;
const BUILTIN_NAME: &str = "polyint";

const POLYINT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "q",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Integrated polynomial coefficient vector.",
}];

const POLYINT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Polynomial coefficient vector.",
}];

const POLYINT_INPUTS_WITH_K: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Polynomial coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Constant of integration.",
    },
];

const POLYINT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "q = polyint(p)",
        inputs: &POLYINT_INPUTS,
        outputs: &POLYINT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "q = polyint(p, k)",
        inputs: &POLYINT_INPUTS_WITH_K,
        outputs: &POLYINT_OUTPUT,
    },
];

const POLYINT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYINT.INVALID_ARGUMENT",
    identifier: Some("RunMat:polyint:InvalidArgument"),
    when: "Input arity or integration-constant argument is malformed.",
    message: "polyint: invalid argument",
};

const POLYINT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYINT.INVALID_INPUT",
    identifier: Some("RunMat:polyint:InvalidInput"),
    when: "Inputs are not single- or double-precision coefficient/constant values, or coefficients do not form a vector.",
    message: "polyint: invalid input",
};

const POLYINT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYINT.INTERNAL",
    identifier: Some("RunMat:polyint:Internal"),
    when: "Runtime fails while building output tensors or provider fallback paths.",
    message: "polyint: internal runtime failure",
};

const POLYINT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    POLYINT_ERROR_INVALID_ARGUMENT,
    POLYINT_ERROR_INVALID_INPUT,
    POLYINT_ERROR_INTERNAL,
];

pub const POLYINT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POLYINT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POLYINT_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::poly::polyint")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "polyint",
    op_kind: GpuOpKind::Custom("polynomial-integral"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("polyint")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers implement the polyint hook for real and complex-interleaved coefficient vectors; complex integration constants fall back to host integration and re-upload.",
};

fn polyint_error(message: impl Into<String>) -> RuntimeError {
    polyint_error_with(message, &POLYINT_ERROR_INVALID_INPUT)
}

fn polyint_argument_error(message: impl Into<String>) -> RuntimeError {
    polyint_error_with(message, &POLYINT_ERROR_INVALID_ARGUMENT)
}

fn polyint_error_with(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::poly::polyint")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "polyint",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Symbolic operation on coefficient vectors; fusion does not apply.",
};

#[runtime_builtin(
    name = "polyint",
    category = "math/poly",
    summary = "Integrate polynomial coefficient vectors and append a constant of integration.",
    keywords = "polyint,polynomial,integral,antiderivative",
    type_resolver(polyint_type),
    descriptor(crate::builtins::math::poly::polyint::POLYINT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::poly::polyint"
)]
async fn polyint_builtin(coeffs: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(polyint_argument_error("polyint: too many input arguments"));
    }
    reject_unsupported_numeric_class(&coeffs, "coefficient")?;
    for value in &rest {
        reject_unsupported_numeric_class(value, "constant")?;
    }

    let constant = match rest.into_iter().next() {
        Some(value) => parse_constant(value).await?,
        None => Complex64::new(0.0, 0.0),
    };

    if let Value::GpuTensor(handle) = &coeffs {
        if let Some(device_result) = try_polyint_gpu(handle, constant)? {
            return Ok(Value::GpuTensor(device_result));
        }
    }

    let source_gpu = match &coeffs {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    polyint_host_value(coeffs, constant, source_gpu).await
}

async fn polyint_host_value(
    coeffs: Value,
    constant: Complex64,
    source_gpu: Option<runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let polynomial = parse_polynomial(coeffs).await?;
    let mut integrated = integrate_coeffs(&polynomial.coeffs);
    if integrated.is_empty() {
        integrated.push(constant);
    } else if let Some(last) = integrated.last_mut() {
        *last += constant;
    }
    let value = coeffs_to_value(&integrated, polynomial.class)?;
    maybe_return_gpu(value, source_gpu.as_ref())
}

fn reject_unsupported_numeric_class(value: &Value, role: &str) -> BuiltinResult<()> {
    let unsupported = match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => !matches!(
            tensor.numeric_dtype(),
            NumericDType::F64 | NumericDType::F32
        ),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
        }
        _ => false,
    };
    if unsupported {
        return Err(polyint_error(format!(
            "polyint: {role} input must be single or double"
        )));
    }
    Ok(())
}

fn try_polyint_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    constant: Complex64,
) -> BuiltinResult<Option<runmat_accelerate_api::GpuTensorHandle>> {
    if constant.im.abs() > EPS {
        return Ok(None);
    }
    ensure_vector_shape(&handle.shape)?;
    let Some(provider) =
        runmat_accelerate_api::provider_for_handle(handle).or_else(runmat_accelerate_api::provider)
    else {
        return Ok(None);
    };
    match provider.polyint(handle, constant.re) {
        Ok(result) => Ok(Some(result)),
        Err(err) => {
            trace!("polyint: provider hook unavailable, falling back to host: {err}");
            Ok(None)
        }
    }
}

fn integrate_coeffs(coeffs: &[Complex64]) -> Vec<Complex64> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(coeffs.len() + 1);
    for (idx, coeff) in coeffs.iter().enumerate() {
        let power = (coeffs.len() - idx) as f64;
        if power <= 0.0 {
            result.push(Complex64::new(0.0, 0.0));
        } else {
            result.push(*coeff / Complex64::new(power, 0.0));
        }
    }
    result.push(Complex64::new(0.0, 0.0));
    result
}

fn maybe_return_gpu(
    value: Value,
    source_gpu: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let Some(source_gpu) = source_gpu else {
        return Ok(value);
    };
    let provider = runmat_accelerate_api::provider_for_handle(source_gpu);
    match value {
        Value::Tensor(tensor) => {
            if let Some(provider) = provider {
                match gpu_helpers::upload_tensor(provider, &tensor) {
                    Ok(handle) => return Ok(Value::GpuTensor(handle)),
                    Err(err) => {
                        warn!("polyint: provider upload failed, keeping result on host: {err}");
                    }
                }
            } else {
                trace!("polyint: no provider available to re-upload result");
            }
            Ok(Value::Tensor(tensor))
        }
        Value::ComplexTensor(tensor) => {
            if let Some(provider) = provider {
                match gpu_helpers::upload_complex_tensor(provider, &tensor) {
                    Ok(handle) => return Ok(gpu_helpers::complex_gpu_value(handle)),
                    Err(err) => {
                        warn!(
                            "polyint: provider complex upload failed, keeping result on host: {err}"
                        );
                    }
                }
            } else {
                trace!("polyint: no provider available to re-upload complex result");
            }
            Ok(Value::ComplexTensor(tensor))
        }
        other => Ok(other),
    }
}

fn coeffs_to_value(coeffs: &[Complex64], class: FloatingClass) -> BuiltinResult<Value> {
    let shape = vec![1, coeffs.len()];
    if coeffs.iter().all(|c| c.im.abs() <= EPS) {
        let tensor = match class {
            FloatingClass::Double => {
                let data = coeffs.iter().map(|c| c.re).collect();
                Tensor::new(data, shape)
            }
            FloatingClass::Single => {
                let data = coeffs.iter().map(|c| c.re as f32).collect();
                Tensor::from_f32(data, shape)
            }
        }
        .map_err(|e| polyint_error(format!("polyint: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let data: Vec<(f64, f64)> = coeffs.iter().map(|c| (c.re, c.im)).collect();
        let tensor =
            ComplexTensor::new(data, shape).map_err(|e| polyint_error(format!("polyint: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

async fn parse_polynomial(value: Value) -> BuiltinResult<Polynomial> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    match gathered {
        Value::Tensor(tensor) => parse_tensor_coeffs(tensor),
        Value::ComplexTensor(tensor) => parse_complex_tensor_coeffs(&tensor),
        Value::Num(n) => Ok(Polynomial {
            coeffs: vec![Complex64::new(n, 0.0)],
            class: FloatingClass::Double,
        }),
        Value::Complex(re, im) => Ok(Polynomial {
            coeffs: vec![Complex64::new(re, im)],
            class: FloatingClass::Double,
        }),
        other => Err(polyint_error(format!(
            "polyint: expected a numeric coefficient vector, got {:?}",
            other
        ))),
    }
}

fn parse_tensor_coeffs(tensor: Tensor) -> BuiltinResult<Polynomial> {
    ensure_vector_shape(&tensor.shape)?;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| polyint_error(format!("polyint: {error}")))?;
    let (coeffs, class) = match storage {
        NumericStorage::F64(values) => (values, FloatingClass::Double),
        NumericStorage::F32(values) => (
            values.into_iter().map(f64::from).collect(),
            FloatingClass::Single,
        ),
        storage => {
            return Err(polyint_error(format!(
                "polyint: coefficient input must be single or double, got {}",
                storage.class_name()
            )))
        }
    };
    Ok(Polynomial {
        coeffs: coeffs.into_iter().map(|v| Complex64::new(v, 0.0)).collect(),
        class,
    })
}

fn parse_complex_tensor_coeffs(tensor: &ComplexTensor) -> BuiltinResult<Polynomial> {
    ensure_vector_shape(&tensor.shape)?;
    Ok(Polynomial {
        coeffs: tensor
            .data
            .iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect(),
        class: FloatingClass::Double,
    })
}

async fn parse_constant(value: Value) -> BuiltinResult<Complex64> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    match gathered {
        Value::Tensor(tensor) => {
            if !tensor::is_scalar_tensor(&tensor) {
                return Err(polyint_error(
                    "polyint: constant of integration must be a scalar",
                ));
            }
            let value = match tensor
                .into_numeric_storage()
                .map_err(|error| polyint_error(format!("polyint: {error}")))?
            {
                NumericStorage::F64(values) => values[0],
                NumericStorage::F32(values) => f64::from(values[0]),
                storage => {
                    return Err(polyint_error(format!(
                        "polyint: constant input must be single or double, got {}",
                        storage.class_name()
                    )))
                }
            };
            Ok(Complex64::new(value, 0.0))
        }
        Value::ComplexTensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(polyint_error(
                    "polyint: constant of integration must be a scalar",
                ));
            }
            let (re, im) = tensor.data[0];
            Ok(Complex64::new(re, im))
        }
        Value::Num(n) => Ok(Complex64::new(n, 0.0)),
        Value::Complex(re, im) => Ok(Complex64::new(re, im)),
        other => Err(polyint_error(format!(
            "polyint: constant of integration must be numeric, got {:?}",
            other
        ))),
    }
}

fn ensure_vector_shape(shape: &[usize]) -> BuiltinResult<()> {
    let non_unit = shape.iter().filter(|&&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(polyint_error("polyint: coefficients must form a vector"))
    }
}

#[derive(Clone)]
struct Polynomial {
    coeffs: Vec<Complex64>,
    class: FloatingClass,
}

#[derive(Clone, Copy)]
enum FloatingClass {
    Double,
    Single,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::gpu_helpers;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_builtins::{IntegerStorage, LogicalArray};

    fn assert_error_contains(err: crate::RuntimeError, needle: &str) {
        assert!(
            err.message().contains(needle),
            "expected error containing '{needle}', got '{}'",
            err.message()
        );
    }

    #[test]
    fn polyint_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = POLYINT_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"q = polyint(p)"));
        assert!(labels.contains(&"q = polyint(p, k)"));
    }

    #[test]
    fn polyint_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = POLYINT_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.POLYINT.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.POLYINT.INVALID_INPUT"));
        assert!(codes.contains(&"RM.POLYINT.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integrates_polynomial_without_constant() {
        let tensor = Tensor::new(vec![3.0, -2.0, 5.0, 7.0], vec![1, 4]).unwrap();
        let result = polyint_builtin(Value::Tensor(tensor), Vec::new()).expect("polyint");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                let expected = [0.75, -2.0 / 3.0, 2.5, 7.0, 0.0];
                assert!(tensor::tensor_values_f64(&t)
                    .iter()
                    .zip(expected.iter())
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integrates_with_constant() {
        let tensor = Tensor::new(vec![4.0, 0.0, -8.0], vec![1, 3]).unwrap();
        let args = vec![Value::Num(3.0)];
        let result = polyint_builtin(Value::Tensor(tensor), args).expect("polyint");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [4.0 / 3.0, 0.0, -8.0, 3.0];
                assert!(tensor::tensor_values_f64(&t)
                    .iter()
                    .zip(expected.iter())
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn polyint_rejects_every_integer_class_for_coefficients_and_constant() {
        let cases = [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for storage in cases {
            let integer = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let err = polyint_builtin(Value::Tensor(integer.clone()), Vec::new())
                .expect_err("integer coefficients must be rejected");
            assert_error_contains(err, "must be single or double");

            let coefficients = Tensor::new(vec![4.0, 0.0, -8.0], vec![1, 3]).unwrap();
            let err = polyint_builtin(Value::Tensor(coefficients), vec![Value::Tensor(integer)])
                .expect_err("integer constant must be rejected");
            assert_error_contains(err, "must be single or double");
        }
    }

    #[test]
    fn polyint_preserves_native_single_output_storage() {
        let tensor = Tensor::from_f32(vec![3.0, -2.0, 5.0], vec![3, 1]).unwrap();
        let constant = Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap();
        let result =
            polyint_builtin(Value::Tensor(tensor), vec![Value::Tensor(constant)]).expect("polyint");
        let Value::Tensor(tensor) = result else {
            panic!("expected native-single tensor");
        };
        assert_eq!(tensor.shape, vec![1, 4]);
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            tensor.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, -1.0, 5.0, 2.0])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integrates_scalar_value() {
        let result = polyint_builtin(Value::Num(5.0), Vec::new()).expect("polyint");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let values = tensor::tensor_values_f64(&t);
                assert!((values[0] - 5.0).abs() < 1e-12);
                assert!(values[1].abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_logical_coefficients_and_constant() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let err = polyint_builtin(Value::LogicalArray(logical), Vec::new())
            .expect_err("logical coefficients must be rejected");
        assert_error_contains(err, "must be single or double");

        let coefficients = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let constant = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let err = polyint_builtin(
            Value::Tensor(coefficients),
            vec![Value::LogicalArray(constant)],
        )
        .expect_err("logical constant must be rejected");
        assert_error_contains(err, "must be single or double");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn returns_row_vector_for_column_input() {
        let tensor = Tensor::new(vec![2.0, 0.0, -6.0], vec![3, 1]).unwrap();
        let result = polyint_builtin(Value::Tensor(tensor), Vec::new()).expect("polyint");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [2.0 / 3.0, 0.0, -6.0, 0.0];
                assert!(tensor::tensor_values_f64(&t)
                    .iter()
                    .zip(expected.iter())
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected row tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integrates_complex_coefficients() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.0), (0.0, 4.0)], vec![1, 3]).unwrap();
        let args = vec![Value::Complex(0.0, -1.0)];
        let result = polyint_builtin(Value::ComplexTensor(tensor), args).expect("polyint");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [(1.0 / 3.0, 2.0 / 3.0), (-1.5, 0.0), (0.0, 4.0), (0.0, -1.0)];
                assert!(t
                    .data
                    .iter()
                    .zip(expected.iter())
                    .all(|((lre, lim), (rre, rim))| {
                        (lre - rre).abs() < 1e-12 && (lim - rim).abs() < 1e-12
                    }));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_matrix_coefficients() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = polyint_builtin(Value::Tensor(tensor), Vec::new()).expect_err("expected error");
        assert_error_contains(err, "vector");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_scalar_constant() {
        let coeffs = Tensor::new(vec![1.0, -4.0, 6.0], vec![1, 3]).unwrap();
        let constant = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err = polyint_builtin(Value::Tensor(coeffs), vec![Value::Tensor(constant)])
            .expect_err("expected error");
        assert_error_contains(err, "constant of integration must be a scalar");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_excess_arguments() {
        let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let err = polyint_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(1.0), Value::Num(2.0)],
        )
        .expect_err("expected error");
        assert_eq!(err.identifier(), POLYINT_ERROR_INVALID_ARGUMENT.identifier);
        assert_error_contains(err, "too many input arguments");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handles_empty_input_as_zero_polynomial() {
        let tensor = Tensor::new(vec![], vec![1, 0]).unwrap();
        let result = polyint_builtin(Value::Tensor(tensor), Vec::new()).expect("polyint");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            Value::Tensor(t) => {
                // Allow tensor fallback if scalar auto-boxing changes in future
                assert_eq!(t.len(), 1);
                assert!(tensor::tensor_value_f64(&t, 0).abs() < 1e-12);
            }
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_input_with_constant() {
        let tensor = Tensor::new(vec![], vec![1, 0]).unwrap();
        let result = polyint_builtin(Value::Tensor(tensor), vec![Value::Complex(1.5, -2.0)])
            .expect("polyint");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(t.data.len(), 1);
                let (re, im) = t.data[0];
                assert!((re - 1.5).abs() < 1e-12);
                assert!((im + 2.0).abs() < 1e-12);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyint_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -4.0, 6.0], vec![1, 3]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = polyint_builtin(Value::GpuTensor(handle), Vec::new()).expect("polyint");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 4]);
                    let expected = [1.0 / 3.0, -2.0, 6.0, 0.0];
                    assert!(tensor::tensor_values_f64(&gathered)
                        .iter()
                        .zip(expected.iter())
                        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[test]
    fn polyint_gpu_rejects_every_integer_class_before_dispatch() {
        test_support::with_test_provider(|provider| {
            let cases = [
                IntegerStorage::I8(vec![1]),
                IntegerStorage::I16(vec![1]),
                IntegerStorage::I32(vec![1]),
                IntegerStorage::I64(vec![1]),
                IntegerStorage::U8(vec![1]),
                IntegerStorage::U16(vec![1]),
                IntegerStorage::U32(vec![1]),
                IntegerStorage::U64(vec![1]),
            ];
            for storage in cases {
                let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
                let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
                let err = polyint_builtin(Value::GpuTensor(handle), Vec::new())
                    .expect_err("integer gpuArray coefficients must be rejected");
                assert_error_contains(err, "must be single or double");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyint_gpu_complex_constant_reuploads_complex_result() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = polyint_builtin(Value::GpuTensor(handle), vec![Value::Complex(0.0, 2.0)])
                .expect("polyint");
            match result {
                Value::GpuTensor(handle) => {
                    assert_eq!(
                        runmat_accelerate_api::handle_storage(&handle),
                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                    );
                    let gathered =
                        block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)))
                            .expect("gather");
                    let Value::ComplexTensor(ct) = gathered else {
                        panic!("expected complex tensor");
                    };
                    assert_eq!(ct.shape, vec![1, 3]);
                    let expected = [(0.5, 0.0), (0.0, 0.0), (0.0, 2.0)];
                    assert!(ct
                        .data
                        .iter()
                        .zip(expected.iter())
                        .all(|((lre, lim), (rre, rim))| {
                            (lre - rre).abs() < 1e-12 && (lim - rim).abs() < 1e-12
                        }));
                }
                other => panic!("expected complex gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn polyint_complex_gpu_coefficients_stay_resident() {
        test_support::with_test_provider(|provider| {
            let coeffs = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &coeffs).expect("upload");
            let result =
                polyint_builtin(Value::GpuTensor(handle), vec![Value::Num(2.0)]).expect("polyint");
            let Value::GpuTensor(handle) = result else {
                panic!("expected complex gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)))
                .expect("gather");
            let Value::ComplexTensor(ct) = gathered else {
                panic!("expected complex tensor");
            };
            assert_eq!(ct.shape, vec![1, 3]);
            let expected = [(0.5, 0.5), (2.0, -1.0), (2.0, 0.0)];
            assert!(ct
                .data
                .iter()
                .zip(expected.iter())
                .all(|((lre, lim), (rre, rim))| {
                    (lre - rre).abs() < 1e-12 && (lim - rim).abs() < 1e-12
                }));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyint_gpu_with_gpu_constant() {
        test_support::with_test_provider(|provider| {
            let coeffs = Tensor::new(vec![2.0, 0.0], vec![1, 2]).unwrap();
            let coeff_handle =
                gpu_helpers::upload_tensor(provider, &coeffs).expect("upload coeffs");
            let constant = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
            let constant_handle =
                gpu_helpers::upload_tensor(provider, &constant).expect("upload constant");
            let result = polyint_builtin(
                Value::GpuTensor(coeff_handle),
                vec![Value::GpuTensor(constant_handle)],
            )
            .expect("polyint");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(handle)).expect("gather result");
                    assert_eq!(gathered.shape, vec![1, 3]);
                    let expected = [1.0, 0.0, 3.0];
                    assert!(tensor::tensor_values_f64(&gathered)
                        .iter()
                        .zip(expected.iter())
                        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
                }
                other => panic!("expected gpu tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn polyint_wgpu_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(vec![3.0, -2.0, 5.0, 7.0], vec![1, 4]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider.as_ref(), &tensor).expect("upload");
        let gpu_value = polyint_builtin(Value::GpuTensor(handle), Vec::new()).expect("polyint gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        let cpu_value =
            polyint_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("polyint cpu");
        let expected = match cpu_value {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu result {other:?}"),
        };
        assert_eq!(gathered.shape, expected.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        tensor::tensor_values_f64(&gathered)
            .iter()
            .zip(tensor::tensor_values_f64(&expected).iter())
            .for_each(|(lhs, rhs)| assert!((lhs - rhs).abs() < tol));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn polyint_wgpu_complex_coefficients_match_cpu() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let coeffs =
            ComplexTensor::new(vec![(3.0, 1.5), (-2.0, 0.5), (5.0, -1.0)], vec![1, 3]).unwrap();
        let cpu_value =
            polyint_builtin(Value::ComplexTensor(coeffs.clone()), vec![Value::Num(2.0)])
                .expect("polyint cpu");
        let cpu = match cpu_value {
            Value::ComplexTensor(t) => t,
            other => panic!("unexpected cpu result {other:?}"),
        };

        let handle = gpu_helpers::upload_complex_tensor(provider, &coeffs).expect("upload");
        let gpu_value =
            polyint_builtin(Value::GpuTensor(handle), vec![Value::Num(2.0)]).expect("polyint gpu");
        let Value::GpuTensor(handle) = gpu_value else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).expect("gather");
        let Value::ComplexTensor(gpu) = gathered else {
            panic!("expected complex tensor");
        };
        assert_eq!(gpu.shape, cpu.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        gpu.data
            .iter()
            .zip(cpu.data.iter())
            .for_each(|((lre, lim), (rre, rim))| {
                assert!((lre - rre).abs() < tol);
                assert!((lim - rim).abs() < tol);
            });
    }

    fn polyint_builtin(coeffs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::polyint_builtin(coeffs, rest))
    }
}
