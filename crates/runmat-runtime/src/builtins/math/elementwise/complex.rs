//! MATLAB-compatible `complex` constructor builtin.
//!
//! `complex(a, b)` constructs `a + 1i*b` element-wise. The real and imaginary
//! parts must have matching sizes unless one input is scalar. `complex(a)`
//! returns real input lifted into complex storage with zero imaginary parts and
//! leaves existing complex input unchanged. Binary inputs must be real numeric.

use runmat_builtins::shape_rules::element_count_if_known;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "complex";

const COMPLEX_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex result.",
}];

const COMPLEX_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric input to lift into complex storage.",
}];

const COMPLEX_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real part operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Imaginary part operand.",
    },
];

const COMPLEX_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Z = complex(A)",
        inputs: &COMPLEX_INPUTS_A,
        outputs: &COMPLEX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Z = complex(A, B)",
        inputs: &COMPLEX_INPUTS_A_B,
        outputs: &COMPLEX_OUTPUT,
    },
];

const COMPLEX_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INVALID_ARGUMENT",
    identifier: Some("RunMat:complex:InvalidArgument"),
    when: "Argument arity is invalid.",
    message: "complex: invalid argument",
};

const COMPLEX_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INVALID_INPUT",
    identifier: Some("RunMat:complex:InvalidInput"),
    when: "Input value cannot be converted into real numeric tensor inputs.",
    message: "complex: invalid input",
};

const COMPLEX_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.SIZE_MISMATCH",
    identifier: Some("RunMat:complex:SizeMismatch"),
    when: "Real and imaginary parts are not compatible for scalar expansion.",
    message: "complex: size mismatch",
};

const COMPLEX_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INTERNAL",
    identifier: Some("RunMat:complex:Internal"),
    when: "Internal complex tensor construction failed.",
    message: "complex: internal error",
};

const COMPLEX_ERRORS: [BuiltinErrorDescriptor; 4] = [
    COMPLEX_ERROR_INVALID_ARGUMENT,
    COMPLEX_ERROR_INVALID_INPUT,
    COMPLEX_ERROR_SIZE_MISMATCH,
    COMPLEX_ERROR_INTERNAL,
];

pub const COMPLEX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPLEX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COMPLEX_ERRORS,
};

fn complex_error_with_detail(
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

#[runtime_builtin(
    name = "complex",
    category = "math/elementwise",
    summary = "Construct complex values from real and imaginary parts.",
    keywords = "complex,construct,imaginary,real,elementwise",
    type_resolver(complex_type),
    descriptor(crate::builtins::math::elementwise::complex::COMPLEX_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::complex"
)]
async fn complex_builtin(real: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    match rest.len() {
        0 => unary_complex(real).await,
        1 => {
            let imag = rest.into_iter().next().expect("rest has one element");
            binary_complex(real, imag).await
        }
        n => Err(complex_error_with_detail(
            &COMPLEX_ERROR_INVALID_ARGUMENT,
            format!("expected 1 or 2 input arguments, got {}", n + 1),
        )),
    }
}

fn complex_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args {
        [] => Type::Unknown,
        [input] => complex_unary_type(input),
        [lhs, rhs] => complex_binary_type(lhs, rhs),
        _ => Type::Unknown,
    }
}

fn complex_unary_type(input: &Type) -> Type {
    match input {
        Type::Tensor { shape } | Type::Logical { shape } => tensor_like_type(shape),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn complex_binary_type(lhs: &Type, rhs: &Type) -> Type {
    if is_real_numeric_scalar(lhs) && is_real_numeric_scalar(rhs) {
        return Type::Num;
    }

    match (numeric_array_shape(lhs), numeric_array_shape(rhs)) {
        (Some(lhs_shape), Some(rhs_shape)) => match (lhs_shape, rhs_shape) {
            (Some(left), Some(right)) if left == right => Type::Tensor {
                shape: Some(left.clone()),
            },
            (None, None) => Type::tensor(),
            _ => Type::Unknown,
        },
        (Some(shape), None) if is_real_numeric_scalar(rhs) => tensor_like_type(shape),
        (None, Some(shape)) if is_real_numeric_scalar(lhs) => tensor_like_type(shape),
        (Some(None), _) | (_, Some(None)) => Type::tensor(),
        _ if matches!(lhs, Type::Unknown) || matches!(rhs, Type::Unknown) => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn tensor_like_type(shape: &Option<Vec<Option<usize>>>) -> Type {
    match shape {
        Some(dims) => match element_count_if_known(dims) {
            Some(1) => Type::Num,
            _ => Type::Tensor {
                shape: Some(dims.clone()),
            },
        },
        None => Type::tensor(),
    }
}

fn is_real_numeric_scalar(ty: &Type) -> bool {
    match ty {
        Type::Num | Type::Int | Type::Bool => true,
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape) == Some(1)
        }
        _ => false,
    }
}

fn numeric_array_shape(ty: &Type) -> Option<&Option<Vec<Option<usize>>>> {
    match ty {
        Type::Tensor { shape } | Type::Logical { shape } => {
            if shape.as_ref().and_then(|dims| element_count_if_known(dims)) == Some(1) {
                None
            } else {
                Some(shape)
            }
        }
        _ => None,
    }
}

async fn unary_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::GpuTensor(handle) => unary_complex_gpu(handle).await,
        other => unary_complex_host(other),
    }
}

fn unary_complex_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        other => {
            let tensor = value_into_real_tensor(other)?;
            let shape = tensor.shape.clone();
            if is_scalar_tensor(&tensor) {
                return Ok(Value::Complex(tensor.data[0], 0.0));
            }
            let data = tensor.data.into_iter().map(|x| (x, 0.0)).collect();
            let ct = ComplexTensor::new(data, shape)
                .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
            Ok(complex_tensor_into_value(ct))
        }
    }
}

async fn binary_complex(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if matches!(lhs, Value::GpuTensor(_)) || matches!(rhs, Value::GpuTensor(_)) {
        match try_binary_complex_gpu(&lhs, &rhs).await {
            Ok(Some(value)) => return Ok(value),
            Ok(None) => {
                let real_value = gather_if_gpu_value(&lhs).await?;
                let imag_value = gather_if_gpu_value(&rhs).await?;
                let real_tensor = value_into_real_tensor(real_value)?;
                let imag_tensor = value_into_real_tensor(imag_value)?;
                return compose_complex(&real_tensor, &imag_tensor);
            }
            Err(err) => return Err(err),
        }
    }
    let real_tensor = value_into_real_tensor(lhs)?;
    let imag_tensor = value_into_real_tensor(rhs)?;
    compose_complex(&real_tensor, &imag_tensor)
}

async fn unary_complex_gpu(handle: runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        return Ok(gpu_helpers::complex_gpu_value(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.complex_from_real(&handle).await {
            return Ok(gpu_helpers::complex_gpu_value(out));
        }
    }

    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
    unary_complex_host(gathered)
}

async fn gather_if_gpu_value(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(value).await,
        other => Ok(other.clone()),
    }
}

async fn try_binary_complex_gpu(lhs: &Value, rhs: &Value) -> BuiltinResult<Option<Value>> {
    let provider = match (lhs, rhs) {
        (Value::GpuTensor(handle), _) | (_, Value::GpuTensor(handle)) => {
            runmat_accelerate_api::provider_for_handle(handle)
        }
        _ => None,
    };
    let Some(provider) = provider else {
        return Ok(None);
    };

    let real = value_to_real_gpu_handle(lhs, provider).await?;
    let imag = match value_to_real_gpu_handle(rhs, provider).await {
        Ok(imag) => imag,
        Err(err) => {
            if real.owned {
                provider.free(&real.handle).ok();
            }
            return Err(err);
        }
    };
    let result = match provider
        .complex_from_real_imag(&real.handle, &imag.handle)
        .await
    {
        Ok(out) => Ok(Some(gpu_helpers::complex_gpu_value(out))),
        Err(_) => Ok(None),
    };
    if real.owned {
        provider.free(&real.handle).ok();
    }
    if imag.owned {
        provider.free(&imag.handle).ok();
    }
    result
}

struct RealGpuOperand {
    handle: runmat_accelerate_api::GpuTensorHandle,
    owned: bool,
}

async fn value_to_real_gpu_handle(
    value: &Value,
    provider: &dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<RealGpuOperand> {
    match value {
        Value::GpuTensor(handle) => {
            let Some(owner) = runmat_accelerate_api::provider_for_handle(handle) else {
                return Err(complex_error_with_detail(
                    &COMPLEX_ERROR_INVALID_INPUT,
                    "GPU input provider is unavailable",
                ));
            };
            if owner.device_id() != provider.device_id() {
                return Err(complex_error_with_detail(
                    &COMPLEX_ERROR_INVALID_INPUT,
                    "GPU inputs must belong to the same provider",
                ));
            }
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                return Err(complex_error_with_detail(
                    &COMPLEX_ERROR_INVALID_INPUT,
                    "inputs must be real",
                ));
            }
            Ok(RealGpuOperand {
                handle: handle.clone(),
                owned: false,
            })
        }
        other => {
            let tensor = value_into_real_tensor(other.clone())?;
            upload_real_tensor(provider, &tensor).map(|handle| RealGpuOperand {
                handle,
                owned: true,
            })
        }
    }
}

fn upload_real_tensor(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: &Tensor,
) -> BuiltinResult<runmat_accelerate_api::GpuTensorHandle> {
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let handle = provider
        .upload(&view)
        .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
    runmat_accelerate_api::set_handle_logical(&handle, false);
    runmat_accelerate_api::set_handle_storage(
        &handle,
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    runmat_accelerate_api::set_handle_precision(&handle, provider.precision());
    Ok(handle)
}

fn compose_complex(real: &Tensor, imag: &Tensor) -> BuiltinResult<Value> {
    let (shape, data) = if real.shape == imag.shape {
        let data: Vec<(f64, f64)> = real
            .data
            .iter()
            .zip(imag.data.iter())
            .map(|(&re, &im)| (re, im))
            .collect();
        (real.shape.clone(), data)
    } else if is_scalar_tensor(real) {
        let re = real.data[0];
        let data: Vec<(f64, f64)> = imag.data.iter().map(|&im| (re, im)).collect();
        (imag.shape.clone(), data)
    } else if is_scalar_tensor(imag) {
        let im = imag.data[0];
        let data: Vec<(f64, f64)> = real.data.iter().map(|&re| (re, im)).collect();
        (real.shape.clone(), data)
    } else {
        return Err(complex_error_with_detail(
            &COMPLEX_ERROR_SIZE_MISMATCH,
            "real and imaginary parts must have the same size, unless one input is scalar",
        ));
    };

    if data.is_empty() {
        let empty = ComplexTensor::new(Vec::new(), shape)
            .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
        return Ok(complex_tensor_into_value(empty));
    }
    let ct = ComplexTensor::new(data, shape)
        .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(ct))
}

fn is_scalar_tensor(tensor: &Tensor) -> bool {
    tensor.data.len() == 1
}

fn value_into_real_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(complex_error_with_detail(
            &COMPLEX_ERROR_INVALID_INPUT,
            "inputs must be real",
        )),
        Value::String(_) | Value::StringArray(_) => Err(complex_error_with_detail(
            &COMPLEX_ERROR_INVALID_INPUT,
            "expected numeric input, got string",
        )),
        Value::CharArray(_) => Err(complex_error_with_detail(
            &COMPLEX_ERROR_INVALID_INPUT,
            "expected numeric input, got char",
        )),
        other => tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INVALID_INPUT, e)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::gpu_helpers;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, StringArray, Tensor, Type, Value};

    fn complex_call(real: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::complex_builtin(real, rest))
    }

    #[test]
    fn complex_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = COMPLEX_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Z = complex(A)"));
        assert!(labels.contains(&"Z = complex(A, B)"));
    }

    #[test]
    fn type_resolver_rejects_non_scalar_shape_expansion() {
        let out = complex_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(1)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Unknown);
    }

    #[test]
    fn type_resolver_preserves_equal_shape() {
        let out = complex_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
            ],
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
    fn type_resolver_scalar_returns_num() {
        let out = complex_type(&[Type::Num, Type::Num], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn type_resolver_scalar_array_uses_array_shape() {
        let out = complex_type(
            &[
                Type::Num,
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(3)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_scalar_pair() {
        let result = complex_call(Value::Num(3.0), vec![Value::Num(4.0)]).expect("complex");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 3.0);
                assert_eq!(im, 4.0);
            }
            other => panic!("expected Complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_string_input_has_stable_identifier() {
        let err = complex_call(Value::from("bad"), vec![]).expect_err("expected error");
        assert_eq!(err.identifier(), COMPLEX_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_row_vector_pair() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let result = complex_call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).expect("complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 3]);
                assert_eq!(ct.data, vec![(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]);
            }
            other => panic!("expected ComplexTensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_scalar_vector_broadcast_real_left() {
        let imag = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = complex_call(Value::Num(0.0), vec![Value::Tensor(imag)]).expect("complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 3]);
                assert_eq!(ct.data, vec![(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]);
            }
            other => panic!("expected ComplexTensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_scalar_vector_broadcast_real_right() {
        let real = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = complex_call(Value::Tensor(real), vec![Value::Num(0.0)]).expect("complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 3]);
                assert_eq!(ct.data, vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
            }
            other => panic!("expected ComplexTensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_column_vectors() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let result = complex_call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).expect("complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                assert_eq!(ct.data, vec![(1.0, 3.0), (2.0, 4.0)]);
            }
            other => panic!("expected ComplexTensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_rejects_non_scalar_implicit_expansion() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let col = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let err = complex_call(Value::Tensor(row), vec![Value::Tensor(col)]).unwrap_err();
        let msg = err.message().to_ascii_lowercase();
        assert!(msg.contains("same size") || msg.contains("scalar"), "{msg}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_shape_mismatch_errors() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err = complex_call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).unwrap_err();
        let msg = err.message().to_ascii_lowercase();
        assert!(msg.contains("dimension") || msg.contains("size"), "{msg}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_rejects_complex_scalar() {
        let err = complex_call(Value::Complex(1.0, 2.0), vec![Value::Num(3.0)]).unwrap_err();
        assert!(
            err.message().contains("must be real"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_rejects_complex_imag_argument() {
        let err = complex_call(Value::Num(1.0), vec![Value::Complex(0.0, 1.0)]).unwrap_err();
        assert!(
            err.message().contains("must be real"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_rejects_complex_tensor_input() {
        let ct = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let err = complex_call(Value::ComplexTensor(ct), vec![Value::Num(0.0)]).unwrap_err();
        assert!(err.message().contains("must be real"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_rejects_string_input() {
        let err = complex_call(Value::from("hello"), vec![Value::Num(0.0)]).unwrap_err();
        assert!(err.message().contains("string"), "{}", err.message());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_rejects_string_array_input() {
        let arr =
            StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2]).expect("array");
        let err = complex_call(Value::Num(0.0), vec![Value::StringArray(arr)]).unwrap_err();
        assert!(err.message().contains("string"), "{}", err.message());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_promotes_integer_inputs() {
        let result = complex_call(
            Value::Int(IntValue::I32(3)),
            vec![Value::Int(IntValue::I32(-4))],
        )
        .expect("complex");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 3.0);
                assert_eq!(im, -4.0);
            }
            other => panic!("expected Complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_unary_scalar_zero_imag() {
        let result = complex_call(Value::Num(5.0), Vec::new()).expect("complex");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 5.0);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected Complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_unary_tensor_zero_imag() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = complex_call(Value::Tensor(tensor), Vec::new()).expect("complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert_eq!(ct.data, vec![(1.0, 0.0), (2.0, 0.0)]);
            }
            other => panic!("expected ComplexTensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_unary_complex_scalar_passthrough() {
        let result = complex_call(Value::Complex(1.0, 2.0), Vec::new()).expect("complex");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, 2.0);
            }
            other => panic!("expected Complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_unary_complex_tensor_passthrough() {
        let tensor = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
        let result =
            complex_call(Value::ComplexTensor(tensor.clone()), Vec::new()).expect("complex");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out, tensor),
            other => panic!("expected ComplexTensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_unary_rejects_string_input() {
        let err = complex_call(Value::from("hi"), Vec::new()).unwrap_err();
        assert!(err.message().contains("string"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_logical_array_input() {
        let lhs = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let result =
            complex_call(Value::LogicalArray(lhs), vec![Value::Tensor(rhs)]).expect("complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
                assert_eq!(
                    ct.data,
                    vec![(1.0, 10.0), (0.0, 20.0), (0.0, 30.0), (1.0, 40.0)]
                );
            }
            other => panic!("expected ComplexTensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_bool_scalar_promotion() {
        let result = complex_call(Value::Bool(true), vec![Value::Bool(false)]).expect("complex");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected Complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_rejects_char_array_input() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let imag = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err = complex_call(Value::CharArray(chars), vec![Value::Tensor(imag)]).unwrap_err();
        assert!(err.message().contains("char"), "{}", err.message());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_empty_tensor_inputs() {
        let lhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let rhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = complex_call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).expect("complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![0, 3]);
                assert!(ct.data.is_empty());
            }
            other => panic!("expected empty ComplexTensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_unary_gpu_stays_resident() {
        test_support::with_test_provider(|provider| {
            let real = Tensor::new(vec![1.0, -2.0, 3.5], vec![3, 1]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &real.data,
                    shape: &real.shape,
                })
                .expect("upload");
            let result = complex_call(Value::GpuTensor(handle), Vec::new()).expect("complex");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident complex gpuArray");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
            let Value::ComplexTensor(ct) = gathered else {
                panic!("expected gathered complex tensor");
            };
            assert_eq!(ct.shape, vec![3, 1]);
            assert_eq!(ct.data, vec![(1.0, 0.0), (-2.0, 0.0), (3.5, 0.0)]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_binary_gpu_stays_resident_with_scalar_expansion() {
        test_support::with_test_provider(|provider| {
            let real = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let real_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &real.data,
                    shape: &real.shape,
                })
                .expect("upload real");
            let result = complex_call(Value::GpuTensor(real_handle), vec![Value::Num(-4.0)])
                .expect("complex");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident complex gpuArray");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
            let Value::ComplexTensor(ct) = gathered else {
                panic!("expected gathered complex tensor");
            };
            assert_eq!(ct.shape, vec![1, 3]);
            assert_eq!(ct.data, vec![(1.0, -4.0), (2.0, -4.0), (3.0, -4.0)]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_empty_gpu_inputs_stay_resident() {
        test_support::with_test_provider(|provider| {
            let real = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
            let imag = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
            let real_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &real.data,
                    shape: &real.shape,
                })
                .expect("upload real");
            let imag_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &imag.data,
                    shape: &imag.shape,
                })
                .expect("upload imag");
            let result = complex_call(
                Value::GpuTensor(real_handle),
                vec![Value::GpuTensor(imag_handle)],
            )
            .expect("complex");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident complex gpuArray");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
            let Value::ComplexTensor(ct) = gathered else {
                panic!("expected gathered complex tensor");
            };
            assert_eq!(ct.shape, vec![0, 3]);
            assert!(ct.data.is_empty());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_gpu_shape_mismatch_fallback_reports_size_error() {
        test_support::with_test_provider(|provider| {
            let real = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let imag = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
            let real_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &real.data,
                    shape: &real.shape,
                })
                .expect("upload real");
            let imag_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &imag.data,
                    shape: &imag.shape,
                })
                .expect("upload imag");
            let err = complex_call(
                Value::GpuTensor(real_handle),
                vec![Value::GpuTensor(imag_handle)],
            )
            .unwrap_err();
            let message = err.message();
            assert!(
                message.contains("same size") || message.contains("scalar"),
                "unexpected error: {message}"
            );
            assert!(
                !message.contains("GpuTensor"),
                "fallback leaked gpuArray host-conversion error: {message}"
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_binary_rejects_complex_gpu_input() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let err = complex_call(Value::GpuTensor(handle), vec![Value::Num(0.0)]).unwrap_err();
            assert!(
                err.message().contains("must be real"),
                "unexpected error: {}",
                err.message()
            );
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn complex_wgpu_binary_matches_cpu_and_stays_resident() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().unwrap();
        let real = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let imag = Tensor::new(vec![-1.0, 0.5, 4.0], vec![3, 1]).unwrap();
        let expected = complex_call(
            Value::Tensor(real.clone()),
            vec![Value::Tensor(imag.clone())],
        )
        .expect("cpu complex");
        let real_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &real.data,
                shape: &real.shape,
            })
            .expect("upload real");
        let imag_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &imag.data,
                shape: &imag.shape,
            })
            .expect("upload imag");
        let result = complex_call(
            Value::GpuTensor(real_handle),
            vec![Value::GpuTensor(imag_handle)],
        )
        .expect("gpu complex");
        let Value::GpuTensor(out) = result else {
            panic!("expected resident complex gpuArray");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
        assert_eq!(gathered, expected);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn complex_wgpu_scalar_real_gpu_imag_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().unwrap();
        let imag = Tensor::new(vec![-1.0, 0.5, 4.0], vec![3, 1]).unwrap();
        let expected =
            complex_call(Value::Num(2.0), vec![Value::Tensor(imag.clone())]).expect("cpu complex");
        let imag_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &imag.data,
                shape: &imag.shape,
            })
            .expect("upload imag");
        let result = complex_call(Value::Num(2.0), vec![Value::GpuTensor(imag_handle)])
            .expect("gpu complex");
        let Value::GpuTensor(out) = result else {
            panic!("expected resident complex gpuArray");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
        assert_eq!(gathered, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_too_many_args_errors() {
        let err =
            complex_call(Value::Num(1.0), vec![Value::Num(2.0), Value::Num(3.0)]).unwrap_err();
        assert!(
            err.message().contains("1 or 2 input arguments"),
            "{}",
            err.message()
        );
    }
}
