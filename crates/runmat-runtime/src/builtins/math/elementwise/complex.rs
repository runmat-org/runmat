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
    ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType, ResolveContext, Tensor,
    Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use crate::builtins::math::elementwise::integer_cast::IntegerTarget;
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

const COMPLEX_ERROR_INTEGER_CLASS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INTEGER_CLASS",
    identifier: Some("RunMat:complex:IntegerClass"),
    when: "An integer input is paired with an unsupported unlike input class.",
    message: "complex: integer inputs require matching integer classes or a scalar double",
};

const COMPLEX_ERRORS: [BuiltinErrorDescriptor; 5] = [
    COMPLEX_ERROR_INVALID_ARGUMENT,
    COMPLEX_ERROR_INVALID_INPUT,
    COMPLEX_ERROR_SIZE_MISMATCH,
    COMPLEX_ERROR_INTERNAL,
    COMPLEX_ERROR_INTEGER_CLASS,
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
            let input = value_into_real_input(other)?;
            let tensor = input.tensor;
            let shape = tensor.shape.clone();
            let is_scalar = is_scalar_tensor(&tensor);
            let storage = tensor
                .into_numeric_storage()
                .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
            match storage.into_integer_storage() {
                Ok(storage) => {
                    let zeros = storage.zeros_like(storage.len());
                    let complex = ComplexTensor::new_integer(
                        IntegerComplexStorage::new(storage, zeros)
                            .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?,
                        shape,
                    )
                    .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
                    Ok(complex_tensor_into_value(complex))
                }
                Err(storage) => {
                    let dtype = storage.numeric_dtype();
                    let values = storage.materialize_f64();
                    if is_scalar && dtype == NumericDType::F64 {
                        return Ok(Value::Complex(values[0], 0.0));
                    }
                    let data = values.into_iter().map(|value| (value, 0.0)).collect();
                    let complex = ComplexTensor::from_f64_values_with_dtype(data, shape, dtype)
                        .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
                    Ok(complex_tensor_into_value(complex))
                }
            }
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
                let real_input = value_into_real_input(real_value)?;
                let imag_input = value_into_real_input(imag_value)?;
                return compose_complex(&real_input, &imag_input);
            }
            Err(err) => return Err(err),
        }
    }
    let real_input = value_into_real_input(lhs)?;
    let imag_input = value_into_real_input(rhs)?;
    compose_complex(&real_input, &imag_input)
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
            let input = value_into_real_input(other.clone())?;
            upload_real_tensor(provider, &input.tensor).map(|handle| RealGpuOperand {
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
    let data = tensor::tensor_values_f64_cow(tensor);
    let view = runmat_accelerate_api::HostTensorView {
        data: &data,
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

struct RealInput {
    tensor: Tensor,
    is_scalar_double: bool,
}

fn compose_complex(real: &RealInput, imag: &RealInput) -> BuiltinResult<Value> {
    if real.tensor.integer_storage().is_some() || imag.tensor.integer_storage().is_some() {
        return compose_integer_complex(real, imag);
    }

    compose_floating_complex(&real.tensor, &imag.tensor)
}

fn compose_floating_complex(real: &Tensor, imag: &Tensor) -> BuiltinResult<Value> {
    let output_dtype =
        if real.numeric_dtype() == NumericDType::F32 && imag.numeric_dtype() == NumericDType::F32 {
            NumericDType::F32
        } else {
            NumericDType::F64
        };
    let real_values = tensor::tensor_values_f64_cow(real);
    let imag_values = tensor::tensor_values_f64_cow(imag);
    let (shape, data) = if real.shape == imag.shape {
        let data: Vec<(f64, f64)> = real_values
            .iter()
            .zip(imag_values.iter())
            .map(|(&re, &im)| (re, im))
            .collect();
        (real.shape.clone(), data)
    } else if is_scalar_tensor(real) {
        let re = real_values[0];
        let data: Vec<(f64, f64)> = imag_values.iter().map(|&im| (re, im)).collect();
        (imag.shape.clone(), data)
    } else if is_scalar_tensor(imag) {
        let im = imag_values[0];
        let data: Vec<(f64, f64)> = real_values.iter().map(|&re| (re, im)).collect();
        (real.shape.clone(), data)
    } else {
        return Err(complex_error_with_detail(
            &COMPLEX_ERROR_SIZE_MISMATCH,
            "real and imaginary parts must have the same size, unless one input is scalar",
        ));
    };

    let ct = ComplexTensor::from_f64_values_with_dtype(data, shape, output_dtype)
        .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(ct))
}

fn compose_integer_complex(real: &RealInput, imag: &RealInput) -> BuiltinResult<Value> {
    let real_storage = real.tensor.integer_storage();
    let imag_storage = imag.tensor.integer_storage();
    let prototype = real_storage
        .or(imag_storage)
        .expect("integer input was checked");

    match (real_storage, imag_storage) {
        (Some(real_storage), Some(imag_storage))
            if real_storage.class_name() != imag_storage.class_name() =>
        {
            return Err(complex_error_with_detail(
                &COMPLEX_ERROR_INTEGER_CLASS,
                format!(
                    "got {} and {}; integer inputs must have the same class",
                    real_storage.class_name(),
                    imag_storage.class_name()
                ),
            ));
        }
        (Some(_), None) if !imag.is_scalar_double => {
            return Err(complex_error_with_detail(
                &COMPLEX_ERROR_INTEGER_CLASS,
                "the noninteger input must be a full scalar double",
            ));
        }
        (None, Some(_)) if !real.is_scalar_double => {
            return Err(complex_error_with_detail(
                &COMPLEX_ERROR_INTEGER_CLASS,
                "the noninteger input must be a full scalar double",
            ));
        }
        _ => {}
    }

    let shape = compatible_complex_shape(&real.tensor, &imag.tensor)?;
    let len = shape.iter().product();
    let target = IntegerTarget::from_storage(prototype);
    let real_values = integer_component_values(&real.tensor, real.is_scalar_double, target, len)?;
    let imag_values = integer_component_values(&imag.tensor, imag.is_scalar_double, target, len)?;
    let storage = IntegerComplexStorage::new(real_values, imag_values)
        .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
    let complex = ComplexTensor::new_integer(storage, shape)
        .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(complex))
}

fn integer_component_values(
    tensor: &Tensor,
    is_scalar_double: bool,
    target: IntegerTarget,
    output_len: usize,
) -> BuiltinResult<IntegerStorage> {
    match tensor.integer_storage() {
        Some(storage) if storage.len() == output_len => Ok(storage.clone()),
        Some(storage) if storage.len() == 1 => {
            let value = storage.value_at(0).expect("one-element integer storage");
            prototype_storage_from_values(storage, vec![value; output_len])
        }
        Some(_) => Err(complex_error_with_detail(
            &COMPLEX_ERROR_SIZE_MISMATCH,
            "real and imaginary parts must have the same size, unless one input is scalar",
        )),
        None if is_scalar_double => Ok(target.storage(
            std::iter::repeat_with(|| {
                target.cast_scalar(
                    tensor
                        .as_f64_slice()
                        .expect("scalar-double input has double storage")[0],
                )
            })
            .take(output_len)
            .collect(),
        )),
        None => Err(complex_error_with_detail(
            &COMPLEX_ERROR_INTEGER_CLASS,
            "the noninteger input must be a full scalar double",
        )),
    }
}

fn prototype_storage_from_values(
    prototype: &IntegerStorage,
    values: Vec<runmat_builtins::IntValue>,
) -> BuiltinResult<IntegerStorage> {
    prototype
        .from_same_class_values(values)
        .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INTERNAL, e))
}

fn compatible_complex_shape(real: &Tensor, imag: &Tensor) -> BuiltinResult<Vec<usize>> {
    if real.shape == imag.shape {
        Ok(real.shape.clone())
    } else if is_scalar_tensor(real) {
        Ok(imag.shape.clone())
    } else if is_scalar_tensor(imag) {
        Ok(real.shape.clone())
    } else {
        Err(complex_error_with_detail(
            &COMPLEX_ERROR_SIZE_MISMATCH,
            "real and imaginary parts must have the same size, unless one input is scalar",
        ))
    }
}

fn is_scalar_tensor(tensor: &Tensor) -> bool {
    tensor::is_scalar_tensor(tensor)
}

fn value_into_real_input(value: Value) -> BuiltinResult<RealInput> {
    let is_scalar_double = matches!(value, Value::Num(_));
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
            .map(|tensor| RealInput {
                tensor,
                is_scalar_double,
            })
            .map_err(|e| complex_error_with_detail(&COMPLEX_ERROR_INVALID_INPUT, e)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::gpu_helpers;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        CharArray, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray,
        Tensor, Type, Value,
    };

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

    #[test]
    fn complex_single_components_preserve_native_complex_single_storage() {
        let real = Tensor::from_f32(vec![0.1, 2.0], vec![1, 2]).unwrap();
        let imag = Tensor::from_f32(vec![0.2, -3.0], vec![1, 2]).unwrap();
        let result = complex_call(Value::Tensor(real), vec![Value::Tensor(imag)]).expect("complex");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            result.as_f32_slice(),
            Some(&[(0.1_f32, 0.2_f32), (2.0_f32, -3.0_f32)][..])
        );
        assert_eq!(
            result.materialize_f64(),
            vec![(f64::from(0.1_f32), f64::from(0.2_f32)), (2.0, -3.0)]
        );
    }

    #[test]
    fn complex_single_scalar_retains_class_as_complex_tensor() {
        let real = Tensor::from_f32(vec![0.1], vec![1, 1]).unwrap();
        let imag = Tensor::from_f32(vec![0.2], vec![1, 1]).unwrap();
        let result = complex_call(Value::Tensor(real), vec![Value::Tensor(imag)]).expect("complex");
        let Value::ComplexTensor(result) = result else {
            panic!("single complex scalar must retain its class");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(result.as_f32_slice(), Some(&[(0.1_f32, 0.2_f32)][..]));
    }

    #[test]
    fn complex_integer_components_preserve_uint64_storage_and_scalar_double_expansion() {
        let real = Tensor::new_integer(
            IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
            vec![1, 2],
        )
        .unwrap();
        let result = complex_call(Value::Tensor(real), vec![Value::Num(1.0)]).expect("complex");
        let Value::ComplexTensor(complex) = result else {
            panic!("integer complex values must retain complex tensor storage");
        };
        assert_eq!(complex.shape, vec![1, 2]);
        assert_eq!(
            complex.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
                    IntegerStorage::U64(vec![1, 1]),
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn complex_integer_components_broadcast_from_storage_without_mirrors() {
        let real =
            Tensor::new_integer(IntegerStorage::I32(vec![-3]), vec![1, 1]).expect("real scalar");
        let imag = Tensor::new_integer(IntegerStorage::I32(vec![7, -8, i32::MAX]), vec![3, 1])
            .expect("imag vector");

        let result = complex_call(Value::Tensor(real), vec![Value::Tensor(imag)])
            .expect("complex integer broadcast");
        let Value::ComplexTensor(output) = result else {
            panic!("expected typed complex integer tensor");
        };
        assert_eq!(output.shape, vec![3, 1]);
        assert_eq!(
            output.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I32(vec![-3, -3, -3]),
                    IntegerStorage::I32(vec![7, -8, i32::MAX]),
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn complex_integer_scalar_keeps_exact_complex_storage() {
        let result = complex_call(
            Value::Int(IntValue::I64(i64::MIN)),
            vec![Value::Int(IntValue::I64(7))],
        )
        .expect("complex");
        let Value::ComplexTensor(complex) = result else {
            panic!("integer complex scalar must retain exact storage");
        };
        assert_eq!(complex.shape, vec![1, 1]);
        assert_eq!(
            complex.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MIN]),
                    IntegerStorage::I64(vec![7]),
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn complex_mixed_gpu_path_uploads_typed_integer_storage_exactly() {
        test_support::with_test_provider(|provider| {
            let real = Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![1, 2])
                .expect("typed integer tensor");
            let imag = Tensor::new(vec![10.0, 20.0], vec![1, 2]).expect("imag tensor");
            let imag_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &imag.materialize_f64(),
                    shape: &imag.shape,
                })
                .expect("upload imag");

            let result = complex_call(Value::Tensor(real), vec![Value::GpuTensor(imag_handle)])
                .expect("complex");

            let Value::GpuTensor(out) = result else {
                panic!("expected resident complex gpuArray");
            };
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
            let Value::ComplexTensor(ct) = gathered else {
                panic!("expected gathered complex tensor");
            };
            assert_eq!(ct.shape, vec![1, 2]);
            assert_eq!(ct.materialize_f64(), vec![(2.0, 10.0), (4.0, 20.0)]);
        });
    }

    #[test]
    fn complex_rejects_mixed_integer_classes_and_non_scalar_double_arrays() {
        let mixed = complex_call(
            Value::Int(IntValue::I16(1)),
            vec![Value::Int(IntValue::U16(2))],
        )
        .expect_err("mixed integer classes should fail");
        assert_eq!(mixed.identifier(), COMPLEX_ERROR_INTEGER_CLASS.identifier);

        let doubles = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let array = complex_call(Value::Int(IntValue::I16(1)), vec![Value::Tensor(doubles)])
            .expect_err("integer inputs only permit scalar doubles as unlike operands");
        assert_eq!(array.identifier(), COMPLEX_ERROR_INTEGER_CLASS.identifier);
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
                assert_eq!(
                    ct.materialize_f64(),
                    vec![(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]
                );
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
                assert_eq!(
                    ct.materialize_f64(),
                    vec![(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
                );
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
                assert_eq!(
                    ct.materialize_f64(),
                    vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
                );
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
                assert_eq!(ct.materialize_f64(), vec![(1.0, 3.0), (2.0, 4.0)]);
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
    fn complex_preserves_integer_inputs() {
        let result = complex_call(
            Value::Int(IntValue::I32(3)),
            vec![Value::Int(IntValue::I32(-4))],
        )
        .expect("complex");
        match result {
            Value::ComplexTensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(
                    tensor.integer_storage().cloned(),
                    Some(
                        IntegerComplexStorage::new(
                            IntegerStorage::I32(vec![3]),
                            IntegerStorage::I32(vec![-4]),
                        )
                        .expect("matching components")
                    )
                );
            }
            other => panic!("expected typed complex integer result, got {other:?}"),
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
                assert_eq!(ct.materialize_f64(), vec![(1.0, 0.0), (2.0, 0.0)]);
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
                    ct.materialize_f64(),
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
                assert!(ct.materialize_f64().is_empty());
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
                    data: &real.materialize_f64(),
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
            assert_eq!(
                ct.materialize_f64(),
                vec![(1.0, 0.0), (-2.0, 0.0), (3.5, 0.0)]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_binary_gpu_stays_resident_with_scalar_expansion() {
        test_support::with_test_provider(|provider| {
            let real = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let real_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &real.materialize_f64(),
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
            assert_eq!(
                ct.materialize_f64(),
                vec![(1.0, -4.0), (2.0, -4.0), (3.0, -4.0)]
            );
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
                    data: &real.materialize_f64(),
                    shape: &real.shape,
                })
                .expect("upload real");
            let imag_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &imag.materialize_f64(),
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
            assert!(ct.materialize_f64().is_empty());
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
                    data: &real.materialize_f64(),
                    shape: &real.shape,
                })
                .expect("upload real");
            let imag_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &imag.materialize_f64(),
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
                data: &real.materialize_f64(),
                shape: &real.shape,
            })
            .expect("upload real");
        let imag_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &imag.materialize_f64(),
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
                data: &imag.materialize_f64(),
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
