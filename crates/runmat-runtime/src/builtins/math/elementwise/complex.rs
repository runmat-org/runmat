//! MATLAB-compatible `complex` constructor builtin.
//!
//! `complex(a, b)` constructs `a + 1i*b` element-wise. The real and imaginary
//! parts must have matching sizes unless one input is scalar. `complex(a)`
//! returns real input lifted into complex storage with zero imaginary parts and
//! leaves existing complex input unchanged. Binary inputs must be real numeric.

use runmat_builtins::shape_rules::element_count_if_known;
use runmat_builtins::{ComplexTensor, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "complex";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "complex",
    category = "math/elementwise",
    summary = "Construct complex values from real and imaginary parts, or lift a real value into complex storage.",
    keywords = "complex,construct,imaginary,real,elementwise",
    type_resolver(complex_type),
    builtin_path = "crate::builtins::math::elementwise::complex"
)]
async fn complex_builtin(real: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    match rest.len() {
        0 => unary_complex(real),
        1 => {
            let imag = rest.into_iter().next().expect("rest has one element");
            binary_complex(real, imag)
        }
        n => Err(builtin_error(format!(
            "complex: expected 1 or 2 input arguments, got {}",
            n + 1
        ))),
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

fn unary_complex(value: Value) -> BuiltinResult<Value> {
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
                .map_err(|e| builtin_error(format!("complex: {e}")))?;
            Ok(complex_tensor_into_value(ct))
        }
    }
}

fn binary_complex(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    let real_tensor = value_into_real_tensor(lhs)?;
    let imag_tensor = value_into_real_tensor(rhs)?;
    compose_complex(&real_tensor, &imag_tensor)
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
        return Err(builtin_error(
            "complex: real and imaginary parts must have the same size, unless one input is scalar",
        ));
    };

    if data.is_empty() {
        let empty = ComplexTensor::new(Vec::new(), shape)
            .map_err(|e| builtin_error(format!("complex: {e}")))?;
        return Ok(complex_tensor_into_value(empty));
    }
    let ct = ComplexTensor::new(data, shape).map_err(|e| builtin_error(format!("complex: {e}")))?;
    Ok(complex_tensor_into_value(ct))
}

fn is_scalar_tensor(tensor: &Tensor) -> bool {
    tensor.data.len() == 1
}

fn value_into_real_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(builtin_error("complex: inputs must be real"))
        }
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("complex: expected numeric input, got string"))
        }
        Value::CharArray(_) => Err(builtin_error("complex: expected numeric input, got char")),
        other => tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map_err(|e| builtin_error(format!("complex: {e}"))),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, StringArray, Tensor, Type, Value};

    fn complex_call(real: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::complex_builtin(real, rest))
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
