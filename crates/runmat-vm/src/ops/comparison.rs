use crate::interpreter::stack::pop2;
use crate::ops::integer_comparison::{
    scalar_order, tensor_element_equals_scalar, tensor_elements_equal,
};
use runmat_builtins::{ComplexTensor, IntValue, IntegerComplexStorage, Tensor, Value};
use runmat_runtime::builtins::common::shape::is_scalar_shape;
use runmat_runtime::RuntimeError;
use std::future::Future;

fn rel_binary_use_builtin(a: &Value, b: &Value) -> bool {
    !matches!(a, Value::Num(_) | Value::Int(_)) || !matches!(b, Value::Num(_) | Value::Int(_))
}

fn logical_bit_equals_int(bit: u8, scalar: &IntValue) -> bool {
    if bit != 0 {
        int_value_is_one(scalar)
    } else {
        scalar.is_zero()
    }
}

fn int_value_is_one(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value == 1,
        IntValue::I16(value) => *value == 1,
        IntValue::I32(value) => *value == 1,
        IntValue::I64(value) => *value == 1,
        IntValue::U8(value) => *value == 1,
        IntValue::U16(value) => *value == 1,
        IntValue::U32(value) => *value == 1,
        IntValue::U64(value) => *value == 1,
    }
}

fn scalar_relation(
    a: &Value,
    b: &Value,
    predicate: fn(f64, f64) -> bool,
) -> Result<bool, RuntimeError> {
    if let Some(ordering) = scalar_order(a, b) {
        let order = match ordering {
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
            std::cmp::Ordering::Greater => 1.0,
        };
        return Ok(predicate(order, 0.0));
    }
    Ok(predicate(a.try_into()?, b.try_into()?))
}

fn integer_value_equals_value(integer: &IntValue, value: &Value) -> bool {
    scalar_order(&Value::Int(integer.clone()), value) == Some(std::cmp::Ordering::Equal)
}

fn integer_value_equals_f64(integer: &IntValue, value: f64) -> bool {
    integer_value_equals_value(integer, &Value::Num(value))
}

fn complex_integer_component_equals(
    left: &IntegerComplexStorage,
    left_index: usize,
    right: &IntegerComplexStorage,
    right_index: usize,
) -> bool {
    let left_real = left
        .real
        .value_at(left_index)
        .expect("left real integer component index");
    let right_real = right
        .real
        .value_at(right_index)
        .expect("right real integer component index");
    let left_imag = left
        .imag
        .value_at(left_index)
        .expect("left imaginary integer component index");
    let right_imag = right
        .imag
        .value_at(right_index)
        .expect("right imaginary integer component index");
    integer_value_equals_value(&left_real, &Value::Int(right_real))
        && integer_value_equals_value(&left_imag, &Value::Int(right_imag))
}

fn complex_integer_element_equals_pair(
    storage: &IntegerComplexStorage,
    index: usize,
    real: f64,
    imag: f64,
) -> bool {
    let storage_real = storage
        .real
        .value_at(index)
        .expect("real integer component index");
    let storage_imag = storage
        .imag
        .value_at(index)
        .expect("imaginary integer component index");
    integer_value_equals_f64(&storage_real, real) && integer_value_equals_f64(&storage_imag, imag)
}

fn complex_integer_element_equals_real_value(
    storage: &IntegerComplexStorage,
    index: usize,
    scalar: &Value,
) -> bool {
    let storage_real = storage
        .real
        .value_at(index)
        .expect("real integer component index");
    let storage_imag = storage
        .imag
        .value_at(index)
        .expect("imaginary integer component index");
    storage_imag.is_zero() && integer_value_equals_value(&storage_real, scalar)
}

fn real_tensor_element_value(tensor: &Tensor, index: usize) -> Value {
    let value = tensor
        .numeric_value_at(index)
        .expect("real tensor element index");
    value
        .into_int_value()
        .map(Value::Int)
        .unwrap_or_else(|| Value::Num(value.materialize_f64()))
}

fn tensor_element_pair(tensor: &ComplexTensor, index: usize) -> (f64, f64) {
    tensor
        .numeric_value_at(index)
        .map(|(real, imag)| (real.materialize_f64(), imag.materialize_f64()))
        .expect("complex tensor index")
}

fn typed_complex_integer_comparison(
    a: &Value,
    b: &Value,
    invert: bool,
) -> Result<Option<Value>, RuntimeError> {
    let make = |matches: bool| -> f64 {
        if matches ^ invert {
            1.0
        } else {
            0.0
        }
    };
    match (a, b) {
        (Value::ComplexTensor(left), Value::ComplexTensor(right))
            if left.integer_storage().is_some() || right.integer_storage().is_some() =>
        {
            if left.shape != right.shape {
                return Err(crate::interpreter::errors::mex(
                    "ShapeMismatch",
                    "shape mismatch for element-wise comparison",
                ));
            }
            let mut out = Vec::with_capacity(left.len());
            match (left.integer_storage(), right.integer_storage()) {
                (Some(left_storage), Some(right_storage)) => {
                    for index in 0..left.len() {
                        out.push(make(complex_integer_component_equals(
                            left_storage,
                            index,
                            right_storage,
                            index,
                        )));
                    }
                }
                (Some(left_storage), None) => {
                    for index in 0..left.len() {
                        let (real, imag) = tensor_element_pair(right, index);
                        out.push(make(complex_integer_element_equals_pair(
                            left_storage,
                            index,
                            real,
                            imag,
                        )));
                    }
                }
                (None, Some(right_storage)) => {
                    for index in 0..left.len() {
                        let (real, imag) = tensor_element_pair(left, index);
                        out.push(make(complex_integer_element_equals_pair(
                            right_storage,
                            index,
                            real,
                            imag,
                        )));
                    }
                }
                (None, None) => unreachable!("guard requires one typed complex integer operand"),
            }
            return Ok(Some(Value::Tensor(
                Tensor::new(out, left.shape.clone())
                    .map_err(|error| format!("complex eq: {error}"))?,
            )));
        }
        (Value::ComplexTensor(tensor), Value::Complex(real, imag))
            if tensor.integer_storage().is_some() =>
        {
            let storage = tensor
                .integer_storage()
                .expect("typed complex tensor storage");
            let out = (0..tensor.len())
                .map(|index| {
                    make(complex_integer_element_equals_pair(
                        storage, index, *real, *imag,
                    ))
                })
                .collect();
            return Ok(Some(Value::Tensor(
                Tensor::new(out, tensor.shape.clone())
                    .map_err(|error| format!("complex scalar eq: {error}"))?,
            )));
        }
        (Value::Complex(real, imag), Value::ComplexTensor(tensor))
            if tensor.integer_storage().is_some() =>
        {
            let storage = tensor
                .integer_storage()
                .expect("typed complex tensor storage");
            let out = (0..tensor.len())
                .map(|index| {
                    make(complex_integer_element_equals_pair(
                        storage, index, *real, *imag,
                    ))
                })
                .collect();
            return Ok(Some(Value::Tensor(
                Tensor::new(out, tensor.shape.clone())
                    .map_err(|error| format!("complex scalar eq: {error}"))?,
            )));
        }
        (Value::ComplexTensor(tensor), Value::Tensor(real))
            if tensor.integer_storage().is_some() =>
        {
            if tensor.shape != real.shape {
                return Err(crate::interpreter::errors::mex(
                    "ShapeMismatch",
                    "shape mismatch for element-wise comparison",
                ));
            }
            let storage = tensor
                .integer_storage()
                .expect("typed complex tensor storage");
            let out = (0..tensor.len())
                .map(|index| {
                    make(complex_integer_element_equals_real_value(
                        storage,
                        index,
                        &real_tensor_element_value(real, index),
                    ))
                })
                .collect();
            return Ok(Some(Value::Tensor(
                Tensor::new(out, tensor.shape.clone())
                    .map_err(|error| format!("complex real eq: {error}"))?,
            )));
        }
        (Value::Tensor(real), Value::ComplexTensor(tensor))
            if tensor.integer_storage().is_some() =>
        {
            if tensor.shape != real.shape {
                return Err(crate::interpreter::errors::mex(
                    "ShapeMismatch",
                    "shape mismatch for element-wise comparison",
                ));
            }
            let storage = tensor
                .integer_storage()
                .expect("typed complex tensor storage");
            let out = (0..tensor.len())
                .map(|index| {
                    make(complex_integer_element_equals_real_value(
                        storage,
                        index,
                        &real_tensor_element_value(real, index),
                    ))
                })
                .collect();
            return Ok(Some(Value::Tensor(
                Tensor::new(out, tensor.shape.clone())
                    .map_err(|error| format!("real complex eq: {error}"))?,
            )));
        }
        (Value::ComplexTensor(tensor), Value::Num(_) | Value::Int(_))
            if tensor.integer_storage().is_some() =>
        {
            let storage = tensor
                .integer_storage()
                .expect("typed complex tensor storage");
            let out = (0..tensor.len())
                .map(|index| make(complex_integer_element_equals_real_value(storage, index, b)))
                .collect();
            return Ok(Some(Value::Tensor(
                Tensor::new(out, tensor.shape.clone())
                    .map_err(|error| format!("complex scalar eq: {error}"))?,
            )));
        }
        (Value::Num(_) | Value::Int(_), Value::ComplexTensor(tensor))
            if tensor.integer_storage().is_some() =>
        {
            let storage = tensor
                .integer_storage()
                .expect("typed complex tensor storage");
            let out = (0..tensor.len())
                .map(|index| make(complex_integer_element_equals_real_value(storage, index, a)))
                .collect();
            return Ok(Some(Value::Tensor(
                Tensor::new(out, tensor.shape.clone())
                    .map_err(|error| format!("scalar complex eq: {error}"))?,
            )));
        }
        _ => {}
    }
    Ok(None)
}

pub struct RelationInvertedSpec {
    pub name: &'static str,
    pub inverse_name: &'static str,
    pub right_name: &'static str,
    pub right_inverse_name: &'static str,
    pub predicate: fn(f64, f64) -> bool,
}

pub async fn relation<CM, CMFut, B, BFut>(
    stack: &mut Vec<Value>,
    name: &'static str,
    reverse_name: &'static str,
    predicate: fn(f64, f64) -> bool,
    mut call_method: CM,
    mut call_builtin: B,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<Value, RuntimeError>>,
    B: FnMut(&'static str, Value, Value) -> BFut,
    BFut: Future<Output = Result<Value, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), name, b.clone()).await {
                Ok(v) => v,
                Err(_) => {
                    if rel_binary_use_builtin(&a, &b) {
                        call_builtin(name, a.clone(), b.clone()).await?
                    } else {
                        Value::Num(if scalar_relation(&a, &b, predicate)? {
                            1.0
                        } else {
                            0.0
                        })
                    }
                }
            }
        }
        (_, Value::Object(obj)) => {
            match call_method(Value::Object(obj.clone()), reverse_name, a.clone()).await {
                Ok(v) => v,
                Err(_) => {
                    if rel_binary_use_builtin(&a, &b) {
                        call_builtin(name, a.clone(), b.clone()).await?
                    } else {
                        Value::Num(if scalar_relation(&a, &b, predicate)? {
                            1.0
                        } else {
                            0.0
                        })
                    }
                }
            }
        }
        _ => {
            if rel_binary_use_builtin(&a, &b) {
                call_builtin(name, a.clone(), b.clone()).await?
            } else {
                Value::Num(if scalar_relation(&a, &b, predicate)? {
                    1.0
                } else {
                    0.0
                })
            }
        }
    };
    stack.push(result);
    Ok(())
}

pub async fn relation_inverted<CM, CMFut, B, BFut, LT, LTFut>(
    stack: &mut Vec<Value>,
    spec: RelationInvertedSpec,
    mut call_method: CM,
    mut call_builtin: B,
    mut logical_truth: LT,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<Value, RuntimeError>>,
    B: FnMut(&'static str, Value, Value) -> BFut,
    BFut: Future<Output = Result<Value, RuntimeError>>,
    LT: FnMut(Value, String) -> LTFut,
    LTFut: Future<Output = Result<bool, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), spec.name, b.clone()).await {
                Ok(v) => v,
                Err(_) => {
                    match call_method(Value::Object(obj.clone()), spec.inverse_name, b.clone())
                        .await
                    {
                        Ok(v) => Value::Num(
                            if !logical_truth(v, "comparison result".to_string()).await? {
                                1.0
                            } else {
                                0.0
                            },
                        ),
                        Err(_) => {
                            if rel_binary_use_builtin(&a, &b) {
                                call_builtin(spec.name, a.clone(), b.clone()).await?
                            } else {
                                Value::Num(if scalar_relation(&a, &b, spec.predicate)? {
                                    1.0
                                } else {
                                    0.0
                                })
                            }
                        }
                    }
                }
            }
        }
        (_, Value::Object(obj)) => {
            match call_method(Value::Object(obj.clone()), spec.right_name, a.clone()).await {
                Ok(v) => v,
                Err(_) => {
                    match call_method(
                        Value::Object(obj.clone()),
                        spec.right_inverse_name,
                        a.clone(),
                    )
                    .await
                    {
                        Ok(v) => Value::Num(
                            if !logical_truth(v, "comparison result".to_string()).await? {
                                1.0
                            } else {
                                0.0
                            },
                        ),
                        Err(_) => {
                            if rel_binary_use_builtin(&a, &b) {
                                call_builtin(spec.name, a.clone(), b.clone()).await?
                            } else {
                                Value::Num(if scalar_relation(&a, &b, spec.predicate)? {
                                    1.0
                                } else {
                                    0.0
                                })
                            }
                        }
                    }
                }
            }
        }
        _ => {
            if rel_binary_use_builtin(&a, &b) {
                call_builtin(spec.name, a.clone(), b.clone()).await?
            } else {
                Value::Num(if scalar_relation(&a, &b, spec.predicate)? {
                    1.0
                } else {
                    0.0
                })
            }
        }
    };
    stack.push(result);
    Ok(())
}

pub async fn equal<CM, CMFut, B, BFut, LT, LTFut>(
    stack: &mut Vec<Value>,
    mut call_method: CM,
    mut call_builtin: B,
    _logical_truth: LT,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<Value, RuntimeError>>,
    B: FnMut(&'static str, Value, Value) -> BFut,
    BFut: Future<Output = Result<Value, RuntimeError>>,
    LT: FnMut(Value, String) -> LTFut,
    LTFut: Future<Output = Result<bool, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    if let Some(result) = typed_complex_integer_comparison(&a, &b, false)? {
        stack.push(result);
        return Ok(());
    }
    let push_logical =
        |data: Vec<u8>, shape: Vec<usize>, stack: &mut Vec<Value>| -> Result<(), RuntimeError> {
            if data.len() == 1 && is_scalar_shape(&shape) {
                stack.push(Value::Bool(data[0] != 0));
                return Ok(());
            }
            let logical =
                runmat_builtins::LogicalArray::new(data, shape).map_err(|e| format!("eq: {e}"))?;
            stack.push(Value::LogicalArray(logical));
            Ok(())
        };
    let logical_eq_scalar = |array: &runmat_builtins::LogicalArray,
                             scalar: f64,
                             stack: &mut Vec<Value>|
     -> Result<(), RuntimeError> {
        let mut out = Vec::with_capacity(array.data.len());
        for &bit in &array.data {
            let val = if bit != 0 { 1.0 } else { 0.0 };
            out.push(if (val - scalar).abs() < 1e-12 { 1 } else { 0 });
        }
        push_logical(out, array.shape.clone(), stack)
    };
    let logical_eq_int_scalar = |array: &runmat_builtins::LogicalArray,
                                 scalar: &IntValue,
                                 stack: &mut Vec<Value>|
     -> Result<(), RuntimeError> {
        let mut out = Vec::with_capacity(array.data.len());
        for &bit in &array.data {
            out.push(if logical_bit_equals_int(bit, scalar) {
                1
            } else {
                0
            });
        }
        push_logical(out, array.shape.clone(), stack)
    };
    let logical_eq_tensor = |array: &runmat_builtins::LogicalArray,
                             tensor: &runmat_builtins::Tensor,
                             stack: &mut Vec<Value>|
     -> Result<(), RuntimeError> {
        if array.shape != tensor.shape {
            return Err(crate::interpreter::errors::mex(
                "ShapeMismatch",
                "shape mismatch for element-wise comparison",
            ));
        }
        let mut out = Vec::with_capacity(array.data.len());
        for i in 0..array.data.len() {
            let val = if array.data[i] != 0 { 1.0 } else { 0.0 };
            out.push(
                if tensor_element_equals_scalar(tensor, i, &Value::Num(val)) {
                    1
                } else {
                    0
                },
            );
        }
        push_logical(out, array.shape.clone(), stack)
    };
    match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "eq", b.clone()).await {
                Ok(v) => stack.push(v),
                Err(_) => {
                    if rel_binary_use_builtin(&a, &b) {
                        stack.push(call_builtin("eq", a.clone(), b.clone()).await?);
                    } else {
                        let aa: f64 = (&a).try_into()?;
                        let bb: f64 = (&b).try_into()?;
                        stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }))
                    }
                }
            }
        }
        (_, Value::Object(obj)) => {
            match call_method(Value::Object(obj.clone()), "eq", a.clone()).await {
                Ok(v) => stack.push(v),
                Err(_) => {
                    if rel_binary_use_builtin(&a, &b) {
                        stack.push(call_builtin("eq", a.clone(), b.clone()).await?);
                    } else {
                        let aa: f64 = (&a).try_into()?;
                        let bb: f64 = (&b).try_into()?;
                        stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }))
                    }
                }
            }
        }
        (Value::HandleObject(_), _) | (_, Value::HandleObject(_)) => {
            stack.push(call_builtin("eq", a.clone(), b.clone()).await?);
        }
        (Value::Symbolic(_), _) | (_, Value::Symbolic(_)) => {
            stack.push(call_builtin("eq", a.clone(), b.clone()).await?);
        }
        (Value::LogicalArray(la), Value::LogicalArray(lb)) => {
            if la.shape != lb.shape {
                return Err(crate::interpreter::errors::mex(
                    "ShapeMismatch",
                    "shape mismatch for element-wise comparison",
                ));
            }
            let mut out = Vec::with_capacity(la.data.len());
            for i in 0..la.data.len() {
                out.push(if la.data[i] == lb.data[i] { 1 } else { 0 });
            }
            push_logical(out, la.shape.clone(), stack)?;
        }
        (Value::LogicalArray(la), Value::Num(n)) => logical_eq_scalar(la, *n, stack)?,
        (Value::LogicalArray(la), Value::Int(i)) => logical_eq_int_scalar(la, i, stack)?,
        (Value::LogicalArray(la), Value::Bool(flag)) => {
            logical_eq_scalar(la, if *flag { 1.0 } else { 0.0 }, stack)?
        }
        (Value::Num(n), Value::LogicalArray(lb)) => logical_eq_scalar(lb, *n, stack)?,
        (Value::Int(i), Value::LogicalArray(lb)) => logical_eq_int_scalar(lb, i, stack)?,
        (Value::Bool(flag), Value::LogicalArray(lb)) => {
            logical_eq_scalar(lb, if *flag { 1.0 } else { 0.0 }, stack)?
        }
        (Value::LogicalArray(la), Value::Tensor(tb)) => logical_eq_tensor(la, tb, stack)?,
        (Value::Tensor(ta), Value::LogicalArray(lb)) => logical_eq_tensor(lb, ta, stack)?,
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                return Err(crate::interpreter::errors::mex(
                    "ShapeMismatch",
                    "shape mismatch for element-wise comparison",
                ));
            }
            let mut out = Vec::with_capacity(ta.len());
            for i in 0..ta.len() {
                out.push(if tensor_elements_equal(ta, tb, i) {
                    1.0
                } else {
                    0.0
                });
            }
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, ta.shape.clone())
                    .map_err(|e| format!("eq: {e}"))?,
            ));
        }
        (Value::Tensor(t), Value::Num(_)) | (Value::Tensor(t), Value::Int(_)) => {
            let out: Vec<f64> = (0..t.len())
                .map(|i| {
                    if tensor_element_equals_scalar(t, i, &b) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, t.shape.clone())
                    .map_err(|e| format!("eq: {e}"))?,
            ));
        }
        (Value::Num(_), Value::Tensor(t)) | (Value::Int(_), Value::Tensor(t)) => {
            let out: Vec<f64> = (0..t.len())
                .map(|i| {
                    if tensor_element_equals_scalar(t, i, &a) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, t.shape.clone())
                    .map_err(|e| format!("eq: {e}"))?,
            ));
        }
        (Value::StringArray(sa), Value::StringArray(sb)) => {
            if sa.shape != sb.shape {
                return Err(crate::interpreter::errors::mex(
                    "ShapeMismatch",
                    "shape mismatch for string array comparison",
                ));
            }
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() {
                out.push(if sa.data[i] == sb.data[i] { 1.0 } else { 0.0 });
            }
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, sa.shape.clone())
                    .map_err(|e| format!("eq: {e}"))?,
            ));
        }
        (Value::StringArray(sa), Value::String(s)) => {
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() {
                out.push(if sa.data[i] == *s { 1.0 } else { 0.0 });
            }
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, sa.shape.clone())
                    .map_err(|e| format!("eq: {e}"))?,
            ));
        }
        (Value::String(s), Value::StringArray(sa)) => {
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() {
                out.push(if *s == sa.data[i] { 1.0 } else { 0.0 });
            }
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, sa.shape.clone())
                    .map_err(|e| format!("eq: {e}"))?,
            ));
        }
        (Value::String(a_s), Value::String(b_s)) => {
            stack.push(Value::Num(if a_s == b_s { 1.0 } else { 0.0 }))
        }
        _ => {
            let equal = if let Some(ordering) = scalar_order(&a, &b) {
                ordering == std::cmp::Ordering::Equal
            } else {
                let aa: f64 = (&a).try_into()?;
                let bb: f64 = (&b).try_into()?;
                aa == bb
            };
            stack.push(Value::Num(if equal { 1.0 } else { 0.0 }));
        }
    }
    Ok(())
}

pub async fn not_equal<CM, CMFut, B, BFut, LT, LTFut>(
    stack: &mut Vec<Value>,
    mut call_method: CM,
    mut call_builtin: B,
    mut logical_truth: LT,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<Value, RuntimeError>>,
    B: FnMut(&'static str, Value, Value) -> BFut,
    BFut: Future<Output = Result<Value, RuntimeError>>,
    LT: FnMut(Value, String) -> LTFut,
    LTFut: Future<Output = Result<bool, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    if let Some(result) = typed_complex_integer_comparison(&a, &b, true)? {
        stack.push(result);
        return Ok(());
    }
    match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "ne", b.clone()).await {
                Ok(v) => stack.push(v),
                Err(_) => match call_method(Value::Object(obj.clone()), "eq", b.clone()).await {
                    Ok(v) => stack.push(Value::Num(
                        if !logical_truth(v, "comparison result".to_string()).await? {
                            1.0
                        } else {
                            0.0
                        },
                    )),
                    Err(_) => {
                        if rel_binary_use_builtin(&a, &b) {
                            stack.push(call_builtin("ne", a.clone(), b.clone()).await?);
                        } else {
                            let aa: f64 = (&a).try_into()?;
                            let bb: f64 = (&b).try_into()?;
                            stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                        }
                    }
                },
            }
        }
        (_, Value::Object(obj)) => {
            match call_method(Value::Object(obj.clone()), "ne", a.clone()).await {
                Ok(v) => stack.push(v),
                Err(_) => match call_method(Value::Object(obj.clone()), "eq", a.clone()).await {
                    Ok(v) => stack.push(Value::Num(
                        if !logical_truth(v, "comparison result".to_string()).await? {
                            1.0
                        } else {
                            0.0
                        },
                    )),
                    Err(_) => {
                        if rel_binary_use_builtin(&a, &b) {
                            stack.push(call_builtin("ne", a.clone(), b.clone()).await?);
                        } else {
                            let aa: f64 = (&a).try_into()?;
                            let bb: f64 = (&b).try_into()?;
                            stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                        }
                    }
                },
            }
        }
        (Value::HandleObject(_), _) | (_, Value::HandleObject(_)) => {
            stack.push(call_builtin("ne", a.clone(), b.clone()).await?)
        }
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                return Err(crate::interpreter::errors::mex(
                    "ShapeMismatch",
                    "shape mismatch for element-wise comparison",
                ));
            }
            let mut out = Vec::with_capacity(ta.len());
            for i in 0..ta.len() {
                out.push(if !tensor_elements_equal(ta, tb, i) {
                    1.0
                } else {
                    0.0
                });
            }
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, ta.shape.clone())
                    .map_err(|e| format!("ne: {e}"))?,
            ));
        }
        (Value::Tensor(t), Value::Num(_)) | (Value::Tensor(t), Value::Int(_)) => {
            let out: Vec<f64> = (0..t.len())
                .map(|i| {
                    if !tensor_element_equals_scalar(t, i, &b) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, t.shape.clone())
                    .map_err(|e| format!("ne: {e}"))?,
            ));
        }
        (Value::Num(_), Value::Tensor(t)) | (Value::Int(_), Value::Tensor(t)) => {
            let out: Vec<f64> = (0..t.len())
                .map(|i| {
                    if !tensor_element_equals_scalar(t, i, &a) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, t.shape.clone())
                    .map_err(|e| format!("ne: {e}"))?,
            ));
        }
        (Value::StringArray(sa), Value::StringArray(sb)) => {
            if sa.shape != sb.shape {
                return Err(crate::interpreter::errors::mex(
                    "ShapeMismatch",
                    "shape mismatch for string array comparison",
                ));
            }
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() {
                out.push(if sa.data[i] != sb.data[i] { 1.0 } else { 0.0 });
            }
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, sa.shape.clone())
                    .map_err(|e| format!("ne: {e}"))?,
            ));
        }
        (Value::StringArray(sa), Value::String(s)) => {
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() {
                out.push(if sa.data[i] != *s { 1.0 } else { 0.0 });
            }
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, sa.shape.clone())
                    .map_err(|e| format!("ne: {e}"))?,
            ));
        }
        (Value::String(s), Value::StringArray(sa)) => {
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() {
                out.push(if *s != sa.data[i] { 1.0 } else { 0.0 });
            }
            stack.push(Value::Tensor(
                runmat_builtins::Tensor::new(out, sa.shape.clone())
                    .map_err(|e| format!("ne: {e}"))?,
            ));
        }
        (Value::String(a_s), Value::String(b_s)) => {
            stack.push(Value::Num(if a_s != b_s { 1.0 } else { 0.0 }))
        }
        _ => {
            let equal = if let Some(ordering) = scalar_order(&a, &b) {
                ordering == std::cmp::Ordering::Equal
            } else {
                let aa: f64 = (&a).try_into()?;
                let bb: f64 = (&b).try_into()?;
                aa == bb
            };
            stack.push(Value::Num(if !equal { 1.0 } else { 0.0 }));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage, LogicalArray};

    async fn unreachable_call_method(
        _receiver: Value,
        _name: &'static str,
        _arg: Value,
    ) -> Result<Value, RuntimeError> {
        panic!("method dispatch should not be used by this test")
    }

    async fn unreachable_call_builtin(
        _name: &'static str,
        _lhs: Value,
        _rhs: Value,
    ) -> Result<Value, RuntimeError> {
        panic!("builtin dispatch should not be used by this test")
    }

    async fn unreachable_logical_truth(
        _value: Value,
        _context: String,
    ) -> Result<bool, RuntimeError> {
        panic!("logical truth should not be used by this test")
    }

    fn logical_array(data: Vec<u8>) -> Value {
        Value::LogicalArray(LogicalArray::new(data, vec![1, 3]).expect("logical array"))
    }

    fn assert_logical_result(value: &Value, expected: &[u8]) {
        let Value::LogicalArray(array) = value else {
            panic!("expected logical array, got {value:?}");
        };
        assert_eq!(array.shape, vec![1, 3]);
        assert_eq!(array.data, expected);
    }

    fn assert_tensor_result(value: &Value, shape: &[usize], expected: &[f64]) {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor result, got {value:?}");
        };
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.materialize_f64(), expected);
    }

    fn complex_uint64(real: Vec<u64>, imag: Vec<u64>, shape: Vec<usize>) -> Value {
        Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(IntegerStorage::U64(real), IntegerStorage::U64(imag))
                    .expect("matching integer components"),
                shape,
            )
            .expect("typed complex integer tensor"),
        )
    }

    #[test]
    fn logical_array_eq_integer_scalar_is_exact_for_large_uint64_rhs() {
        let mut stack = vec![
            logical_array(vec![0, 1, 1]),
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
        ];

        block_on(equal(
            &mut stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect("eq");

        assert_eq!(stack.len(), 1);
        assert_logical_result(&stack[0], &[0, 0, 0]);
    }

    #[test]
    fn logical_array_eq_integer_scalar_is_exact_for_large_uint64_lhs() {
        let mut stack = vec![
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
            logical_array(vec![0, 1, 0]),
        ];

        block_on(equal(
            &mut stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect("eq");

        assert_eq!(stack.len(), 1);
        assert_logical_result(&stack[0], &[0, 0, 0]);
    }

    #[test]
    fn logical_array_eq_integer_scalar_matches_zero_and_one_exactly() {
        let mut zero_stack = vec![logical_array(vec![0, 1, 0]), Value::Int(IntValue::U64(0))];
        block_on(equal(
            &mut zero_stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect("eq zero");
        assert_logical_result(&zero_stack[0], &[1, 0, 1]);

        let mut one_stack = vec![Value::Int(IntValue::I64(1)), logical_array(vec![0, 1, 1])];
        block_on(equal(
            &mut one_stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect("eq one");
        assert_logical_result(&one_stack[0], &[0, 1, 1]);
    }

    #[test]
    fn logical_array_eq_typed_integer_tensor_reads_exact_storage() {
        let integer = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 1, (1_u64 << 53) + 1]),
            vec![1, 3],
        )
        .expect("integer tensor");
        let mut stack = vec![logical_array(vec![0, 1, 1]), Value::Tensor(integer)];

        block_on(equal(
            &mut stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect("eq");

        assert_eq!(stack.len(), 1);
        assert_logical_result(&stack[0], &[1, 1, 0]);
    }

    #[test]
    fn typed_complex_integer_eq_is_exact_above_f64_precision() {
        let large = (1_u64 << 53) + 1;
        let mut stack = vec![
            complex_uint64(vec![large, u64::MAX], vec![0, 7], vec![1, 2]),
            complex_uint64(vec![large, u64::MAX - 1], vec![0, 7], vec![1, 2]),
        ];

        block_on(equal(
            &mut stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect("eq");

        assert_eq!(stack.len(), 1);
        assert_tensor_result(&stack[0], &[1, 2], &[1.0, 0.0]);
    }

    #[test]
    fn typed_complex_integer_eq_typed_real_tensor_reads_exact_storage() {
        let large = (1_u64 << 53) + 1;
        let real = Tensor::new_integer(IntegerStorage::U64(vec![large, u64::MAX]), vec![1, 2])
            .expect("integer tensor");
        let mut stack = vec![
            complex_uint64(vec![large, u64::MAX], vec![0, 1], vec![1, 2]),
            Value::Tensor(real),
        ];

        block_on(equal(
            &mut stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect("eq");

        assert_eq!(stack.len(), 1);
        assert_tensor_result(&stack[0], &[1, 2], &[1.0, 0.0]);
    }

    #[test]
    fn typed_complex_integer_ne_inverts_exact_component_comparison() {
        let mut stack = vec![
            complex_uint64(vec![42, 42, 42], vec![0, 1, 2], vec![1, 3]),
            complex_uint64(vec![42, 42, 43], vec![0, 9, 2], vec![1, 3]),
        ];

        block_on(not_equal(
            &mut stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect("ne");

        assert_eq!(stack.len(), 1);
        assert_tensor_result(&stack[0], &[1, 3], &[0.0, 1.0, 1.0]);
    }

    #[test]
    fn typed_complex_integer_eq_real_requires_zero_imaginary_component() {
        let mut stack = vec![
            complex_uint64(vec![(1_u64 << 53) + 1, 5], vec![0, 1], vec![1, 2]),
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
        ];

        block_on(equal(
            &mut stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect("eq real scalar");

        assert_eq!(stack.len(), 1);
        assert_tensor_result(&stack[0], &[1, 2], &[1.0, 0.0]);
    }

    #[test]
    fn typed_complex_integer_eq_rejects_shape_mismatch() {
        let mut stack = vec![
            complex_uint64(vec![1, 2], vec![0, 0], vec![1, 2]),
            complex_uint64(vec![1, 2], vec![0, 0], vec![2, 1]),
        ];

        let error = block_on(equal(
            &mut stack,
            unreachable_call_method,
            unreachable_call_builtin,
            unreachable_logical_truth,
        ))
        .expect_err("shape mismatch");

        assert!(error.to_string().contains("shape mismatch"));
    }
}
