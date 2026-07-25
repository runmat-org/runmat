use crate::interpreter::stack::pop2;
use crate::ops::integer_comparison::{
    scalar_order, tensor_element_equals_scalar, tensor_elements_equal,
};
use runmat_builtins::{IntValue, Value};
use runmat_runtime::builtins::common::shape::is_scalar_shape;
use runmat_runtime::RuntimeError;
use std::future::Future;

fn rel_binary_use_builtin(a: &Value, b: &Value) -> bool {
    !matches!(a, Value::Num(_) | Value::Int(_)) || !matches!(b, Value::Num(_) | Value::Int(_))
}

fn reject_typed_complex_integer_comparison(a: &Value, b: &Value) -> Result<(), RuntimeError> {
    if matches!(a, Value::ComplexTensor(tensor) if tensor.integer_data.is_some())
        || matches!(b, Value::ComplexTensor(tensor) if tensor.integer_data.is_some())
    {
        return Err(crate::interpreter::errors::mex(
            "ComplexIntegerComparison",
            "operations involving complex numbers with integer types are not supported",
        ));
    }
    Ok(())
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
    reject_typed_complex_integer_comparison(&a, &b)?;
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
            out.push(if (val - tensor.data[i]).abs() < 1e-12 {
                1
            } else {
                0
            });
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
            let mut out = Vec::with_capacity(ta.data.len());
            for i in 0..ta.data.len() {
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
            let out: Vec<f64> = t
                .data
                .iter()
                .enumerate()
                .map(|(i, _)| {
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
            let out: Vec<f64> = t
                .data
                .iter()
                .enumerate()
                .map(|(i, _)| {
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
    reject_typed_complex_integer_comparison(&a, &b)?;
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
            let mut out = Vec::with_capacity(ta.data.len());
            for i in 0..ta.data.len() {
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
            let out: Vec<f64> = t
                .data
                .iter()
                .enumerate()
                .map(|(i, _)| {
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
            let out: Vec<f64> = t
                .data
                .iter()
                .enumerate()
                .map(|(i, _)| {
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
    use runmat_builtins::LogicalArray;

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
}
