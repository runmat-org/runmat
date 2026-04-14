use crate::interpreter::stack::pop2;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use runmat_runtime::builtins::common::shape::is_scalar_shape;
use std::future::Future;

fn rel_binary_use_builtin(a: &Value, b: &Value) -> bool {
    !matches!(a, Value::Num(_) | Value::Int(_)) || !matches!(b, Value::Num(_) | Value::Int(_))
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
        (Value::Object(obj), _) => match call_method(Value::Object(obj.clone()), name, b.clone()).await {
            Ok(v) => v,
            Err(_) => Value::Num(if predicate((&a).try_into()?, (&b).try_into()?) { 1.0 } else { 0.0 }),
        },
        (_, Value::Object(obj)) => match call_method(Value::Object(obj.clone()), reverse_name, a.clone()).await {
            Ok(v) => v,
            Err(_) => Value::Num(if predicate((&a).try_into()?, (&b).try_into()?) { 1.0 } else { 0.0 }),
        },
        _ => {
            if rel_binary_use_builtin(&a, &b) {
                call_builtin(name, a.clone(), b.clone()).await?
            } else {
                Value::Num(if predicate((&a).try_into()?, (&b).try_into()?) { 1.0 } else { 0.0 })
            }
        }
    };
    stack.push(result);
    Ok(())
}

pub async fn relation_inverted<CM, CMFut, B, BFut, LT, LTFut>(
    stack: &mut Vec<Value>,
    name: &'static str,
    inverse_name: &'static str,
    right_name: &'static str,
    right_inverse_name: &'static str,
    predicate: fn(f64, f64) -> bool,
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
        (Value::Object(obj), _) => match call_method(Value::Object(obj.clone()), name, b.clone()).await {
            Ok(v) => v,
            Err(_) => match call_method(Value::Object(obj.clone()), inverse_name, b.clone()).await {
                Ok(v) => Value::Num(if !logical_truth(v, "comparison result".to_string()).await? { 1.0 } else { 0.0 }),
                Err(_) => Value::Num(if predicate((&a).try_into()?, (&b).try_into()?) { 1.0 } else { 0.0 }),
            },
        },
        (_, Value::Object(obj)) => match call_method(Value::Object(obj.clone()), right_name, a.clone()).await {
            Ok(v) => v,
            Err(_) => match call_method(Value::Object(obj.clone()), right_inverse_name, a.clone()).await {
                Ok(v) => Value::Num(if !logical_truth(v, "comparison result".to_string()).await? { 1.0 } else { 0.0 }),
                Err(_) => Value::Num(if predicate((&a).try_into()?, (&b).try_into()?) { 1.0 } else { 0.0 }),
            },
        },
        _ => {
            if rel_binary_use_builtin(&a, &b) {
                call_builtin(name, a.clone(), b.clone()).await?
            } else {
                Value::Num(if predicate((&a).try_into()?, (&b).try_into()?) { 1.0 } else { 0.0 })
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
    let push_logical = |data: Vec<u8>, shape: Vec<usize>, stack: &mut Vec<Value>| -> Result<(), RuntimeError> {
        if data.len() == 1 && is_scalar_shape(&shape) {
            stack.push(Value::Bool(data[0] != 0));
            return Ok(());
        }
        let logical = runmat_builtins::LogicalArray::new(data, shape).map_err(|e| format!("eq: {e}"))?;
        stack.push(Value::LogicalArray(logical));
        Ok(())
    };
    let logical_eq_scalar = |array: &runmat_builtins::LogicalArray, scalar: f64, stack: &mut Vec<Value>| -> Result<(), RuntimeError> {
        let mut out = Vec::with_capacity(array.data.len());
        for &bit in &array.data {
            let val = if bit != 0 { 1.0 } else { 0.0 };
            out.push(if (val - scalar).abs() < 1e-12 { 1 } else { 0 });
        }
        push_logical(out, array.shape.clone(), stack)
    };
    let logical_eq_tensor = |array: &runmat_builtins::LogicalArray, tensor: &runmat_builtins::Tensor, stack: &mut Vec<Value>| -> Result<(), RuntimeError> {
        if array.shape != tensor.shape {
            return Err(crate::interpreter::errors::mex("ShapeMismatch", "shape mismatch for element-wise comparison"));
        }
        let mut out = Vec::with_capacity(array.data.len());
        for i in 0..array.data.len() {
            let val = if array.data[i] != 0 { 1.0 } else { 0.0 };
            out.push(if (val - tensor.data[i]).abs() < 1e-12 { 1 } else { 0 });
        }
        push_logical(out, array.shape.clone(), stack)
    };
    match (&a, &b) {
        (Value::Object(obj), _) => match call_method(Value::Object(obj.clone()), "eq", b.clone()).await {
            Ok(v) => stack.push(v),
            Err(_) => {
                let aa: f64 = (&a).try_into()?;
                let bb: f64 = (&b).try_into()?;
                stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }))
            }
        },
        (_, Value::Object(obj)) => match call_method(Value::Object(obj.clone()), "eq", a.clone()).await {
            Ok(v) => stack.push(v),
            Err(_) => {
                let aa: f64 = (&a).try_into()?;
                let bb: f64 = (&b).try_into()?;
                stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }))
            }
        },
        (Value::HandleObject(_), _) | (_, Value::HandleObject(_)) => {
            stack.push(call_builtin("eq", a.clone(), b.clone()).await?);
        }
        (Value::LogicalArray(la), Value::LogicalArray(lb)) => {
            if la.shape != lb.shape {
                return Err(crate::interpreter::errors::mex("ShapeMismatch", "shape mismatch for element-wise comparison"));
            }
            let mut out = Vec::with_capacity(la.data.len());
            for i in 0..la.data.len() {
                out.push(if la.data[i] == lb.data[i] { 1 } else { 0 });
            }
            push_logical(out, la.shape.clone(), stack)?;
        }
        (Value::LogicalArray(la), Value::Num(n)) => logical_eq_scalar(la, *n, stack)?,
        (Value::LogicalArray(la), Value::Int(i)) => logical_eq_scalar(la, i.to_f64(), stack)?,
        (Value::LogicalArray(la), Value::Bool(flag)) => logical_eq_scalar(la, if *flag { 1.0 } else { 0.0 }, stack)?,
        (Value::Num(n), Value::LogicalArray(lb)) => logical_eq_scalar(lb, *n, stack)?,
        (Value::Int(i), Value::LogicalArray(lb)) => logical_eq_scalar(lb, i.to_f64(), stack)?,
        (Value::Bool(flag), Value::LogicalArray(lb)) => logical_eq_scalar(lb, if *flag { 1.0 } else { 0.0 }, stack)?,
        (Value::LogicalArray(la), Value::Tensor(tb)) => logical_eq_tensor(la, tb, stack)?,
        (Value::Tensor(ta), Value::LogicalArray(lb)) => logical_eq_tensor(lb, ta, stack)?,
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                return Err(crate::interpreter::errors::mex("ShapeMismatch", "shape mismatch for element-wise comparison"));
            }
            let mut out = Vec::with_capacity(ta.data.len());
            for i in 0..ta.data.len() {
                out.push(if (ta.data[i] - tb.data[i]).abs() < 1e-12 { 1.0 } else { 0.0 });
            }
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, ta.shape.clone()).map_err(|e| format!("eq: {e}"))?));
        }
        (Value::Tensor(t), Value::Num(_)) | (Value::Tensor(t), Value::Int(_)) => {
            let s = match &b { Value::Num(n) => *n, Value::Int(i) => i.to_f64(), _ => 0.0 };
            let out: Vec<f64> = t.data.iter().map(|x| if (*x - s).abs() < 1e-12 { 1.0 } else { 0.0 }).collect();
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, t.shape.clone()).map_err(|e| format!("eq: {e}"))?));
        }
        (Value::Num(_), Value::Tensor(t)) | (Value::Int(_), Value::Tensor(t)) => {
            let s = match &a { Value::Num(n) => *n, Value::Int(i) => i.to_f64(), _ => 0.0 };
            let out: Vec<f64> = t.data.iter().map(|x| if (s - *x).abs() < 1e-12 { 1.0 } else { 0.0 }).collect();
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, t.shape.clone()).map_err(|e| format!("eq: {e}"))?));
        }
        (Value::StringArray(sa), Value::StringArray(sb)) => {
            if sa.shape != sb.shape {
                return Err(crate::interpreter::errors::mex("ShapeMismatch", "shape mismatch for string array comparison"));
            }
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() {
                out.push(if sa.data[i] == sb.data[i] { 1.0 } else { 0.0 });
            }
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, sa.shape.clone()).map_err(|e| format!("eq: {e}"))?));
        }
        (Value::StringArray(sa), Value::String(s)) => {
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() { out.push(if sa.data[i] == *s { 1.0 } else { 0.0 }); }
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, sa.shape.clone()).map_err(|e| format!("eq: {e}"))?));
        }
        (Value::String(s), Value::StringArray(sa)) => {
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() { out.push(if *s == sa.data[i] { 1.0 } else { 0.0 }); }
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, sa.shape.clone()).map_err(|e| format!("eq: {e}"))?));
        }
        (Value::String(a_s), Value::String(b_s)) => stack.push(Value::Num(if a_s == b_s { 1.0 } else { 0.0 })),
        _ => {
            let bb: f64 = (&b).try_into()?;
            let aa: f64 = (&a).try_into()?;
            stack.push(Value::Num(if aa == bb { 1.0 } else { 0.0 }));
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
    match (&a, &b) {
        (Value::Object(obj), _) => match call_method(Value::Object(obj.clone()), "ne", b.clone()).await {
            Ok(v) => stack.push(v),
            Err(_) => match call_method(Value::Object(obj.clone()), "eq", b.clone()).await {
                Ok(v) => stack.push(Value::Num(if !logical_truth(v, "comparison result".to_string()).await? { 1.0 } else { 0.0 })),
                Err(_) => {
                    let aa: f64 = (&a).try_into()?;
                    let bb: f64 = (&b).try_into()?;
                    stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                }
            },
        },
        (_, Value::Object(obj)) => match call_method(Value::Object(obj.clone()), "ne", a.clone()).await {
            Ok(v) => stack.push(v),
            Err(_) => match call_method(Value::Object(obj.clone()), "eq", a.clone()).await {
                Ok(v) => stack.push(Value::Num(if !logical_truth(v, "comparison result".to_string()).await? { 1.0 } else { 0.0 })),
                Err(_) => {
                    let aa: f64 = (&a).try_into()?;
                    let bb: f64 = (&b).try_into()?;
                    stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
                }
            },
        },
        (Value::HandleObject(_), _) | (_, Value::HandleObject(_)) => stack.push(call_builtin("ne", a.clone(), b.clone()).await?),
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                return Err(crate::interpreter::errors::mex("ShapeMismatch", "shape mismatch for element-wise comparison"));
            }
            let mut out = Vec::with_capacity(ta.data.len());
            for i in 0..ta.data.len() { out.push(if (ta.data[i] - tb.data[i]).abs() >= 1e-12 { 1.0 } else { 0.0 }); }
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, ta.shape.clone()).map_err(|e| format!("ne: {e}"))?));
        }
        (Value::Tensor(t), Value::Num(_)) | (Value::Tensor(t), Value::Int(_)) => {
            let s = match &b { Value::Num(n) => *n, Value::Int(i) => i.to_f64(), _ => 0.0 };
            let out: Vec<f64> = t.data.iter().map(|x| if (*x - s).abs() >= 1e-12 { 1.0 } else { 0.0 }).collect();
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, t.shape.clone()).map_err(|e| format!("ne: {e}"))?));
        }
        (Value::Num(_), Value::Tensor(t)) | (Value::Int(_), Value::Tensor(t)) => {
            let s = match &a { Value::Num(n) => *n, Value::Int(i) => i.to_f64(), _ => 0.0 };
            let out: Vec<f64> = t.data.iter().map(|x| if (s - *x).abs() >= 1e-12 { 1.0 } else { 0.0 }).collect();
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, t.shape.clone()).map_err(|e| format!("ne: {e}"))?));
        }
        (Value::StringArray(sa), Value::StringArray(sb)) => {
            if sa.shape != sb.shape {
                return Err(crate::interpreter::errors::mex("ShapeMismatch", "shape mismatch for string array comparison"));
            }
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() { out.push(if sa.data[i] != sb.data[i] { 1.0 } else { 0.0 }); }
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, sa.shape.clone()).map_err(|e| format!("ne: {e}"))?));
        }
        (Value::StringArray(sa), Value::String(s)) => {
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() { out.push(if sa.data[i] != *s { 1.0 } else { 0.0 }); }
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, sa.shape.clone()).map_err(|e| format!("ne: {e}"))?));
        }
        (Value::String(s), Value::StringArray(sa)) => {
            let mut out = Vec::with_capacity(sa.data.len());
            for i in 0..sa.data.len() { out.push(if *s != sa.data[i] { 1.0 } else { 0.0 }); }
            stack.push(Value::Tensor(runmat_builtins::Tensor::new(out, sa.shape.clone()).map_err(|e| format!("ne: {e}"))?));
        }
        (Value::String(a_s), Value::String(b_s)) => stack.push(Value::Num(if a_s != b_s { 1.0 } else { 0.0 })),
        _ => {
            let bb: f64 = (&b).try_into()?;
            let aa: f64 = (&a).try_into()?;
            stack.push(Value::Num(if aa != bb { 1.0 } else { 0.0 }));
        }
    }
    Ok(())
}
