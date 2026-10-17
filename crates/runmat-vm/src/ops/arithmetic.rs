use crate::interpreter::errors::mex;
use crate::interpreter::stack::pop2;
use runmat_runtime::builtins::common::shape::is_scalar_shape;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::object::dispatch::OperatorMethodDispatch;
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::future::Future;

fn reject_typed_complex_integer_arithmetic(values: &[&Value]) -> Result<(), RuntimeError> {
    if values.iter().any(
        |value| matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some()),
    ) {
        return Err(mex(
            "ComplexIntegerArithmetic",
            "complex integer arithmetic is not supported",
        ));
    }
    Ok(())
}

pub async fn add<CM, CMFut, RM, RMFut, F, FFut>(
    stack: &mut Vec<Value>,
    mut call_method: CM,
    mut right_method: RM,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    RM: FnMut(Value, Value, &'static str) -> RMFut,
    RMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    reject_typed_complex_integer_arithmetic(&[&a, &b])?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "plus", b.clone()).await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        (_, Value::Object(obj)) => {
            match right_method(a.clone(), Value::Object(obj.clone()), "plus").await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        _ => fallback(a.clone(), b.clone()).await?,
    };
    result.with_owned(|value| stack.push(value));
    Ok(())
}

pub async fn sub<CM, CMFut, RM, RMFut, F, FFut>(
    stack: &mut Vec<Value>,
    mut call_method: CM,
    mut right_method: RM,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    RM: FnMut(Value, Value, &'static str) -> RMFut,
    RMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    reject_typed_complex_integer_arithmetic(&[&a, &b])?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "minus", b.clone()).await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        (_, Value::Object(obj)) => {
            match right_method(a.clone(), Value::Object(obj.clone()), "minus").await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        _ => fallback(a.clone(), b.clone()).await?,
    };
    result.with_owned(|value| stack.push(value));
    Ok(())
}

pub async fn mul<CM, CMFut, RM, RMFut, F, FFut>(
    stack: &mut Vec<Value>,
    mut call_method: CM,
    mut right_method: RM,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    RM: FnMut(Value, Value, &'static str) -> RMFut,
    RMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    reject_typed_complex_integer_arithmetic(&[&a, &b])?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "mtimes", b.clone()).await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        (_, Value::Object(obj)) => {
            match right_method(a.clone(), Value::Object(obj.clone()), "mtimes").await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        _ => fallback(a.clone(), b.clone()).await?,
    };
    result.with_owned(|value| stack.push(value));
    Ok(())
}

pub async fn binary_method_ordered<CM, CMFut, RM, RMFut, F, FFut>(
    stack: &mut Vec<Value>,
    method: &'static str,
    mut call_method: CM,
    mut call_right_method: RM,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    RM: FnMut(Value, Value, &'static str) -> RMFut,
    RMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    reject_typed_complex_integer_arithmetic(&[&a, &b])?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), method, b.clone()).await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        (_, Value::Object(obj)) => {
            match call_right_method(a.clone(), Value::Object(obj.clone()), method).await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        _ => fallback(a.clone(), b.clone()).await?,
    };
    result.with_owned(|value| stack.push(value));
    Ok(())
}

pub async fn binary_fallback<F, FFut>(
    stack: &mut Vec<Value>,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    reject_typed_complex_integer_arithmetic(&[&a, &b])?;
    fallback(a, b).await?.with_owned(|value| stack.push(value));
    Ok(())
}

pub async fn power<CM, CMFut, RM, RMFut, F, FFut>(
    stack: &mut Vec<Value>,
    mut call_method: CM,
    mut call_right_method: RM,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    RM: FnMut(Value, Value, &'static str) -> RMFut,
    RMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    reject_typed_complex_integer_arithmetic(&[&a, &b])?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "power", b.clone()).await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        (_, Value::Object(obj)) => {
            match call_right_method(a.clone(), Value::Object(obj.clone()), "power").await? {
                Some(value) => value,
                None => fallback(a.clone(), b.clone()).await?,
            }
        }
        _ => fallback(a.clone(), b.clone()).await?,
    };
    result.with_owned(|value| stack.push(value));
    Ok(())
}

pub async fn unary<UF, UFut>(stack: &mut Vec<Value>, mut op: UF) -> Result<(), RuntimeError>
where
    UF: FnMut(Value) -> UFut,
    UFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let value = stack
        .pop()
        .ok_or(mex("StackUnderflow", "stack underflow"))?;
    let _input_roots = crate::call::root_values([&value])?;
    op(value).await?.with_owned(|result| stack.push(result));
    Ok(())
}

pub async fn unary_arithmetic<UF, UFut>(
    stack: &mut Vec<Value>,
    mut op: UF,
) -> Result<(), RuntimeError>
where
    UF: FnMut(Value) -> UFut,
    UFut: Future<Output = Result<runmat_runtime::execution::RootedValue, RuntimeError>>,
{
    let value = stack
        .pop()
        .ok_or(mex("StackUnderflow", "stack underflow"))?;
    reject_typed_complex_integer_arithmetic(&[&value])?;
    let _input_roots = crate::call::root_values([&value])?;
    op(value).await?.with_owned(|result| stack.push(result));
    Ok(())
}

pub fn is_scalarish_for_division(value: &Value) -> bool {
    match value {
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::Symbolic(_) => true,
        Value::SymbolicArray(array) => is_scalar_shape(&array.shape),
        Value::LogicalArray(arr) => is_scalar_shape(&arr.shape),
        Value::Tensor(tensor) => is_scalar_shape(tensor.shape()),
        Value::ComplexTensor(tensor) => is_scalar_shape(tensor.shape()),
        Value::GpuTensor(handle) => is_scalar_shape(handle.shape()),
        _ => false,
    }
}

pub async fn execute_right_division<CM, CMFut, RM, RMFut, SF, SFFut, MF, MFFut>(
    lhs: &Value,
    rhs: &Value,
    mut call_method: CM,
    mut call_right_method: RM,
    mut scalarish_fallback: SF,
    mut matrix_fallback: MF,
) -> Result<RootedValue, RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    RM: FnMut(Value, Value, &'static str) -> RMFut,
    RMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    SF: FnMut(Value, Value) -> SFFut,
    SFFut: Future<Output = Result<RootedValue, RuntimeError>>,
    MF: FnMut(Value, Value) -> MFFut,
    MFFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    match (lhs, rhs) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "mrdivide", rhs.clone()).await? {
                Some(value) => Ok(value),
                None => {
                    if is_scalarish_for_division(rhs) {
                        scalarish_fallback(lhs.clone(), rhs.clone()).await
                    } else {
                        matrix_fallback(lhs.clone(), rhs.clone()).await
                    }
                }
            }
        }
        (_, Value::Object(obj)) => {
            match call_right_method(lhs.clone(), Value::Object(obj.clone()), "mrdivide").await? {
                Some(value) => Ok(value),
                None => {
                    if is_scalarish_for_division(rhs) {
                        scalarish_fallback(lhs.clone(), rhs.clone()).await
                    } else {
                        matrix_fallback(lhs.clone(), rhs.clone()).await
                    }
                }
            }
        }
        _ => {
            if is_scalarish_for_division(rhs) {
                scalarish_fallback(lhs.clone(), rhs.clone()).await
            } else {
                matrix_fallback(lhs.clone(), rhs.clone()).await
            }
        }
    }
}

pub async fn execute_left_division<CM, CMFut, RM, RMFut, SF, SFFut, MF, MFFut>(
    lhs: &Value,
    rhs: &Value,
    mut call_method: CM,
    mut call_right_method: RM,
    mut scalarish_fallback: SF,
    mut matrix_fallback: MF,
) -> Result<RootedValue, RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    RM: FnMut(Value, Value, &'static str) -> RMFut,
    RMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    SF: FnMut(Value, Value) -> SFFut,
    SFFut: Future<Output = Result<RootedValue, RuntimeError>>,
    MF: FnMut(Value, Value) -> MFFut,
    MFFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    match (lhs, rhs) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "mldivide", rhs.clone()).await? {
                Some(value) => Ok(value),
                None => {
                    if is_scalarish_for_division(lhs) {
                        scalarish_fallback(lhs.clone(), rhs.clone()).await
                    } else {
                        matrix_fallback(lhs.clone(), rhs.clone()).await
                    }
                }
            }
        }
        (_, Value::Object(obj)) => {
            match call_right_method(lhs.clone(), Value::Object(obj.clone()), "mldivide").await? {
                Some(value) => Ok(value),
                None => {
                    if is_scalarish_for_division(lhs) {
                        scalarish_fallback(lhs.clone(), rhs.clone()).await
                    } else {
                        matrix_fallback(lhs.clone(), rhs.clone()).await
                    }
                }
            }
        }
        _ => {
            if is_scalarish_for_division(lhs) {
                scalarish_fallback(lhs.clone(), rhs.clone()).await
            } else {
                matrix_fallback(lhs.clone(), rhs.clone()).await
            }
        }
    }
}

#[cfg(test)]
#[path = "arithmetic/replay_tests.rs"]
mod replay_tests;
