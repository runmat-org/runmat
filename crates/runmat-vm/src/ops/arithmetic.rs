use crate::interpreter::errors::mex;
use crate::interpreter::stack::pop2;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use std::future::Future;

pub async fn add<CM, CMFut, F, FFut>(
    stack: &mut Vec<Value>,
    mut call_method: CM,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<Value, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<Value, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => match call_method(Value::Object(obj.clone()), "plus", b.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        (_, Value::Object(obj)) => match call_method(Value::Object(obj.clone()), "plus", a.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        _ => fallback(a.clone(), b.clone()).await?,
    };
    stack.push(result);
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
    CMFut: Future<Output = Result<Value, RuntimeError>>,
    RM: FnMut(Value, Value) -> RMFut,
    RMFut: Future<Output = Result<Value, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<Value, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => match call_method(Value::Object(obj.clone()), "minus", b.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        (_, Value::Object(obj)) => match right_method(Value::Object(obj.clone()), a.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        _ => fallback(a.clone(), b.clone()).await?,
    };
    stack.push(result);
    Ok(())
}

pub async fn mul<CM, CMFut, F, FFut>(
    stack: &mut Vec<Value>,
    mut call_method: CM,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<Value, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<Value, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => match call_method(Value::Object(obj.clone()), "mtimes", b.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        (_, Value::Object(obj)) => match call_method(Value::Object(obj.clone()), "mtimes", a.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        _ => fallback(a.clone(), b.clone()).await?,
    };
    stack.push(result);
    Ok(())
}

pub async fn binary_method<CM, CMFut, F, FFut>(
    stack: &mut Vec<Value>,
    method: &'static str,
    mut call_method: CM,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<Value, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<Value, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => match call_method(Value::Object(obj.clone()), method, b.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        (_, Value::Object(obj)) => match call_method(Value::Object(obj.clone()), method, a.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        _ => fallback(a.clone(), b.clone()).await?,
    };
    stack.push(result);
    Ok(())
}

pub async fn binary_fallback<F, FFut>(stack: &mut Vec<Value>, mut fallback: F) -> Result<(), RuntimeError>
where
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<Value, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    stack.push(fallback(a, b).await?);
    Ok(())
}

pub async fn power<CM, CMFut, F, FFut>(
    stack: &mut Vec<Value>,
    mut call_method: CM,
    mut fallback: F,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<Value, RuntimeError>>,
    F: FnMut(Value, Value) -> FFut,
    FFut: Future<Output = Result<Value, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => match call_method(Value::Object(obj.clone()), "power", b.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        (_, Value::Object(obj)) => match call_method(Value::Object(obj.clone()), "power", a.clone()).await {
            Ok(v) => v,
            Err(_) => fallback(a.clone(), b.clone()).await?,
        },
        _ => fallback(a.clone(), b.clone()).await?,
    };
    stack.push(result);
    Ok(())
}

pub async fn unary<UF, UFut>(stack: &mut Vec<Value>, mut op: UF) -> Result<(), RuntimeError>
where
    UF: FnMut(Value) -> UFut,
    UFut: Future<Output = Result<Value, RuntimeError>>,
{
    let value = stack.pop().ok_or(mex("StackUnderflow", "stack underflow"))?;
    stack.push(op(value).await?);
    Ok(())
}
