use crate::interpreter::stack::pop2;
use crate::ops::integer_comparison::scalar_order;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::object::dispatch::OperatorMethodDispatch;
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::future::Future;

fn rel_binary_use_builtin(a: &Value, b: &Value) -> bool {
    !matches!(a, Value::Num(_) | Value::Int(_)) || !matches!(b, Value::Num(_) | Value::Int(_))
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

fn rooted_logical(value: bool) -> Result<RootedValue, RuntimeError> {
    RootedValue::new(Value::Bool(value)).map_err(|error| error.into_runtime_error())
}

pub async fn relation<CM, CMFut, B, BFut>(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    name: &'static str,
    reverse_name: &'static str,
    predicate: fn(f64, f64) -> bool,
    mut call_method: CM,
    mut call_builtin: B,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    B: FnMut(&'static str, Value, Value) -> BFut,
    BFut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), name, b.clone()).await? {
                Some(v) => v,
                None => {
                    if rel_binary_use_builtin(&a, &b) {
                        call_builtin(name, a.clone(), b.clone()).await?
                    } else {
                        rooted_logical(scalar_relation(&a, &b, predicate)?)?
                    }
                }
            }
        }
        (_, Value::Object(obj)) => {
            match call_method(Value::Object(obj.clone()), reverse_name, a.clone()).await? {
                Some(v) => v,
                None => {
                    if rel_binary_use_builtin(&a, &b) {
                        call_builtin(name, a.clone(), b.clone()).await?
                    } else {
                        rooted_logical(scalar_relation(&a, &b, predicate)?)?
                    }
                }
            }
        }
        _ => {
            if rel_binary_use_builtin(&a, &b) {
                call_builtin(name, a.clone(), b.clone()).await?
            } else {
                rooted_logical(scalar_relation(&a, &b, predicate)?)?
            }
        }
    };
    result.with_owned(|value| stack.try_push(value))
}

pub async fn relation_inverted<CM, CMFut, B, BFut, LT, LTFut>(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    spec: RelationInvertedSpec,
    mut call_method: CM,
    mut call_builtin: B,
    mut logical_truth: LT,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    B: FnMut(&'static str, Value, Value) -> BFut,
    BFut: Future<Output = Result<RootedValue, RuntimeError>>,
    LT: FnMut(Value, String) -> LTFut,
    LTFut: Future<Output = Result<bool, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    let result = match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), spec.name, b.clone()).await? {
                Some(v) => v,
                None => {
                    match call_method(Value::Object(obj.clone()), spec.inverse_name, b.clone())
                        .await?
                    {
                        Some(v) => rooted_logical(
                            !v.with_owned_async(|value| {
                                logical_truth(value, "comparison result".to_string())
                            })
                            .await?,
                        )?,
                        None => {
                            if rel_binary_use_builtin(&a, &b) {
                                call_builtin(spec.name, a.clone(), b.clone()).await?
                            } else {
                                rooted_logical(scalar_relation(&a, &b, spec.predicate)?)?
                            }
                        }
                    }
                }
            }
        }
        (_, Value::Object(obj)) => {
            match call_method(Value::Object(obj.clone()), spec.right_name, a.clone()).await? {
                Some(v) => v,
                None => {
                    match call_method(
                        Value::Object(obj.clone()),
                        spec.right_inverse_name,
                        a.clone(),
                    )
                    .await?
                    {
                        Some(v) => rooted_logical(
                            !v.with_owned_async(|value| {
                                logical_truth(value, "comparison result".to_string())
                            })
                            .await?,
                        )?,
                        None => {
                            if rel_binary_use_builtin(&a, &b) {
                                call_builtin(spec.name, a.clone(), b.clone()).await?
                            } else {
                                rooted_logical(scalar_relation(&a, &b, spec.predicate)?)?
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
                rooted_logical(scalar_relation(&a, &b, spec.predicate)?)?
            }
        }
    };
    result.with_owned(|value| stack.try_push(value))
}

pub async fn equal<CM, CMFut, B, BFut, LT, LTFut>(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    mut call_method: CM,
    mut call_builtin: B,
    _logical_truth: LT,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    B: FnMut(&'static str, Value, Value) -> BFut,
    BFut: Future<Output = Result<RootedValue, RuntimeError>>,
    LT: FnMut(Value, String) -> LTFut,
    LTFut: Future<Output = Result<bool, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "eq", b.clone()).await? {
                Some(v) => v.with_owned(|value| stack.try_push(value)),
                None => call_builtin("eq", a.clone(), b.clone())
                    .await?
                    .with_owned(|value| stack.try_push(value)),
            }
        }
        (_, Value::Object(obj)) => {
            match call_method(Value::Object(obj.clone()), "eq", a.clone()).await? {
                Some(v) => v.with_owned(|value| stack.try_push(value)),
                None => call_builtin("eq", a.clone(), b.clone())
                    .await?
                    .with_owned(|value| stack.try_push(value)),
            }
        }
        _ if rel_binary_use_builtin(&a, &b) => call_builtin("eq", a, b)
            .await?
            .with_owned(|value| stack.try_push(value)),
        _ => {
            let equal = if let Some(ordering) = scalar_order(&a, &b) {
                ordering == std::cmp::Ordering::Equal
            } else {
                let aa: f64 = (&a).try_into()?;
                let bb: f64 = (&b).try_into()?;
                aa == bb
            };
            stack.try_push(Value::Bool(equal))
        }
    }?;
    Ok(())
}

pub async fn not_equal<CM, CMFut, B, BFut, LT, LTFut>(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    mut call_method: CM,
    mut call_builtin: B,
    mut logical_truth: LT,
) -> Result<(), RuntimeError>
where
    CM: FnMut(Value, &'static str, Value) -> CMFut,
    CMFut: Future<Output = Result<OperatorMethodDispatch, RuntimeError>>,
    B: FnMut(&'static str, Value, Value) -> BFut,
    BFut: Future<Output = Result<RootedValue, RuntimeError>>,
    LT: FnMut(Value, String) -> LTFut,
    LTFut: Future<Output = Result<bool, RuntimeError>>,
{
    let (a, b) = pop2(stack)?;
    let _input_roots = crate::call::root_values([&a, &b])?;
    match (&a, &b) {
        (Value::Object(obj), _) => {
            match call_method(Value::Object(obj.clone()), "ne", b.clone()).await? {
                Some(v) => v.with_owned(|value| stack.try_push(value)),
                None => match call_method(Value::Object(obj.clone()), "eq", b.clone()).await? {
                    Some(v) => stack.try_push(Value::Bool(
                        !v.with_owned_async(|value| {
                            logical_truth(value, "comparison result".to_string())
                        })
                        .await?,
                    )),
                    None => call_builtin("ne", a.clone(), b.clone())
                        .await?
                        .with_owned(|value| stack.try_push(value)),
                },
            }
        }
        (_, Value::Object(obj)) => {
            match call_method(Value::Object(obj.clone()), "ne", a.clone()).await? {
                Some(v) => v.with_owned(|value| stack.try_push(value)),
                None => match call_method(Value::Object(obj.clone()), "eq", a.clone()).await? {
                    Some(v) => stack.try_push(Value::Bool(
                        !v.with_owned_async(|value| {
                            logical_truth(value, "comparison result".to_string())
                        })
                        .await?,
                    )),
                    None => call_builtin("ne", a.clone(), b.clone())
                        .await?
                        .with_owned(|value| stack.try_push(value)),
                },
            }
        }
        _ if rel_binary_use_builtin(&a, &b) => call_builtin("ne", a, b)
            .await?
            .with_owned(|value| stack.try_push(value)),
        _ => {
            let equal = if let Some(ordering) = scalar_order(&a, &b) {
                ordering == std::cmp::Ordering::Equal
            } else {
                let aa: f64 = (&a).try_into()?;
                let bb: f64 = (&b).try_into()?;
                aa == bb
            };
            stack.try_push(Value::Bool(!equal))
        }
    }?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntValue;

    async fn unreachable_call_method(
        _receiver: Value,
        _name: &'static str,
        _arg: Value,
    ) -> Result<OperatorMethodDispatch, RuntimeError> {
        panic!("method dispatch should not be used by this test")
    }

    async fn unreachable_call_builtin(
        _name: &'static str,
        _lhs: Value,
        _rhs: Value,
    ) -> Result<RootedValue, RuntimeError> {
        panic!("builtin dispatch should not be used by this test")
    }

    async fn unreachable_logical_truth(
        _value: Value,
        _context: String,
    ) -> Result<bool, RuntimeError> {
        panic!("logical truth should not be used by this test")
    }

    #[test]
    fn equality_real_scalar_fast_path_preserves_exact_wide_integers() {
        let wide = (1_u64 << 53) + 1;
        for (left, right, expected) in [
            (
                Value::Int(IntValue::U64(wide)),
                Value::Num((wide - 1) as f64),
                false,
            ),
            (
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::I64(-1)),
                false,
            ),
            (
                Value::Int(IntValue::I64(255)),
                Value::Int(IntValue::U64(255)),
                true,
            ),
            (Value::Num(f64::NAN), Value::Num(f64::NAN), false),
        ] {
            for (a, b) in [(left.clone(), right.clone()), (right, left)] {
                let mut eq_stack =
                    crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
                        a.clone(),
                        b.clone(),
                    ]);
                block_on(equal(
                    &mut eq_stack,
                    unreachable_call_method,
                    unreachable_call_builtin,
                    unreachable_logical_truth,
                ))
                .unwrap();
                assert_eq!(eq_stack.as_slice(), &[Value::Bool(expected)]);
                let mut ne_stack =
                    crate::interpreter::stack::InterpreterStack::from_unbounded(vec![a, b]);
                block_on(not_equal(
                    &mut ne_stack,
                    unreachable_call_method,
                    unreachable_call_builtin,
                    unreachable_logical_truth,
                ))
                .unwrap();
                assert_eq!(ne_stack.as_slice(), &[Value::Bool(!expected)]);
            }
        }
    }
}

#[cfg(test)]
#[path = "comparison/replay_tests.rs"]
mod replay_tests;
