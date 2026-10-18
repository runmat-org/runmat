use super::*;
use futures::executor::block_on;
use std::cell::Cell;
use std::future::ready;
use std::rc::Rc;

fn object() -> Value {
    Value::Object(runmat_value::ObjectInstance::new("OperatorReplayFixture"))
}

#[test]
fn selected_operator_error_is_not_replayed_through_fallback() {
    let method_calls = Rc::new(Cell::new(0));
    let fallback_calls = Rc::new(Cell::new(0));
    let mut stack = crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
        object(),
        Value::Num(2.0),
    ]);
    let error = block_on(add(
        &mut stack,
        {
            let method_calls = Rc::clone(&method_calls);
            move |_, _, _| {
                method_calls.set(method_calls.get() + 1);
                ready(Err(crate::interpreter::errors::mex(
                    "OperatorBodyFailed",
                    "selected operator failed after invocation",
                )))
            }
        },
        |_, _, _| ready(Ok(None)),
        {
            let fallback_calls = Rc::clone(&fallback_calls);
            move |_, _| {
                fallback_calls.set(fallback_calls.get() + 1);
                ready(RootedValue::new(Value::Num(3.0)).map_err(|error| error.into_runtime_error()))
            }
        },
    ))
    .expect_err("selected callee error must propagate");

    assert_eq!(error.identifier(), Some("RunMat:OperatorBodyFailed"));
    assert_eq!(method_calls.get(), 1);
    assert_eq!(fallback_calls.get(), 0);
}

#[test]
fn certified_missing_operator_uses_fallback_once() {
    let fallback_calls = Rc::new(Cell::new(0));
    let mut stack = crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
        object(),
        Value::Num(2.0),
    ]);
    block_on(add(
        &mut stack,
        |_, _, _| ready(Ok(None)),
        |_, _, _| ready(Ok(None)),
        {
            let fallback_calls = Rc::clone(&fallback_calls);
            move |_, _| {
                fallback_calls.set(fallback_calls.get() + 1);
                ready(RootedValue::new(Value::Num(3.0)).map_err(|error| error.into_runtime_error()))
            }
        },
    ))
    .expect("certified absence permits fallback");

    assert_eq!(fallback_calls.get(), 1);
    assert_eq!(stack.as_slice(), &[Value::Num(3.0)]);
}
