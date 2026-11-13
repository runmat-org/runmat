use super::*;
use futures::executor::block_on;
use std::cell::Cell;
use std::future::ready;
use std::rc::Rc;

fn object() -> Value {
    Value::Object(runmat_value::ObjectInstance::new("ComparisonReplayFixture"))
}

#[test]
fn selected_comparison_error_is_not_replayed_through_builtin() {
    let method_calls = Rc::new(Cell::new(0));
    let builtin_calls = Rc::new(Cell::new(0));
    let mut stack = crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
        object(),
        Value::Num(2.0),
    ]);
    let error = block_on(equal(
        &mut stack,
        {
            let method_calls = Rc::clone(&method_calls);
            move |_, _, _| {
                method_calls.set(method_calls.get() + 1);
                ready(Err(crate::interpreter::errors::mex(
                    "ComparisonBodyFailed",
                    "selected comparison failed after invocation",
                )))
            }
        },
        {
            let builtin_calls = Rc::clone(&builtin_calls);
            move |_, _, _| {
                builtin_calls.set(builtin_calls.get() + 1);
                ready(
                    RootedValue::new(Value::Bool(false))
                        .map_err(|error| error.into_runtime_error()),
                )
            }
        },
        |_, _| ready(Ok(false)),
    ))
    .expect_err("selected callee error must propagate");

    assert_eq!(error.identifier(), Some("RunMat:ComparisonBodyFailed"));
    assert_eq!(method_calls.get(), 1);
    assert_eq!(builtin_calls.get(), 0);
}

#[test]
fn certified_missing_comparison_uses_builtin_once() {
    let builtin_calls = Rc::new(Cell::new(0));
    let mut stack = crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
        object(),
        Value::Num(2.0),
    ]);
    block_on(equal(
        &mut stack,
        |_, _, _| ready(Ok(None)),
        {
            let builtin_calls = Rc::clone(&builtin_calls);
            move |_, _, _| {
                builtin_calls.set(builtin_calls.get() + 1);
                ready(
                    RootedValue::new(Value::Bool(false))
                        .map_err(|error| error.into_runtime_error()),
                )
            }
        },
        |_, _| ready(Ok(false)),
    ))
    .expect("certified absence permits builtin fallback");

    assert_eq!(builtin_calls.get(), 1);
    assert_eq!(stack.as_slice(), &[Value::Bool(false)]);
}
