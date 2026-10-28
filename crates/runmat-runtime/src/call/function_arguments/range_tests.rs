use super::*;
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use runmat_value::{Closure, HandleRef};
use std::rc::Rc;

fn captured(supplied: Vec<Value>) -> RootedFunctionArguments {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let callable = RetainedValueReadLease::prepare_owned(
        Value::Closure(Closure::new(
            "range_fixture".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            None,
            vec![Value::Num(1.), Value::Num(2.)],
            None,
        )),
        &runtime,
    )
    .unwrap();
    RootedFunctionArguments::captured(callable, RootedValues::new(supplied).unwrap()).unwrap()
}

#[test]
fn method_receivers_are_function_inputs_not_lexical_slots() {
    for (role, lexical_count, input_count) in [
        (runmat_types::FunctionSourceRole::Anonymous, 1, 1),
        (runmat_types::FunctionSourceRole::Method, 0, 2),
    ] {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let callable = RetainedValueReadLease::prepare_owned(
            Value::Closure(Closure::new(
                "input_owner_fixture".into(),
                None,
                role,
                None,
                vec![Value::Num(7.)],
                None,
            )),
            &runtime,
        )
        .unwrap();
        let args = RootedFunctionArguments::captured(
            callable,
            RootedValues::new(vec![Value::Num(11.)]).unwrap(),
        )
        .unwrap();
        assert_eq!(args.lexical_capture_count(), lexical_count);
        let original = args.get(lexical_count).unwrap() as *const Value;
        assert_eq!(args.function_inputs().len(), input_count);
        assert_eq!(
            args.function_inputs().get(0).unwrap() as *const Value,
            original
        );
        let inputs = args.into_function_inputs();
        assert_eq!(inputs.len(), input_count);
        assert_eq!(inputs.get(0).unwrap() as *const Value, original);
    }
}

#[test]
fn borrowed_and_owning_ranges_keep_original_segments_and_exact_iterators() {
    let args = captured(vec![Value::Num(3.), Value::Num(4.)]);
    let capture = args.get(1).unwrap() as *const Value;
    let supplied = args.get(2).unwrap() as *const Value;
    let ranged = BuiltinArguments::new(args).into_range(1..=2).unwrap();
    assert_eq!(ranged.len(), 2);
    assert_eq!(ranged.get(0).unwrap() as *const Value, capture);
    assert_eq!(ranged.get(1).unwrap() as *const Value, supplied);
    let alias = ranged.fork();
    assert_eq!(alias.get(0).unwrap() as *const Value, capture);
    let view = alias.view();
    let mut iter = view.iter();
    assert_eq!(iter.len(), 2);
    assert_eq!(iter.next_back(), Some(&Value::Num(3.)));
    assert_eq!(iter.next(), Some(&Value::Num(2.)));
    assert_eq!(iter.next(), None);
    assert_eq!(view.chunks_exact(1).len(), 2);
    let (first, tail) = view.split_first().unwrap();
    assert_eq!(first as *const Value, capture);
    assert_eq!(tail.first().unwrap() as *const Value, supplied);
    let (last, prefix) = view.split_last().unwrap();
    assert_eq!(last as *const Value, supplied);
    assert_eq!(prefix.first().unwrap() as *const Value, capture);
    assert!(prefix.range(..0).unwrap().split_first().is_none());
    assert!(tail.range(..0).unwrap().split_last().is_none());
    assert!(view.range(3..).is_err());
    assert!(view.range(..=usize::MAX).is_err());
    let mut projected = ranged.into_iter();
    assert_eq!(projected.next(), Some(Value::Num(2.)));
    assert_eq!(projected.next_back(), Some(Value::Num(3.)));
    assert_eq!(projected.len(), 0);
    let transferred = alias.into_function_arguments();
    assert_eq!(transferred.capture_count(), 1);
    assert_eq!(transferred.supplied_count(), 1);
    assert!(std::ptr::eq(
        transferred.capture_lease(0).unwrap().value(),
        transferred.get(0).unwrap()
    ));
}
#[test]
fn routed_range_edits_do_not_revive_hidden_arguments() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let args = captured(vec![Value::Num(3.), Value::Num(4.)])
        .into_range(1..3)
        .unwrap()
        .prepend_routing_in_context(Value::Num(0.), &runtime)
        .unwrap();
    assert_eq!(
        args.iter().cloned().collect::<Vec<_>>(),
        vec![Value::Num(0.), Value::Num(2.), Value::Num(3.)]
    );
    let args = args.remove_first_routing_in_context(&runtime).unwrap();
    let mut args = args.try_into_values().unwrap();
    assert_eq!(args.len(), 2);
    assert_eq!(args.next().unwrap().value(), &Value::Num(2.));
    assert_eq!(args.next().unwrap().value(), &Value::Num(3.));
    assert!(args.next().is_none());
}
#[test]
fn partial_builtin_iteration_retains_original_capture_roots() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let root = runmat_gc::gc_allocate_rooted(Value::Num(17.)).unwrap();
        let handle = root.handle();
        let callable = RetainedValueReadLease::prepare_owned(
            Value::Closure(Closure::new(
                "root_fixture".into(),
                None,
                runmat_types::FunctionSourceRole::Anonymous,
                None,
                vec![Value::HandleObject(HandleRef {
                    class_name: "RootFixture".into(),
                    target: handle,
                    valid: true,
                })],
                None,
            )),
            &runtime,
        )
        .unwrap();
        let args = RootedFunctionArguments::captured(
            callable,
            RootedValues::new(vec![Value::Num(1.)]).unwrap(),
        )
        .unwrap();
        root.unroot().unwrap();
        let mut iter = BuiltinArguments::new(args)
            .into_range(..1)
            .unwrap()
            .into_iter();
        let value = iter.next().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        drop(value);
        drop(iter);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[test]
fn forked_supplied_iteration_keeps_original_storage_and_root_forest() {
    runmat_gc::gc_test_context(|| {
        let initial = runmat_gc::gc_allocate_rooted(Value::Num(23.)).unwrap();
        let handle = initial.handle();
        let original = RootedValues::new(vec![
            Value::Num(1.),
            Value::HandleObject(HandleRef {
                class_name: "SuppliedRootFixture".into(),
                target: handle,
                valid: true,
            }),
        ])
        .unwrap();
        let alias = original.fork();
        assert!(std::ptr::eq(&original[0], &alias[0]));
        initial.unroot().unwrap();
        let mut values = RootedFunctionArguments::ordinary(alias)
            .into_range(1..)
            .unwrap()
            .try_into_values()
            .unwrap();
        drop(original);
        let projected = values.next().unwrap();
        assert_eq!(values.len(), 0);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        drop(projected);
        drop(values);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
