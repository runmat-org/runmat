//! Public dispatch must use one runtime's registry and invocation state.
use futures::executor::block_on;
use runmat_runtime::{call_builtin_async_with_outputs, call_feval_async_with_outputs};
use runmat_value::Value;
use std::sync::{Arc, Mutex};

#[test]
fn memoized_public_feval_validates_count_and_never_replays_or_caches_shortage() {
    for name in ["memoize", "MemoizedFunction.subsref"] {
        let canonical = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name).len();
        let legacy = runmat_builtins::builtin_functions()
            .into_iter()
            .filter(|binding| binding.name == name)
            .count();
        assert_eq!(
            canonical + legacy,
            1,
            "single-library binding precondition: {name}"
        );
    }
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let calls = Arc::new(Mutex::new(0));
    let observed = calls.clone();
    let _invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |_, args, requested| {
            let observed = observed.clone();
            let first = args.first().cloned().unwrap();
            Box::pin(async move {
                *observed.lock().unwrap() += 1;
                if requested == 2 {
                    runmat_runtime::sequence::ValueSequence::comma_separated(vec![
                        first,
                        Value::Num(4.0),
                    ])
                    .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
                } else {
                    runmat_runtime::sequence::single_value_sequence(first)
                }
            })
        }),
    ));
    let memoized = block_on(call_builtin_async_with_outputs(
        "memoize",
        &[Value::BoundFunctionHandle {
            name: "fixture".into(),
            function: 42,
        }],
        1,
    ))
    .unwrap();
    let _caller = runmat_runtime::output_count::push_output_count(Some(7));
    let invalid = block_on(call_feval_async_with_outputs(
        memoized.clone(),
        &[Value::Num(2.0)],
        runmat_value::MAX_VALUE_SEQUENCE_OUTPUTS + 1,
    ))
    .unwrap_err();
    assert_eq!(
        invalid.identifier(),
        Some("RunMat:OutputSequenceLimitExceeded")
    );
    assert_eq!(*calls.lock().unwrap(), 0);
    for expected in 1..=2 {
        let error = block_on(call_feval_async_with_outputs(
            memoized.clone(),
            &[Value::Num(2.0)],
            3,
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:CommaSeparatedListOutputShortage"),
            "{error:?}"
        );
        assert_eq!(*calls.lock().unwrap(), expected);
        assert_eq!(
            runmat_runtime::output_count::current_output_count(),
            Some(7)
        );
        let Value::HandleObject(handle) = &memoized else {
            panic!("memoized handle")
        };
        runmat_gc::gc_with_value(&handle.target, |target| {
            let Value::Object(object) = target else {
                panic!("object")
            };
            let Value::Cell(cache) = &object.properties["__runmat_memoize_cache"] else {
                panic!("cache")
            };
            assert!(
                cache.data.is_empty(),
                "shortage cannot publish a cache entry"
            );
        })
        .unwrap();
    }
    for _ in 0..2 {
        for (count, expected) in [
            (0, Value::OutputList(vec![])),
            (1, Value::Num(2.0)),
            (2, Value::OutputList(vec![Value::Num(2.0), Value::Num(4.0)])),
        ] {
            assert_eq!(
                block_on(call_feval_async_with_outputs(
                    memoized.clone(),
                    &[Value::Num(2.0)],
                    count
                ))
                .unwrap(),
                expected
            );
            assert_eq!(
                runmat_runtime::output_count::current_output_count(),
                Some(7)
            );
        }
        assert_eq!(*calls.lock().unwrap(), 5);
    }
    assert_eq!(
        block_on(call_feval_async_with_outputs(
            memoized,
            &[Value::String(":".into())],
            1
        ))
        .unwrap(),
        Value::String(":".into())
    );
    assert_eq!(*calls.lock().unwrap(), 6);
    assert_eq!(
        runmat_runtime::output_count::current_output_count(),
        Some(7)
    );
}
