//! The same public residual control exercises actual immutable working-set reuse.
use super::*;
fn x(value: &Value) -> Vec<f64> {
    let Value::Object(object) = value else {
        panic!("warm result")
    };
    values(object.active_set_warm_start_payload().unwrap().x())
}
fn args(ws: Value, d: f64, lower: Value, aeq: Value, beq: Value) -> Vec<Value> {
    vec![
        Value::Num(1.),
        Value::Num(d),
        empty(),
        empty(),
        aeq,
        beq,
        lower,
        empty(),
        ws,
    ]
}
pub(super) fn check() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let selected = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "optimoptions",
        &[
            Value::from("lsqlin"),
            Value::from("Algorithm"),
            Value::from("active-set"),
            Value::from("Display"),
            Value::from("off"),
        ],
    ))
    .unwrap();
    let initial = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "optimwarmstart",
        &[
            matrix(vec![4.], 1, 1),
            selected.value().clone(),
            Value::from("MaxLinearInequalities"),
            Value::Num(8.),
            Value::from("MaxLinearEqualities"),
            Value::Num(4.),
        ],
    ))
    .unwrap();
    let alias = initial.value().clone();
    let first = run_in(
        &runtime,
        args(
            initial.value().clone(),
            -1.,
            Value::Num(0.),
            empty(),
            empty(),
        ),
    );
    assert_eq!(values(&first[3]), vec![1.]);
    assert!(x(&first[0])[0].abs() < 1e-7);
    let Value::Object(object) = &first[0] else {
        panic!("warm")
    };
    let state = object.active_set_warm_start_payload().unwrap();
    assert!(state.working().unwrap().active().iter().any(|&v| v));
    assert_eq!(state.limits().equalities, Some(4));
    assert_eq!(state.limits().inequalities, Some(8));
    // A previously active lower bound must be released for the new objective.
    let second = run_in(
        &runtime,
        args(first[0].clone(), 2., Value::Num(0.), empty(), empty()),
    );
    assert_eq!(values(&second[3]), vec![1.]);
    assert!((x(&second[0])[0] - 2.).abs() < 1e-7);
    assert_eq!(x(&alias), vec![4.]);
    assert!(x(&first[0])[0].abs() < 1e-7);
    // Changed bound geometry and a new equality invalidate saved row authority.
    let changed = run_in(
        &runtime,
        args(first[0].clone(), 2., Value::Num(3.), empty(), empty()),
    );
    assert_eq!(values(&changed[3]), vec![1.]);
    assert!((x(&changed[0])[0] - 3.).abs() < 1e-7);
    let equality = run_in(
        &runtime,
        args(
            changed[0].clone(),
            2.,
            empty(),
            Value::Num(1.),
            Value::Num(1.),
        ),
    );
    assert_eq!(values(&equality[3]), vec![1.]);
    assert!((x(&equality[0])[0] - 1.).abs() < 1e-7);
    let mut path = runmat_value::UnboundedStructValueBuilder::new();
    path.insert("type", Value::from("."));
    path.insert("subs", Value::from("X"));
    let path = Value::Struct(path.finish());
    let read = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "LsqlinWarmStart.subsref",
        &[alias.clone(), path.clone()],
    ))
    .unwrap();
    assert_eq!(values(read.value()), vec![4.]);
    let error = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "LsqlinWarmStart.subsasgn",
        &[alias.clone(), path, Value::Num(99.)],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:PropertyReadOnly"));
    // Dimension rejection precedes numerical solving.
    let bad = vec![
        matrix(vec![1., 1.], 1, 2),
        Value::Num(2.),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        alias.clone(),
    ];
    let error = block_on(execute(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(bad),
    ))
    .unwrap_err();
    assert!(error.to_string().contains("same number of variables"));
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    let source = CancellationSource::new();
    let cancelled = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(CancellationReason::User);
    let error = block_on(execute(
        Some(&cancelled),
        crate::builtins::common::test_support::builtin_arguments(args(
            alias,
            1.,
            Value::Num(0.),
            empty(),
            empty(),
        )),
    ))
    .unwrap_err();
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
}
