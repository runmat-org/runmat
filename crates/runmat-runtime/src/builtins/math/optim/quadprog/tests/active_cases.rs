//! Existing public control extended with actual active geometry and warm forms.
use super::*;
fn active(extra: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    options(
        [
            ("Algorithm", Value::from("active-set")),
            ("Display", Value::from("off")),
        ]
        .into_iter()
        .chain(extra),
    )
}
fn arguments([h, f, aeq, beq, lower, upper, x0, opts]: [Value; 8]) -> Vec<Value> {
    vec![h, f, empty(), empty(), aeq, beq, lower, upper, x0, opts]
}
fn warm_x(value: &Value) -> Vec<f64> {
    let Value::Object(object) = value else {
        panic!("actual warm output");
    };
    assert!(object
        .class_name()
        .is(runmat_builtins::QUADPROG_WARM_START_CLASS));
    values(
        object
            .active_set_warm_start_payload()
            .expect("immutable math data")
            .x(),
    )
}
fn public(runtime: &crate::context::RuntimeContext, args: &[Value]) -> Vec<Value> {
    let result = block_on(crate::call_builtin_async_with_outputs_in_context(
        runtime, "quadprog", args, 5,
    ))
    .unwrap();
    let Value::OutputList(values) = result.value() else {
        panic!("five public outputs");
    };
    values.to_vec()
}
pub(super) fn check() {
    // H is indefinite off the equality affine space, but restricted curvature
    // is PSD. Dependent rows are allowed, inconsistent RHS is not.
    let h = matrix(vec![-1., 0., 0., 2.], 2, 2);
    for (aeq, beq, expected) in [
        (matrix(vec![1., 0.], 1, 2), Value::Num(1.), 1.),
        (
            matrix(vec![1., 2., 0., 0.], 2, 2),
            matrix(vec![1., 2.], 2, 1),
            1.,
        ),
        (
            matrix(vec![1., 2., 0., 0.], 2, 2),
            matrix(vec![1., 3.], 2, 1),
            -2.,
        ),
    ] {
        let result = run(arguments([
            h.clone(),
            matrix(vec![0., -4.], 2, 1),
            aeq,
            beq,
            empty(),
            empty(),
            matrix(vec![1., 0.], 2, 1),
            active([]),
        ]));
        assert_eq!(values(&result[2]), vec![expected]);
        if expected == 1. {
            let x = values(&result[0]);
            assert!((x[0] - 1.).abs() < 1e-7 && (x[1] - 2.).abs() < 1e-7);
            assert!((values(&result[1])[0] + 4.5).abs() < 1e-7);
            assert_eq!(field(&result[3], "algorithm"), &Value::from("active-set"));
        }
    }
    let flat = run(arguments([
        matrix(vec![0., 1., 1., 0.], 2, 2),
        matrix(vec![0., 0.], 2, 1),
        matrix(vec![1., 0.], 1, 2),
        Value::Num(1.),
        empty(),
        empty(),
        matrix(vec![1., 0.], 2, 1),
        active([]),
    ]));
    assert_eq!(values(&flat[2]), vec![-3.]);
    assert!(values(&flat[0]).is_empty());
    // A recession proposal is blocked four units away, not capped at one.
    let blocked = run(arguments([
        Value::Num(0.),
        Value::Num(-1.),
        empty(),
        empty(),
        empty(),
        Value::Num(4.),
        Value::Num(0.),
        active([]),
    ]));
    assert_eq!(values(&blocked[2]), vec![1.]);
    assert!((values(&blocked[0])[0] - 4.).abs() < 1e-7);
    let tiny = run(arguments([
        Value::Num(f64::from_bits(1)),
        Value::Num(-1.),
        empty(),
        empty(),
        empty(),
        empty(),
        Value::Num(0.),
        active([]),
    ]));
    assert_ne!(
        values(&tiny[2]),
        vec![-3.],
        "nonzero original curvature is not a flat ray"
    );
    let threshold = run(arguments([
        Value::Num(2.),
        Value::Num(-4.),
        empty(),
        empty(),
        empty(),
        empty(),
        Value::Num(0.),
        active([("ObjectiveLimit", Value::Num(-1.))]),
    ]));
    assert_eq!(values(&threshold[2]), vec![-3.]);
    assert!((values(&threshold[1])[0] + 4.).abs() < 1e-7);
    assert!(
        matches!(field(&threshold[3],"message"),Value::String(runmat_value::StringElement::Text(s)) if s.as_str().contains("threshold"))
    );
    let timed = run(arguments([
        Value::Num(2.),
        Value::Num(-4.),
        empty(),
        empty(),
        empty(),
        empty(),
        Value::Num(0.),
        active([("MaxTime", Value::Num(0.))]),
    ]));
    assert_eq!(values(&timed[2]), vec![0.]);
    assert_eq!(values(field(&timed[3], "iterations")), vec![0.]);

    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let opts = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "optimoptions",
        &[
            Value::from("quadprog"),
            Value::from("Algorithm"),
            Value::from("active-set"),
            Value::from("Display"),
            Value::from("off"),
        ],
    ))
    .unwrap();
    let Value::Struct(fields) = opts.value() else {
        panic!("typed options");
    };
    assert!(matches!(fields.fields.get("MaxIter"),Some(Value::Tensor(t)) if t.is_empty()));
    let initial = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "optimwarmstart",
        &[Value::Num(4.), opts.value().clone()],
    ))
    .unwrap();
    let alias = initial.value().clone();
    let first = public(
        &runtime,
        &[
            Value::Num(1.),
            Value::Num(1.),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.),
            empty(),
            initial.value().clone(),
        ],
    );
    assert_eq!(values(&first[2]), vec![1.]);
    assert!(warm_x(&first[0])[0].abs() < 1e-7);
    let path = options([("type", Value::from(".")), ("subs", Value::from("X"))]);
    let read = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "QuadprogWarmStart.subsref",
        &[alias.clone(), path.clone()],
    ))
    .unwrap();
    assert_eq!(values(read.value()), vec![4.]);
    let error = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "QuadprogWarmStart.subsasgn",
        &[alias.clone(), path, Value::Num(99.)],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:PropertyReadOnly"));
    let Value::Object(object) = &first[0] else {
        panic!("native warm");
    };
    assert!(object
        .active_set_warm_start_payload()
        .unwrap()
        .working()
        .unwrap()
        .active()
        .iter()
        .any(|&v| v));
    // Same row state, changed objective: release the lower-bound working row.
    let second = public(
        &runtime,
        &[
            Value::Num(1.),
            Value::Num(-2.),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.),
            empty(),
            first[0].clone(),
        ],
    );
    assert_eq!(values(&second[2]), vec![1.]);
    assert!((warm_x(&second[0])[0] - 2.).abs() < 1e-7);
    assert_eq!(warm_x(&alias), vec![4.]);
    assert!(warm_x(&first[0])[0].abs() < 1e-7);
}
