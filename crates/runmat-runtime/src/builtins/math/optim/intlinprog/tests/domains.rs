use super::*;
use crate::builtins::math::optim::integer_constraint::schema::{Property, CLASS};
fn descriptor(properties: impl IntoIterator<Item = (Property, Value)>) -> Value {
    let mut object = runmat_value::UnboundedObjectBuilder::new(CLASS.display_name());
    for key in Property::ALL {
        object.insert(key.name().into(), empty());
    }
    for (key, value) in properties {
        object.insert(key.name().into(), value);
    }
    Value::Object(object.finish())
}
fn one(property: Property, l: f64, u: f64, a: Value, b: Value) -> Vec<Value> {
    vec![
        Value::Num(1.0),
        descriptor([(property, Value::Num(1.0))]),
        a,
        b,
        empty(),
        empty(),
        Value::Num(l),
        Value::Num(u),
        empty(),
        options([]),
    ]
}
#[test]
fn semi_zero_alternative_gap_branch_and_integer_rounding_are_real_domains() {
    let runtime = context();
    let zero = run(
        &runtime,
        one(Property::SemiContinuous, 2.0, 5.0, empty(), empty()),
    );
    assert_eq!(values(&zero[0]), vec![0.0]);
    assert_eq!(values(&zero[2]), vec![1.0]);
    assert!(!values(field(&zero[3], "numnodes")).is_empty());
    let active = run(
        &runtime,
        one(
            Property::SemiContinuous,
            2.5,
            5.0,
            Value::Num(-1.0),
            Value::Num(-0.5),
        ),
    );
    let x = values(&active[0]);
    assert_eq!(x.len(), 1);
    assert!(x[0].is_finite() && (2.5..=5.0).contains(&x[0]));
    assert_eq!(
        values(&active[1]),
        x,
        "original f=1 objective is consistent"
    );
    // The active continuous optimum is 2.5. Use an explicit strong objective
    // gap rather than claiming exact binary64 equality for a continuous solve.
    assert!(x[0] - 2.5 <= 1e-6);
    assert_eq!(values(&active[2]), vec![1.0]);
    let integer = run(
        &runtime,
        one(
            Property::SemiInteger,
            1.25,
            4.75,
            Value::Num(-1.0),
            Value::Num(-1.5),
        ),
    );
    assert_eq!(values(&integer[0]), vec![2.0]);
    assert_eq!(values(&integer[2]), vec![1.0]);
}
#[test]
fn empty_active_domain_preserves_zero_and_gap_equality_cannot_prove_unbounded_milp() {
    let runtime = context();
    let zero = run(
        &runtime,
        one(Property::SemiInteger, 2.1, 2.9, empty(), empty()),
    );
    assert_eq!(values(&zero[0]), vec![0.0]);
    let args = vec![
        matrix(vec![0.0, -1.0], 2, 1),
        descriptor([(Property::SemiContinuous, Value::Num(1.0))]),
        empty(),
        empty(),
        matrix(vec![1.0, 0.0], 1, 2),
        Value::Num(0.5),
        matrix(vec![2.0, f64::NEG_INFINITY], 2, 1),
        matrix(vec![5.0, f64::INFINITY], 2, 1),
        empty(),
        options([]),
    ];
    let infeasible = run(&runtime, args);
    assert_eq!(values(&infeasible[2]), vec![-2.0]);
    assert!(values(&infeasible[0]).is_empty());
}
#[test]
fn mixed_descriptor_properties_and_invalid_bounds_are_checked_against_problem() {
    let runtime = context();
    let intcon = descriptor([
        (Property::Integer, Value::Num(1.0)),
        (Property::SemiContinuous, Value::Num(2.0)),
        (Property::SemiInteger, Value::Num(3.0)),
    ]);
    let result = run(
        &runtime,
        vec![
            matrix(vec![1.0; 3], 3, 1),
            intcon,
            matrix(vec![0.0, 0.0, -1.0, 0.0, 0.0, -1.0], 2, 3),
            matrix(vec![-0.5, -2.1], 2, 1),
            empty(),
            empty(),
            matrix(vec![1.0, 2.5, 1.2], 3, 1),
            matrix(vec![4.0, 5.0, 4.0], 3, 1),
            empty(),
            options([]),
        ],
    );
    assert_eq!(values(&result[0]), vec![1.0, 2.5, 3.0]);
    let error = block_on(execute(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(one(
            Property::SemiContinuous,
            0.0,
            5.0,
            empty(),
            empty(),
        )),
    ))
    .unwrap_err();
    assert!(error.to_string().contains("strictly positive"));
}
