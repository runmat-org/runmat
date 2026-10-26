//! Same residual/terminal control, now through actual feasible working sets.
use super::*;
fn controls(extra: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    options(
        [
            ("Algorithm", Value::from("active-set")),
            ("Display", Value::from("off")),
        ]
        .into_iter()
        .chain(extra),
    )
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
    assert_eq!(values(field(selected.value(), "MaxIter")), vec![2000.]);
    assert_eq!(values(field(selected.value(), "TolX")), vec![1e-8]);
    let result = run_in(
        &runtime,
        vec![
            matrix(vec![1., 0., 1., 0., 1., 1.], 3, 2),
            matrix(vec![2., 3., 5.], 3, 1),
            empty(),
            empty(),
            empty(),
            empty(),
            matrix(vec![0., 0.], 2, 1),
            matrix(vec![1., 10.], 2, 1),
            matrix(vec![-100., 0.], 2, 1),
            selected.value().clone(),
        ],
    );
    assert_eq!(values(&result[3]), vec![1.]);
    let x = values(&result[0]);
    assert!((x[0] - 1.).abs() < 1e-6 && (x[1] - 3.5).abs() < 1e-6);
    assert!((values(&result[1])[0] - 1.5).abs() < 1e-6);
    assert!((values(field(&result[5], "upper"))[0] - 1.5).abs() < 1e-6);
    assert!(matches!(&result[4],Value::Struct(v)
        if !v.fields.contains_key("dualitygap") && !v.fields.contains_key("linearsolver")
        && !v.fields.contains_key("cgiterations")));
    // Original redundant equalities and an infeasible x0 use the same phase budget.
    let equality = run_in(
        &runtime,
        vec![
            matrix(vec![1., 0., 0., 1.], 2, 2),
            matrix(vec![2., 3.], 2, 1),
            empty(),
            empty(),
            matrix(vec![1., 2., 1., 2.], 2, 2),
            matrix(vec![1., 2.], 2, 1),
            empty(),
            empty(),
            matrix(vec![0., 0.], 2, 1),
            controls([]),
        ],
    );
    assert_eq!(values(&equality[3]), vec![1.]);
    let x = values(&equality[0]);
    assert!(x[0].abs() < 1e-6 && (x[1] - 1.).abs() < 1e-6);
    let infeasible = run_in(
        &runtime,
        vec![
            Value::Num(1.),
            Value::Num(2.),
            matrix(vec![1., -1.], 2, 1),
            matrix(vec![0., -1.], 2, 1),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.),
            controls([]),
        ],
    );
    assert_eq!(values(&infeasible[3]), vec![-2.]);
    assert!(values(&infeasible[0]).is_empty());
    let stopped = run_in(
        &runtime,
        vec![
            Value::Num(1.),
            Value::Num(0.),
            Value::Num(1.),
            Value::Num(0.),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(5.),
            controls([("MaxIterations", Value::Num(0.))]),
        ],
    );
    assert_eq!(values(&stopped[3]), vec![0.]);
    assert_eq!(values(&stopped[0]), vec![5.]);
    assert_eq!(values(&stopped[1]), vec![25.]);
    assert!(values(&stopped[5]).is_empty());
    assert_eq!(values(field(&stopped[4], "constrviolation")), vec![5.]);
    let threshold = run_in(
        &runtime,
        vec![
            Value::Num(1.),
            Value::Num(2.),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(2.),
            controls([("ObjectiveLimit", Value::Num(1.))]),
        ],
    );
    assert_eq!(values(&threshold[3]), vec![-3.]);
    assert_eq!(values(&threshold[0]), vec![2.]);
    assert!(values(&threshold[5]).is_empty());
    let nonbinary = run_in(
        &runtime,
        vec![
            Value::Num(3.),
            Value::Num(1.),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.),
            controls([]),
        ],
    );
    assert_eq!(values(&nonbinary[3]), vec![1.]);
    assert!((values(&nonbinary[0])[0] - 1. / 3.).abs() < 1e-12);
    let tiny = run_in(
        &runtime,
        vec![
            Value::Num(1e-20),
            Value::Num(1e-20),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.),
            controls([]),
        ],
    );
    // A rank/difficulty refusal is valid; unit-floor success at x=0 is not.
    assert!(values(&tiny[3]) != vec![1.] || (values(&tiny[0])[0] - 1.).abs() < 1e-6);
    // A rank-deficient, underdetermined residual target needs no normal
    // equations or unchecked full-column-rank assumption.
    let under = run_in(
        &runtime,
        vec![
            matrix(vec![1., 1.], 1, 2),
            Value::Num(1.),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            matrix(vec![0., 0.], 2, 1),
            controls([]),
        ],
    );
    assert_eq!(values(&under[3]), vec![1.]);
    let x = values(&under[0]);
    assert!((x[0] + x[1] - 1.).abs() < 1e-7);
    assert!(values(&under[2])[0].abs() < 1e-7);
    assert!(values(field(&under[5], "lower"))
        .iter()
        .all(|v| v.abs() < 1e-7));
    assert!(values(field(&under[5], "upper"))
        .iter()
        .all(|v| v.abs() < 1e-7));
    let mut c = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, Some(&runtime));
    let args = vec![
        Value::SparseTensor(
            runmat_value::SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![1.]).unwrap(),
        ),
        Value::Num(1.),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        Value::Num(0.),
        controls([]),
    ];
    assert!(block_on(super::super::parse::problem(
        crate::builtins::common::test_support::builtin_arguments(args),
        &mut c
    ))
    .is_err());
}
