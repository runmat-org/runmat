//! The existing residual control also exercises the reflective numerical path.
use super::*;
pub(super) fn check() {
    let original = crate::user_functions::test_support::exact_fixture_runtime();
    let selected = block_on(crate::call_builtin_async_in_context(
        &original,
        "optimoptions",
        &[
            Value::from("lsqlin"),
            Value::from("Algorithm"),
            Value::from("trust-region-reflective"),
            Value::from("Display"),
            Value::from("off"),
        ],
    ))
    .unwrap();
    assert_eq!(
        values(field(selected.value(), "OptimalityTolerance")),
        vec![100. * f64::EPSILON]
    );
    assert_eq!(
        values(field(selected.value(), "FunctionTolerance")),
        vec![100. * f64::EPSILON]
    );
    let options = |subproblem: &str, bandwidth: f64| {
        super::options([
            ("Algorithm", Value::from("trust-region-reflective")),
            ("Display", Value::from("off")),
            ("OptimalityTolerance", Value::Num(1e-8)),
            ("FunctionTolerance", Value::Num(0.)),
            (
                "MaxPCGIter",
                if subproblem == "factorization" || bandwidth == f64::INFINITY {
                    Value::Cell(runmat_value::CellArray::new(vec![], 1, 0).unwrap())
                } else {
                    Value::Num(2.)
                },
            ),
            ("SubproblemAlgorithm", Value::from(subproblem)),
            ("PrecondBandWidth", Value::Num(bandwidth)),
        ])
    };
    for matrix in [
        matrix(vec![1., 0., 1., 0., 1., 1.], 3, 2),
        Value::SparseTensor(
            runmat_value::SparseTensor::new(
                3,
                2,
                vec![0, 2, 4],
                vec![0, 2, 1, 2],
                vec![1., 1., 1., 1.],
            )
            .unwrap(),
        ),
    ] {
        for (subproblem, bandwidth) in [
            ("cg", 0.),
            ("cg", 1.),
            ("factorization", 0.),
            ("cg", f64::INFINITY),
        ] {
            let result = run_in(
                &original,
                vec![
                    matrix.clone(),
                    super::matrix(vec![2., 3., 5.], 3, 1),
                    empty(),
                    empty(),
                    empty(),
                    empty(),
                    super::matrix(vec![0., 0.], 2, 1),
                    super::matrix(vec![1., 10.], 2, 1),
                    empty(),
                    options(subproblem, bandwidth),
                ],
            );
            assert_eq!(
                values(&result[3]),
                vec![1.],
                "subproblem={subproblem}, bandwidth={bandwidth}, information={:?}",
                result[4]
            );
            let x = values(&result[0]);
            assert!((x[0] - 1.).abs() < 1e-6 && (x[1] - 3.5).abs() < 1e-6);
            assert!((values(&result[1])[0] - 1.5).abs() < 1e-6);
            assert_eq!(
                field(&result[4], "algorithm"),
                &Value::from("trust-region-reflective")
            );
            let count = values(field(&result[4], "cgiterations"))[0];
            if subproblem == "cg" && bandwidth.is_finite() {
                assert!(count > 0.);
            } else {
                assert_eq!(count, 0.);
            }
            assert!(
                matches!(&result[4],Value::Struct(value)if !value.fields.contains_key("linearsolver")&&!value.fields.contains_key("constrviolation"))
            );
            assert!((values(field(&result[5], "upper"))[0] - 1.5).abs() < 1e-6);
        }
    }
    let stopped = run_in(
        &original,
        vec![
            Value::Num(1.),
            Value::Num(2.),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.),
            Value::Num(1.),
            Value::Num(100.),
            super::options([
                ("Algorithm", Value::from("trust-region-reflective")),
                ("Display", Value::from("off")),
                ("MaxIterations", Value::Num(0.)),
            ]),
        ],
    );
    assert_eq!(values(&stopped[3]), vec![0.]);
    assert_eq!(values(&stopped[0]), vec![0.5]);
    assert!(values(&stopped[5]).is_empty());
    let fallback = run_in(
        &original,
        vec![
            Value::Num(1.),
            Value::Num(2.),
            Value::Num(1.),
            Value::Num(1.),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            super::options([
                ("Algorithm", Value::from("trust-region-reflective")),
                ("Display", Value::from("off")),
                (
                    "SubproblemAlgorithm",
                    Value::Cell(runmat_value::CellArray::new(vec![], 1, 0).unwrap()),
                ),
            ]),
        ],
    );
    assert_eq!(values(&fallback[3]), vec![1.]);
    assert_eq!(
        field(&fallback[4], "algorithm"),
        &Value::from("interior-point")
    );
    let mut c = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, Some(&original));
    for (name, value) in [
        ("SubproblemAlgorithm", Value::from("invalid")),
        ("PrecondBandWidth", Value::Num(-1.)),
        ("PrecondBandWidth", Value::Num(0.5)),
    ] {
        assert!(block_on(super::super::options::Controls::parse(
            super::options([
                ("Algorithm", Value::from("trust-region-reflective")),
                (name, value)
            ]),
            false,
            &mut c
        ))
        .is_err());
    }
    super::super::reflective::check_fold();
}
