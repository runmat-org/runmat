use super::*;
use crate::builtins::math::optim::common::conic::terminal;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, SparseTensor};

fn column(values: Vec<f64>) -> Value {
    vector_value(values)
}
fn call(f: Vec<f64>, a: Value, b: Vec<f64>, rest: Vec<Value>) -> Vec<Value> {
    let _outputs = crate::output_count::push_output_count(Some(5)).unwrap();
    let Value::OutputList(output) =
        block_on(linprog_builtin(column(f), a, column(b), rest)).unwrap()
    else {
        panic!("five outputs")
    };
    output
        .try_into_owned()
        .unwrap()
        .try_into_unbounded_vec()
        .unwrap()
}
fn values(value: &Value) -> Vec<f64> {
    match value {
        Value::Tensor(t) => t.materialize_f64(),
        Value::Num(v) => vec![*v],
        _ => panic!("real numeric"),
    }
}
fn number(value: &Value) -> f64 {
    values(value)[0]
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(s) = value else {
        panic!("struct")
    };
    &s.fields[name]
}
fn dense(rows: usize, cols: usize, values: Vec<f64>) -> Value {
    Value::Tensor(Tensor::new(values, vec![rows, cols]).unwrap())
}

#[test]
fn public_shadow_prices_keep_original_rows_and_bound_signs() {
    let output = call(
        vec![-3.0, -5.0],
        dense(3, 2, vec![1.0, 0.0, 3.0, 0.0, 2.0, 2.0]),
        vec![4.0, 12.0, 18.0],
        vec![
            empty_double(),
            empty_double(),
            column(vec![0.0, 0.0]),
            empty_double(),
        ],
    );
    assert_eq!(number(&output[2]), 1.0);
    let x = values(&output[0]);
    assert!((x[0] - 2.0).abs() < 1e-6 && (x[1] - 6.0).abs() < 1e-6);
    assert!((number(&output[1]) + 36.0).abs() < 1e-6);
    let lambda = &output[4];
    let z = values(field(lambda, "ineqlin"));
    for (actual, want) in z.iter().zip([0.0, 1.5, 1.0]) {
        assert!((actual - want).abs() < 1e-6);
    }
    let lower = values(field(lambda, "lower"));
    let upper = values(field(lambda, "upper"));
    assert!((-3.0 + z[0] + 3.0 * z[2] - lower[0] + upper[0]).abs() < 1e-6);
    assert!((-5.0 + 2.0 * z[1] + 2.0 * z[2] - lower[1] + upper[1]).abs() < 1e-6);
    assert_eq!(values(field(lambda, "eqlin")).len(), 0);
    assert_eq!(upper, vec![0.0, 0.0]);
    assert!(number(field(&output[3], "dualitygap")) < 1e-6);
}

#[test]
fn redundant_equalities_fixed_bounds_and_small_objective_have_real_duals() {
    let output = call(
        vec![1e-20, -2e-20],
        empty_double(),
        vec![],
        vec![
            dense(2, 2, vec![1.0, 2.0, 1.0, 2.0]),
            column(vec![3.0, 6.0]),
            column(vec![1.0, 0.0]),
            column(vec![1.0, 4.0]),
        ],
    );
    assert_eq!(number(&output[2]), 1.0);
    let x = values(&output[0]);
    assert!((x[0] - 1.0).abs() < 1e-7 && (x[1] - 2.0).abs() < 1e-7);
    assert!((number(&output[1]) + 3e-20).abs() < 1e-27);
    let eq = values(field(&output[4], "eqlin"));
    let lower = values(field(&output[4], "lower"));
    let upper = values(field(&output[4], "upper"));
    for (i, cost) in [1e-20, -2e-20].iter().enumerate() {
        assert!((cost + eq[0] + 2.0 * eq[1] - lower[i] + upper[i]).abs() < 1e-27);
    }
}

#[test]
fn sparse_storage_horizons_have_verified_physical_balance_without_enumeration() {
    for t in [6, 24, 168] {
        // The reported storage model, with a flat positive price. Its exact
        // optimum consumes initial energy once: no profitable lossy cycle.
        let n = 3 * t;
        let mut pointers = vec![0];
        let mut rows = Vec::new();
        let mut coefficients = Vec::new();
        for col in 0..n {
            if col < 2 * t {
                let first = col % t;
                let coefficient = if col < t { -0.92 } else { 1.0 / 0.92 };
                for row in first..t {
                    rows.push(row);
                    coefficients.push(coefficient);
                }
            } else {
                rows.push(col - 2 * t);
                coefficients.push(1.0);
            }
            pointers.push(rows.len());
        }
        let aeq =
            Value::SparseTensor(SparseTensor::new(t, n, pointers, rows, coefficients).unwrap());
        let mut f = vec![1.0; t];
        f.extend(vec![-1.0; t]);
        f.extend(vec![0.0; t]);
        let mut ub = vec![10.0; 2 * t];
        ub.extend(vec![40.0; t]);
        let output = call(
            f,
            empty_double(),
            vec![],
            vec![aeq, column(vec![20.0; t]), column(vec![0.0; n]), column(ub)],
        );
        assert_eq!(number(&output[2]), 1.0, "horizon {t}");
        let x = values(&output[0]);
        let mut energy = 20.0;
        for i in 0..t {
            assert!(x[i] >= -1e-6 && x[i] <= 10.0 + 1e-6);
            assert!(x[t + i] >= -1e-6 && x[t + i] <= 10.0 + 1e-6);
            energy += 0.92 * x[i] - x[t + i] / 0.92;
            assert!((x[2 * t + i] - energy).abs() < 1e-5);
            assert!((-1e-5..=40.0 + 1e-5).contains(&energy));
        }
        assert!((number(&output[1]) + 18.4).abs() < 1e-4);
        assert!(number(field(&output[3], "firstorderopt")) < 1e-6);
        assert!(number(field(&output[3], "dualitygap")) < 1e-4);
        assert!(number(field(&output[3], "iterations")) < 1000.0);
    }
}

#[test]
fn infeasible_unbounded_and_stopped_forms_do_not_fabricate_lambda() {
    let cases = [
        (
            vec![1.0],
            dense(2, 1, vec![1.0, -1.0]),
            vec![0.0, -1.0],
            vec![],
            -2.0,
        ),
        (vec![-1.0], empty_double(), vec![], vec![], -3.0),
        // A descent direction exists in x2, but x1 is inconsistent. A dual
        // infeasibility status alone is not an unboundedness certificate.
        (
            vec![0.0, -1.0],
            dense(2, 2, vec![1.0, -1.0, 0.0, 0.0]),
            vec![0.0, -1.0],
            vec![],
            -2.0,
        ),
        // Feasibility is not the clipped-zero seed, requiring the auxiliary
        // feasibility solve if the backend supplies a descent certificate.
        (
            vec![0.0, -1.0],
            empty_double(),
            vec![],
            vec![dense(1, 2, vec![1.0, 0.0]), column(vec![1.0])],
            -3.0,
        ),
        (
            vec![0.0, -1.0],
            empty_double(),
            vec![],
            vec![dense(1, 2, vec![3.0, 0.0]), column(vec![1.0])],
            -3.0,
        ),
    ];
    for (f, a, b, rest, flag) in cases {
        let output = call(f, a, b, rest);
        assert_eq!(number(&output[2]), flag);
        assert!(values(&output[0]).is_empty() && values(&output[1]).is_empty());
        for name in ["ineqlin", "eqlin", "lower", "upper"] {
            assert!(values(field(&output[4], name)).is_empty());
        }
    }
    for limit in [0.0, 1.0] {
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("MaxIterations", Value::Num(limit));
        let output = call(
            vec![-1.0],
            empty_double(),
            vec![],
            vec![
                empty_double(),
                empty_double(),
                column(vec![0.0]),
                column(vec![1.0]),
                Value::Struct(options.finish()),
            ],
        );
        assert_eq!(number(&output[2]), 0.0);
        assert!(number(field(&output[3], "iterations")) <= limit);
        let x = values(&output[0]);
        if !x.is_empty() {
            assert!(x[0] >= -1e-8 && x[0] <= 1.0 + 1e-8);
            assert!((number(&output[1]) + x[0]).abs() < 1e-12);
        }
        assert!(values(field(&output[4], "lower")).is_empty());
        assert!(number(field(&output[3], "firstorderopt")).is_nan());
    }
}

#[test]
fn sparse_single_and_exact_integer_admission_preserve_classes_and_reject_rounding() {
    let a =
        Value::SparseTensor(SparseTensor::new_f32(1, 1, vec![0, 1], vec![0], vec![2.0]).unwrap());
    let output = call(vec![-1.0], a, vec![4.0], vec![]);
    assert_eq!(number(&output[2]), 1.0);
    assert!((values(&output[0])[0] - 2.0).abs() < 1e-7);
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let a = Value::SparseTensor(
        SparseTensor::new_integer(
            1,
            1,
            vec![0, 1],
            vec![0],
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
        )
        .unwrap(),
    );
    let error = block_on(linprog_builtin(
        column(vec![-1.0]),
        a,
        column(vec![1.0]),
        vec![],
    ))
    .unwrap_err();
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn original_scale_validation_rejects_tiny_constraint_and_objective_false_certificates() {
    let mut cooperation =
        ScalarCooperation::for_builtin(runmat_builtins::LINPROG_CATALOG_ENTRY.identity, None);
    let model = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![-1e-20],
            a_ineq: MatrixInput::new(1, 1, vec![0, 1], vec![0], vec![1e-20]),
            b_ineq: vec![0.0],
            a_eq: MatrixInput::zeros((0, 1)),
            b_eq: vec![],
            lb: vec![f64::NEG_INFINITY],
            ub: vec![f64::INFINITY],
        },
        &mut cooperation,
    ))
    .unwrap();
    assert!(
        block_on(certificates::primal(&model, &[1.0], &mut cooperation))
            .unwrap()
            .unwrap()
            .normalized
            > 0.1
    );
    assert!(!block_on(terminal::descent_ray(
        &certificates::conic_view(&model, &model.cones()),
        &[1.0],
        &mut cooperation
    ))
    .unwrap());
    assert!(
        !block_on(certificates::solution(
            &model,
            &[0.0],
            &[0.0],
            &mut cooperation
        ))
        .unwrap()
        .unwrap()
        .valid
    );
}

#[test]
fn terminal_certificates_require_exact_original_dyadic_relations() {
    let mut cooperation =
        ScalarCooperation::for_builtin(runmat_builtins::LINPROG_CATALOG_ENTRY.identity, None);
    let make = |a, b| LinearProgram {
        f: vec![0.0],
        a_ineq: MatrixInput::new(2, 1, vec![0, 2], vec![0, 1], a),
        b_ineq: b,
        a_eq: MatrixInput::zeros((0, 1)),
        b_eq: vec![],
        lb: vec![f64::NEG_INFINITY],
        ub: vec![f64::INFINITY],
    };
    let feasible = block_on(model::ConicModel::new(
        make(vec![-1e-20, 1.0], vec![-1.0, 1e30]),
        &mut cooperation,
    ))
    .unwrap();
    assert!(
        block_on(certificates::primal(&feasible, &[1e20], &mut cooperation))
            .unwrap()
            .unwrap()
            .normalized
            <= TOL
    );
    let infeasible = block_on(model::ConicModel::new(
        make(vec![-1.0, 1.0], vec![-1.0, 0.0]),
        &mut cooperation,
    ))
    .unwrap();
    let zero_rows = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![0.0],
            a_ineq: MatrixInput::zeros((2, 1)),
            b_ineq: vec![1.0, 10.0 / TOL],
            a_eq: MatrixInput::zeros((0, 1)),
            b_eq: vec![],
            lb: vec![f64::NEG_INFINITY],
            ub: vec![f64::INFINITY],
        },
        &mut cooperation,
    ))
    .unwrap();
    let weighted_negative = block_on(model::ConicModel::new(
        make(vec![1e-9, 1.0], vec![1.0, 1e30]),
        &mut cooperation,
    ))
    .unwrap();
    // Both models have feasible x=0, but an approximately cone-positive
    // candidate obtains negative b'z by amplifying a tiny negative weight.
    for (model, candidate) in [
        (&zero_rows, [1.0, -TOL / 2.0]),
        (&weighted_negative, [1.0, -1e-9]),
    ] {
        assert!(
            block_on(certificates::primal(model, &[0.0], &mut cooperation))
                .unwrap()
                .unwrap()
                .normalized
                <= TOL
        );
        assert!(!block_on(terminal::primal_infeasible(
            &certificates::conic_view(model, &model.cones()),
            &candidate,
            &mut cooperation
        ))
        .unwrap());
    }
    // x1=0 and x1+1e-20*x2<=0 imply x2<=0: min(-x2)=0.
    // A zero d1 must not let the unrelated coefficient of x1 hide d2>0.
    let bounded = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![0.0, -1.0],
            a_ineq: MatrixInput::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 1e-20]),
            b_ineq: vec![0.0],
            a_eq: MatrixInput::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]),
            b_eq: vec![0.0],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        },
        &mut cooperation,
    ))
    .unwrap();
    let unbounded = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![0.0, -1.0],
            a_ineq: MatrixInput::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]),
            b_ineq: vec![0.0],
            a_eq: MatrixInput::zeros((0, 2)),
            b_eq: vec![],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        },
        &mut cooperation,
    ))
    .unwrap();
    let epsilon = 2.0_f64.powi(-30);
    let near_feasible = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![0.0, 0.0],
            a_ineq: MatrixInput::new(
                2,
                2,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![1.0, -1.0 + epsilon, 1.0, -1.0],
            ),
            b_ineq: vec![-1.0, 0.0],
            a_eq: MatrixInput::zeros((0, 2)),
            b_eq: vec![],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        },
        &mut cooperation,
    ))
    .unwrap();
    // This exact feasible witness escapes along the nearly dependent rows.
    // The old weighted tolerance accepted z=[1,1] with a nonzero A'z.
    assert_eq!(1.0 + (-1.0 + epsilon), epsilon);
    assert!(epsilon / (2.0 - epsilon) < TOL);
    assert!(
        block_on(certificates::primal(
            &near_feasible,
            &[-2.0_f64.powi(30), 2.0_f64.powi(30) - 1.0],
            &mut cooperation
        ))
        .unwrap()
        .unwrap()
        .normalized
            <= TOL
    );
    // x1=x2 and x1+(-1+epsilon)*x2<=0 imply x2<=0. Its
    // objective -x2 is bounded below, despite an approximately feasible ray.
    let near_bounded = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![0.0, -1.0],
            a_ineq: MatrixInput::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, -1.0 + epsilon]),
            b_ineq: vec![0.0],
            a_eq: MatrixInput::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, -1.0]),
            b_eq: vec![0.0],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        },
        &mut cooperation,
    ))
    .unwrap();
    let infeasible_base = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![0.0, -1.0],
            a_ineq: MatrixInput::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]),
            b_ineq: vec![-1e-20],
            a_eq: MatrixInput::zeros((0, 2)),
            b_eq: vec![],
            lb: vec![0.0, f64::NEG_INFINITY],
            ub: vec![f64::INFINITY; 2],
        },
        &mut cooperation,
    ))
    .unwrap();
    // An exact ray cannot establish unboundedness from a merely tolerance-
    // feasible base. These bounds and row are actually inconsistent.
    assert!(
        block_on(certificates::primal(
            &infeasible_base,
            &infeasible_base.seed,
            &mut cooperation
        ))
        .unwrap()
        .unwrap()
        .normalized
            <= TOL
    );
    assert!(block_on(terminal::descent_ray(
        &certificates::conic_view(&infeasible_base, &infeasible_base.cones()),
        &[0.0, 1.0],
        &mut cooperation
    ))
    .unwrap());
    assert!(!block_on(terminal::feasible_point(
        &certificates::conic_view(&infeasible_base, &infeasible_base.cones()),
        &infeasible_base.seed,
        &mut cooperation
    ))
    .unwrap());
    assert!(block_on(terminal::feasible_point(
        &certificates::conic_view(&unbounded, &unbounded.cones()),
        &unbounded.seed,
        &mut cooperation
    ))
    .unwrap());
    assert!(block_on(terminal::feasible_point(
        &certificates::conic_view(&near_feasible, &near_feasible.cones()),
        &[-2.0_f64.powi(30), 2.0_f64.powi(30) - 1.0],
        &mut cooperation
    ))
    .unwrap());
    // Continued fractions propose rational witnesses, not rounded outputs.
    // Full exact revalidation recovers an ordinary non-dyadic equality.
    let thirds = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![0.0, -1.0],
            a_ineq: MatrixInput::zeros((0, 2)),
            b_ineq: vec![],
            a_eq: MatrixInput::new(1, 2, vec![0, 1, 1], vec![0], vec![3.0]),
            b_eq: vec![1.0],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        },
        &mut cooperation,
    ))
    .unwrap();
    for error in [-1e-12, 0.0, 1e-12] {
        assert!(block_on(terminal::feasible_point(
            &certificates::conic_view(&thirds, &thirds.cones()),
            &[1.0 / 3.0 + error, 0.0],
            &mut cooperation
        ))
        .unwrap());
    }
    assert!(block_on(terminal::descent_ray(
        &certificates::conic_view(&thirds, &thirds.cones()),
        &[0.0, 1.0],
        &mut cooperation
    ))
    .unwrap());
    let rational_farkas = block_on(model::ConicModel::new(
        make(vec![3.0, -1.0], vec![0.0, -1.0]),
        &mut cooperation,
    ))
    .unwrap();
    assert!(block_on(terminal::primal_infeasible(
        &certificates::conic_view(&rational_farkas, &rational_farkas.cones()),
        &[1.0 / 3.0 + 1e-12, 1.0],
        &mut cooperation
    ))
    .unwrap());
    let rational_ray = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![0.0, -1.0],
            a_ineq: MatrixInput::zeros((0, 2)),
            b_ineq: vec![],
            a_eq: MatrixInput::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![3.0, -1.0]),
            b_eq: vec![0.0],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        },
        &mut cooperation,
    ))
    .unwrap();
    assert!(block_on(terminal::descent_ray(
        &certificates::conic_view(&rational_ray, &rational_ray.cones()),
        &[1.0 / 3.0 - 1e-12, 1.0],
        &mut cooperation
    ))
    .unwrap());
    for scale in [f64::from_bits(1), 1e-100, 1.0, 1e100, f64::MAX] {
        assert!(!block_on(terminal::primal_infeasible(
            &certificates::conic_view(&near_feasible, &near_feasible.cones()),
            &[scale, scale],
            &mut cooperation
        ))
        .unwrap());
        assert!(!block_on(terminal::descent_ray(
            &certificates::conic_view(&near_bounded, &near_bounded.cones()),
            &[scale, scale],
            &mut cooperation
        ))
        .unwrap());
        assert!(!block_on(terminal::primal_infeasible(
            &certificates::conic_view(&feasible, &feasible.cones()),
            &[scale, 0.0],
            &mut cooperation
        ))
        .unwrap());
        assert!(!block_on(terminal::descent_ray(
            &certificates::conic_view(&bounded, &bounded.cones()),
            &[0.0, scale],
            &mut cooperation
        ))
        .unwrap());
        assert!(block_on(terminal::descent_ray(
            &certificates::conic_view(&unbounded, &unbounded.cones()),
            &[0.0, scale],
            &mut cooperation
        ))
        .unwrap());
        assert!(block_on(terminal::primal_infeasible(
            &certificates::conic_view(&infeasible, &infeasible.cones()),
            &[scale, scale],
            &mut cooperation
        ))
        .unwrap());
    }
}

#[test]
fn owned_backend_cooperation_delivers_original_cancellation_between_iterations() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::{cell::Cell, num::NonZeroU64, rc::Rc};
    let mut initial =
        ScalarCooperation::for_builtin(runmat_builtins::LINPROG_CATALOG_ENTRY.identity, None);
    let model = block_on(model::ConicModel::new(
        LinearProgram {
            f: vec![-1.0],
            a_ineq: MatrixInput::zeros((0, 1)),
            b_ineq: vec![],
            a_eq: MatrixInput::zeros((0, 1)),
            b_eq: vec![],
            lb: vec![0.0],
            ub: vec![1.0],
        },
        &mut initial,
    ))
    .unwrap();
    let source = Rc::new(CancellationSource::new());
    let cancelled = source.clone();
    let turns = Rc::new(Cell::new(0));
    let observed = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            observed.set(observed.get() + 1);
            if observed.get() == 2 {
                cancelled.cancel(CancellationReason::User);
            }
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = ScalarCooperation::with_builtin_budget(
        runmat_builtins::LINPROG_CATALOG_ENTRY.identity,
        budget,
    );
    let mut controls = options::Controls::default();
    let error = block_on(solve::run(&model, false, &mut cooperation, &mut controls))
        .err()
        .expect("original cancellation");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    assert!(error.source.is_some());
    assert_eq!(turns.get(), 2);
    assert!(controls.iterations() <= 1);
}
