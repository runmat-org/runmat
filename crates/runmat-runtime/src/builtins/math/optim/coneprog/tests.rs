// Existing public controls, with the corrected MATLAB cone-sign fixtures.
use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

#[test]
fn registered_solver_checks_requested_outputs_and_invocation_cancellation_first() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::rc::Rc;
    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(CancellationReason::User);
    let error = block_on(coneprog_builtin_registered(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0), empty()]),
    ))
    .expect_err("cancelled solver");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    let error = block_on(secondordercone_builtin_registered(
        None,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("constructor arity is grammar, not numeric input");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    for value in [Value::Bool(true), Value::Complex(1.0, 1.0)] {
        for field in [
            runmat_builtins::OptimOptionField::TolX,
            runmat_builtins::OptimOptionField::TolFun,
            runmat_builtins::OptimOptionField::MaxIter,
        ] {
            crate::builtins::math::optim::common::preflight_option_scalar(CONEPROG, &value, field)
                .expect_err("known non-real/non-numeric option metadata");
        }
    }
    let _outputs = crate::output_count::push_output_count(Some(6)).expect("output-count scope");
    let error = block_on(coneprog_builtin_registered(
        None,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("output preflight before argument parsing");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn conic_model_observes_budget_cancellation_without_solver_replay() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::{num::NonZeroU64, rc::Rc};
    let source = Rc::new(CancellationSource::new());
    let cancelled = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancelled.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = ScalarCooperation::with_builtin_budget(
        runmat_builtins::CONEPROG_CATALOG_ENTRY.identity,
        budget,
    );
    let problem = ConeProgram {
        f: vec![1.0],
        soc: Vec::new(),
        a_ineq: MatrixInput::zeros((0, 1)),
        b_ineq: Vec::new(),
        a_eq: MatrixInput::zeros((0, 1)),
        b_eq: Vec::new(),
        lb: vec![0.0],
        ub: vec![1.0],
        options: options::Controls::default(),
    };
    let error = block_on(solve_coneprog(problem, &mut cooperation))
        .err()
        .expect("cancellation at numerical boundary");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    assert!(error.source.is_some());
}

pub(super) fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

pub(super) fn empty() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

pub(super) fn run(args: Vec<Value>, outputs: usize) -> Vec<Value> {
    let _guard = crate::output_count::push_output_count(Some(outputs)).expect("output-count scope");
    let value = block_on(coneprog_builtin(args)).expect("coneprog");
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => vec![other],
    }
}

pub(super) fn soc_unit_disk() -> Value {
    block_on(secondordercone_builtin(
        tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
        tensor(vec![0.0, 0.0], 2, 1),
        tensor(vec![0.0, 0.0], 2, 1),
        Value::Num(-1.0),
    ))
    .unwrap()
}

#[test]
fn secondordercone_constructs_descriptor() {
    let Value::Struct(st) = soc_unit_disk() else {
        panic!("expected struct");
    };
    assert!(st.fields.contains_key("A"));
    assert!(st.fields.contains_key("b"));
    assert!(st.fields.contains_key("d"));
    assert_eq!(st.fields.get("Type"), Some(&Value::from("secondordercone")));
}

#[test]
fn coneprog_solves_soc_with_linear_bound() {
    let outputs = run(
        vec![
            tensor(vec![-1.0, 0.0], 2, 1),
            soc_unit_disk(),
            tensor(vec![1.0, 0.0], 1, 2),
            Value::Num(0.75),
            empty(),
            empty(),
            empty(),
            empty(),
        ],
        5,
    );
    let Value::Tensor(x) = &outputs[0] else {
        panic!("expected x");
    };
    assert!((x.materialize_f64()[0] - 0.75).abs() < 1.0e-3, "{x:?}");
    assert!(x.materialize_f64()[1].abs() < 1.0e-3, "{x:?}");
    assert!(matches!(&outputs[1], Value::Num(fval) if (*fval + 0.75).abs() < 1.0e-3));
    assert!(matches!(&outputs[2], Value::Num(flag) if *flag == 1.0));
    assert!(matches!(&outputs[3], Value::Struct(st) if st.fields.contains_key("algorithm")));
    let Value::Struct(lambda) = &outputs[4] else {
        panic!("expected lambda struct");
    };
    let Value::Tensor(ineqlin) = lambda.fields.get("ineqlin").unwrap() else {
        panic!("expected ineqlin multipliers");
    };
    assert!(
        (ineqlin.materialize_f64()[0] - 1.0).abs() < 1.0e-2,
        "{ineqlin:?}"
    );
    assert!(lambda.fields.contains_key("soc"));
}

#[test]
fn coneprog_accepts_problem_struct_and_options() {
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    options.insert("Solver", Value::from("coneprog"));
    options.insert("TolFun", Value::Num(1.0e-8));
    options.insert("MaxIter", Value::Num(120.0));

    let mut problem = runmat_value::UnboundedStructValueBuilder::new();
    problem.insert("f", tensor(vec![0.0, 1.0], 2, 1));
    problem.insert("socConstraints", soc_unit_disk());
    problem.insert("Aineq", tensor(vec![0.0, -1.0], 1, 2));
    problem.insert("bineq", Value::Num(0.25));
    problem.insert("options", Value::Struct(options.finish()));

    let outputs = run(vec![Value::Struct(problem.finish())], 3);
    let Value::Tensor(x) = &outputs[0] else {
        panic!("expected x");
    };
    assert!((x.materialize_f64()[1] + 0.25).abs() < 1.0e-3, "{x:?}");
    assert!(matches!(&outputs[2], Value::Num(flag) if *flag == 1.0));
}

#[test]
fn coneprog_numeric_parsers_read_typed_integer_storage_exactly() {
    let f = block_on(numeric_vector(
        "f",
        poisoned_integer_tensor(IntegerStorage::I16(vec![1, -2]), vec![2, 1]),
        FiniteMode::Finite,
    ))
    .expect("vector");
    assert_eq!(f, vec![1.0, -2.0]);

    let bounds = block_on(numeric_vector(
        "lb",
        poisoned_integer_tensor(IntegerStorage::I16(vec![-10, 5]), vec![1, 2]),
        FiniteMode::Bounds,
    ))
    .expect("bounds");
    assert_eq!(bounds, vec![-10.0, 5.0]);

    let matrix = block_on(numeric_matrix(
        "A",
        poisoned_integer_tensor(IntegerStorage::U16(vec![1, 2, 3, 4]), vec![2, 2]),
        &mut test_cooperation(),
    ))
    .expect("matrix")
    .expect("nonempty matrix");
    assert_eq!(matrix.m, 2);
    assert_eq!(matrix.n, 2);
    assert_eq!(matrix.nzval, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn coneprog_reports_infeasible_bounds() {
    let outputs = run(
        vec![
            tensor(vec![1.0, 0.0], 2, 1),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            tensor(vec![2.0, 0.0], 2, 1),
            tensor(vec![1.0, 1.0], 2, 1),
        ],
        4,
    );
    assert!(matches!(&outputs[0], Value::Tensor(t) if t.materialize_f64().is_empty()));
    assert!(matches!(&outputs[2], Value::Num(flag) if *flag == -2.0));
}

#[test]
fn coneprog_validates_cone_dimensions() {
    let err = block_on(secondordercone_builtin(
        tensor(vec![1.0, 0.0], 1, 2),
        tensor(vec![0.0, 0.0], 2, 1),
        tensor(vec![0.0, 0.0], 2, 1),
        Value::Num(1.0),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:coneprog:InvalidInput"));
}

fn test_cooperation() -> ScalarCooperation {
    ScalarCooperation::for_builtin(runmat_builtins::CONEPROG_CATALOG_ENTRY.identity, None)
}

async fn numeric_vector(label: &str, value: Value, mode: FiniteMode) -> BuiltinResult<Vec<f64>> {
    numeric::vector(label, value, mode, &mut test_cooperation()).await
}
