//! Sparse conic linear programming with original-row multiplier certificates.

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, LINPROG_ERROR_INVALID_ARGUMENT,
    LINPROG_ERROR_INVALID_INPUT, LINPROG_GPU_INPUT_EXTENSION, LINPROG_INTEGER_AEQ_EXTENSION,
    LINPROG_INTEGER_A_EXTENSION, LINPROG_INTEGER_BEQ_EXTENSION, LINPROG_INTEGER_B_EXTENSION,
    LINPROG_INTEGER_F_EXTENSION, LINPROG_INTEGER_LB_EXTENSION, LINPROG_INTEGER_UB_EXTENSION,
};
pub use runmat_builtins::{LINPROG_DESCRIPTOR, LINPROG_EXTENSIONS, LINPROG_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "linprog";
const ALGORITHM: &str = "interior-point (Clarabel 0.11.1)";
const TOL: f64 = 1.0e-8;

mod certificates;
pub(in crate::builtins::math::optim) mod compiled;
mod matrix;
mod model;
mod options;
mod solve;
#[cfg(test)]
mod backend_tests;
use crate::builtins::math::optim::common::ScalarCooperation;
use matrix::MatrixInput;
use model::{LinearProgram, LinprogOutcome};
use solve::solve_linprog;

fn linprog_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("linprog:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::linprog")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "linprog",
    op_kind: GpuOpKind::Custom("linear-programming"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host sparse interior-point LP solver. GPU-resident numeric inputs are gathered before solving.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::linprog")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "linprog",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Linear programming is a solver boundary and terminates fusion planning.",
};

#[runtime_builtin(
    name = "linprog",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::linprog"
)]
async fn linprog_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs_in(
        &LINPROG_DESCRIPTOR,
        &LINPROG_ERROR_INVALID_ARGUMENT,
        NAME,
        runtime.as_ref(),
    )?;
    if !(3..=8).contains(&args.len()) {
        return Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_ARGUMENT,
            "expected f,A,b and optionally Aeq,beq,lb,ub,options",
        ));
    }
    let mut args = args.into_iter();
    let f = args.next().expect("admitted arity");
    let a = args.next().expect("admitted arity");
    let b = args.next().expect("admitted arity");
    let rest = args.collect();
    linprog_in_runtime(runtime.as_ref(), f, a, b, rest).await
}

#[cfg(test)]
async fn linprog_builtin(f: Value, a: Value, b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    linprog_in_runtime(None, f, a, b, rest).await
}

async fn linprog_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    f: Value,
    a: Value,
    b: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME, "finite host-memory execution is unsupported; this optimizer has no complete workspace and factorization bound")?;
    crate::output_count::validate_bounded_outputs(
        &LINPROG_DESCRIPTOR,
        &LINPROG_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let mut cooperation =
        ScalarCooperation::for_builtin(runmat_builtins::LINPROG_CATALOG_ENTRY.identity, runtime);
    cooperation.completed(0).await?;
    if rest.len() > 5 {
        return Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    if rest.len() == 1 && !is_empty_value(&rest[0]) {
        return Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_ARGUMENT,
            "Aeq requires a matching beq argument",
        ));
    }

    crate::builtins::math::optim::common::preflight_catalog_inputs(
        &runmat_builtins::LINPROG_CATALOG_ENTRY,
        &LINPROG_ERROR_INVALID_INPUT,
        [&f, &a, &b].into_iter().chain(rest.iter()),
    )?;

    let mut controls = options::Controls::parse(rest.get(4), runtime, &mut cooperation).await?;

    use runmat_builtins::OptimNumericInput as Role;
    for (label, value, role) in [
        ("f", &f, Role::NonemptyVector),
        ("A", &a, Role::OptionalSparseMatrix),
        ("b", &b, Role::OptionalVector),
    ] {
        preflight_numeric(label, value, role)?;
    }
    for ((label, role), value) in [
        ("Aeq", Role::OptionalSparseMatrix),
        ("beq", Role::OptionalVector),
        ("lb", Role::OptionalVector),
        ("ub", Role::OptionalVector),
    ]
    .into_iter()
    .zip(&rest)
    {
        preflight_numeric(label, value, role)?;
    }

    ensure_linprog_input_policy("f", &f, &LINPROG_INTEGER_F_EXTENSION).await?;
    ensure_linprog_input_policy("A", &a, &LINPROG_INTEGER_A_EXTENSION).await?;
    ensure_linprog_input_policy("b", &b, &LINPROG_INTEGER_B_EXTENSION).await?;
    if let Some(value) = rest.first() {
        ensure_linprog_input_policy("Aeq", value, &LINPROG_INTEGER_AEQ_EXTENSION).await?;
    }
    if let Some(value) = rest.get(1) {
        ensure_linprog_input_policy("beq", value, &LINPROG_INTEGER_BEQ_EXTENSION).await?;
    }
    if let Some(value) = rest.get(2) {
        ensure_linprog_input_policy("lb", value, &LINPROG_INTEGER_LB_EXTENSION).await?;
    }
    if let Some(value) = rest.get(3) {
        ensure_linprog_input_policy("ub", value, &LINPROG_INTEGER_UB_EXTENSION).await?;
    }

    let f = numeric_vector("f", f, FiniteMode::Finite).await?;
    if f.is_empty() {
        return Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_INPUT,
            "f must be a nonempty numeric vector",
        ));
    }
    let n = f.len();

    let (a_ineq, b_ineq) = parse_constraint_pair("A", a, "b", b, n, &mut cooperation).await?;
    let (a_eq, b_eq) =
        parse_optional_equality(rest.first(), rest.get(1), n, &mut cooperation).await?;
    let (lb, ub) = parse_bounds(rest.get(2), rest.get(3), n).await?;
    // Normalized numeric backing now owns the model; do not retain gathered
    // original optional inputs throughout factorization and certificate work.
    drop(rest);

    for i in 0..n {
        if lb[i] > ub[i] || lb[i] == f64::INFINITY || ub[i] == f64::NEG_INFINITY {
            return Ok(finalize(LinprogOutcome::infeasible(
                "No finite feasible point exists within the supplied bounds.",
            )));
        }
    }

    let problem = LinearProgram {
        f,
        a_ineq,
        b_ineq,
        a_eq,
        b_eq,
        lb,
        ub,
    };
    let products = compiled::Products::for_output_count(
        crate::output_count::current_output_count_in(cooperation.runtime()),
    );
    let outcome = solve_linprog(problem, &mut cooperation, &mut controls, products).await?;
    controls.summary(&outcome, cooperation.runtime())?;
    Ok(finalize(outcome))
}

fn preflight_numeric(
    label: &str,
    value: &Value,
    role: runmat_builtins::OptimNumericInput,
) -> BuiltinResult<()> {
    let fact = crate::value_fact::value_fact(value);
    if matches!(fact.kind, runmat_types::ValueKindFact::Logical) || !role.accepts_metadata(&fact) {
        return Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_INPUT,
            format!("{label} has unsupported real numeric class or geometry"),
        ));
    }
    Ok(())
}

async fn ensure_linprog_input_policy(
    label: &str,
    value: &Value,
    integer_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(integer_extension, NAME)?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
        {
            return Err(linprog_error_with_detail(
                &LINPROG_ERROR_INVALID_INPUT,
                format!("{label} integer values must be exactly representable as double"),
            ));
        }
    }
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(&LINPROG_GPU_INPUT_EXTENSION, NAME)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum FiniteMode {
    Finite,
    Bounds,
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(|err| {
            if crate::runtime_error::error_has_control_cancellation_origin(&err) {
                return err;
            }
            let mut error = linprog_error_with_detail(&LINPROG_ERROR_INVALID_INPUT, err.message());
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error.failure_origin = err.failure_origin;
            error.source = Some(Box::new(err));
            error
        })
}

fn is_empty_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if tensor::tensor_element_len(t) == 0)
        || matches!(value, Value::SparseTensor(t) if t.rows == 0 || t.cols == 0)
}

async fn numeric_vector(
    label: &str,
    value: Value,
    finite_mode: FiniteMode,
) -> BuiltinResult<Vec<f64>> {
    let value = gather(value).await?;
    if is_empty_value(&value) {
        return Ok(Vec::new());
    }
    let data = match value {
        Value::Num(n) => vec![n],
        Value::Int(i) => vec![i.to_f64()],
        Value::Tensor(t) => {
            let dims = t.shape().len();
            if dims > 2 || (t.rows() != 1 && t.cols() != 1) {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{label} must be a vector"),
                ));
            }
            // Solver input is deliberately normalized to double, but typed
            // integer tensors must be read from their exact storage rather
            // than the compatibility f64 mirror.
            tensor::tensor_into_values_f64(t)
        }
        other => {
            return Err(linprog_error_with_detail(
                &LINPROG_ERROR_INVALID_INPUT,
                format!("{label} must be a real numeric vector, got {other:?}"),
            ))
        }
    };
    validate_numbers(label, &data, finite_mode)?;
    Ok(data)
}

async fn numeric_matrix(
    label: &str,
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<MatrixInput>> {
    matrix::parse(label, value, cooperation).await
}

fn validate_numbers(label: &str, data: &[f64], finite_mode: FiniteMode) -> BuiltinResult<()> {
    for value in data {
        match finite_mode {
            FiniteMode::Finite if !value.is_finite() => {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{label} values must be finite"),
                ))
            }
            FiniteMode::Bounds if value.is_nan() => {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{label} bounds cannot be NaN"),
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

async fn parse_constraint_pair(
    matrix_label: &str,
    matrix: Value,
    rhs_label: &str,
    rhs: Value,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(MatrixInput, Vec<f64>)> {
    let matrix = numeric_matrix(matrix_label, matrix, cooperation).await?;
    let rhs = numeric_vector(rhs_label, rhs, FiniteMode::Finite).await?;
    match (matrix, rhs.is_empty()) {
        (None, true) => Ok((MatrixInput::zeros((0, n)), Vec::new())),
        (None, false) => Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_INPUT,
            format!("{matrix_label} cannot be empty when {rhs_label} is nonempty"),
        )),
        (Some(matrix), _) => {
            if matrix.n != n {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{matrix_label} must have one column per element of f"),
                ));
            }
            if rhs.len() != matrix.m {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{rhs_label} length must match rows of {matrix_label}"),
                ));
            }
            Ok((matrix, rhs))
        }
    }
}

async fn parse_optional_equality(
    aeq: Option<&Value>,
    beq: Option<&Value>,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(MatrixInput, Vec<f64>)> {
    match (aeq, beq) {
        (None, None) => Ok((MatrixInput::zeros((0, n)), Vec::new())),
        (Some(aeq), None) if is_empty_value(aeq) => Ok((MatrixInput::zeros((0, n)), Vec::new())),
        (Some(_), None) => Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_ARGUMENT,
            "Aeq requires a matching beq argument",
        )),
        (None, Some(_)) => Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_ARGUMENT,
            "beq requires a matching Aeq argument",
        )),
        (Some(aeq), Some(beq)) => {
            parse_constraint_pair("Aeq", aeq.clone(), "beq", beq.clone(), n, cooperation).await
        }
    }
}

async fn parse_bounds(
    lb: Option<&Value>,
    ub: Option<&Value>,
    n: usize,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let lb = match lb {
        None => vec![f64::NEG_INFINITY; n],
        Some(value) if is_empty_value(value) => vec![f64::NEG_INFINITY; n],
        Some(value) => {
            let values = numeric_vector("lb", value.clone(), FiniteMode::Bounds).await?;
            normalize_bound("lb", values, n)?
        }
    };
    let ub = match ub {
        None => vec![f64::INFINITY; n],
        Some(value) if is_empty_value(value) => vec![f64::INFINITY; n],
        Some(value) => {
            let values = numeric_vector("ub", value.clone(), FiniteMode::Bounds).await?;
            normalize_bound("ub", values, n)?
        }
    };
    Ok((lb, ub))
}

fn normalize_bound(label: &str, values: Vec<f64>, n: usize) -> BuiltinResult<Vec<f64>> {
    if values.len() == n {
        Ok(values)
    } else {
        Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_INPUT,
            format!("{label} length must match f"),
        ))
    }
}

fn finalize(outcome: LinprogOutcome) -> Value {
    let output = Value::Struct(build_output_struct(&outcome));
    let requested = crate::output_count::current_output_count();
    let lambda = if requested.is_some_and(|count| count >= 5) {
        Some(match outcome.lambda {
            Some(lambda) => lambda.into_value(),
            None => empty_lambda(),
        })
    } else {
        None
    };
    let x = outcome.x.map(vector_value).unwrap_or_else(empty_double);
    let fval = outcome.fval.map(Value::Num).unwrap_or_else(empty_double);
    let exitflag = Value::Num(outcome.exitflag as f64);

    match requested {
        None => x,
        Some(0) => Value::output_list(Vec::new()),
        Some(1) => crate::output_count::output_list_with_padding(1, vec![x]),
        Some(2) => crate::output_count::output_list_with_padding(2, vec![x, fval]),
        Some(3) => crate::output_count::output_list_with_padding(3, vec![x, fval, exitflag]),
        Some(n) if n >= 5 => crate::output_count::output_list_with_padding(
            n,
            vec![x, fval, exitflag, output, lambda.expect("requested lambda")],
        ),
        Some(n) if n >= 4 => {
            crate::output_count::output_list_with_padding(n, vec![x, fval, exitflag, output])
        }
        Some(_) => x,
    }
}

fn empty_lambda() -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    for name in ["ineqlin", "eqlin", "lower", "upper"] {
        fields.insert(name, empty_double());
    }
    Value::Struct(fields.finish())
}

fn vector_value(values: Vec<f64>) -> Value {
    let n = values.len();
    Tensor::new(values, vec![n, 1])
        .map(Value::Tensor)
        .unwrap_or_else(|_| empty_double())
}

fn empty_double() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn build_output_struct(outcome: &LinprogOutcome) -> StructValue {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iterations", Value::Num(outcome.iterations as f64));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("constrviolation", Value::Num(outcome.constrviolation));
    fields.insert("primalfeasibility", Value::Num(outcome.primalfeasibility));
    fields.insert("dualfeasibility", Value::Num(outcome.dualfeasibility));
    fields.insert("dualitygap", Value::Num(outcome.dualitygap));
    fields.insert("firstorderopt", Value::Num(outcome.firstorderopt));
    fields.insert("message", Value::from(outcome.message.clone()));
    fields.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Value as V};

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
        let args = vec![V::Num(1.0), empty(), empty()];
        let error = block_on(linprog_builtin_registered(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect_err("cancelled solver");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let error = block_on(linprog_builtin_registered(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                V::Num(1.0),
                empty(),
                empty(),
                V::Num(1.0),
            ]),
        ))
        .expect_err("lone nonempty Aeq remains a grammar failure");
        assert_eq!(
            error.identifier(),
            LINPROG_ERROR_INVALID_ARGUMENT.identifier
        );
        assert!(error.message().contains("Aeq requires a matching beq"));
        let _outputs = crate::output_count::push_output_count(Some(6)).expect("output-count scope");
        let error = block_on(linprog_builtin_registered(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("output preflight before argument parsing");
        assert_eq!(
            error.identifier(),
            LINPROG_ERROR_INVALID_ARGUMENT.identifier
        );
    }

    #[test]
    fn sparse_model_observes_budget_cancellation_without_solver_replay() {
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
            runmat_builtins::LINPROG_CATALOG_ENTRY.identity,
            budget,
        );
        let problem = LinearProgram {
            f: vec![1.0],
            a_ineq: MatrixInput::from(&[[-1.0]]),
            b_ineq: vec![0.0],
            a_eq: MatrixInput::zeros((0, 1)),
            b_eq: Vec::new(),
            lb: vec![f64::NEG_INFINITY],
            ub: vec![f64::INFINITY],
        };
        let error = block_on(solve_linprog(
            problem,
            &mut cooperation,
            &mut options::Controls::default(),
            compiled::Products::Primal,
        ))
        .err()
        .expect("cancellation at numerical boundary");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(error.source.is_some());
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> V {
        V::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn empty() -> V {
        V::Tensor(Tensor::zeros(vec![0, 0]))
    }

    #[test]
    fn numeric_inputs_read_exact_typed_integer_storage() {
        let vector =
            Tensor::new_integer(IntegerStorage::I64(vec![-7, 3]), vec![2, 1]).expect("vector");
        assert_eq!(
            block_on(numeric_vector("f", V::Tensor(vector), FiniteMode::Finite)).unwrap(),
            vec![-7.0, 3.0]
        );
    }

    fn run(f: V, a: V, b: V, rest: Vec<V>, outputs: usize) -> Vec<V> {
        let _guard =
            crate::output_count::push_output_count(Some(outputs)).expect("output-count scope");
        let value = block_on(linprog_builtin(f, a, b, rest)).expect("linprog");
        match value {
            V::OutputList(values) => values
                .try_into_owned()
                .expect("unique output list")
                .try_into_unbounded_vec()
                .expect("unbounded output list"),
            other => vec![other],
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn public_invocation_accepts_rank_zero_and_rank_one_objective_vectors() {
        for shape in [vec![], vec![1]] {
            let objective = V::Tensor(Tensor::new(vec![1.0], shape).expect("objective"));
            let output = block_on(linprog_builtin(
                objective,
                empty(),
                empty(),
                vec![empty(), empty(), V::Num(0.0), V::Num(1.0)],
            ))
            .expect("bounded scalar linear program");
            let V::Tensor(solution) = output else {
                panic!("expected solution tensor");
            };
            assert_eq!(solution.shape(), &[1, 1]);
            assert!(solution.materialize_f64()[0].abs() < 1e-7);
        }
    }

    #[test]
    fn solves_bounded_feasible_problem() {
        let outputs = run(
            tensor(vec![-1.0, -2.0], 2, 1),
            tensor(vec![1.0, 1.0], 1, 2),
            V::Num(4.0),
            vec![empty(), empty(), tensor(vec![0.0, 0.0], 2, 1), empty()],
            3,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.materialize_f64()[0] - 0.0).abs() < 1.0e-7, "{x:?}");
                assert!((x.materialize_f64()[1] - 4.0).abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval + 8.0).abs() < 1.0e-7));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == 1.0));
    }

    #[test]
    fn solves_equality_constrained_problem() {
        let outputs = run(
            tensor(vec![1.0, 2.0], 2, 1),
            empty(),
            empty(),
            vec![
                tensor(vec![1.0, 1.0], 1, 2),
                V::Num(3.0),
                tensor(vec![1.0, 0.0], 2, 1),
                empty(),
            ],
            2,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.materialize_f64()[0] - 3.0).abs() < 1.0e-7, "{x:?}");
                assert!((x.materialize_f64()[1] - 0.0).abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval - 3.0).abs() < 1.0e-7));
    }

    #[test]
    fn reports_infeasible_bounds() {
        let outputs = run(
            V::Num(1.0),
            empty(),
            empty(),
            vec![empty(), empty(), V::Num(2.0), V::Num(1.0)],
            4,
        );
        assert!(matches!(&outputs[0], V::Tensor(t) if t.materialize_f64().is_empty()));
        assert!(matches!(&outputs[1], V::Tensor(t) if t.materialize_f64().is_empty()));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == -2.0));
        assert!(matches!(&outputs[3], V::Struct(s) if s.fields.contains_key("message")));
    }

    #[test]
    fn reports_unbounded_problem() {
        let outputs = run(V::Num(-1.0), empty(), empty(), Vec::new(), 3);
        assert!(matches!(&outputs[0], V::Tensor(t) if t.materialize_f64().is_empty()));
        assert!(matches!(&outputs[1], V::Tensor(t) if t.materialize_f64().is_empty()));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == -3.0));
    }

    #[test]
    fn accepts_empty_optional_placeholders() {
        let outputs = run(
            tensor(vec![1.0, 1.0], 2, 1),
            empty(),
            empty(),
            vec![empty(), empty(), tensor(vec![2.0, 3.0], 2, 1), empty()],
            2,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.materialize_f64()[0] - 2.0).abs() < 1.0e-7, "{x:?}");
                assert!((x.materialize_f64()[1] - 3.0).abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval - 5.0).abs() < 1.0e-7));
    }

    #[test]
    fn solves_one_sided_bound_with_fewer_rows_than_variables() {
        let outputs = run(
            tensor(vec![1.0, 0.0], 2, 1),
            empty(),
            empty(),
            vec![
                empty(),
                empty(),
                tensor(vec![2.0, f64::NEG_INFINITY], 2, 1),
                empty(),
            ],
            3,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.materialize_f64()[0] - 2.0).abs() < 1.0e-7, "{x:?}");
                assert!(x.materialize_f64()[1].abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval - 2.0).abs() < 1.0e-7));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == 1.0));
    }

    #[test]
    fn optimizes_along_equality_face_when_particular_solution_is_suboptimal() {
        let outputs = run(
            tensor(vec![-1.0, 0.0, 0.0], 3, 1),
            tensor(vec![1.0, 0.0, 0.0], 1, 3),
            V::Num(1.0),
            vec![
                tensor(vec![0.0, 0.0, 1.0], 1, 3),
                V::Num(0.0),
                empty(),
                empty(),
            ],
            3,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.materialize_f64()[0] - 1.0).abs() < 1.0e-7, "{x:?}");
                assert!(x.materialize_f64()[1].abs() < 1.0e-7, "{x:?}");
                assert!(x.materialize_f64()[2].abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval + 1.0).abs() < 1.0e-7));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == 1.0));
    }

    #[test]
    fn validates_matrix_dimensions() {
        let err = block_on(linprog_builtin(
            tensor(vec![1.0, 1.0], 2, 1),
            tensor(vec![1.0, 1.0, 1.0], 1, 3),
            V::Num(1.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:linprog:InvalidInput"));
    }

    #[test]
    fn linprog_typed_integer_roles_are_gated_in_matlab_mode() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let f = Tensor::new_integer(IntegerStorage::I32(vec![1]), vec![1, 1]).unwrap();
        let error = block_on(linprog_builtin(V::Tensor(f), empty(), empty(), Vec::new()))
            .expect_err("typed f must be gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:LinprogIntegerFExtension")
        );
    }

    #[test]
    fn linprog_runmat_integer_boundary_rejects_lossy_uint64() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let f = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
            .unwrap();
        let error = block_on(linprog_builtin(V::Tensor(f), empty(), empty(), Vec::new()))
            .expect_err("lossy integer boundary must reject");
        assert!(error.message().contains("exactly representable as double"));
    }
}
