//! MATLAB-compatible `coneprog` and `secondordercone` builtins.

use nalgebra::{DMatrix, DVector};
use runmat_builtins::{
    BuiltinErrorDescriptor, CONEPROG_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    CONEPROG_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
};
pub use runmat_builtins::{CONEPROG_DESCRIPTOR, SECONDORDERCONE_DESCRIPTOR};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, StructValue, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::{lookup_option, option_f64, option_usize};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod solve;
use crate::builtins::math::optim::common::ScalarCooperation;
use solve::solve_coneprog;

const CONEPROG: &str = "coneprog";
const ALGORITHM: &str = "interior-point conic barrier";
const DEFAULT_TOL: f64 = 1.0e-7;
const DEFAULT_MAX_ITER: usize = 200;
const STRICT_MARGIN: f64 = 1.0e-7;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::coneprog")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "coneprog",
    op_kind: GpuOpKind::Custom("conic-programming"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Host conic interior-point solver. GPU-resident numeric inputs are gathered before solving.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::coneprog")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "coneprog",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Conic programming is a solver boundary and terminates fusion planning.",
};

#[runtime_builtin(
    name = "secondordercone",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::optim::coneprog"
)]
async fn secondordercone_builtin_registered(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &SECONDORDERCONE_DESCRIPTOR,
        &ERROR_INVALID_ARGUMENT,
        "secondordercone",
    )?;
    if args.len() != 4 {
        return Err(coneprog_error(
            &ERROR_INVALID_ARGUMENT,
            "secondordercone: expected A, b, d, and gamma",
        ));
    }
    crate::builtins::math::optim::common::preflight_catalog_inputs(
        &runmat_builtins::SECONDORDERCONE_CATALOG_ENTRY,
        &ERROR_INVALID_INPUT,
        &args,
    )?;
    let [a, b, d, gamma]: [Value; 4] = args.try_into().map_err(|_| {
        coneprog_error(
            &ERROR_INVALID_ARGUMENT,
            "secondordercone: expected A, b, d, and gamma",
        )
    })?;
    secondordercone_builtin(a, b, d, gamma).await
}

async fn secondordercone_builtin(
    a: Value,
    b: Value,
    d: Value,
    gamma: Value,
) -> BuiltinResult<Value> {
    use runmat_builtins::OptimNumericInput as Role;
    for (label, value, role) in [
        ("A", &a, Role::NonemptyMatrix),
        ("b", &b, Role::OptionalVector),
        ("d", &d, Role::OptionalVector),
        ("gamma", &gamma, Role::Scalar),
    ] {
        preflight::numeric(label, value, role)?;
    }
    let a = numeric_matrix("A", a).await?.ok_or_else(|| {
        coneprog_error(&ERROR_INVALID_INPUT, "secondordercone: A must be nonempty")
    })?;
    let b = numeric_vector("b", b, FiniteMode::Finite).await?;
    let d = numeric_vector("d", d, FiniteMode::Finite).await?;
    let gamma = numeric_scalar("gamma", gamma, FiniteMode::Finite).await?;
    validate_soc_dimensions(&a, &b, &d, gamma)?;

    let mut st = StructValue::new();
    st.insert("A", matrix_value(&a)?);
    st.insert("b", vector_value_with_len(b));
    st.insert("d", vector_value_with_len(d));
    st.insert("gamma", Value::Num(gamma));
    st.insert("Type", Value::from("secondordercone"));
    Ok(Value::Struct(st))
}

#[runtime_builtin(
    name = "coneprog",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::coneprog"
)]
async fn coneprog_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    coneprog_in_runtime(runtime.as_ref(), args).await
}

#[cfg(test)]
async fn coneprog_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    coneprog_in_runtime(None, args).await
}

async fn coneprog_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &CONEPROG_DESCRIPTOR,
        &ERROR_INVALID_ARGUMENT,
        CONEPROG,
    )?;
    let mut cooperation =
        ScalarCooperation::for_builtin(runmat_builtins::CONEPROG_CATALOG_ENTRY.identity, runtime);
    cooperation.completed(0).await?;
    preflight::arguments(&args)?;
    crate::builtins::math::optim::common::preflight_catalog_inputs(
        &runmat_builtins::CONEPROG_CATALOG_ENTRY,
        &ERROR_INVALID_INPUT,
        &args,
    )?;
    let problem = parse_problem(args).await?;
    Ok(finalize(solve_coneprog(&problem, &mut cooperation).await?))
}

mod preflight;

#[derive(Clone, Copy)]
enum FiniteMode {
    Finite,
    Bounds,
}

#[derive(Clone)]
struct MatrixInput {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl MatrixInput {
    fn row(&self, row: usize) -> Vec<f64> {
        (0..self.cols)
            .map(|col| self.data[row + col * self.rows])
            .collect()
    }

    fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        (0..self.rows).map(|row| dot(&self.row(row), x)).collect()
    }
}

#[derive(Clone)]
struct SocConstraint {
    a: MatrixInput,
    b: Vec<f64>,
    d: Vec<f64>,
    gamma: f64,
}

#[derive(Clone)]
struct ConeProgram {
    f: Vec<f64>,
    soc: Vec<SocConstraint>,
    a_ineq: Vec<Vec<f64>>,
    b_ineq: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    options: SolverOptions,
}

#[derive(Clone, Copy)]
struct SolverOptions {
    tol: f64,
    max_iter: usize,
}

#[derive(Clone)]
struct ConeOutcome {
    x: Option<Vec<f64>>,
    fval: Option<f64>,
    exitflag: i32,
    iterations: usize,
    constrviolation: f64,
    message: String,
    lambda: LambdaEstimate,
}

#[derive(Clone)]
struct LambdaEstimate {
    ineqlin: Vec<f64>,
    eqlin: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    soc: Vec<f64>,
}

impl ConeOutcome {
    fn infeasible(problem: &ConeProgram, message: &str) -> Self {
        Self {
            x: None,
            fval: None,
            exitflag: -2,
            iterations: 0,
            constrviolation: 0.0,
            message: message.to_string(),
            lambda: LambdaEstimate::zeros(problem),
        }
    }

    fn unbounded(problem: &ConeProgram) -> Self {
        Self {
            x: None,
            fval: None,
            exitflag: -3,
            iterations: 0,
            constrviolation: 0.0,
            message: "Problem is unbounded.".to_string(),
            lambda: LambdaEstimate::zeros(problem),
        }
    }
}

impl LambdaEstimate {
    fn zeros(problem: &ConeProgram) -> Self {
        Self {
            ineqlin: vec![0.0; problem.a_ineq.len()],
            eqlin: vec![0.0; problem.a_eq.len()],
            lower: vec![0.0; problem.f.len()],
            upper: vec![0.0; problem.f.len()],
            soc: vec![0.0; problem.soc.len()],
        }
    }
}

async fn parse_problem(args: Vec<Value>) -> BuiltinResult<ConeProgram> {
    if args.len() == 1 {
        let value = args.into_iter().next().unwrap();
        let Value::Struct(problem) = value else {
            return Err(coneprog_error(
                &ERROR_INVALID_ARGUMENT,
                "coneprog: single-argument form expects a problem struct",
            ));
        };
        return parse_problem_struct(problem).await;
    }
    if args.len() < 2 {
        return Err(coneprog_error(
            &ERROR_INVALID_ARGUMENT,
            "coneprog: expected f and socConstraints",
        ));
    }
    let mut args = args.into_iter();
    let f_value = args.next().unwrap();
    let soc_value = args.next().unwrap();
    let rest = args.collect::<Vec<_>>();

    let f = numeric_vector("f", f_value, FiniteMode::Finite).await?;
    if f.is_empty() {
        return Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            "coneprog: f must be a nonempty numeric vector",
        ));
    }
    let n = f.len();
    let soc = parse_soc_constraints(soc_value, n).await?;
    let (rest, options) = split_options(rest).await?;
    let (a_ineq, b_ineq, a_eq, b_eq, lb, ub) = parse_positional_constraints(rest, n).await?;
    Ok(ConeProgram {
        f,
        soc,
        a_ineq,
        b_ineq,
        a_eq,
        b_eq,
        lb,
        ub,
        options,
    })
}

async fn parse_problem_struct(problem: StructValue) -> BuiltinResult<ConeProgram> {
    let f = required_field(&problem, "f")?;
    let f = numeric_vector("f", f, FiniteMode::Finite).await?;
    if f.is_empty() {
        return Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            "coneprog: f must be a nonempty numeric vector",
        ));
    }
    let n = f.len();
    let soc_value = optional_field(&problem, "socConstraints")
        .or_else(|| optional_field(&problem, "socConstraints".to_ascii_lowercase().as_str()))
        .unwrap_or_else(empty_value);
    let soc = parse_soc_constraints(soc_value, n).await?;
    let a = optional_field(&problem, "Aineq")
        .or_else(|| optional_field(&problem, "A"))
        .unwrap_or_else(empty_value);
    let b = optional_field(&problem, "bineq")
        .or_else(|| optional_field(&problem, "b"))
        .unwrap_or_else(empty_value);
    let (a_ineq, b_ineq) = parse_constraint_pair("A", a, "b", b, n).await?;
    let aeq = optional_field(&problem, "Aeq").unwrap_or_else(empty_value);
    let beq = optional_field(&problem, "beq").unwrap_or_else(empty_value);
    let (a_eq, b_eq) = parse_constraint_pair("Aeq", aeq, "beq", beq, n).await?;
    let lb = optional_field(&problem, "lb").unwrap_or_else(empty_value);
    let ub = optional_field(&problem, "ub").unwrap_or_else(empty_value);
    let (lb, ub) = parse_bounds(Some(&lb), Some(&ub), n).await?;
    let options = match optional_field(&problem, "options") {
        Some(Value::Struct(st)) => gathered_options(&st).await?,
        Some(other) => {
            return Err(coneprog_error(
                &ERROR_INVALID_INPUT,
                format!("coneprog: options must be a struct, got {other:?}"),
            ))
        }
        None => SolverOptions::default(),
    };
    Ok(ConeProgram {
        f,
        soc,
        a_ineq,
        b_ineq,
        a_eq,
        b_eq,
        lb,
        ub,
        options,
    })
}

async fn split_options(rest: Vec<Value>) -> BuiltinResult<(Vec<Value>, SolverOptions)> {
    let Some(last) = rest.last() else {
        return Ok((rest, SolverOptions::default()));
    };
    if let Value::Struct(st) = last {
        if is_options_struct(st) {
            let options = gathered_options(st).await?;
            let mut without = rest;
            without.pop();
            return Ok((without, options));
        }
    }
    Ok((rest, SolverOptions::default()))
}

async fn parse_positional_constraints(
    rest: Vec<Value>,
    n: usize,
) -> BuiltinResult<(
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    if !matches!(rest.len(), 0 | 2 | 4 | 6) {
        return Err(coneprog_error(
            &ERROR_INVALID_ARGUMENT,
            "coneprog: optional linear constraints must be A,b,Aeq,beq,lb,ub",
        ));
    }
    let a = rest.first().cloned().unwrap_or_else(empty_value);
    let b = rest.get(1).cloned().unwrap_or_else(empty_value);
    let (a_ineq, b_ineq) = parse_constraint_pair("A", a, "b", b, n).await?;
    let aeq = rest.get(2).cloned().unwrap_or_else(empty_value);
    let beq = rest.get(3).cloned().unwrap_or_else(empty_value);
    let (a_eq, b_eq) = parse_constraint_pair("Aeq", aeq, "beq", beq, n).await?;
    let (lb, ub) = parse_bounds(rest.get(4), rest.get(5), n).await?;
    Ok((a_ineq, b_ineq, a_eq, b_eq, lb, ub))
}

fn is_options_struct(st: &StructValue) -> bool {
    st.fields
        .keys()
        .any(|name| runmat_builtins::OptimSolver::Coneprog.recognizes_options_field(name))
}

async fn gathered_options(options: &StructValue) -> BuiltinResult<SolverOptions> {
    let mut gathered = StructValue::new();
    for field in [
        runmat_builtins::OptimOptionField::TolFun,
        runmat_builtins::OptimOptionField::TolX,
        runmat_builtins::OptimOptionField::MaxIter,
    ] {
        if let Some(value) = lookup_option(options, field.name()) {
            gathered.insert(field.name(), gather(value.clone()).await?);
        }
    }
    parse_options(Some(&gathered))
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            tol: DEFAULT_TOL,
            max_iter: DEFAULT_MAX_ITER,
        }
    }
}

fn parse_options(options: Option<&StructValue>) -> BuiltinResult<SolverOptions> {
    let tol = option_f64(CONEPROG, options, "TolFun", DEFAULT_TOL)?.min(option_f64(
        CONEPROG,
        options,
        "TolX",
        DEFAULT_TOL,
    )?);
    let max_iter = option_usize(CONEPROG, options, "MaxIter", DEFAULT_MAX_ITER)?.max(1);
    Ok(SolverOptions {
        tol: tol.max(1.0e-12),
        max_iter,
    })
}

async fn parse_soc_constraints(value: Value, n: usize) -> BuiltinResult<Vec<SocConstraint>> {
    let value = if matches!(value, Value::GpuTensor(_)) {
        gather(value).await?
    } else {
        value
    };
    if is_empty_value(&value) {
        return Ok(Vec::new());
    }
    match value {
        Value::Struct(st) => parse_single_soc(st, n).await.map(|constraint| vec![constraint]),
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for value in cell.data {
                let Value::Struct(st) = value else {
                    return Err(coneprog_error(
                        &ERROR_INVALID_INPUT,
                        "coneprog: socConstraints cell entries must be secondordercone structs",
                    ));
                };
                out.push(parse_single_soc(st, n).await?);
            }
            Ok(out)
        }
        Value::OutputList(values) => {
            let mut out = Vec::with_capacity(values.len());
            let values = values.try_into_owned().map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?;
            for value in values {
                let Value::Struct(st) = value else {
                    return Err(coneprog_error(
                        &ERROR_INVALID_INPUT,
                        "coneprog: socConstraints entries must be secondordercone structs",
                    ));
                };
                out.push(parse_single_soc(st, n).await?);
            }
            Ok(out)
        }
        other => Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            format!("coneprog: socConstraints must be a secondordercone struct or cell array, got {other:?}"),
        )),
    }
}

async fn parse_single_soc(st: StructValue, n: usize) -> BuiltinResult<SocConstraint> {
    let a = numeric_matrix("socConstraints.A", required_field(&st, "A")?)
        .await?
        .ok_or_else(|| coneprog_error(&ERROR_INVALID_INPUT, "coneprog: cone A must be nonempty"))?;
    let b = numeric_vector(
        "socConstraints.b",
        required_field(&st, "b")?,
        FiniteMode::Finite,
    )
    .await?;
    let d = numeric_vector(
        "socConstraints.d",
        required_field(&st, "d")?,
        FiniteMode::Finite,
    )
    .await?;
    let gamma = numeric_scalar(
        "socConstraints.gamma",
        required_field(&st, "gamma")?,
        FiniteMode::Finite,
    )
    .await?;
    validate_soc_dimensions(&a, &b, &d, gamma)?;
    if a.cols != n || d.len() != n {
        return Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            "coneprog: cone A columns and d length must match f",
        ));
    }
    Ok(SocConstraint { a, b, d, gamma })
}

fn validate_soc_dimensions(a: &MatrixInput, b: &[f64], d: &[f64], gamma: f64) -> BuiltinResult<()> {
    if b.len() != a.rows {
        return Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            "secondordercone: b length must match rows of A",
        ));
    }
    if d.len() != a.cols {
        return Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            "secondordercone: d length must match columns of A",
        ));
    }
    if !gamma.is_finite() {
        return Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            "secondordercone: gamma must be finite",
        ));
    }
    Ok(())
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(|err| {
            let mut error = coneprog_error(&ERROR_INVALID_INPUT, err.message());
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error.source = Some(Box::new(err));
            error
        })
}

fn required_field(st: &StructValue, name: &str) -> BuiltinResult<Value> {
    optional_field(st, name).ok_or_else(|| {
        coneprog_error(
            &ERROR_INVALID_INPUT,
            format!("coneprog: problem struct is missing {name}"),
        )
    })
}

fn optional_field(st: &StructValue, name: &str) -> Option<Value> {
    st.fields
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}

fn is_empty_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if tensor::tensor_element_len(t) == 0)
}

fn empty_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

async fn numeric_scalar(label: &str, value: Value, finite_mode: FiniteMode) -> BuiltinResult<f64> {
    let values = numeric_vector(label, value, finite_mode).await?;
    if values.len() == 1 {
        Ok(values[0])
    } else {
        Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            format!("coneprog: {label} must be a scalar"),
        ))
    }
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
        Value::Bool(flag) => vec![if flag { 1.0 } else { 0.0 }],
        Value::Tensor(tensor) => {
            if tensor.shape().len() > 2 || (tensor.rows() != 1 && tensor.cols() != 1) {
                return Err(coneprog_error(
                    &ERROR_INVALID_INPUT,
                    format!("coneprog: {label} must be a vector"),
                ));
            }
            tensor::tensor_into_values_f64(tensor)
        }
        other => {
            return Err(coneprog_error(
                &ERROR_INVALID_INPUT,
                format!("coneprog: {label} must be a real numeric vector, got {other:?}"),
            ))
        }
    };
    validate_numbers(label, &data, finite_mode)?;
    Ok(data)
}

async fn numeric_matrix(label: &str, value: Value) -> BuiltinResult<Option<MatrixInput>> {
    let value = gather(value).await?;
    if is_empty_value(&value) {
        return Ok(None);
    }
    match value {
        Value::Num(n) => {
            validate_numbers(label, &[n], FiniteMode::Finite)?;
            Ok(Some(MatrixInput {
                rows: 1,
                cols: 1,
                data: vec![n],
            }))
        }
        Value::Int(i) => {
            let n = i.to_f64();
            validate_numbers(label, &[n], FiniteMode::Finite)?;
            Ok(Some(MatrixInput {
                rows: 1,
                cols: 1,
                data: vec![n],
            }))
        }
        Value::Tensor(tensor) => {
            if tensor.shape().len() > 2 {
                return Err(coneprog_error(
                    &ERROR_INVALID_INPUT,
                    format!("coneprog: {label} must be a numeric matrix"),
                ));
            }
            let rows = tensor.rows();
            let cols = tensor.cols();
            let data = tensor::tensor_into_values_f64(tensor);
            validate_numbers(label, &data, FiniteMode::Finite)?;
            Ok(Some(MatrixInput { rows, cols, data }))
        }
        other => Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            format!("coneprog: {label} must be a real numeric matrix, got {other:?}"),
        )),
    }
}

fn validate_numbers(label: &str, data: &[f64], finite_mode: FiniteMode) -> BuiltinResult<()> {
    for value in data {
        match finite_mode {
            FiniteMode::Finite if !value.is_finite() => {
                return Err(coneprog_error(
                    &ERROR_INVALID_INPUT,
                    format!("coneprog: {label} values must be finite"),
                ))
            }
            FiniteMode::Bounds if value.is_nan() => {
                return Err(coneprog_error(
                    &ERROR_INVALID_INPUT,
                    format!("coneprog: {label} bounds cannot be NaN"),
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
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let matrix = numeric_matrix(matrix_label, matrix).await?;
    let rhs = numeric_vector(rhs_label, rhs, FiniteMode::Finite).await?;
    match (matrix, rhs.is_empty()) {
        (None, true) => Ok((Vec::new(), Vec::new())),
        (None, false) => Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            format!("coneprog: {matrix_label} cannot be empty when {rhs_label} is nonempty"),
        )),
        (Some(matrix), _) => {
            if matrix.cols != n {
                return Err(coneprog_error(
                    &ERROR_INVALID_INPUT,
                    format!("coneprog: {matrix_label} must have one column per element of f"),
                ));
            }
            if rhs.len() != matrix.rows {
                return Err(coneprog_error(
                    &ERROR_INVALID_INPUT,
                    format!("coneprog: {rhs_label} length must match rows of {matrix_label}"),
                ));
            }
            let rows = (0..matrix.rows).map(|row| matrix.row(row)).collect();
            Ok((rows, rhs))
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
        Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            format!("coneprog: {label} length must match f"),
        ))
    }
}

fn impossible_bounds(problem: &ConeProgram) -> Option<&'static str> {
    for i in 0..problem.f.len() {
        if problem.lb[i] > problem.ub[i] + problem.options.tol {
            return Some("No feasible point found: lower bound exceeds upper bound.");
        }
    }
    None
}

fn append_bound_rows(problem: &ConeProgram, rows: &mut Vec<Vec<f64>>, rhs: &mut Vec<f64>) {
    for i in 0..problem.f.len() {
        if problem.lb[i].is_finite() {
            let mut row = vec![0.0; problem.f.len()];
            row[i] = -1.0;
            rows.push(row);
            rhs.push(-problem.lb[i]);
        }
        if problem.ub[i].is_finite() {
            let mut row = vec![0.0; problem.f.len()];
            row[i] = 1.0;
            rows.push(row);
            rhs.push(problem.ub[i]);
        }
    }
}

fn barrier_value(
    problem: &ConeProgram,
    linear_rows: &[Vec<f64>],
    linear_rhs: &[f64],
    x: &[f64],
    t: f64,
) -> f64 {
    let mut value = t * dot(&problem.f, x);
    for (row, rhs) in linear_rows.iter().zip(linear_rhs) {
        let slack = rhs - dot(row, x);
        if slack <= 0.0 {
            return f64::INFINITY;
        }
        value -= slack.ln();
    }
    for constraint in &problem.soc {
        let z = cone_z(constraint, x);
        let q = dot(&constraint.d, x) + constraint.gamma;
        let margin = q * q - dot(&z, &z);
        if q <= 0.0 || margin <= 0.0 {
            return f64::INFINITY;
        }
        value -= margin.ln();
    }
    value
}

fn barrier_gradient(
    problem: &ConeProgram,
    linear_rows: &[Vec<f64>],
    linear_rhs: &[f64],
    x: &[f64],
    t: f64,
) -> Vec<f64> {
    let mut grad = problem.f.iter().map(|value| t * value).collect::<Vec<_>>();
    for (row, rhs) in linear_rows.iter().zip(linear_rhs) {
        let slack = (rhs - dot(row, x)).max(STRICT_MARGIN);
        for (gi, ai) in grad.iter_mut().zip(row) {
            *gi += ai / slack;
        }
    }
    for constraint in &problem.soc {
        let z = cone_z(constraint, x);
        let q = dot(&constraint.d, x) + constraint.gamma;
        let margin = (q * q - dot(&z, &z)).max(STRICT_MARGIN);
        let mut margin_grad = constraint
            .d
            .iter()
            .map(|di| 2.0 * q * di)
            .collect::<Vec<_>>();
        for col in 0..constraint.a.cols {
            let a_col_z = (0..constraint.a.rows)
                .map(|row| constraint.a.data[row + col * constraint.a.rows] * z[row])
                .sum::<f64>();
            margin_grad[col] -= 2.0 * a_col_z;
        }
        for (gi, mi) in grad.iter_mut().zip(margin_grad) {
            *gi -= mi / margin;
        }
    }
    grad
}

fn project_gradient(a_eq: &[Vec<f64>], grad: &[f64]) -> Vec<f64> {
    if a_eq.is_empty() {
        return grad.to_vec();
    }
    let mut projected = grad.to_vec();
    let rhs = a_eq.iter().map(|row| dot(row, grad)).collect::<Vec<_>>();
    if let Some(rowspace) = minimum_norm_equality_correction(a_eq, &rhs, grad.len()) {
        for (pi, ri) in projected.iter_mut().zip(rowspace) {
            *pi -= ri;
        }
    }
    projected
}

fn project_to_equalities(a_eq: &[Vec<f64>], b_eq: &[f64], x: &mut [f64]) {
    if a_eq.is_empty() {
        return;
    }
    let residual = a_eq
        .iter()
        .zip(b_eq)
        .map(|(row, rhs)| rhs - dot(row, x))
        .collect::<Vec<_>>();
    if let Some(correction) = minimum_norm_equality_correction(a_eq, &residual, x.len()) {
        for (xi, ci) in x.iter_mut().zip(correction) {
            *xi += ci;
        }
    }
}

fn minimum_norm_equality_correction(rows: &[Vec<f64>], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    if rows.is_empty() {
        return Some(vec![0.0; n]);
    }
    let a = dmatrix_from_rows(rows, n);
    let rhs = DVector::from_column_slice(rhs);
    let gram = &a * a.transpose();
    let svd = gram.svd(true, true);
    let y = svd.solve(&rhs, DEFAULT_TOL).ok()?;
    let correction = a.transpose() * y;
    Some(correction.iter().copied().collect())
}

fn pseudo_solve(rows: &[Vec<f64>], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    if rows.is_empty() {
        return Some(vec![0.0; n]);
    }
    let matrix = dmatrix_from_rows(rows, n);
    let rhs_vec = DVector::from_column_slice(rhs);
    let svd = matrix.svd(true, true);
    let x = svd.solve(&rhs_vec, DEFAULT_TOL).ok()?;
    let out = x.iter().copied().collect::<Vec<_>>();
    let residual = rows
        .iter()
        .zip(rhs)
        .map(|(row, target)| (dot(row, &out) - target).abs())
        .fold(0.0, f64::max);
    (residual <= 1.0e-7).then_some(out)
}

fn dmatrix_from_rows(rows: &[Vec<f64>], n: usize) -> DMatrix<f64> {
    let data = rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    DMatrix::from_row_slice(rows.len(), n, &data)
}

fn is_strictly_feasible(
    problem: &ConeProgram,
    linear_rows: &[Vec<f64>],
    linear_rhs: &[f64],
    x: &[f64],
) -> bool {
    linear_rows
        .iter()
        .zip(linear_rhs)
        .all(|(row, rhs)| rhs - dot(row, x) > STRICT_MARGIN)
        && problem.soc.iter().all(|constraint| {
            let z = cone_z(constraint, x);
            let q = dot(&constraint.d, x) + constraint.gamma;
            q > 0.0 && q - norm(&z) > STRICT_MARGIN
        })
        && equality_violation(problem, x) <= 1.0e-6
}

fn constraint_violation(problem: &ConeProgram, x: &[f64]) -> f64 {
    let mut violation = equality_violation(problem, x);
    for (row, rhs) in problem.a_ineq.iter().zip(&problem.b_ineq) {
        violation = violation.max((dot(row, x) - rhs).max(0.0));
    }
    for i in 0..x.len() {
        violation = violation.max((problem.lb[i] - x[i]).max(0.0));
        violation = violation.max((x[i] - problem.ub[i]).max(0.0));
    }
    for constraint in &problem.soc {
        let z = cone_z(constraint, x);
        let q = dot(&constraint.d, x) + constraint.gamma;
        violation = violation.max((norm(&z) - q).max(0.0));
    }
    violation
}

fn equality_violation(problem: &ConeProgram, x: &[f64]) -> f64 {
    problem
        .a_eq
        .iter()
        .zip(&problem.b_eq)
        .map(|(row, rhs)| (dot(row, x) - rhs).abs())
        .fold(0.0, f64::max)
}

fn cone_z(constraint: &SocConstraint, x: &[f64]) -> Vec<f64> {
    let mut z = constraint.a.mul_vec(x);
    for (zi, bi) in z.iter_mut().zip(&constraint.b) {
        *zi += bi;
    }
    z
}

fn cone_violation_gradient(constraint: &SocConstraint, z: &[f64], z_norm: f64) -> Vec<f64> {
    let mut grad = constraint.d.iter().map(|di| -*di).collect::<Vec<_>>();
    if z_norm > STRICT_MARGIN {
        for col in 0..constraint.a.cols {
            let a_col_z = (0..constraint.a.rows)
                .map(|row| constraint.a.data[row + col * constraint.a.rows] * z[row])
                .sum::<f64>();
            grad[col] += a_col_z / z_norm;
        }
    }
    grad
}

fn optimal_outcome(
    problem: &ConeProgram,
    x: Vec<f64>,
    fval: f64,
    iterations: usize,
) -> ConeOutcome {
    let lambda = estimate_lambda(problem, &x);
    ConeOutcome {
        x: Some(x.clone()),
        fval: Some(fval),
        exitflag: 1,
        iterations,
        constrviolation: constraint_violation(problem, &x),
        message: "Optimal solution found.".to_string(),
        lambda,
    }
}

#[derive(Clone, Copy)]
enum ActiveConstraint {
    Ineq(usize),
    Eq(usize),
    Lower(usize),
    Upper(usize),
    Soc(usize),
}

fn estimate_lambda(problem: &ConeProgram, x: &[f64]) -> LambdaEstimate {
    let mut lambda = LambdaEstimate::zeros(problem);
    let mut columns = Vec::new();
    let mut kinds = Vec::new();
    let active_tol = problem.options.tol.max(1.0e-6);

    for (idx, (row, rhs)) in problem.a_ineq.iter().zip(&problem.b_ineq).enumerate() {
        if rhs - dot(row, x) <= active_tol {
            columns.push(DVector::from_column_slice(row));
            kinds.push(ActiveConstraint::Ineq(idx));
        }
    }
    for (idx, row) in problem.a_eq.iter().enumerate() {
        columns.push(DVector::from_column_slice(row));
        kinds.push(ActiveConstraint::Eq(idx));
    }
    for idx in 0..problem.f.len() {
        if problem.lb[idx].is_finite() && x[idx] - problem.lb[idx] <= active_tol {
            let mut column = vec![0.0; problem.f.len()];
            column[idx] = -1.0;
            columns.push(DVector::from_column_slice(&column));
            kinds.push(ActiveConstraint::Lower(idx));
        }
        if problem.ub[idx].is_finite() && problem.ub[idx] - x[idx] <= active_tol {
            let mut column = vec![0.0; problem.f.len()];
            column[idx] = 1.0;
            columns.push(DVector::from_column_slice(&column));
            kinds.push(ActiveConstraint::Upper(idx));
        }
    }
    for (idx, constraint) in problem.soc.iter().enumerate() {
        let z = cone_z(constraint, x);
        let z_norm = norm(&z);
        let q = dot(&constraint.d, x) + constraint.gamma;
        if q - z_norm <= active_tol {
            let column = cone_violation_gradient(constraint, &z, z_norm);
            if norm(&column) > DEFAULT_TOL {
                columns.push(DVector::from_column_slice(&column));
                kinds.push(ActiveConstraint::Soc(idx));
            }
        }
    }

    if columns.is_empty() {
        return lambda;
    }

    let matrix = DMatrix::from_columns(&columns);
    let rhs = DVector::from_iterator(problem.f.len(), problem.f.iter().map(|value| -*value));
    let Some(solution) = matrix.svd(true, true).solve(&rhs, DEFAULT_TOL).ok() else {
        return lambda;
    };
    for (value, kind) in solution.iter().zip(kinds) {
        match kind {
            ActiveConstraint::Ineq(idx) => lambda.ineqlin[idx] = (*value).max(0.0),
            ActiveConstraint::Eq(idx) => lambda.eqlin[idx] = *value,
            ActiveConstraint::Lower(idx) => lambda.lower[idx] = (*value).max(0.0),
            ActiveConstraint::Upper(idx) => lambda.upper[idx] = (*value).max(0.0),
            ActiveConstraint::Soc(idx) => lambda.soc[idx] = (*value).max(0.0),
        }
    }
    lambda
}

fn finalize(outcome: ConeOutcome) -> Value {
    let x = outcome
        .x
        .clone()
        .map(vector_value_with_len)
        .unwrap_or_else(empty_value);
    let fval = outcome.fval.map(Value::Num).unwrap_or_else(empty_value);
    let exitflag = Value::Num(outcome.exitflag as f64);
    let output = Value::Struct(build_output_struct(&outcome));
    let lambda = Value::Struct(build_lambda_struct(&outcome));

    match crate::output_count::current_output_count() {
        None => x,
        Some(0) => Value::output_list(Vec::new()),
        Some(1) => crate::output_count::output_list_with_padding(1, vec![x]),
        Some(2) => crate::output_count::output_list_with_padding(2, vec![x, fval]),
        Some(3) => crate::output_count::output_list_with_padding(3, vec![x, fval, exitflag]),
        Some(4) => {
            crate::output_count::output_list_with_padding(4, vec![x, fval, exitflag, output])
        }
        Some(n) => crate::output_count::output_list_with_padding(
            n,
            vec![x, fval, exitflag, output, lambda],
        ),
    }
}

fn build_output_struct(outcome: &ConeOutcome) -> StructValue {
    let mut fields = StructValue::new();
    fields.insert("iterations", Value::Num(outcome.iterations as f64));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("constrviolation", Value::Num(outcome.constrviolation));
    fields.insert("message", Value::from(outcome.message.clone()));
    fields
}

fn build_lambda_struct(outcome: &ConeOutcome) -> StructValue {
    let mut fields = StructValue::new();
    fields.insert(
        "ineqlin",
        vector_value_with_len(outcome.lambda.ineqlin.clone()),
    );
    fields.insert("eqlin", vector_value_with_len(outcome.lambda.eqlin.clone()));
    fields.insert("lower", vector_value_with_len(outcome.lambda.lower.clone()));
    fields.insert("upper", vector_value_with_len(outcome.lambda.upper.clone()));
    let soc = outcome
        .lambda
        .soc
        .iter()
        .map(|value| Value::Num(*value))
        .collect::<Vec<_>>();
    fields.insert(
        "soc",
        Value::Cell(
            CellArray::new(soc, outcome.lambda.soc.len().max(1), 1).unwrap_or_else(|_| {
                CellArray::new(Vec::new(), 0, 0).expect("empty lambda soc cell")
            }),
        ),
    );
    fields
}

fn vector_value_with_len(values: Vec<f64>) -> Value {
    let n = values.len();
    Tensor::new(values, vec![n, 1])
        .map(Value::Tensor)
        .unwrap_or_else(|_| empty_value())
}

fn matrix_value(matrix: &MatrixInput) -> BuiltinResult<Value> {
    Tensor::new(matrix.data.clone(), vec![matrix.rows, matrix.cols])
        .map(Value::Tensor)
        .map_err(|err| coneprog_error(&ERROR_INVALID_INPUT, err))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn coneprog_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("coneprog:") || detail.starts_with("secondordercone:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(CONEPROG);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
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
            vec![Value::Num(1.0), empty()],
        ))
        .expect_err("cancelled solver");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let error = block_on(secondordercone_builtin_registered(Vec::new()))
            .expect_err("constructor arity is grammar, not numeric input");
        assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
        for value in [Value::Bool(true), Value::Complex(1.0, 1.0)] {
            for field in [
                runmat_builtins::OptimOptionField::TolX,
                runmat_builtins::OptimOptionField::TolFun,
                runmat_builtins::OptimOptionField::MaxIter,
            ] {
                crate::builtins::math::optim::common::preflight_option_scalar(
                    CONEPROG, &value, field,
                )
                .expect_err("known non-real/non-numeric option metadata");
            }
        }
        let _outputs = crate::output_count::push_output_count(Some(6));
        let error = block_on(coneprog_builtin_registered(None, Vec::new()))
            .expect_err("output preflight before argument parsing");
        assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn active_barrier_observes_budget_cancellation_without_solver_replay() {
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
            a_ineq: Vec::new(),
            b_ineq: Vec::new(),
            a_eq: Vec::new(),
            b_eq: Vec::new(),
            lb: vec![0.0],
            ub: vec![1.0],
            options: SolverOptions::default(),
        };
        let error = block_on(solve_coneprog(&problem, &mut cooperation))
            .err()
            .expect("cancellation at numerical boundary");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(error.source.is_some());
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn empty() -> Value {
        Value::Tensor(Tensor::zeros(vec![0, 0]))
    }

    fn run(args: Vec<Value>, outputs: usize) -> Vec<Value> {
        let _guard = crate::output_count::push_output_count(Some(outputs));
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

    fn soc_unit_disk() -> Value {
        block_on(secondordercone_builtin(
            tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
            tensor(vec![0.0, 0.0], 2, 1),
            tensor(vec![0.0, 0.0], 2, 1),
            Value::Num(1.0),
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
        let mut options = StructValue::new();
        options.insert("Solver", Value::from("coneprog"));
        options.insert("TolFun", Value::Num(1.0e-8));
        options.insert("MaxIter", Value::Num(120.0));

        let mut problem = StructValue::new();
        problem.insert("f", tensor(vec![0.0, 1.0], 2, 1));
        problem.insert("socConstraints", soc_unit_disk());
        problem.insert("Aineq", tensor(vec![0.0, -1.0], 1, 2));
        problem.insert("bineq", Value::Num(0.25));
        problem.insert("options", Value::Struct(options));

        let outputs = run(vec![Value::Struct(problem)], 3);
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
        ))
        .expect("matrix")
        .expect("nonempty matrix");
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
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
}
