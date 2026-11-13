//! MATLAB-compatible `lsqnonlin` builtin for nonlinear least-squares problems.

use runmat_builtins::{
    BuiltinErrorDescriptor, LSQNONLIN_DESCRIPTOR, LSQNONLIN_ERROR_INVALID_ARGUMENT,
    LSQNONLIN_ERROR_INVALID_INPUT, LSQNONLIN_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::{initial_guess_with_cooperation, lookup_option};
use crate::builtins::math::optim::least_squares::{
    LeastSquaresBounds, LeastSquaresEvaluator, LeastSquaresOptions, LeastSquaresResult, Method,
    ResidualFuture,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(in crate::builtins::math::optim) mod compiled;
mod evaluation;
use evaluation::NonlinEvaluator;

const NAME: &str = "lsqnonlin";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_TOL_FUN: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;
const DEFAULT_MAX_FUN_EVALS_FACTOR: usize = 100;
const DEFAULT_ALGORITHM: &str = "trust-region-reflective";

fn lsq_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("lsqnonlin:") {
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

fn map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = lsq_error_with_detail(fallback, err.message());
        mapped.gpu_gather_retry = err.gpu_gather_retry;
        mapped.failure_origin = err.failure_origin;
        mapped.source = Some(Box::new(err));
        mapped
    }
}

#[runtime_builtin(
    name = "lsqnonlin",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::lsqnonlin"
)]
async fn lsqnonlin_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    lsqnonlin_in_runtime(runtime.as_ref(), first, rest).await
}

#[cfg(test)]
async fn lsqnonlin_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let runtime = crate::context::legacy::active();
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    lsqnonlin_in_runtime(runtime.as_ref(), first, rest.into()).await
}

async fn lsqnonlin_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    validate_requested_outputs(runtime)?;
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Lsqnonlin,
        runtime,
    );
    cooperation.completed(0).await?;
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        NAME,
        "nonlinear residual Jacobian and factorization workspaces require unbounded host memory",
    )?;
    let _option_sources = rest.fork();
    let first_source = crate::execution::RootedValue::in_optional_context(first, runtime)?;
    let parsed = ParsedProblem::parse(first_source.value().clone(), rest)
        .await
        .map_err(|err| map_error(err, &LSQNONLIN_ERROR_INVALID_ARGUMENT))?;
    let ParsedProblem {
        function,
        x0,
        lb,
        ub,
        options,
    } = parsed;
    let prepared_options = crate::builtins::math::optim::common::prepare_solver_options(
        runmat_builtins::OptimSolver::Lsqnonlin,
        options,
        None,
        None,
        runtime,
        &mut cooperation,
    )
    .await
    .map_err(|err| map_error(err, &LSQNONLIN_ERROR_INVALID_ARGUMENT))?;
    let guess = initial_guess_with_cooperation(NAME, x0, &mut cooperation)
        .await
        .map_err(|err| map_error(err, &LSQNONLIN_ERROR_INVALID_INPUT))?;
    let n = guess.values.len();
    let mut options = LsqOptions::from_struct(
        prepared_options.as_ref().map(|options| options.structure()),
        n,
    )
    .map_err(|err| map_error(err, &LSQNONLIN_ERROR_INVALID_ARGUMENT))?;
    let outputs = crate::output_count::current_output_count_in(cooperation.runtime()).unwrap_or(1);
    options.solver.final_gradient = outputs >= 5;
    options.solver.final_jacobian = outputs >= 7;
    let bounds = ParsedProblem::bounds_from_values(lb, ub, n, &mut cooperation)
        .await
        .map_err(|err| map_error(err, &LSQNONLIN_ERROR_INVALID_ARGUMENT))?;
    let fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct_for_solver(
        prepared_options.as_ref().map(|options| options.structure()),
        runmat_builtins::OptimSolver::Lsqnonlin,
    );
    let multiply = if options.supplied && options.method == Method::Reflective {
        fields.exact(runmat_builtins::OptimOptionField::JacobianMultiplyFcn)
    } else {
        None
    };
    let mut evaluator = NonlinEvaluator::new(
        cooperation,
        function,
        guess.shape.clone(),
        guess.scalar,
        options.supplied,
        multiply,
    )?;
    let mut result = crate::builtins::math::optim::least_squares::solve_controlled(
        NAME,
        &mut evaluator,
        guess.values,
        &bounds,
        &options.solver,
        options.method,
        options.display,
        prepared_options.as_ref().map(|value| value.structure()),
    )
    .await
    .map_err(|mut err| {
        err.gpu_gather_retry = crate::GpuGatherRetry::Never;
        map_error(err, &LSQNONLIN_ERROR_INVALID_INPUT)
    })?;
    if result.method == Method::LevenbergMarquardt {
        crate::builtins::math::optim::least_squares::original_derivative_columns(
            NAME,
            &mut result,
            &bounds,
            Some(&mut evaluator.cooperation),
        )
        .await?;
    }
    let residual_shape = evaluator
        .residual_shape()
        .unwrap_or(&[result.residual_len])
        .to_vec();
    finalize(
        result,
        &guess.shape,
        guess.scalar,
        &residual_shape,
        &bounds,
        &mut evaluator.cooperation,
    )
    .await
    .map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        map_error(error, &LSQNONLIN_ERROR_INVALID_INPUT)
    })
}

fn validate_requested_outputs(
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    crate::output_count::validate_bounded_outputs_in(
        &LSQNONLIN_DESCRIPTOR,
        &LSQNONLIN_ERROR_TOO_MANY_OUTPUTS,
        NAME,
        runtime,
    )
}

struct ParsedProblem {
    function: Value,
    x0: Value,
    lb: Option<Value>,
    ub: Option<Value>,
    options: Option<StructValue>,
}

impl ParsedProblem {
    async fn parse(first: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        if rest.is_empty() {
            if let Value::Struct(problem) = first {
                return Self::from_problem_struct(problem);
            }
        }
        Self::from_function_args(first, rest).await
    }

    async fn from_function_args(
        function: Value,
        rest: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Self> {
        if rest.is_empty() {
            return Err(lsq_error_with_detail(
                &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                "lsqnonlin: expected initial guess x0",
            ));
        }
        let remaining = rest.len() - 1;
        let mut values = rest.into_iter();
        let x0 = values.next().expect("initial guess was checked above");
        match remaining {
            0 => Ok(Self {
                function,
                x0,
                lb: None,
                ub: None,
                options: None,
            }),
            2 | 3 => {
                let lb = values.next();
                let ub = values.next();
                let options = parse_options(values.next())?;
                Ok(Self {
                    function,
                    x0,
                    lb,
                    ub,
                    options,
                })
            }
            6..=8 => {
                let lb = values.next();
                let ub = values.next();
                let a = values.next();
                let b = values.next();
                let aeq = values.next();
                let beq = values.next();
                ensure_empty_constraint("A", a.as_ref()).await?;
                ensure_empty_constraint("b", b.as_ref()).await?;
                ensure_empty_constraint("Aeq", aeq.as_ref()).await?;
                ensure_empty_constraint("beq", beq.as_ref()).await?;
                if let Some(nonlcon) = values.next() {
                    ensure_empty_constraint("nonlcon", Some(&nonlcon)).await?;
                }
                let options = parse_options(values.next())?;
                Ok(Self {
                    function,
                    x0,
                    lb,
                    ub,
                    options,
                })
            }
            _ => Err(lsq_error_with_detail(
                &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                "lsqnonlin: expected fun,x0 or fun,x0,lb,ub[,options]",
            )),
        }
    }

    fn from_problem_struct(problem: StructValue) -> BuiltinResult<Self> {
        if let Some(solver) = lookup_option(&problem, "solver") {
            let solver = string_scalar("problem.solver", solver)?;
            if runmat_builtins::OptimSolver::parse(&solver)
                != Some(runmat_builtins::OptimSolver::Lsqnonlin)
            {
                return Err(lsq_error_with_detail(
                    &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                    "problem.solver must be 'lsqnonlin'",
                ));
            }
        }
        if has_nonempty_problem_field(&problem, "A")
            || has_nonempty_problem_field(&problem, "b")
            || has_nonempty_problem_field(&problem, "Aineq")
            || has_nonempty_problem_field(&problem, "bineq")
            || has_nonempty_problem_field(&problem, "Aeq")
            || has_nonempty_problem_field(&problem, "beq")
            || has_nonempty_problem_field(&problem, "nonlcon")
        {
            return Err(lsq_error_with_detail(
                &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                "linear and nonlinear constraints require the interior-point infrastructure and are not supported yet",
            ));
        }
        let function = lookup_option(&problem, "objective")
            .or_else(|| lookup_option(&problem, "fun"))
            .cloned()
            .ok_or_else(|| {
                lsq_error_with_detail(
                    &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                    "problem.objective is required",
                )
            })?;
        let x0 = lookup_option(&problem, "x0").cloned().ok_or_else(|| {
            lsq_error_with_detail(&LSQNONLIN_ERROR_INVALID_ARGUMENT, "problem.x0 is required")
        })?;
        let lb = lookup_option(&problem, "lb").cloned();
        let ub = lookup_option(&problem, "ub").cloned();
        let options = match lookup_option(&problem, "options") {
            None => None,
            Some(Value::Struct(options)) => Some(options.clone()),
            Some(other) if is_empty_value(other) => None,
            Some(other) => {
                return Err(lsq_error_with_detail(
                    &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                    format!("problem.options must be a struct, got {other:?}"),
                ))
            }
        };
        Ok(Self {
            function,
            x0,
            lb,
            ub,
            options,
        })
    }

    async fn bounds_from_values(
        lb: Option<Value>,
        ub: Option<Value>,
        n: usize,
        cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
    ) -> BuiltinResult<LeastSquaresBounds> {
        let mut bounds = LeastSquaresBounds::unbounded(n);
        if let Some(lb) = lb {
            bounds.lower = bound_vector_with_cooperation(
                "lower bounds",
                lb,
                n,
                f64::NEG_INFINITY,
                cooperation,
            )
            .await?;
        }
        if let Some(ub) = ub {
            bounds.upper =
                bound_vector_with_cooperation("upper bounds", ub, n, f64::INFINITY, cooperation)
                    .await?;
        }
        bounds.validate(NAME, n)?;
        Ok(bounds)
    }
}

fn parse_options(value: Option<Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options)),
        Some(other) if is_empty_value(&other) => Ok(None),
        Some(other) => Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
}

async fn ensure_empty_constraint(label: &str, value: Option<&Value>) -> BuiltinResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if crate::value_fact::value_fact(value)
        .shape
        .element_count()
        .is_some_and(|count| count != 0)
    {
        return Err(lsq_error_with_detail(&LSQNONLIN_ERROR_INVALID_ARGUMENT, format!("{label} constraints require the interior-point infrastructure and are not supported yet")));
    }
    let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
    if is_empty_value(&gathered) {
        Ok(())
    } else {
        Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_ARGUMENT,
            format!("{label} constraints require the interior-point infrastructure and are not supported yet"),
        ))
    }
}

fn has_nonempty_problem_field(problem: &StructValue, name: &str) -> bool {
    lookup_option(problem, name).is_some_and(|value| !is_empty_value(value))
}

fn string_scalar(label: &str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(value)) => Ok(value.as_str().to_owned()),
        Value::StringArray(values) if values.data().len() == 1 => {
            values.data()[0].as_str().map(str::to_owned).ok_or_else(|| {
                lsq_error_with_detail(
                    &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                    format!("{label} must be nonmissing text"),
                )
            })
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|error| {
                lsq_error_with_detail(
                    &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                    format!("{label} contains {error}"),
                )
            })
        }
        _ => Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_ARGUMENT,
            format!("{label} must be a character vector or string scalar"),
        )),
    }
}

struct LsqOptions {
    supplied: bool,
    solver: LeastSquaresOptions,
    method: Method,
    display: runmat_builtins::OptimDisplay,
}

impl LsqOptions {
    fn from_struct(options: Option<&StructValue>, variable_len: usize) -> BuiltinResult<Self> {
        let fields =
            crate::builtins::math::optim::common::ScalarOptionFields::from_struct_for_solver(
                options,
                runmat_builtins::OptimSolver::Lsqnonlin,
            );
        let supplied = fields.switch(
            NAME,
            runmat_builtins::OptimOptionField::SpecifyObjectiveGradient,
        )?;
        let display = fields.text(runmat_builtins::OptimOptionField::Display, "off")?;
        let display = runmat_builtins::OptimDisplay::parse(&display)
            .filter(|value| {
                matches!(
                    value,
                    runmat_builtins::OptimDisplay::Off
                        | runmat_builtins::OptimDisplay::None
                        | runmat_builtins::OptimDisplay::Final
                        | runmat_builtins::OptimDisplay::Iter
                )
            })
            .ok_or_else(|| {
                lsq_error_with_detail(
                    &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                    "Display must be off, none, final or iter",
                )
            })?;
        let algorithm = fields.text(
            runmat_builtins::OptimOptionField::Algorithm,
            DEFAULT_ALGORITHM,
        )?;
        let method = match runmat_builtins::OptimAlgorithm::parse(&algorithm) {
            Some(runmat_builtins::OptimAlgorithm::LevenbergMarquardt) => Method::LevenbergMarquardt,
            Some(runmat_builtins::OptimAlgorithm::TrustRegionReflective) => Method::Reflective,
            _ => {
                return Err(lsq_error_with_detail(
                    &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                    "Algorithm must be levenberg-marquardt or trust-region-reflective",
                ))
            }
        };
        let tol_x = fields.real(NAME, runmat_builtins::OptimOptionField::TolX, DEFAULT_TOL_X)?;
        let legacy = crate::builtins::math::optim::common::option_f64_value(
            NAME,
            fields.exact(runmat_builtins::OptimOptionField::TolFun),
            "TolFun",
            DEFAULT_TOL_FUN,
        )?;
        let tol_fun = crate::builtins::math::optim::common::option_f64_value(
            NAME,
            fields.exact(runmat_builtins::OptimOptionField::FunctionTolerance),
            "FunctionTolerance",
            legacy,
        )?;
        if fields
            .exact(runmat_builtins::OptimOptionField::TolFun)
            .is_some()
            && fields
                .exact(runmat_builtins::OptimOptionField::FunctionTolerance)
                .is_some()
            && legacy != tol_fun
        {
            return Err(lsq_error_with_detail(
                &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                "conflicting TolFun and FunctionTolerance controls",
            ));
        }
        if tol_x < 0.0 || tol_fun < 0.0 || legacy < 0.0 {
            return Err(lsq_error_with_detail(
                &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                "StepTolerance, FunctionTolerance and TolFun must be nonnegative",
            ));
        }
        let max_iter = fields.count(
            NAME,
            runmat_builtins::OptimOptionField::MaxIter,
            DEFAULT_MAX_ITER,
        )?;
        let max_fun_evals = fields.count_product_default(
            NAME,
            runmat_builtins::OptimOptionField::MaxFunEvals,
            if method == Method::Reflective {
                DEFAULT_MAX_FUN_EVALS_FACTOR
            } else {
                2 * DEFAULT_MAX_FUN_EVALS_FACTOR
            },
            variable_len,
        )?;
        Ok(Self {
            supplied,
            solver: LeastSquaresOptions {
                criteria: Some(crate::builtins::math::optim::least_squares::Criteria {
                    function: tol_fun, optimality: legacy, step: tol_x,
                }),
                tol_x,
                tol_fun,
                max_iter,
                max_fun_evals,
                final_gradient: true,
                final_jacobian: true,
                scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling::from_fields(NAME,&fields,!supplied)?,
            },
            method,
            display,
        })
    }
}

#[derive(Clone)]
struct RealArray {
    values: Vec<f64>,
    shape: Vec<usize>,
}

#[cfg(test)]
async fn real_array(label: &str, value: Value) -> BuiltinResult<RealArray> {
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Lsqnonlin,
        None,
    );
    real_array_with_cooperation(label, value, &mut cooperation).await
}

async fn real_array_with_cooperation(
    label: &str,
    value: Value,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<RealArray> {
    crate::builtins::math::optim::common::preflight_real_array(NAME, &value, label, false)?;
    let value =
        crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime()).await?;
    match value {
        Value::Num(n) => finite_array(label, vec![n], vec![1, 1], cooperation).await,
        Value::Int(i) => finite_array(label, vec![i.to_f64()], vec![1, 1], cooperation).await,
        Value::Bool(flag) => {
            finite_array(
                label,
                vec![if flag { 1.0 } else { 0.0 }],
                vec![1, 1],
                cooperation,
            )
            .await
        }
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let values =
                crate::builtins::math::optim::common::tensor_values(&tensor, cooperation).await?;
            finite_array(label, values, shape, cooperation).await
        }
        Value::LogicalArray(LogicalArray { data, shape }) => {
            finite_array(
                label,
                crate::builtins::math::optim::common::logical_values(&data, cooperation).await?,
                shape.as_slice().to_vec(),
                cooperation,
            )
            .await
        }
        other => Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_INPUT,
            format!("{label} must be real numeric, got {other:?}"),
        )),
    }
}

async fn finite_array(
    label: &str,
    values: Vec<f64>,
    shape: Vec<usize>,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<RealArray> {
    if values.is_empty() {
        return Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_INPUT,
            format!("{label} must not be empty"),
        ));
    }
    if let Some(value) =
        crate::builtins::math::optim::common::first_nonfinite(&values, cooperation).await?
    {
        return Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_INPUT,
            format!("{label} must be finite, got {value}"),
        ));
    }
    Ok(RealArray { values, shape })
}

fn x_value(x: &[f64], shape: &[usize], scalar: bool) -> BuiltinResult<Value> {
    if scalar {
        Ok(Value::Num(x[0]))
    } else {
        Tensor::new(x.to_vec(), shape.to_vec())
            .map(Value::Tensor)
            .map_err(|err| lsq_error_with_detail(&LSQNONLIN_ERROR_INVALID_INPUT, err))
    }
}

#[cfg(test)]
async fn bound_vector(
    label: &str,
    value: Value,
    n: usize,
    default: f64,
) -> BuiltinResult<Vec<f64>> {
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Lsqnonlin,
        None,
    );
    bound_vector_with_cooperation(label, value, n, default, &mut cooperation).await
}

async fn bound_vector_with_cooperation(
    label: &str,
    value: Value,
    n: usize,
    default: f64,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    crate::builtins::math::optim::common::preflight_real_array(NAME, &value, label, true)?;
    if crate::value_fact::value_fact(&value)
        .shape
        .element_count()
        .is_some_and(|count| count > n)
    {
        return Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_ARGUMENT,
            format!("{label} must be empty or contain no more elements than x0"),
        ));
    }
    let value =
        crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime()).await?;
    if is_empty_value(&value) {
        return Ok(vec![default; n]);
    }
    let values = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(flag) => vec![if flag { 1.0 } else { 0.0 }],
        Value::Tensor(tensor) => {
            crate::builtins::math::optim::common::tensor_values(&tensor, cooperation).await?
        }
        Value::LogicalArray(LogicalArray { data, .. }) => {
            crate::builtins::math::optim::common::logical_values(&data, cooperation).await?
        }
        other => {
            return Err(lsq_error_with_detail(
                &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                format!("{label} must be real numeric or empty, got {other:?}"),
            ))
        }
    };
    if values.len() > n {
        return Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_ARGUMENT,
            format!("{label} must be empty or contain no more elements than x0"),
        ));
    }
    let mut out = vec![default; n];
    for (target, source) in out.chunks_mut(4096).zip(values.chunks(4096)) {
        if source.iter().any(|value| value.is_nan()) {
            return Err(lsq_error_with_detail(
                &LSQNONLIN_ERROR_INVALID_ARGUMENT,
                format!("{label} must not contain NaN"),
            ));
        }
        target[..source.len()].copy_from_slice(source);
        cooperation.completed(source.len() as u64).await?;
    }
    Ok(out)
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 0,
        Value::LogicalArray(LogicalArray { data, .. }) => data.is_empty(),
        _ => false,
    }
}

async fn finalize(
    result: LeastSquaresResult,
    x_shape: &[usize],
    x_scalar: bool,
    residual_shape: &[usize],
    bounds: &LeastSquaresBounds,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Value> {
    let requested = match crate::output_count::current_output_count_in(cooperation.runtime()) {
        None => return into_x_value(result.x, x_shape, x_scalar),
        Some(0) => return Ok(Value::output_list(Vec::new())),
        Some(count @ 1..=7) => count,
        Some(_) => {
            return Err(lsq_error_with_detail(
                &LSQNONLIN_ERROR_TOO_MANY_OUTPUTS,
                "lsqnonlin: too many output arguments; maximum is 7",
            ))
        }
    };
    // Borrow derivative consumers before moving the actual numerical buffers
    // into their requested output carriers. No residual/x payload clone is
    // manufactured merely to keep the terminal record readable.
    let information = (requested >= 5).then(|| Value::Struct(output_struct(&result)));
    let lambda = if requested >= 6 {
        Some(Value::Struct(
            lambda_struct(&result, x_shape, x_scalar, bounds, cooperation).await?,
        ))
    } else {
        None
    };
    let jacobian = if requested >= 7 {
        Some(
            crate::builtins::math::optim::least_squares::jacobian_value(NAME, &result, cooperation)
                .await?,
        )
    } else {
        None
    };
    let unevaluated = result.point_evidence
        == crate::builtins::math::optim::least_squares::PointEvidence::Supplied;
    let mut outputs = Vec::with_capacity(requested);
    outputs.push(into_x_value(result.x, x_shape, x_scalar)?);
    if requested >= 2 {
        outputs.push(Value::Num(result.resnorm));
    }
    if requested >= 3 {
        outputs.push(if unevaluated {
            crate::builtins::math::optim::common::output_callback::empty(cooperation.runtime())?
        } else {
            tensor_value(result.residual, residual_shape.to_vec())?
        });
    }
    if requested >= 4 {
        outputs.push(Value::Num(result.exitflag as f64));
    }
    outputs.extend(information);
    outputs.extend(lambda);
    outputs.extend(jacobian);
    Ok(crate::output_count::output_list_with_padding(
        requested, outputs,
    ))
}

fn into_x_value(values: Vec<f64>, shape: &[usize], scalar: bool) -> BuiltinResult<Value> {
    if scalar {
        Ok(Value::Num(values[0]))
    } else {
        tensor_value(values, shape.to_vec())
    }
}

fn tensor_value(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(values, shape)
        .map(Value::Tensor)
        .map_err(|err| lsq_error_with_detail(&LSQNONLIN_ERROR_INVALID_INPUT, err))
}

fn output_struct(result: &LeastSquaresResult) -> StructValue {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iterations", Value::Num(result.iterations as f64));
    fields.insert("funcCount", Value::Num(result.func_count as f64));
    fields.insert("firstorderopt", Value::Num(result.first_order_optimality));
    fields.insert("stepsize", Value::Num(result.step_size));
    fields.insert("cgiterations", Value::Num(result.cg_iterations as f64));
    fields.insert(
        "lssteplength",
        Value::Num(if result.step_size == 0.0 { 0.0 } else { 1.0 }),
    );
    fields.insert("algorithm", Value::from(result.method.name()));
    if let Method::NotStarted(requested) = result.method {
        fields.insert("requestedalgorithm", Value::from(requested.name()));
    }
    fields.insert("message", Value::from(result.message.clone()));
    fields.finish()
}

async fn lambda_struct(
    result: &LeastSquaresResult,
    x_shape: &[usize],
    x_scalar: bool,
    bounds: &LeastSquaresBounds,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<StructValue> {
    let compiled::BoundMultipliers {
        lower,
        upper,
        finite_gradient,
    } = compiled::bound_multipliers(result, bounds, cooperation).await?;
    if !finite_gradient {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert(
            "lower",
            Value::Tensor(
                Tensor::new(Vec::new(), vec![0, 0])
                    .map_err(|e| lsq_error_with_detail(&LSQNONLIN_ERROR_INVALID_INPUT, e))?,
            ),
        );
        fields.insert(
            "upper",
            Value::Tensor(
                Tensor::new(Vec::new(), vec![0, 0])
                    .map_err(|e| lsq_error_with_detail(&LSQNONLIN_ERROR_INVALID_INPUT, e))?,
            ),
        );
        return Ok(fields.finish());
    }
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("lower", multiplier_value(lower, x_shape, x_scalar)?);
    fields.insert("upper", multiplier_value(upper, x_shape, x_scalar)?);
    Ok(fields.finish())
}

fn multiplier_value(values: Vec<f64>, x_shape: &[usize], x_scalar: bool) -> BuiltinResult<Value> {
    if x_scalar {
        Ok(Value::Num(values[0]))
    } else {
        tensor_value(values, x_shape.to_vec())
    }
}

#[cfg(test)]
mod tests;
