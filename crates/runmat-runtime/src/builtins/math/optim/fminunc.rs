//! MATLAB-compatible `fminunc` builtin for unconstrained smooth minimization.

use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{
    FMINUNC_CALLBACK_NUMERIC_EXTENSION, FMINUNC_DESCRIPTOR, FMINUNC_ERROR_INVALID_ARGUMENT,
    FMINUNC_ERROR_INVALID_INPUT, FMINUNC_INPUT_NUMERIC_EXTENSION, FMINUNC_OPTION_EXTENSION,
    FMINUNC_RESIDENT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::{call_function, vector_to_value};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fminunc";
const ALGORITHM: &str = "quasi-newton bfgs";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_TOL_FUN: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;
const DEFAULT_MAX_FUN_EVALS: usize = 40000;
const MAX_ITER_LIMIT: usize = 1_000_000;
const MAX_FUN_EVAL_LIMIT: usize = 10_000_000;
const MAX_DENSE_BFGS_VARIABLES: usize = 2048;
const C1: f64 = 1.0e-4;
const C2: f64 = 0.9;
const MAX_LINE_SEARCH_ITERS: usize = 24;

fn fminunc_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("fminunc:") {
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

fn fminunc_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = fminunc_error_with_detail(fallback, err.message());
        mapped.gpu_gather_retry = err.gpu_gather_retry;
        mapped.failure_origin = err.failure_origin;
        mapped.source = Some(Box::new(err));
        mapped
    }
}

#[runtime_builtin(
    name = "fminunc",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::fminunc"
)]
async fn fminunc_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    function: Value,
    x0: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    fminunc_in_runtime(runtime.as_ref(), function, x0, rest).await
}

#[cfg(test)]
async fn fminunc_builtin(function: Value, x0: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    fminunc_in_runtime(None, function, x0, rest).await
}

async fn fminunc_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    x0: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &FMINUNC_DESCRIPTOR,
        &FMINUNC_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Fminunc,
        runtime,
    );
    cooperation.completed(0).await?;
    if rest.len() > 1 {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options_struct = parse_options(rest.first())
        .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_ARGUMENT))?;
    let options_struct = crate::builtins::math::optim::common::prepare_solver_options(
        runmat_builtins::OptimSolver::Fminunc,
        options_struct,
        Some(&FMINUNC_OPTION_EXTENSION),
        Some(&FMINUNC_RESIDENT_EXTENSION),
        &mut cooperation,
    )
    .await
    .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_ARGUMENT))?;
    let guess = crate::builtins::math::optim::common::initial_guess_with_extensions(
        NAME,
        x0,
        &FMINUNC_INPUT_NUMERIC_EXTENSION,
        &FMINUNC_RESIDENT_EXTENSION,
        &mut cooperation,
    )
    .await
    .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_INPUT))?;
    let options = FminuncOptions::from_struct(options_struct.as_ref(), guess.values.len())
        .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_ARGUMENT))?;
    let outcome = minimize(
        &function,
        guess.values,
        &guess.shape,
        guess.scalar,
        &options,
        &mut cooperation,
    )
    .await
    .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_INPUT))?;
    finalize(
        outcome,
        &guess.shape,
        guess.scalar,
        &options,
        &mut cooperation,
    )
    .await
}

fn parse_options(value: Option<&Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
}

type DisplayMode = runmat_builtins::OptimDisplay;

fn parse_display(text: &str) -> BuiltinResult<DisplayMode> {
    DisplayMode::parse(text).ok_or_else(|| {
        fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option Display must be 'off', 'iter', 'notify', or 'final', got '{text}'"),
        )
    })
}

#[derive(Debug, Clone)]
struct FminuncOptions {
    tol_x: f64,
    tol_fun: f64,
    max_iter: usize,
    max_fun_evals: usize,
    display: DisplayMode,
    specify_objective_gradient: bool,
}

impl FminuncOptions {
    fn from_struct(options: Option<&StructValue>, variables: usize) -> BuiltinResult<Self> {
        let fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct(options);
        let display =
            parse_display(&fields.text(runmat_builtins::OptimOptionField::Display, "off")?)?;
        let algorithm =
            fields.text(runmat_builtins::OptimOptionField::Algorithm, "quasi-newton")?;
        if !matches!(
            runmat_builtins::OptimAlgorithm::parse(&algorithm),
            Some(
                runmat_builtins::OptimAlgorithm::QuasiNewton
                    | runmat_builtins::OptimAlgorithm::Bfgs
            )
        ) {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "fminunc: Algorithm must be 'quasi-newton' or 'bfgs'",
            ));
        }
        let tol_x = fields.real(NAME, runmat_builtins::OptimOptionField::TolX, DEFAULT_TOL_X)?;
        let tol_fun = fields.real(
            NAME,
            runmat_builtins::OptimOptionField::TolFun,
            DEFAULT_TOL_FUN,
        )?;
        if tol_x <= 0.0 || tol_fun <= 0.0 {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "options TolX and TolFun must be positive",
            ));
        }
        let max_iter = bounded_option_usize(
            &fields,
            runmat_builtins::OptimOptionField::MaxIter,
            DEFAULT_MAX_ITER,
            MAX_ITER_LIMIT,
        )?
        .max(1);
        let max_fun_evals = bounded_option_usize(
            &fields,
            runmat_builtins::OptimOptionField::MaxFunEvals,
            DEFAULT_MAX_FUN_EVALS,
            MAX_FUN_EVAL_LIMIT,
        )?
        .max(1);
        let specify_objective_gradient = fields
            .get(runmat_builtins::OptimOptionField::SpecifyObjectiveGradient)
            .map(|value| {
                bool_value(
                    runmat_builtins::OptimOptionField::SpecifyObjectiveGradient.name(),
                    value,
                )
            })
            .transpose()?
            .unwrap_or(false);
        validate_problem_size(variables)?;
        Ok(Self {
            tol_x,
            tol_fun,
            max_iter,
            max_fun_evals,
            display,
            specify_objective_gradient,
        })
    }
}

fn bounded_option_usize(
    fields: &crate::builtins::math::optim::common::ScalarOptionFields<'_>,
    field: runmat_builtins::OptimOptionField,
    default: usize,
    maximum: usize,
) -> BuiltinResult<usize> {
    let raw = fields.get(field);
    let field = field.name();
    if let Some(value) = raw {
        if let Some(integer) = tensor::scalar_integer_value(value) {
            let Some(value) = integer.try_to_usize() else {
                return Err(fminunc_error_with_detail(
                    &FMINUNC_ERROR_INVALID_ARGUMENT,
                    format!("option {field} must be non-negative"),
                ));
            };
            if value > maximum {
                return Err(fminunc_error_with_detail(
                    &FMINUNC_ERROR_INVALID_ARGUMENT,
                    format!("option {field} must be no greater than {maximum}"),
                ));
            }
            return Ok(value);
        }
    }
    let value =
        crate::builtins::math::optim::common::option_f64_value(NAME, raw, field, default as f64)?;
    if value < 0.0 {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be non-negative"),
        ));
    }
    if value > maximum as f64 {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be no greater than {maximum}"),
        ));
    }
    if value.fract() != 0.0 {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be an integer"),
        ));
    }
    Ok(value.floor() as usize)
}

fn validate_problem_size(variables: usize) -> BuiltinResult<()> {
    if variables > MAX_DENSE_BFGS_VARIABLES {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_INPUT,
            format!(
                "problem has {variables} variables; current dense BFGS implementation supports at most {MAX_DENSE_BFGS_VARIABLES}"
            ),
        ));
    }
    algorithm::dense_len(variables)?;
    Ok(())
}

fn bool_value(field: &str, value: &Value) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                format!("option {field} must be logical 0 or 1"),
            )),
        };
    }
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(n) => bool_from_number(field, *n),
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            bool_from_number(field, tensor::tensor_value_f64(tensor, 0))
        }
        Value::String(s) => bool_from_text(field, s),
        Value::StringArray(sa) if sa.data.len() == 1 => bool_from_text(field, &sa.data[0]),
        Value::CharArray(chars) if chars.rows == 1 => {
            let text: String = chars.data.iter().collect();
            bool_from_text(field, &text)
        }
        other => Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be logical, got {other:?}"),
        )),
    }
}

fn bool_from_number(field: &str, value: f64) -> BuiltinResult<bool> {
    if !value.is_finite() {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be finite"),
        ));
    }
    if value == 0.0 {
        Ok(false)
    } else if value == 1.0 {
        Ok(true)
    } else {
        Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be logical 0 or 1"),
        ))
    }
}

fn bool_from_text(field: &str, value: &str) -> BuiltinResult<bool> {
    match runmat_builtins::OptimSwitch::parse(value) {
        Some(value) => Ok(value.enabled()),
        None => Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!(
                "option {field} must be 'on' or 'off', got '{}'",
                value.trim().to_ascii_lowercase()
            ),
        )),
    }
}

mod algorithm;
use algorithm::{finalize, minimize};

#[cfg(test)]
mod tests;
