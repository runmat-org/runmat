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
use crate::builtins::math::optim::common::{call_function_in_context, vector_to_value};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fminunc";
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
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    fminunc_in_runtime(runtime.as_ref(), function, x0, rest).await
}

#[cfg(test)]
async fn fminunc_builtin(function: Value, x0: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    legacy_test_value(fminunc_in_runtime(None, function, x0, rest.into()).await?)
}

async fn fminunc_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    x0: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME, "finite host-memory execution is unsupported; this optimizer has no complete workspace and factorization bound")?;
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
        runtime,
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
    let options = FminuncOptions::from_struct(
        options_struct.as_ref().map(|options| options.structure()),
        guess.values.len(),
    )
    .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_ARGUMENT))?;
    let callback_fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct(
        options_struct.as_ref().map(|value| value.structure()),
    );
    if options.trust_region_fallback {
        reporting::trust_region_fallback(runtime)?;
    }
    if options.trust_region.is_some() {
        let outcome = trust_region::minimize(
            &function,
            guess.values,
            &guess.shape,
            guess.scalar,
            &options,
            callback_fields.exact(runmat_builtins::OptimOptionField::OutputFcn),
            &mut cooperation,
        )
        .await
        .map_err(|error| fminunc_map_error(error, &FMINUNC_ERROR_INVALID_INPUT))?;
        return trust_region::finalize(
            outcome,
            &guess.shape,
            guess.scalar,
            &options,
            &mut cooperation,
        )
        .await;
    }
    let outcome = minimize(
        &function,
        guess.values,
        &guess.shape,
        guess.scalar,
        &options,
        callback_fields.get(runmat_builtins::OptimOptionField::OutputFcn),
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

fn publish_result(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let sequence = match value {
        Value::OutputList(values) => runmat_value::ValueSequence::from_output_storage(values),
        value => runmat_value::ValueSequence::single(value),
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    match runtime {
        Some(runtime) => crate::execution::RootedValueSequence::in_context(sequence, runtime),
        None => crate::execution::RootedValueSequence::new(sequence)
            .map_err(|source| source.into_runtime_error()),
    }
}

#[cfg(test)]
fn legacy_test_value(result: crate::execution::RootedValueSequence) -> BuiltinResult<Value> {
    result.with_owned_sequence(|sequence| {
        let values = sequence.into_output_storage();
        if crate::output_count::current_output_count().is_some() {
            Ok(Value::OutputList(values))
        } else {
            crate::sequence::into_first_output(values)?.ok_or_else(|| {
                fminunc_error_with_detail(&FMINUNC_ERROR_INVALID_INPUT, "missing test result")
            })
        }
    })
}

fn parse_options(value: Option<&Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Tensor(value)) if value.is_empty() => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
}

type DisplayMode = runmat_builtins::OptimDisplay;

fn parse_display(text: &str) -> BuiltinResult<DisplayMode> {
    DisplayMode::parse(text)
        .filter(|display| runmat_builtins::OptimSolver::Fminunc.accepts_display(*display))
        .ok_or_else(|| {
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
    function_tolerance: f64,
    trust_region: Option<trust_region::Controls>,
    trust_region_fallback: bool,
    update: runmat_builtins::OptimQuasiNewtonUpdate,
    objective_limit: f64,
    parallel: crate::execution::CallbackParallelMode,
    scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling,
}

impl FminuncOptions {
    fn from_struct(options: Option<&StructValue>, variables: usize) -> BuiltinResult<Self> {
        let fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct(options);
        let display =
            parse_display(&fields.text(runmat_builtins::OptimOptionField::Display, "off")?)?;
        use runmat_builtins::{OptimAlgorithm, OptimOptionField as Field};
        let algorithm = fields.text(Field::Algorithm, "quasi-newton")?;
        let algorithm = OptimAlgorithm::parse(&algorithm)
            .filter(|algorithm| {
                matches!(
                    algorithm,
                    OptimAlgorithm::QuasiNewton
                        | OptimAlgorithm::Bfgs
                        | OptimAlgorithm::TrustRegion
                )
            })
            .ok_or_else(|| {
                fminunc_error_with_detail(
                    &FMINUNC_ERROR_INVALID_ARGUMENT,
                    "Algorithm must be 'quasi-newton', 'bfgs', or 'trust-region'",
                )
            })?;
        let specify_objective_gradient = fields
            .exact(Field::SpecifyObjectiveGradient)
            .map(|value| bool_value(Field::SpecifyObjectiveGradient.name(), value))
            .transpose()?
            .unwrap_or(false);
        let actual_tr = algorithm == OptimAlgorithm::TrustRegion && specify_objective_gradient;
        let tol_x = crate::builtins::math::optim::common::option_f64_value(
            NAME,
            fields.exact(Field::TolX),
            "StepTolerance",
            DEFAULT_TOL_X,
        )?;
        let legacy = fields.exact(Field::TolFun);
        let modern_field = if actual_tr {
            Field::FunctionTolerance
        } else {
            Field::OptimalityTolerance
        };
        let modern = fields.exact(modern_field);
        if let (Some(legacy), Some(modern)) = (legacy, modern) {
            let legacy = crate::builtins::math::optim::common::option_f64_value(
                NAME,
                Some(legacy),
                "TolFun",
                DEFAULT_TOL_FUN,
            )?;
            let modern = crate::builtins::math::optim::common::option_f64_value(
                NAME,
                Some(modern),
                modern_field.name(),
                DEFAULT_TOL_FUN,
            )?;
            if legacy != modern {
                return Err(fminunc_error_with_detail(
                    &FMINUNC_ERROR_INVALID_ARGUMENT,
                    format!(
                        "TolFun conflicts with {} for the effective algorithm",
                        modern_field.name()
                    ),
                ));
            }
        }
        let tol_fun = crate::builtins::math::optim::common::option_f64_value(
            NAME,
            fields
                .exact(Field::OptimalityTolerance)
                .or(if actual_tr { None } else { legacy }),
            "OptimalityTolerance",
            DEFAULT_TOL_FUN,
        )?;
        // FunctionTolerance has no QN stopping role. Dormant TR operands are
        // not parsed merely because an imported/default struct retains them.
        let function_tolerance = if actual_tr {
            crate::builtins::math::optim::common::option_f64_value(
                NAME,
                modern.or(legacy),
                "FunctionTolerance",
                DEFAULT_TOL_FUN,
            )?
        } else {
            DEFAULT_TOL_FUN
        };
        if tol_x < 0.0 || tol_fun < 0.0 || function_tolerance < 0.0 {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "step, function and optimality tolerances must be nonnegative",
            ));
        }
        let max_iter =
            bounded_option_usize(&fields, Field::MaxIter, DEFAULT_MAX_ITER, MAX_ITER_LIMIT)?;
        let evaluation_default = if actual_tr {
            variables.checked_mul(100).ok_or_else(|| {
                fminunc_error_with_detail(
                    &FMINUNC_ERROR_INVALID_ARGUMENT,
                    "default MaxFunEvals extent overflow",
                )
            })?
        } else {
            DEFAULT_MAX_FUN_EVALS
        };
        let max_fun_evals = bounded_option_usize(
            &fields,
            Field::MaxFunEvals,
            evaluation_default,
            MAX_FUN_EVAL_LIMIT,
        )?;
        if max_fun_evals == 0 {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "MaxFunEvals must be positive",
            ));
        }
        let update = if actual_tr {
            runmat_builtins::OptimQuasiNewtonUpdate::Bfgs
        } else {
            quasi_newton::controls(&fields)?
        };
        let objective_limit = fields
            .exact(Field::ObjectiveLimit)
            .map(|value| {
                crate::builtins::math::optim::common::quasi_newton_controls::objective_limit(
                    NAME, value,
                )
            })
            .transpose()?
            .unwrap_or(-1e20);
        let parallel = if !actual_tr && !specify_objective_gradient {
            fields
                .exact(Field::UseParallel)
                .map(|value| {
                    crate::builtins::math::optim::common::quasi_newton_controls::parallel_mode(
                        NAME, value,
                    )
                })
                .transpose()?
                .unwrap_or_default()
        } else {
            crate::execution::CallbackParallelMode::Off
        };
        let scaling = crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling::from_fields(NAME,&fields,!specify_objective_gradient)?;
        scaling.validate_variables(NAME, variables)?;
        let trust_region = if actual_tr {
            Some(trust_region::Controls::from_fields(&fields, variables)?)
        } else {
            if update.dense() {
                validate_problem_size(variables)?;
            }
            None
        };
        Ok(Self {
            tol_x,
            tol_fun,
            max_iter,
            max_fun_evals,
            display,
            specify_objective_gradient,
            function_tolerance,
            trust_region,
            update,
            objective_limit,
            parallel,
            scaling,
            trust_region_fallback: algorithm == OptimAlgorithm::TrustRegion
                && !specify_objective_gradient,
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
        Value::String(runmat_value::StringElement::Text(s)) => bool_from_text(field, s),
        Value::StringArray(sa) if sa.data().len() == 1 => {
            let text = sa.data()[0].as_str().ok_or_else(|| {
                fminunc_error_with_detail(
                    &FMINUNC_ERROR_INVALID_ARGUMENT,
                    format!("option {field} must be nonmissing text"),
                )
            })?;
            bool_from_text(field, text)
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let text = runmat_value::decode_utf16_code_units(chars.data()).map_err(|error| {
                fminunc_error_with_detail(
                    &FMINUNC_ERROR_INVALID_ARGUMENT,
                    format!("option {field} contains {error}"),
                )
            })?;
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
mod evaluation;
pub(in crate::builtins::math::optim) mod compiled;
mod quasi_newton;
mod reporting;
mod trust_region;
use algorithm::{finalize, minimize};

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "fminunc/output_tests.rs"]
mod output_tests;
