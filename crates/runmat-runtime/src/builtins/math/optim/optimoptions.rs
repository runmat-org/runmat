//! MATLAB-compatible `optimoptions` options struct builder.

use runmat_builtins::{
    BuiltinErrorDescriptor, OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver as Solver,
    OptimSwitch, OPTIMOPTIONS_ERROR_FLOW, OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
    OPTIMOPTIONS_ERROR_INVALID_OPTION_NAME, OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    OPTIMOPTIONS_ERROR_INVALID_SOLVER, OPTIMOPTIONS_ERROR_MISSING_OPTION_VALUE,
    OPTIMOPTIONS_ERROR_UNKNOWN_OPTION,
    OPTIMOPTIONS_INTEGER_OPTION_EXTENSION as INTEGER_OPTION_EXTENSION,
    OPTIMOPTIONS_RESIDENT_OPTION_EXTENSION as RESIDENT_OPTION_EXTENSION,
};
pub use runmat_builtins::{
    OPTIMOPTIONS_DESCRIPTOR, OPTIMOPTIONS_EXTENSIONS as EXTENSIONS,
    OPTIMOPTIONS_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::CharArray;
use runmat_value::{LogicalArray, StructValue, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "optimoptions";
mod preflight;
mod state;
use state::{OptionKey, OptionState};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::optimoptions")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "optimoptions",
    op_kind: GpuOpKind::Custom("optimization-options"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host metadata construction. gpuArray option values are gathered before validation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::optimoptions")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "optimoptions",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Option struct construction is host metadata work and terminates fusion planning.",
};

#[runtime_builtin(
    name = "optimoptions",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimoptions"
)]
async fn optimoptions_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &OPTIMOPTIONS_DESCRIPTOR,
        &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    ensure_optimoptions_extensions(rest.view())?;
    preflight::arguments(rest.view())?;
    let gathered = rest
        .try_map_preserving_roots_async(rest.len(), runtime.as_ref(), |index| {
            gather_if_needed_async(&rest[index])
        })
        .await
        .map_err(|err| {
            remap_optimoptions_flow(&OPTIMOPTIONS_ERROR_FLOW, err, |source| {
                format!("optimoptions: {}", source.message())
            })
        })?;
    drop(rest);
    let first = gathered.first().ok_or_else(|| {
        optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
            "optimoptions: expected a solver name or options struct",
        )
    })?;

    let mut solver;
    let explicit_solver;
    let mut options = match first {
        Value::Struct(existing) => {
            explicit_solver = false;
            solver = solver_from_options(existing)?;
            canonicalize_existing_options(runtime.as_ref(), existing, solver)?
        }
        other => {
            explicit_solver = true;
            solver = parse_solver(other)?;
            default_options(runtime.as_ref(), solver)?
        }
    };

    let mut queue = gathered.iter().skip(1);
    while let Some(arg) = queue.next() {
        match arg {
            Value::Struct(existing) => {
                if explicit_solver {
                    let next_solver = solver_from_options(existing)?;
                    let skip_defaults_from = match next_solver {
                        Solver::Generic => None,
                        other => Some(other),
                    };
                    apply_struct_fields(
                        runtime.as_ref(),
                        existing,
                        &mut options,
                        solver,
                        false,
                        skip_defaults_from,
                    )?;
                    options.set_solver(state::text(runtime.as_ref(), solver.name())?);
                    continue;
                } else {
                    let next_solver = solver_from_options(existing)?;
                    let skip_defaults_from;
                    if next_solver != Solver::Generic && next_solver != solver {
                        options = if solver == Solver::Generic {
                            merge_generic_into_defaults(runtime.as_ref(), &options, next_solver)?
                        } else {
                            default_options(runtime.as_ref(), next_solver)?
                        };
                        solver = next_solver;
                        skip_defaults_from = Some(next_solver);
                    } else if next_solver != Solver::Generic {
                        solver = next_solver;
                        skip_defaults_from = Some(next_solver);
                    } else {
                        skip_defaults_from = None;
                    }
                    apply_struct_fields(
                        runtime.as_ref(),
                        existing,
                        &mut options,
                        solver,
                        true,
                        skip_defaults_from,
                    )?;
                    continue;
                }
            }
            name_value => {
                let name = expect_string_scalar(
                    name_value,
                    "optimoptions: option names must be character vectors or string scalars",
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_NAME,
                )?;
                let value = queue.next().ok_or_else(|| {
                    optimoptions_error_with(
                        &OPTIMOPTIONS_ERROR_MISSING_OPTION_VALUE,
                        format!("optimoptions: missing value for option '{name}'"),
                    )
                })?;
                set_option_field(runtime.as_ref(), &mut options, solver, &name, value)?;
            }
        }
    }

    options.finish(runtime.as_ref())
}

fn ensure_optimoptions_extensions(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for (index, value) in args.iter().enumerate() {
        let is_payload = index > 0 || matches!(value, Value::Struct(_));
        if !is_payload {
            continue;
        }
        if crate::builtins::common::validation::value_contains_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_OPTION_EXTENSION,
                NAME,
            )?;
        }
        if crate::builtins::common::validation::value_contains_explicit_gpu(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &RESIDENT_OPTION_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

fn optimoptions_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_optimoptions_flow<F>(
    error: &'static BuiltinErrorDescriptor,
    err: RuntimeError,
    message: F,
) -> RuntimeError
where
    F: FnOnce(&RuntimeError) -> String,
{
    let mut builder = build_runtime_error(message(&err))
        .with_builtin(NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn parse_solver(value: &Value) -> BuiltinResult<Solver> {
    let text = expect_string_scalar(
        value,
        "optimoptions: solver must be a character vector or string scalar",
        &OPTIMOPTIONS_ERROR_INVALID_SOLVER,
    )?;
    parse_solver_name(&text)
}

fn parse_solver_name(text: &str) -> BuiltinResult<Solver> {
    Solver::parse(text).ok_or_else(|| {
        optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_SOLVER,
            format!(
                "optimoptions: unsupported solver '{}'",
                text.trim().to_ascii_lowercase()
            ),
        )
    })
}

fn solver_from_options(options: &StructValue) -> BuiltinResult<Solver> {
    let Some(value) = lookup_case_insensitive(options, "Solver") else {
        return Ok(Solver::Generic);
    };
    parse_solver(value)
}

fn default_options(
    runtime: Option<&crate::context::RuntimeContext>,
    solver: Solver,
) -> BuiltinResult<OptionState> {
    let mut out = OptionState::new();
    if solver != Solver::Generic {
        out.set_solver(state::text(runtime, solver.name())?);
    }
    match solver {
        Solver::Coneprog => {
            out.set(
                OptimOptionField::Algorithm,
                state::text(runtime, "interior-point")?,
            );
            out.set(OptimOptionField::TolX, Value::Num(1.0e-7));
            out.set(OptimOptionField::TolFun, Value::Num(1.0e-7));
            out.set(OptimOptionField::MaxIter, Value::Num(200.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(20000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Fminbnd => {
            out.set(OptimOptionField::TolX, Value::Num(1.0e-4));
            out.set(OptimOptionField::MaxIter, Value::Num(500.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(500.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Fminunc => {
            out.set(
                OptimOptionField::Algorithm,
                state::text(runtime, "quasi-newton")?,
            );
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::TolFun, Value::Num(1.0e-6));
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(40000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
            out.set(
                OptimOptionField::SpecifyObjectiveGradient,
                Value::Bool(false),
            );
        }
        Solver::Fzero => {
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(500.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Fsolve => {
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::TolFun, Value::Num(1.0e-6));
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(40000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Lsqcurvefit => {
            out.set(
                OptimOptionField::Algorithm,
                state::text(runtime, "levenberg-marquardt")?,
            );
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::TolFun, Value::Num(1.0e-6));
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(40000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Lsqnonlin => {
            out.set(
                OptimOptionField::Algorithm,
                state::text(runtime, "trust-region-reflective")?,
            );
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::TolFun, Value::Num(1.0e-6));
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(40000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Generic => {}
    }
    Ok(out)
}

fn canonicalize_existing_options(
    runtime: Option<&crate::context::RuntimeContext>,
    existing: &StructValue,
    solver: Solver,
) -> BuiltinResult<OptionState> {
    let mut out = if solver == Solver::Generic {
        OptionState::new()
    } else {
        default_options(runtime, solver)?
    };
    apply_struct_fields(runtime, existing, &mut out, solver, true, None)?;
    Ok(out)
}

fn merge_generic_into_defaults(
    runtime: Option<&crate::context::RuntimeContext>,
    generic: &OptionState,
    solver: Solver,
) -> BuiltinResult<OptionState> {
    let mut out = default_options(runtime, solver)?;
    for (key, value) in generic.fields() {
        let OptionKey::Field(field) = key else {
            continue;
        };
        if !solver.accepts(field) {
            continue;
        }
        let normalized = normalized_option_value(runtime, solver, field, value);
        if field == OptimOptionField::Display && normalized.is_err() {
            continue;
        }
        out.set(field, normalized?);
    }
    Ok(out)
}

fn apply_struct_fields(
    runtime: Option<&crate::context::RuntimeContext>,
    source: &StructValue,
    target: &mut OptionState,
    solver: Solver,
    copy_solver_field: bool,
    skip_defaults_from: Option<Solver>,
) -> BuiltinResult<()> {
    let source_defaults = skip_defaults_from
        .map(|solver| default_options(runtime, solver))
        .transpose()?;
    for (key, value) in &source.fields {
        if key.eq_ignore_ascii_case("Solver") {
            if !copy_solver_field {
                continue;
            }
            let parsed = parse_solver(value)?;
            target.set_solver(state::text(runtime, parsed.name())?);
            continue;
        }
        let field = admitted_option_field(solver, key)?;
        let normalized = normalized_option_value(runtime, solver, field, value)?;
        if let Some(defaults) = &source_defaults {
            if defaults
                .get(field)
                .is_some_and(|default| matches_source_default(field, default, &normalized))
            {
                continue;
            }
        }
        target.set(field, normalized);
    }
    Ok(())
}

fn matches_source_default(field: OptimOptionField, default: &Value, normalized: &Value) -> bool {
    if default == normalized {
        return true;
    }
    if !matches!(
        field,
        OptimOptionField::MaxIter | OptimOptionField::MaxFunEvals
    ) {
        return false;
    }
    match (default, normalized) {
        (Value::Num(default), Value::Int(integer)) => {
            crate::builtins::common::validation::integer_is_exact_f64(integer)
                && *default == integer.to_f64()
        }
        _ => false,
    }
}

fn set_option_field(
    runtime: Option<&crate::context::RuntimeContext>,
    options: &mut OptionState,
    solver: Solver,
    name: &str,
    value: &Value,
) -> BuiltinResult<()> {
    let field = admitted_option_field(solver, name)?;
    options.set(
        field,
        normalized_option_value(runtime, solver, field, value)?,
    );
    Ok(())
}

fn admitted_option_field(solver: Solver, name: &str) -> BuiltinResult<OptimOptionField> {
    OptimOptionField::parse(name)
        .filter(|field| solver.accepts(*field))
        .ok_or_else(|| {
            optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_UNKNOWN_OPTION,
                format!(
                    "optimoptions: option '{}' is not supported for {}",
                    name,
                    solver_label(solver)
                ),
            )
        })
}

fn normalized_option_value(
    runtime: Option<&crate::context::RuntimeContext>,
    solver: Solver,
    field: OptimOptionField,
    value: &Value,
) -> BuiltinResult<Value> {
    prepared_option_value(solver, field, value)?.into_value(runtime)
}

fn prepared_option_value(
    solver: Solver,
    field: OptimOptionField,
    value: &Value,
) -> BuiltinResult<state::PreparedOptionValue> {
    use state::PreparedOptionValue;
    match field {
        OptimOptionField::TolX | OptimOptionField::TolFun => Ok(PreparedOptionValue::Number(
            positive_finite_scalar(field.name(), value)?,
        )),
        OptimOptionField::MaxIter | OptimOptionField::MaxFunEvals => {
            let count = positive_integer_scalar(field.name(), value)?;
            Ok(tensor::scalar_integer_value(value)
                .map(PreparedOptionValue::Integer)
                .unwrap_or_else(|| PreparedOptionValue::Number(count as f64)))
        }
        OptimOptionField::Display => {
            Ok(PreparedOptionValue::Display(display_value(solver, value)?))
        }
        OptimOptionField::Algorithm => Ok(PreparedOptionValue::Algorithm(algorithm_value(
            solver, value,
        )?)),
        OptimOptionField::SpecifyObjectiveGradient => Ok(PreparedOptionValue::Logical(
            logical_value(field.name(), value)?,
        )),
    }
}

fn solver_label(solver: Solver) -> &'static str {
    match solver {
        Solver::Generic => "optimization solvers",
        _ => solver.name(),
    }
}

fn positive_finite_scalar(field: &str, value: &Value) -> BuiltinResult<f64> {
    let parsed = numeric_scalar(field, value)?;
    if parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be a finite positive scalar"),
        ))
    }
}

fn positive_integer_scalar(field: &str, value: &Value) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        if let Some(parsed) = integer.try_to_usize() {
            if parsed == 0 {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    format!("optimoptions: option {field} must be a finite positive scalar"),
                ));
            }
            return Ok(parsed);
        }
        if integer.try_to_i64().is_some_and(|value| value < 0) {
            return Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!("optimoptions: option {field} must be a finite positive scalar"),
            ));
        }
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} is too large"),
        ));
    }
    let parsed = positive_finite_scalar(field, value)?;
    if parsed.fract() != 0.0 {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be an integer scalar"),
        ));
    }
    if parsed >= 2f64.powi(usize::BITS as i32) {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} is too large"),
        ));
    }
    Ok(parsed as usize)
}

fn numeric_scalar(field: &str, value: &Value) -> BuiltinResult<f64> {
    if crate::builtins::common::validation::value_contains_native_integer_class(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
    {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: integer option {field} must be exactly representable as double"),
        ));
    }
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            return Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!("optimoptions: option {field} must be a numeric scalar, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be finite"),
        ))
    }
}

fn logical_value(field: &str, value: &Value) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!("optimoptions: option {field} must be logical 0 or 1"),
            )),
        };
    }
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        Value::Num(n) => logical_from_number(field, *n),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            logical_from_number(field, tensor::tensor_value_f64(tensor, 0))
        }
        Value::String(s) => logical_from_text(field, s),
        Value::StringArray(sa) if sa.data().len() == 1 => logical_from_text(field, &sa.data()[0]),
        Value::CharArray(array) if array.rows == 1 => {
            let text = runmat_value::decode_utf16_code_units(array.data()).map_err(|error| {
                optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    format!("optimoptions: option {field} contains {error}"),
                )
            })?;
            logical_from_text(field, &text)
        }
        other => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be logical, got {other:?}"),
        )),
    }
}

fn logical_from_number(field: &str, value: f64) -> BuiltinResult<bool> {
    if value == 0.0 {
        Ok(false)
    } else if value == 1.0 {
        Ok(true)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be logical 0 or 1"),
        ))
    }
}

fn logical_from_text(field: &str, value: &str) -> BuiltinResult<bool> {
    OptimSwitch::parse(value)
        .map(OptimSwitch::enabled)
        .ok_or_else(|| {
            optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!(
                    "optimoptions: option {field} must be 'on' or 'off', got '{}'",
                    value.trim().to_ascii_lowercase()
                ),
            )
        })
}

fn display_value(solver: Solver, value: &Value) -> BuiltinResult<OptimDisplay> {
    let display = expect_string_scalar(
        value,
        "optimoptions: Display must be a character vector or string scalar",
        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?
    .trim()
    .to_ascii_lowercase();
    if let Some(display) =
        OptimDisplay::parse(&display).filter(|display| solver.accepts_display(*display))
    {
        Ok(display)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "optimoptions: unsupported Display '{}' for {}",
                display,
                solver_label(solver)
            ),
        ))
    }
}

fn algorithm_value(solver: Solver, value: &Value) -> BuiltinResult<OptimAlgorithm> {
    let algorithm = expect_string_scalar(
        value,
        "optimoptions: Algorithm must be a character vector or string scalar",
        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?
    .trim()
    .to_ascii_lowercase();
    if let Some(algorithm) =
        OptimAlgorithm::parse(&algorithm).filter(|algorithm| solver.accepts_algorithm(*algorithm))
    {
        Ok(algorithm)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "optimoptions: unsupported Algorithm '{}' for {}",
                algorithm,
                solver_label(solver)
            ),
        ))
    }
}

fn expect_string_scalar(
    value: &Value,
    context: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.as_str().to_owned()),
        Value::StringArray(sa) if sa.data().len() == 1 => Ok(sa.data()[0].as_str().to_owned()),
        Value::CharArray(array) if array.rows == 1 => {
            runmat_value::decode_utf16_code_units(array.data()).map_err(|decode_error| {
                optimoptions_error_with(error, format!("{context}: {decode_error}"))
            })
        }
        _ => Err(optimoptions_error_with(error, context)),
    }
}

fn lookup_case_insensitive<'a>(options: &'a StructValue, name: &str) -> Option<&'a Value> {
    options
        .fields
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::call_builtin_async;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntValue, IntegerStorage, Tensor, Utf16CodeUnit};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn option_text_decodes_surrogate_pairs_and_rejects_unpaired_units() {
        let valid = Value::CharArray(
            CharArray::from_code_units(
                vec![
                    Utf16CodeUnit::from_raw(0xd83d),
                    Utf16CodeUnit::from_raw(0xde80),
                ],
                1,
                2,
            )
            .expect("valid character geometry"),
        );
        assert_eq!(
            expect_string_scalar(&valid, "option", &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE)
                .expect("surrogate pair must decode"),
            "🚀"
        );

        let invalid = Value::CharArray(
            CharArray::from_code_units(vec![Utf16CodeUnit::from_raw(0xd83d)], 1, 1)
                .expect("valid character geometry"),
        );
        let error =
            expect_string_scalar(&invalid, "option", &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE)
                .expect_err("unpaired surrogate must retain the option error");
        assert_eq!(
            error.identifier(),
            OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE.identifier
        );
    }

    fn run_optimoptions(rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?;
        let rooted = block_on(optimoptions_builtin(None, rest.into()))?;
        // Normalized option output contains only scalar numeric/logical/text metadata.
        Ok(rooted.value().clone())
    }

    fn run_call_builtin(
        name: &str,
        args: &[Value],
    ) -> BuiltinResult<crate::execution::RootedValue> {
        block_on(call_builtin_async(name, args))
    }

    fn struct_result(value: Value) -> StructValue {
        match value {
            Value::Struct(options) => options,
            other => panic!("expected struct, got {other:?}"),
        }
    }

    fn num_field(options: &StructValue, field: &str) -> f64 {
        match options.fields.get(field) {
            Some(Value::Num(value)) => *value,
            other => panic!("expected numeric field {field}, got {other:?}"),
        }
    }

    fn string_field<'a>(options: &'a StructValue, field: &str) -> &'a str {
        match options.fields.get(field) {
            Some(Value::String(value)) => value.as_str(),
            other => panic!("expected string field {field}, got {other:?}"),
        }
    }

    fn bool_field(options: &StructValue, field: &str) -> bool {
        match options.fields.get(field) {
            Some(Value::Bool(value)) => *value,
            other => panic!("expected bool field {field}, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_descriptor_signatures_and_errors_cover_core_forms() {
        let labels: Vec<&str> = OPTIMOPTIONS_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "options = optimoptions(solver)",
                "options = optimoptions(solver, name, value, ...)",
                "options = optimoptions(oldopts, name, value, ...)",
            ]
        );

        let codes: Vec<&str> = OPTIMOPTIONS_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec![
                "RM.OPTIMOPTIONS.INVALID_ARGUMENT",
                "RM.OPTIMOPTIONS.INVALID_SOLVER",
                "RM.OPTIMOPTIONS.INVALID_OPTION_NAME",
                "RM.OPTIMOPTIONS.MISSING_OPTION_VALUE",
                "RM.OPTIMOPTIONS.UNKNOWN_OPTION",
                "RM.OPTIMOPTIONS.INVALID_OPTION_VALUE",
                "RM.OPTIMOPTIONS.FLOW",
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fminbnd_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("fminbnd")]).expect("optimoptions fminbnd"),
        );
        assert_eq!(string_field(&options, "Solver"), "fminbnd");
        assert_eq!(num_field(&options, "TolX"), 1.0e-4);
        assert_eq!(num_field(&options, "MaxIter"), 500.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 500.0);
        assert_eq!(string_field(&options, "Display"), "notify");
        assert!(!options.fields.contains_key("TolFun"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_coneprog_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("coneprog")]).expect("optimoptions coneprog"),
        );
        assert_eq!(string_field(&options, "Solver"), "coneprog");
        assert_eq!(string_field(&options, "Algorithm"), "interior-point");
        assert_eq!(num_field(&options, "TolX"), 1.0e-7);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-7);
        assert_eq!(num_field(&options, "MaxIter"), 200.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fzero_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("fzero")]).expect("optimoptions fzero"),
        );
        assert_eq!(string_field(&options, "Solver"), "fzero");
        assert_eq!(num_field(&options, "TolX"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxIter"), 400.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 500.0);
        assert_eq!(string_field(&options, "Display"), "off");
        assert!(!options.fields.contains_key("TolFun"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fsolve_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("fsolve")]).expect("optimoptions fsolve"),
        );
        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-6);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxIter"), 400.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 40000.0);
        assert_eq!(string_field(&options, "Display"), "off");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fminunc_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("fminunc")]).expect("optimoptions fminunc"),
        );
        assert_eq!(string_field(&options, "Solver"), "fminunc");
        assert_eq!(string_field(&options, "Algorithm"), "quasi-newton");
        assert_eq!(num_field(&options, "TolX"), 1.0e-6);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxIter"), 400.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 40000.0);
        assert_eq!(string_field(&options, "Display"), "off");
        assert!(!bool_field(&options, "SpecifyObjectiveGradient"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_fminunc_accepts_gradient_and_algorithm_options() {
        let options = struct_result(
            run_optimoptions(vec![
                Value::from("fminunc"),
                Value::from("SpecifyObjectiveGradient"),
                Value::from("on"),
                Value::from("Algorithm"),
                Value::from("bfgs"),
                Value::from("Display"),
                Value::from("notify"),
            ])
            .expect("optimoptions fminunc"),
        );
        assert!(bool_field(&options, "SpecifyObjectiveGradient"));
        assert_eq!(string_field(&options, "Algorithm"), "bfgs");
        assert_eq!(string_field(&options, "Display"), "notify");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_lsqcurvefit_defaults_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![Value::from("lsqcurvefit")]).expect("optimoptions lsqcurvefit"),
        );
        assert_eq!(string_field(&options, "Solver"), "lsqcurvefit");
        assert_eq!(string_field(&options, "Algorithm"), "levenberg-marquardt");
        assert_eq!(num_field(&options, "TolX"), 1.0e-6);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxIter"), 400.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 40000.0);
        assert_eq!(string_field(&options, "Display"), "off");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_lsqcurvefit_accepts_modern_tolerance_aliases_and_algorithm() {
        let options = struct_result(
            run_optimoptions(vec![
                Value::from("lsqcurvefit"),
                Value::from("FunctionTolerance"),
                Value::Num(1.0e-9),
                Value::from("StepTolerance"),
                Value::Num(1.0e-8),
                Value::from("MaxFunctionEvaluations"),
                Value::Num(7.0),
                Value::from("Algorithm"),
                Value::from("trust-region-reflective"),
            ])
            .expect("optimoptions lsqcurvefit aliases"),
        );
        assert_eq!(num_field(&options, "TolFun"), 1.0e-9);
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxFunEvals"), 7.0);
        assert!(!options.fields.contains_key("MaxFunctionEvaluations"));
        assert_eq!(
            string_field(&options, "Algorithm"),
            "trust-region-reflective"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_lsqnonlin_defaults_and_aliases_match_solver() {
        let options = struct_result(
            run_optimoptions(vec![
                Value::from("lsqnonlin"),
                Value::from("FunctionTolerance"),
                Value::Num(1.0e-9),
                Value::from("StepTolerance"),
                Value::Num(1.0e-8),
                Value::from("Algorithm"),
                Value::from("levenberg-marquardt"),
            ])
            .expect("optimoptions lsqnonlin"),
        );
        assert_eq!(string_field(&options, "Solver"), "lsqnonlin");
        assert_eq!(string_field(&options, "Algorithm"), "levenberg-marquardt");
        assert_eq!(num_field(&options, "TolFun"), 1.0e-9);
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxIter"), 400.0);
        assert_eq!(num_field(&options, "MaxFunEvals"), 40000.0);
        assert_eq!(string_field(&options, "Display"), "off");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_name_value_pairs_are_case_insensitive() {
        let options = struct_result(
            run_optimoptions(vec![
                Value::from("fsolve"),
                Value::from("tolx"),
                Value::Num(1.0e-8),
                Value::from("DISPLAY"),
                Value::from("Final"),
            ])
            .expect("optimoptions overrides"),
        );
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(string_field(&options, "Display"), "final");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_updates_existing_options_with_pairs() {
        let base = run_optimoptions(vec![
            Value::from("fzero"),
            Value::from("TolX"),
            Value::Num(1.0e-5),
        ])
        .expect("base options");
        let options = struct_result(
            run_optimoptions(vec![base, Value::from("MaxIter"), Value::Num(25.0)])
                .expect("updated options"),
        );
        assert_eq!(string_field(&options, "Solver"), "fzero");
        assert_eq!(num_field(&options, "TolX"), 1.0e-5);
        assert_eq!(num_field(&options, "MaxIter"), 25.0);
        assert_eq!(
            options.fields.keys().collect::<Vec<_>>(),
            ["Solver", "TolX", "MaxIter", "MaxFunEvals", "Display"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_merges_existing_options_structs() {
        let first = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-5),
        ])
        .expect("first");
        let second = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("MaxIter"),
            Value::Num(30.0),
        ])
        .expect("second");
        let options = struct_result(run_optimoptions(vec![first, second]).expect("merged options"));
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxIter"), 30.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_same_solver_struct_merge_preserves_prior_overrides() {
        let first = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxFunEvals"),
            Value::Num(2000.0),
        ])
        .expect("first");
        let second = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
        ])
        .expect("second");

        let options = struct_result(run_optimoptions(vec![first, second]).expect("merged options"));

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxFunEvals"), 2000.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_solver_form_same_solver_struct_preserves_prior_overrides() {
        let later = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
        ])
        .expect("later options");

        let options = struct_result(
            run_optimoptions(vec![
                Value::from("fsolve"),
                Value::from("MaxFunEvals"),
                Value::Num(2000.0),
                later,
            ])
            .expect("merged options"),
        );

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxFunEvals"), 2000.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_default_skipping_compares_normalized_values() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let first = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxFunEvals"),
            Value::Num(2000.0),
            Value::from("Display"),
            Value::from("final"),
        ])
        .expect("first");

        let mut later = runmat_value::UnboundedStructValueBuilder::new();
        later.insert("Solver", Value::from("fsolve"));
        later.insert("TolX", Value::Num(1.0e-8));
        later.insert("MaxFunEvals", Value::Int(IntValue::I32(40000)));
        later.insert("Display", Value::CharArray(CharArray::new_row("off")));

        let options = struct_result(
            run_optimoptions(vec![first, Value::Struct(later.finish())]).expect("merged options"),
        );

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxFunEvals"), 2000.0);
        assert_eq!(string_field(&options, "Display"), "final");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_generic_to_concrete_solver_preserves_valid_generic_overrides() {
        let mut generic = runmat_value::UnboundedStructValueBuilder::new();
        generic.insert("MaxFunEvals", Value::Num(2000.0));
        generic.insert("Display", Value::from("final"));
        let generic = run_optimoptions(vec![Value::Struct(generic.finish())])
            .expect("normalized generic options");
        let Value::Struct(generic_fields) = &generic else {
            panic!("expected generic option structure");
        };
        assert_eq!(
            generic_fields.fields.keys().collect::<Vec<_>>(),
            ["MaxFunEvals", "Display"]
        );

        let later = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
        ])
        .expect("later options");

        let options =
            struct_result(run_optimoptions(vec![generic, later]).expect("merged options"));

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxFunEvals"), 2000.0);
        assert_eq!(string_field(&options, "Display"), "final");
        assert_eq!(
            options.fields.keys().collect::<Vec<_>>(),
            [
                "Solver",
                "TolX",
                "TolFun",
                "MaxIter",
                "MaxFunEvals",
                "Display"
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_solver_form_keeps_requested_solver_when_struct_has_solver() {
        let fzero_options = run_optimoptions(vec![
            Value::from("fzero"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("MaxIter"),
            Value::Num(30.0),
        ])
        .expect("fzero options");

        let options = struct_result(
            run_optimoptions(vec![Value::from("fsolve"), fzero_options])
                .expect("merged into fsolve options"),
        );

        assert_eq!(string_field(&options, "Solver"), "fsolve");
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "MaxIter"), 30.0);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
        assert_eq!(num_field(&options, "MaxFunEvals"), 40000.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_rejects_unknown_option_names() {
        let err = run_optimoptions(vec![
            Value::from("fzero"),
            Value::from("TolFun"),
            Value::Num(1.0e-8),
        ])
        .expect_err("TolFun is not accepted by fzero");
        assert_eq!(err.identifier(), Some("RunMat:optimoptions:UnknownOption"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_rejects_missing_option_values() {
        let err = run_optimoptions(vec![Value::from("fsolve"), Value::from("TolX")])
            .expect_err("missing option value");
        assert_eq!(
            err.identifier(),
            Some("RunMat:optimoptions:MissingOptionValue")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_rejects_invalid_option_values() {
        let err = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxIter"),
            Value::Num(1.5),
        ])
        .expect_err("noninteger MaxIter should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:optimoptions:InvalidOptionValue")
        );
    }

    #[test]
    fn optimoptions_numeric_options_read_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let max_iter =
            Tensor::new_integer(IntegerStorage::U16(vec![5]), vec![1, 1]).expect("MaxIter");

        let options = struct_result(
            run_optimoptions(vec![
                Value::from("fsolve"),
                Value::from("MaxIter"),
                Value::Tensor(max_iter),
            ])
            .expect("optimoptions"),
        );
        assert_eq!(
            options.fields.get("MaxIter"),
            Some(&Value::Int(IntValue::U16(5)))
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn optimoptions_wide_native_counts_survive_roundtrip_merge_and_solver_decode() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let count = (1_u64 << 53) + 1;
        for integer in [IntValue::I64(count as i64), IntValue::U64(count)] {
            let original = struct_result(
                run_optimoptions(vec![
                    Value::from("coneprog"),
                    Value::from("MaxIter"),
                    Value::Int(integer.clone()),
                ])
                .expect("exact count"),
            );
            let roundtrip =
                struct_result(run_optimoptions(vec![Value::Struct(original)]).expect("roundtrip"));
            let merged = struct_result(
                run_optimoptions(vec![
                    Value::from("coneprog"),
                    Value::Struct(roundtrip),
                    Value::from("Display"),
                    Value::from("off"),
                ])
                .expect("merge"),
            );
            assert_eq!(merged.fields.get("MaxIter"), Some(&Value::Int(integer)));
            assert_eq!(
                crate::builtins::math::optim::common::option_usize_value(
                    "coneprog",
                    merged.fields.get("MaxIter"),
                    "MaxIter",
                    200,
                )
                .expect("solver count decoding"),
                count as usize
            );
        }
    }

    #[test]
    fn optimoptions_strict_mode_rejects_typed_integer_option_before_normalization() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let max_iter =
            Tensor::new_integer(IntegerStorage::U16(vec![5]), vec![1, 1]).expect("MaxIter");

        let error = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxIter"),
            Value::Tensor(max_iter),
        ])
        .expect_err("typed integer option is a RunMat-only extension");

        assert_eq!(
            error.identifier(),
            INTEGER_OPTION_EXTENSION.error_identifier
        );
    }

    #[test]
    fn optimoptions_rejects_wide_integer_tolerance_before_float_conversion() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tolerance =
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
                .expect("TolX");

        let error = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Tensor(tolerance),
        ])
        .expect_err("wide integer tolerance cannot cross exactly");

        assert_eq!(
            error.identifier(),
            Some("RunMat:optimoptions:InvalidOptionValue")
        );
        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn optimoptions_automatic_resident_option_gathers_and_explicit_option_is_gated() {
        test_support::with_test_provider(|provider| {
            let values = [0.25];
            let shape = [1, 1];
            let automatic = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("automatic upload");
            let automatic = automatic
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("admit provenance view");
            let options = struct_result(
                run_optimoptions(vec![
                    Value::from("fsolve"),
                    Value::from("TolX"),
                    Value::GpuTensor(automatic),
                ])
                .expect("automatic option gathers"),
            );
            assert_eq!(num_field(&options, "TolX"), 0.25);

            let explicit = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("explicit upload");
            let explicit = explicit
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("admit provenance view");
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = run_optimoptions(vec![
                Value::from("fsolve"),
                Value::from("TolX"),
                Value::GpuTensor(explicit),
            ])
            .expect_err("explicit option is gated before gather");
            assert_eq!(
                error.identifier(),
                RESIDENT_OPTION_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn optimoptions_rejects_negative_typed_integer_options_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let max_iter =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("MaxIter");

        let err = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxIter"),
            Value::Tensor(max_iter),
        ])
        .expect_err("negative MaxIter should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:optimoptions:InvalidOptionValue")
        );
    }

    #[test]
    fn optimoptions_logical_options_read_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let gradient = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1])
            .expect("SpecifyObjectiveGradient");

        let options = struct_result(
            run_optimoptions(vec![
                Value::from("fminunc"),
                Value::from("SpecifyObjectiveGradient"),
                Value::Tensor(gradient),
            ])
            .expect("optimoptions"),
        );
        assert_eq!(
            options.fields.get("SpecifyObjectiveGradient"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn optimoptions_rejects_wide_typed_integer_logicals_despite_poisoned_mirror() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let gradient =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("SpecifyObjectiveGradient");

        let err = run_optimoptions(vec![
            Value::from("fminunc"),
            Value::from("SpecifyObjectiveGradient"),
            Value::Tensor(gradient),
        ])
        .expect_err("wide integer is not a logical scalar");
        assert_eq!(
            err.identifier(),
            Some("RunMat:optimoptions:InvalidOptionValue")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn optimoptions_rejects_out_of_range_integer_options() {
        let err = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("MaxIter"),
            Value::Num(2f64.powi(usize::BITS as i32)),
        ])
        .expect_err("out-of-range MaxIter should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:optimoptions:InvalidOptionValue")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminbnd_accepts_optimoptions_output() {
        let options = run_optimoptions(vec![
            Value::from("fminbnd"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("Display"),
            Value::from("off"),
        ])
        .expect("optimoptions");
        let result = run_call_builtin(
            "fminbnd",
            &[
                Value::FunctionHandle("cos".into()),
                Value::Num(0.0),
                Value::Num(std::f64::consts::PI),
                options,
            ],
        )
        .expect("fminbnd");
        match result.value() {
            Value::Num(value) => assert!((*value - std::f64::consts::PI).abs() < 1.0e-4),
            other => panic!("unexpected fminbnd result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fzero_accepts_optimoptions_output() {
        let options = run_optimoptions(vec![
            Value::from("fzero"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
        ])
        .expect("optimoptions");
        let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result = run_call_builtin(
            "fzero",
            &[
                Value::FunctionHandle("sin".into()),
                Value::Tensor(bracket),
                options,
            ],
        )
        .expect("fzero");
        match result.value() {
            Value::Num(value) => assert!((*value - std::f64::consts::PI).abs() < 1.0e-6),
            other => panic!("unexpected fzero result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fsolve_accepts_optimoptions_output() {
        let options = run_optimoptions(vec![
            Value::from("fsolve"),
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("TolFun"),
            Value::Num(1.0e-8),
        ])
        .expect("optimoptions");
        let result = run_call_builtin(
            "fsolve",
            &[
                Value::FunctionHandle("sin".into()),
                Value::Num(3.0),
                options,
            ],
        )
        .expect("fsolve");
        match result.value() {
            Value::Num(value) => assert!((*value - std::f64::consts::PI).abs() < 1.0e-6),
            other => panic!("unexpected fsolve result {other:?}"),
        }
    }
}
