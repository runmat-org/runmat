//! Shared constructor/solver option grammar for surrogate optimization.
//! Requested parallel execution masks the inactive vectorized operand before
//! any carrier facts or optional provider access for that operand.
use super::{constrained_numeric as numeric, ScalarCooperation};
use crate::{
    execution::{CallbackParallelMode, RootedValue},
    BuiltinResult,
};
use runmat_builtins::{
    BuiltinCatalogIdentity, BuiltinErrorDescriptor, BuiltinExtensionDescriptor, OptimDisplay,
    OptimOptionField as Field, OptimSolver,
};
use runmat_value::Value;

#[derive(Clone, Copy)]
pub(crate) struct Policy {
    pub identity: BuiltinCatalogIdentity,
    pub error: &'static BuiltinErrorDescriptor,
    pub integer: Option<&'static BuiltinExtensionDescriptor>,
    pub resident: Option<&'static BuiltinExtensionDescriptor>,
}
impl Policy {
    pub(crate) fn constructor() -> Self {
        Self {
            identity: BuiltinCatalogIdentity {
                name: "optimoptions",
            },
            error: &runmat_builtins::OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            integer: Some(&runmat_builtins::OPTIMOPTIONS_INTEGER_OPTION_EXTENSION),
            resident: Some(&runmat_builtins::OPTIMOPTIONS_RESIDENT_OPTION_EXTENSION),
        }
    }
    pub(crate) fn invalid(self, detail: impl std::fmt::Display) -> crate::RuntimeError {
        crate::build_runtime_error(format!("{}: {detail}", self.identity.name))
            .with_builtin(self.identity.name)
            .with_identifier(self.error.identifier.expect("canonical option identifier"))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    }
    fn numeric(self, integers: bool) -> numeric::Policy {
        numeric::Policy {
            owner: self.identity,
            invalid: Some(self.error),
            numeric: if integers { self.integer } else { None },
            single: None,
            resident: self.resident,
            sparse: None,
        }
    }
}
pub(crate) fn empty(value: &Value) -> bool {
    matches!(value, Value::Tensor(v) if v.is_empty())
}
fn text(policy: Policy, value: &Value) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .ok_or_else(|| policy.invalid("option requires scalar text"))
}
pub(crate) fn default_value(
    field: Field,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Option<Value>> {
    let number = match field {
        Field::BatchUpdateInterval => Some(1.0),
        Field::TolCon => Some(1e-3),
        Field::MaxTime => Some(f64::INFINITY),
        Field::MinSampleDistance => Some(1e-6),
        Field::ObjectiveLimit => Some(f64::NEG_INFINITY),
        _ => None,
    };
    if let Some(number) = number {
        return Ok(Some(Value::Num(number)));
    }
    if field == Field::UseVectorized {
        return Ok(Some(Value::Bool(false)));
    }
    let name = match field {
        Field::Display => Some("final"),
        Field::UseParallel => Some("off"),
        Field::PlotFcn => Some("optimplotfvalconstr"),
        Field::CheckpointFile => Some(""),
        _ => None,
    };
    if let Some(name) = name {
        return crate::call::argument_text::copy_str(name, runtime)
            .map(|text| Value::String(text.into()))
            .map(Some);
    }
    if matches!(
        field,
        Field::InitialPoints | Field::MaxFunEvals | Field::MinSurrogatePoints | Field::OutputFcn
    ) {
        return super::output_callback::empty(runtime).map(Some);
    }
    Ok(None)
}
fn initial_metadata(
    policy: Policy,
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    let check = |value: &Value, role: &str, geometry| -> BuiltinResult<()> {
        if numeric::preflight(value, role, geometry, &policy.numeric(false), runtime)? {
            Ok(())
        } else {
            Err(policy.invalid(format!("InitialPoints.{role} must be real double")))
        }
    };
    if let Value::Struct(fields) = value {
        if !fields.fields.contains_key("X") {
            return Err(policy.invalid("InitialPoints structure requires X"));
        }
        for (name, value) in &fields.fields {
            let geometry = match name {
                "X" | "Ineq" => numeric::Geometry::MatrixAny,
                "Fval" => numeric::Geometry::Vector,
                _ => return Err(policy.invalid(format!("unsupported InitialPoints field {name}"))),
            };
            check(value, name, geometry)?;
        }
        Ok(())
    } else {
        check(value, "X", numeric::Geometry::MatrixAny)
    }
}
pub(crate) fn metadata(
    policy: Policy,
    field: Field,
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if !OptimSolver::Surrogateopt.accepts(field) {
        return Err(policy.invalid("option is not active for surrogateopt"));
    }
    if empty(value) {
        return Ok(());
    }
    match field {
        Field::InitialPoints => return initial_metadata(policy, value, runtime),
        Field::OutputFcn | Field::PlotFcn => {
            return super::output_callback::validate_for_builtin(
                policy.identity.name,
                value,
                super::output_callback::CallbackForm::HandlesOrNames,
            )
            .map_err(|error| policy.invalid(error.message));
        }
        Field::Display | Field::CheckpointFile => return validate(policy, field, value),
        Field::MaxTime if crate::builtins::duration::is_duration_object(value) => {
            return validate(policy, field, value)
        }
        Field::UseParallel | Field::UseVectorized
            if matches!(
                value,
                Value::Bool(_)
                    | Value::LogicalArray(_)
                    | Value::String(_)
                    | Value::StringArray(_)
                    | Value::CharArray(_)
            ) =>
        {
            return validate(policy, field, value)
        }
        _ => {}
    }
    // The concrete-carrier check in the shared numerical owner precedes
    // recursive facts and optional provider access. No struct/capture walk.
    if !numeric::preflight(
        value,
        field.name(),
        numeric::Geometry::Scalar,
        &policy.numeric(true),
        runtime,
    )? {
        return Err(policy.invalid(format!("{} must be real double", field.name())));
    }
    validate(policy, field, value)
}
pub(crate) fn validate(policy: Policy, field: Field, value: &Value) -> BuiltinResult<()> {
    if empty(value) || matches!(value, Value::GpuTensor(_)) {
        return Ok(());
    }
    let fail = |error: crate::RuntimeError| policy.invalid(error.message);
    match field {
        Field::BatchUpdateInterval | Field::MinSurrogatePoints | Field::MaxFunEvals => {
            let count =
                super::option_usize_value(policy.identity.name, Some(value), field.name(), 0)
                    .map_err(fail)?;
            if count == 0 && field != Field::MaxFunEvals {
                return Err(policy.invalid(format!("{} must be positive", field.name())));
            }
        }
        Field::TolCon | Field::MinSampleDistance => {
            let number =
                super::option_f64_value(policy.identity.name, Some(value), field.name(), 0.0)
                    .map_err(fail)?;
            if number < 0.0 {
                return Err(policy.invalid(format!("{} must be nonnegative", field.name())));
            }
        }
        Field::ObjectiveLimit => {
            super::quasi_newton_controls::objective_limit(policy.identity.name, value)
                .map_err(fail)?;
        }
        Field::MaxTime => {
            super::global_time::parse(
                super::global_time::Policy {
                    identity: policy.identity,
                    error: policy.error,
                },
                value,
            )?;
        }
        Field::Display => {
            if OptimDisplay::parse(&text(policy, value)?)
                .filter(|v| OptimSolver::Surrogateopt.accepts_display(*v))
                .is_none()
            {
                return Err(policy.invalid("Display must be off, none, iter, or final"));
            }
        }
        Field::CheckpointFile => {
            text(policy, value)?;
        }
        Field::UseParallel => {
            super::quasi_newton_controls::parallel_mode(policy.identity.name, value)
                .map_err(fail)?;
        }
        Field::UseVectorized => {
            super::scalar_options::switch_value(policy.identity.name, field.name(), value)
                .map_err(fail)?;
        }
        Field::InitialPoints | Field::OutputFcn | Field::PlotFcn => {}
        _ => return Err(policy.invalid("inactive surrogate control")),
    }
    Ok(())
}
pub(crate) async fn prepare(
    policy: Policy,
    value: Value,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let value = if empty(&value) {
        Value::Struct(runmat_value::UnboundedStructValueBuilder::new().finish())
    } else {
        value
    };
    let source = RootedValue::in_optional_context(value, c.runtime())?;
    let Value::Struct(fields) = source.value() else {
        return Err(policy.invalid("options must be a scalar structure or []"));
    };
    let solver = OptimSolver::Surrogateopt;
    let mut seen = Vec::new();
    let mut parallel = None;
    // Validate all active outer operands before the first transfer. The
    // vectorized value is intentionally not even inspected until the actual
    // requested parallel mode has been materialized.
    for (name, value) in &fields.fields {
        if name.eq_ignore_ascii_case("Solver") {
            if OptimSolver::parse(&text(policy, value)?) != Some(solver) {
                return Err(policy.invalid("options Solver must be surrogateopt"));
            }
            continue;
        }
        let field = Field::parse_for_solver(name, solver)
            .ok_or_else(|| policy.invalid(format!("unsupported option {name}")))?;
        if seen.contains(&field) {
            return Err(policy.invalid("conflicting option aliases"));
        }
        seen.push(field);
        if field == Field::UseVectorized {
            continue;
        }
        metadata(policy, field, value, c.runtime())?;
        if field == Field::UseParallel {
            parallel = Some(value);
        }
    }
    let parallel = match parallel {
        Some(v) if !empty(v) => {
            let value =
                crate::dispatcher::gather_if_needed_in_context_async(v, c.runtime()).await?;
            validate(policy, Field::UseParallel, &value)?;
            super::quasi_newton_controls::parallel_mode(policy.identity.name, &value)?
        }
        _ => CallbackParallelMode::Off,
    };
    if parallel == CallbackParallelMode::Off {
        for (name, value) in &fields.fields {
            if Field::parse_for_solver(name, solver) == Some(Field::UseVectorized) {
                metadata(policy, Field::UseVectorized, value, c.runtime())?;
            }
        }
    }
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    out.insert("Solver", Value::from(solver.name()));
    out.insert(
        "UseParallel",
        Value::from(match parallel {
            CallbackParallelMode::Off => "off",
            CallbackParallelMode::Auto => "auto",
            CallbackParallelMode::On => "on",
        }),
    );
    for (name, value) in &fields.fields {
        let Some(field) = Field::parse_for_solver(name, solver) else {
            continue;
        };
        if field == Field::UseParallel {
            continue;
        }
        if field == Field::UseVectorized && parallel != CallbackParallelMode::Off {
            continue;
        }
        let value = if matches!(
            field,
            Field::InitialPoints
                | Field::OutputFcn
                | Field::PlotFcn
                | Field::Display
                | Field::CheckpointFile
        ) || crate::builtins::duration::is_duration_object(value)
        {
            value.clone()
        } else {
            crate::dispatcher::gather_if_needed_in_context_async(value, c.runtime()).await?
        };
        validate(policy, field, &value)?;
        out.insert(field.name_for_solver(solver), value);
        c.completed(1).await?;
    }
    RootedValue::in_optional_context(Value::Struct(out.finish()), c.runtime())
}
