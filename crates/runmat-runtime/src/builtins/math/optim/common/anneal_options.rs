//! One typed active option owner for constructor and simulated annealing.
use super::ScalarCooperation;
use crate::{execution::RootedValue, BuiltinResult, RuntimeError};
use runmat_builtins::{OptimOptionField as Field, OptimSolver};
use runmat_value::Value;
#[derive(Clone, Copy)]
pub(crate) enum Owner {
    Constructor,
    Solver,
}
pub(crate) fn default_value(
    field: Field,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Option<Value>> {
    let number = match field {
        Field::FunctionTolerance => Some(1e-6),
        Field::DisplayInterval => Some(10.0),
        Field::PlotInterval => Some(1.0),
        Field::ReannealInterval => Some(100.0),
        Field::InitialTemperature => Some(100.0),
        Field::MaxIter | Field::MaxTime => Some(f64::INFINITY),
        Field::ObjectiveLimit => Some(f64::NEG_INFINITY),
        _ => None,
    };
    if let Some(value) = number {
        return Ok(Some(Value::Num(value)));
    }
    let text = match field {
        Field::DataType => Some("double"),
        Field::Display => Some("final"),
        Field::AcceptanceFcn => Some("acceptancesa"),
        Field::AnnealingFcn => Some("annealingfast"),
        Field::TemperatureFcn => Some("temperatureexp"),
        Field::HybridInterval => Some("end"),
        _ => None,
    };
    if let Some(text) = text {
        return crate::call::argument_text::copy_str(text, runtime)
            .map(|text| Value::String(text.into()))
            .map(Some);
    }
    if matches!(
        field,
        Field::MaxFunEvals
            | Field::MaxStallIterations
            | Field::HybridFcn
            | Field::OutputFcn
            | Field::PlotFcn
    ) {
        return super::output_callback::empty(runtime).map(Some);
    }
    Ok(None)
}
impl Owner {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Constructor => "optimoptions",
            Self::Solver => "simulannealbnd",
        }
    }
}
pub(crate) fn invalid(owner: Owner, message: impl Into<String>) -> RuntimeError {
    let descriptor = match owner {
        Owner::Constructor => &runmat_builtins::OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
        Owner::Solver => &runmat_builtins::SIMULANNEALBND_ERROR_INVALID_ARGUMENT,
    };
    crate::build_runtime_error(message)
        .with_builtin(owner.name())
        .with_identifier(descriptor.identifier.expect("canonical identifier"))
        .build()
}
pub(crate) fn text(owner: Owner, value: &Value) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .ok_or_else(|| invalid(owner, "control must be scalar text"))
}
pub(crate) fn empty(value: &Value) -> bool {
    matches!(value,Value::Tensor(v)if v.is_empty())
}
pub(crate) fn number(owner: Owner, value: &Value) -> BuiltinResult<f64> {
    if super::is_typed_integer_value(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
    {
        return Err(invalid(
            owner,
            "numeric control integer is not exactly representable as double",
        ));
    }
    match value {
        Value::Num(v) => Ok(*v),
        Value::Int(v) => Ok(v.to_f64()),
        Value::Tensor(v) if v.len() == 1 => {
            Ok(crate::builtins::common::tensor::tensor_value_f64(v, 0))
        }
        _ => Err(invalid(owner, "control must be a real host scalar")),
    }
}
pub(crate) fn handle(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}
pub(crate) fn plot_form() -> super::output_callback::CallbackForm {
    const PLOTS: &[runmat_builtins::BuiltinCatalogIdentity] = &[
        runmat_builtins::BuiltinCatalogIdentity {
            name: "saplotbestf",
        },
        runmat_builtins::BuiltinCatalogIdentity {
            name: "saplotbestx",
        },
        runmat_builtins::BuiltinCatalogIdentity { name: "saplotf" },
        runmat_builtins::BuiltinCatalogIdentity { name: "saplotx" },
        runmat_builtins::BuiltinCatalogIdentity {
            name: "saplotstopping",
        },
        runmat_builtins::BuiltinCatalogIdentity {
            name: "saplottemperature",
        },
    ];
    super::output_callback::CallbackForm::HandlesOrPredefined(PLOTS)
}
pub(crate) fn observer(owner: Owner, value: &Value, plot: bool) -> BuiltinResult<()> {
    if empty(value) {
        return Ok(());
    }
    let one = |value: &Value| -> BuiltinResult<()> {
        // An explicitly nested cell supplies trailing parameters to one
        // selected callback; the outer cell remains the observer sequence.
        let target = if let Value::Cell(v) = value {
            v.data()
                .first()
                .ok_or_else(|| invalid(owner, "empty callback parameter cell"))?
        } else {
            value
        };
        let form = if plot {
            plot_form()
        } else {
            super::output_callback::CallbackForm::Handles
        };
        super::output_callback::validate_for(target, form)
            .map_err(|source| invalid(owner, source.message))
    };
    if let Value::Cell(values) = value {
        for v in values.data() {
            one(v)?;
        }
    } else {
        one(value)?;
    }
    Ok(())
}
pub(crate) fn metadata(
    owner: Owner,
    field: Field,
    value: &Value,
    resident: Option<&'static runmat_builtins::BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if empty(value) {
        return Ok(());
    }
    match field {
        Field::AcceptanceFcn | Field::AnnealingFcn | Field::TemperatureFcn => {
            if !handle(value) {
                let name = text(owner, value)?;
                let valid = match field {
                    Field::AcceptanceFcn => {
                        runmat_builtins::OptimAcceptance::parse(&name).is_some()
                    }
                    Field::AnnealingFcn => !name.trim().is_empty(),
                    _ => runmat_builtins::OptimTemperature::parse(&name).is_some(),
                };
                if !valid {
                    return Err(invalid(owner, "unknown predefined annealing operator"));
                }
            }
        }
        Field::OutputFcn => observer(owner, value, false)?,
        Field::PlotFcn => observer(owner, value, true)?,
        Field::HybridFcn => {
            super::swarm_options::hybrid(super::swarm_options::Owner::Constructor, value)
                .map_err(|source| invalid(owner, source.message))?;
        }
        Field::DataType => {
            if runmat_builtins::OptimAnnealData::parse(&text(owner, value)?).is_none() {
                return Err(invalid(owner, "DataType must be double or custom"));
            }
        }
        Field::Display => {
            if runmat_builtins::OptimDisplay::parse(&text(owner, value)?)
                .filter(|mode| OptimSolver::Simulannealbnd.accepts_display(*mode))
                .is_none()
            {
                return Err(invalid(
                    owner,
                    "Display must be off, final, iter, or diagnose",
                ));
            }
        }
        Field::HybridInterval
            if matches!(
                value,
                Value::String(_) | Value::CharArray(_) | Value::StringArray(_)
            ) =>
        {
            if !matches!(
                text(owner, value)?.to_ascii_lowercase().as_str(),
                "never" | "end"
            ) {
                return Err(invalid(owner, "HybridInterval text must be never or end"));
            }
        }
        Field::MaxTime if crate::builtins::duration::is_duration_object(value) => {
            time(owner, value)?;
        }
        _ => {
            if !matches!(
                value,
                Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
            ) {
                return Err(invalid(
                    owner,
                    "numeric control requires a concrete real numeric carrier",
                ));
            }
            let fact = crate::value_fact::value_fact(value);
            let real = matches!(fact.kind,runmat_types::ValueKindFact::Numeric(v)if v.domain==runmat_types::NumericDomain::Real);
            let geometry = if field == Field::InitialTemperature {
                fact.shape.known_dims().is_none_or(|d| {
                    d.len() <= 2 && d.iter().filter(|v| v.is_some_and(|v| v != 1)).count() <= 1
                })
            } else {
                fact.shape.element_count().is_none_or(|n| n == 1)
            };
            if !real || !geometry || fact.storage == runmat_types::StorageFact::Sparse {
                return Err(invalid(
                    owner,
                    "control has invalid real numeric geometry or class",
                ));
            }
            super::preflight_host_residency(owner.name(), value, resident, runtime)?;
        }
    }
    validate(owner, field, value)
}
pub(crate) fn validate(owner: Owner, field: Field, value: &Value) -> BuiltinResult<()> {
    if empty(value) || matches!(value, Value::GpuTensor(_)) {
        return Ok(());
    }
    if matches!(
        field,
        Field::MaxIter
            | Field::MaxFunEvals
            | Field::MaxStallIterations
            | Field::ReannealInterval
            | Field::DisplayInterval
            | Field::PlotInterval
    ) || field == Field::HybridInterval
        && crate::builtins::common::validation::value_text_scalar_string(value).is_none()
    {
        if field == Field::MaxIter && infinite(value) {
            return Ok(());
        }
        let count = super::option_usize_value(owner.name(), Some(value), field.name(), 0)
            .map_err(|source| invalid(owner, source.message))?;
        if count == 0
            && matches!(
                field,
                Field::DisplayInterval | Field::PlotInterval | Field::HybridInterval
            )
        {
            return Err(invalid(owner, "selected interval must be positive"));
        }
        return Ok(());
    }
    if field == Field::InitialTemperature {
        let valid = |v: f64, scalar: bool| v.is_finite() && if scalar { v >= 0.0 } else { v > 0.0 };
        match value {
            Value::Tensor(v) => {
                if super::is_typed_integer_value(value)
                    && !crate::builtins::common::validation::native_integer_value_is_exact_f64(
                        value,
                    )
                {
                    return Err(invalid(
                        owner,
                        "InitialTemperature integer is not exact double",
                    ));
                }
                for i in 0..v.len() {
                    if !valid(
                        crate::builtins::common::tensor::tensor_value_f64(v, i),
                        v.len() == 1,
                    ) {
                        return Err(invalid(
                            owner,
                            "InitialTemperature must be nonnegative scalar or positive vector",
                        ));
                    }
                }
            }
            _ => {
                if !valid(number(owner, value)?, true) {
                    return Err(invalid(owner, "InitialTemperature must be nonnegative"));
                }
            }
        }
        return Ok(());
    }
    let valid = match field {
        Field::FunctionTolerance => {
            let v = number(owner, value)?;
            v.is_finite() && v >= 0.0
        }
        Field::ObjectiveLimit => !number(owner, value)?.is_nan(),
        Field::MaxTime => {
            time(owner, value)?;
            true
        }
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(owner, format!("invalid {} range", field.name())))
    }
}
pub(crate) fn infinite(value: &Value) -> bool {
    matches!(value,Value::Num(v)if *v==f64::INFINITY)
        || matches!(value,Value::Tensor(v)if v.len()==1&&v.integer_storage().is_none()
            &&crate::builtins::common::tensor::tensor_value_f64(v,0)==f64::INFINITY)
}
pub(crate) use super::global_time::TimeLimit;
pub(crate) fn time(owner: Owner, value: &Value) -> BuiltinResult<TimeLimit> {
    let error = match owner {
        Owner::Constructor => &runmat_builtins::OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
        Owner::Solver => &runmat_builtins::SIMULANNEALBND_ERROR_INVALID_ARGUMENT,
    };
    super::global_time::parse(
        super::global_time::Policy {
            identity: runmat_builtins::BuiltinCatalogIdentity { name: owner.name() },
            error,
        },
        value,
    )
}
pub(crate) async fn prepare(value: Value, c: &mut ScalarCooperation) -> BuiltinResult<RootedValue> {
    let owner = Owner::Solver;
    let value = if empty(&value) {
        Value::Struct(runmat_value::UnboundedStructValueBuilder::new().finish())
    } else {
        value
    };
    let source = RootedValue::in_optional_context(value, c.runtime())?;
    let Value::Struct(options) = source.value() else {
        return Err(invalid(owner, "options must be a scalar structure or []"));
    };
    let solver = OptimSolver::Simulannealbnd;
    let mut seen = Vec::new();
    // Admit every active outer carrier before even the first optional gather.
    for (name, value) in &options.fields {
        if name.eq_ignore_ascii_case("Solver") {
            if OptimSolver::parse(&text(owner, value)?) != Some(solver) {
                return Err(invalid(owner, "options Solver must be simulannealbnd"));
            }
            continue;
        }
        let field = Field::parse_for_solver(name, solver)
            .ok_or_else(|| invalid(owner, format!("unsupported option {name}")))?;
        if seen.contains(&field) {
            return Err(invalid(owner, "conflicting option aliases"));
        }
        seen.push(field);
        metadata(
            owner,
            field,
            value,
            Some(&runmat_builtins::SIMULANNEALBND_RESIDENT_OPTIONS_EXTENSION),
            c.runtime(),
        )?;
        if !empty(value) && super::is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                c.runtime(),
                &runmat_builtins::SIMULANNEALBND_INTEGER_OPTIONS_EXTENSION,
                owner.name(),
            )?;
        }
    }
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    out.insert("Solver", Value::from(solver.name()));
    for (name, value) in &options.fields {
        let Some(field) = Field::parse_for_solver(name, solver) else {
            continue;
        };
        let value = if field.is_callback()
            || matches!(field, Field::DataType | Field::Display)
            || crate::builtins::duration::is_duration_object(value)
            || (field == Field::HybridInterval
                && crate::builtins::common::validation::value_text_scalar_string(value).is_some())
        {
            value.clone()
        } else {
            crate::dispatcher::gather_if_needed_in_context_async(value, c.runtime()).await?
        };
        validate(owner, field, &value)?;
        out.insert(field.name_for_solver(solver), value);
        c.completed(1).await?;
    }
    RootedValue::in_optional_context(Value::Struct(out.finish()), c.runtime())
}
