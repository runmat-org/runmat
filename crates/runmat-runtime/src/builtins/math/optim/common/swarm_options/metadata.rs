//! One metadata/value owner for the actual particle-swarm control profile.
use super::{invalid, Field, Owner};
use crate::BuiltinResult;
use runmat_value::{NumericDType, Value};
fn handle(value: &Value) -> bool {
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
pub(super) fn numeric(owner: Owner, value: &Value, vector: bool) -> BuiltinResult<()> {
    if !matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    ) {
        return Err(invalid(
            owner,
            "control must be a concrete real numeric carrier",
        ));
    }
    let fact = crate::value_fact::value_fact(value);
    let real = matches!(fact.kind,runmat_types::ValueKindFact::Numeric(n)if n.domain==runmat_types::NumericDomain::Real);
    let shape = if vector {
        fact.shape.known_dims().is_none_or(|dims| {
            dims.len() <= 2 && dims.iter().filter(|n| n.is_some_and(|n| n != 1)).count() <= 1
        })
    } else {
        fact.shape.element_count().is_none_or(|n| n == 1)
    };
    if !real
        || !shape
        || fact.shape.element_count() == Some(0)
        || fact.storage == runmat_types::StorageFact::Sparse
    {
        return Err(invalid(
            owner,
            "control has invalid real numeric class or geometry",
        ));
    }
    Ok(())
}
fn flag_metadata(owner: Owner, value: &Value) -> BuiltinResult<()> {
    if !matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Tensor(_)
            | Value::GpuTensor(_)
            | Value::Bool(_)
            | Value::LogicalArray(_)
    ) {
        return Err(invalid(
            owner,
            "flag must be a concrete real numeric or logical scalar",
        ));
    }
    let fact = crate::value_fact::value_fact(value);
    let kind = matches!(fact.kind, runmat_types::ValueKindFact::Logical)
        || matches!(fact.kind,runmat_types::ValueKindFact::Numeric(n)if n.domain==runmat_types::NumericDomain::Real);
    if !kind
        || fact.shape.element_count().is_some_and(|n| n != 1)
        || fact.storage == runmat_types::StorageFact::Sparse
    {
        return Err(invalid(
            owner,
            "flag must be a full real numeric or logical scalar",
        ));
    }
    Ok(())
}
pub(crate) fn double_matrix(owner: Owner, value: &Value) -> BuiltinResult<()> {
    if !matches!(
        value,
        Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_)
    ) {
        return Err(invalid(
            owner,
            "initial points and values must be concrete full double arrays",
        ));
    }
    let fact = crate::value_fact::value_fact(value);
    if !matches!(fact.kind,runmat_types::ValueKindFact::Numeric(n)if n.class==runmat_types::NumericClass::Double&&n.domain==runmat_types::NumericDomain::Real)
        || fact.storage == runmat_types::StorageFact::Sparse
        || fact.shape.known_dims().is_some_and(|dims| dims.len() > 2)
    {
        return Err(invalid(
            owner,
            "initial points and values must be full real double matrices",
        ));
    }
    Ok(())
}
pub(crate) fn metadata(
    owner: Owner,
    field: Field,
    value: &Value,
    resident: &'static runmat_builtins::BuiltinExtensionDescriptor,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if matches!(value,Value::Tensor(v)if v.is_empty()) {
        return Ok(());
    }
    match field {
        Field::CreationFcn => {
            if !handle(value)
                && (!matches!(
                    value,
                    Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
                ) || runmat_builtins::OptimSwarmCreation::parse(&super::text(
                    owner, value, field,
                )?)
                .is_none())
            {
                return Err(invalid(
                    owner,
                    "CreationFcn must be pswcreationuniform or one function handle",
                ));
            }
        }
        Field::HybridFcn => {
            super::hybrid(owner, value)?;
        }
        Field::OutputFcn => super::own_parse(
            owner,
            crate::builtins::math::optim::common::output_callback::validate(value),
        )?,
        Field::PlotFcn => super::own_parse(
            owner,
            crate::builtins::math::optim::common::output_callback::validate_for(
                value,
                super::plot_form(),
            ),
        )?,
        Field::Display => {
            let mode = runmat_builtins::OptimDisplay::parse(&super::text(owner, value, field)?)
                .filter(|mode| runmat_builtins::OptimSolver::Particleswarm.accepts_display(*mode));
            if mode.is_none() {
                return Err(invalid(owner, "unsupported Display value"));
            }
        }
        Field::FunValCheck => {
            if runmat_builtins::OptimSwitch::parse(&super::text(owner, value, field)?).is_none() {
                return Err(invalid(owner, "FunValCheck must be on or off"));
            }
        }
        Field::UseParallel => {
            if !matches!(value, Value::GpuTensor(_)) {
                super::own_parse(
                    owner,
                    super::super::quasi_newton_controls::parallel_mode(owner.name(), value),
                )?;
            } else {
                flag_metadata(owner, value)?;
            }
        }
        Field::UseVectorized => {
            if matches!(
                value,
                Value::Bool(_)
                    | Value::LogicalArray(_)
                    | Value::String(_)
                    | Value::StringArray(_)
                    | Value::CharArray(_)
            ) {
                super::switch(owner, value, field)?;
            } else {
                flag_metadata(owner, value)?;
            }
        }
        Field::InitialPoints => {
            let validate_array = |array: &Value| -> BuiltinResult<()> {
                double_matrix(owner, array)?;
                super::super::preflight_host_residency(owner.name(), array, Some(resident), runtime)
            };
            if let Value::Struct(points) = value {
                if !points.fields.contains_key("X")
                    || points.fields.keys().any(|n| n != "X" && n != "Fval")
                {
                    return Err(invalid(
                        owner,
                        "InitialPoints structure requires X and optional Fval only",
                    ));
                }
                for array in points.fields.values() {
                    validate_array(array)?;
                }
            } else {
                validate_array(value)?;
            }
        }
        Field::MaxTime | Field::MaxStallTime
            if crate::builtins::duration::is_duration_object(value) =>
        {
            let days = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
            if days.len() != 1 || days.numeric_dtype() != NumericDType::F64 {
                return Err(invalid(owner, "time limit duration must be scalar"));
            }
        }
        Field::InitialSwarmSpan | Field::InertiaRange => numeric(owner, value, true)?,
        Field::FunctionTolerance
        | Field::MaxIter
        | Field::DisplayInterval
        | Field::MaxStallIterations
        | Field::MaxStallTime
        | Field::MaxTime
        | Field::MinNeighborsFraction
        | Field::ObjectiveLimit
        | Field::SelfAdjustmentWeight
        | Field::SocialAdjustmentWeight
        | Field::SwarmSize => numeric(owner, value, false)?,
        _ => {
            return Err(invalid(
                owner,
                "option is outside the particle-swarm profile",
            ))
        }
    }
    if !field.is_callback() {
        super::super::preflight_host_residency(owner.name(), value, Some(resident), runtime)?;
    }
    Ok(())
}
pub(crate) fn validate(owner: Owner, field: Field, value: &Value) -> BuiltinResult<()> {
    if matches!(value,Value::Tensor(v)if v.is_empty()) || matches!(value, Value::GpuTensor(_)) {
        return Ok(());
    }
    match field {
        Field::InitialPoints => {}
        Field::CreationFcn
        | Field::HybridFcn
        | Field::OutputFcn
        | Field::PlotFcn
        | Field::Display
        | Field::FunValCheck => {}
        Field::UseParallel => {
            super::own_parse(
                owner,
                super::super::quasi_newton_controls::parallel_mode(owner.name(), value),
            )?;
        }
        Field::UseVectorized => {
            super::switch(owner, value, field)?;
        }
        Field::MaxIter | Field::DisplayInterval | Field::MaxStallIterations | Field::SwarmSize => {
            let n = super::own_parse(
                owner,
                super::super::option_usize_value(owner.name(), Some(value), field.name(), 0),
            )?;
            if (field == Field::SwarmSize && n < 2)
                || (matches!(field, Field::DisplayInterval | Field::MaxStallIterations) && n == 0)
                || (field == Field::MaxStallIterations && n.checked_add(1).is_none())
            {
                return Err(invalid(owner, "invalid swarm count range"));
            }
        }
        Field::InitialSwarmSpan => {
            let values = super::floating(owner, value)?;
            if values.iter().any(|v| !v.is_finite() || v <= 0.0) {
                return Err(invalid(
                    owner,
                    "InitialSwarmSpan must be finite and positive",
                ));
            }
        }
        Field::InertiaRange => {
            let values = super::floating(owner, value)?;
            if values.len() != 2
                || values.iter().any(|v| !v.is_finite())
                || values.get(0) > values.get(1)
                || values.get(0) * values.get(1) < 0.0
            {
                return Err(invalid(
                    owner,
                    "InertiaRange must have two increasing same-sign finite values",
                ));
            }
        }
        Field::MaxTime | Field::MaxStallTime => {
            super::time_limit(owner, value)?;
        }
        Field::ObjectiveLimit => {
            super::own_parse(
                owner,
                super::super::quasi_newton_controls::objective_limit(owner.name(), value),
            )?;
        }
        Field::FunctionTolerance
        | Field::MinNeighborsFraction
        | Field::SelfAdjustmentWeight
        | Field::SocialAdjustmentWeight => {
            let n = super::own_parse(
                owner,
                super::super::option_f64_value(owner.name(), Some(value), field.name(), 0.0),
            )?;
            if (field == Field::FunctionTolerance && n < 0.0)
                || (field == Field::MinNeighborsFraction && !(0.0..=1.0).contains(&n))
            {
                return Err(invalid(
                    owner,
                    "invalid swarm tolerance or neighborhood fraction",
                ));
            }
        }
        _ => return Err(invalid(owner, "invalid particle-swarm field")),
    }
    Ok(())
}
