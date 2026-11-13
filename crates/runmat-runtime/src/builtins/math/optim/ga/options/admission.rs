//! Constructor/solver scalar grammar shares one typed option owner. Data
//! depending on PopulationType and nonlinear activation is admitted in prepare.
use super::{common, real, switch, text, Field, Operator, Owner};
use crate::{context::RuntimeContext, BuiltinResult};
use runmat_builtins::*;
use runmat_value::Value;

#[derive(Clone, Copy)]
pub(in super::super) enum Role {
    Count,
    Counts,
    Real,
    Flag,
    Time,
    Range,
    Deferred,
    Opaque,
}
pub(super) fn role(field: Field) -> Role {
    match field {
        Field::PopulationSize => Role::Counts,
        Field::EliteCount
        | Field::MaxIter
        | Field::MaxStallIterations
        | Field::MigrationInterval
        | Field::DisplayInterval
        | Field::PlotInterval => Role::Count,
        Field::FunctionTolerance
        | Field::TolCon
        | Field::ObjectiveLimit
        | Field::CrossoverFraction
        | Field::MigrationFraction
        | Field::InitialPenalty
        | Field::PenaltyFactor => Role::Real,
        Field::UseVectorized | Field::UseParallel => Role::Flag,
        Field::MaxTime | Field::MaxStallTime => Role::Time,
        Field::InitialPopulationRange => Role::Range,
        Field::InitialPopulationMatrix | Field::InitialScoresMatrix => Role::Deferred,
        _ => Role::Opaque,
    }
}
pub(in crate::builtins::math::optim) fn metadata(
    owner: Owner,
    field: Field,
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    if super::super::inputs::raw::unspecified(value) {
        return Ok(());
    }
    if matches!(role(field), Role::Opaque | Role::Deferred) {
        validate(owner, field, value)?;
        if field == Field::CreationFcn {
            super::nonlinear_creation::preflight(owner, value, runtime)?;
        }
        return super::parameters::preflight(owner, field, value, runtime);
    }
    if matches!(role(field), Role::Time) && crate::builtins::duration::is_duration_object(value) {
        return validate(owner, field, value);
    }
    if matches!(role(field), Role::Flag)
        && crate::builtins::common::validation::value_text_scalar_string(value).is_some()
    {
        return validate(owner, field, value);
    }
    numeric_metadata(owner, role(field), value, runtime)?;
    validate(owner, field, value)
}

/// Numeric role admission shared by options and mutable state flags. This is
/// class/intent metadata, not callback, effect or allocation permission.
pub(in super::super) fn numeric_metadata(
    owner: Owner,
    role: Role,
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    if matches!(role, Role::Opaque | Role::Deferred) {
        return Err(owner.invalid("field does not have a numeric control role"));
    }
    let numeric = matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    );
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_));
    if !(numeric || matches!(role, Role::Flag) && logical) {
        return Err(owner.invalid("option requires a concrete real numeric carrier"));
    }
    let fact = crate::value_fact::value_fact(value);
    let kind = matches!(fact.kind,runmat_types::ValueKindFact::Numeric(n)if n.domain==runmat_types::NumericDomain::Real)
        || matches!(role, Role::Flag) && matches!(fact.kind, runmat_types::ValueKindFact::Logical);
    let shape = match role {
        Role::Counts => fact.shape.known_dims().is_none_or(|d| {
            d.len() <= 2 && d.iter().filter(|axis| axis.is_some_and(|n| n != 1)).count() <= 1
        }),
        Role::Range => fact
            .shape
            .known_dims()
            .is_none_or(|d| d.len() == 2 && d[0].is_none_or(|n| n == 2)),
        _ => fact.shape.element_count().is_none_or(|n| n == 1),
    };
    if !kind
        || !shape
        || fact.storage == runmat_types::StorageFact::Sparse
        || fact.shape.element_count() == Some(0)
    {
        return Err(owner.invalid("option has invalid numeric class or geometry"));
    }
    common::preflight_host_residency(owner.name(), value, None, runtime)?;
    if common::is_typed_integer_value(value) {
        let extension = match owner {
            Owner::Constructor => &OPTIMOPTIONS_INTEGER_OPTION_EXTENSION,
            Owner::Solver => &GA_INTEGER_OPTIONS_EXTENSION,
        };
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            extension,
            owner.name(),
        )?;
    }
    Ok(())
}

pub(in crate::builtins::math::optim) fn validate(
    owner: Owner,
    field: Field,
    value: &Value,
) -> BuiltinResult<()> {
    if super::super::inputs::raw::unspecified(value) {
        return Ok(());
    }
    if matches!(value, Value::GpuTensor(_)) {
        return if matches!(role(field), Role::Opaque) {
            Err(owner.invalid("text/callback option cannot be a resident numeric carrier"))
        } else {
            Ok(())
        };
    }
    let invalid = || {
        owner.invalid(format!(
            "invalid {} value",
            field.name_for_solver(OptimSolver::Ga)
        ))
    };
    match field {
        Field::InitialPopulationMatrix | Field::InitialScoresMatrix => Ok(()), // Selected solver context owns activation/class.
        Field::PopulationType => OptimGeneticPopulation::parse(&text(value)?)
            .map(|_| ())
            .ok_or_else(invalid),
        Field::NonlinearConstraintAlgorithm => OptimGeneticConstraint::parse(&text(value)?)
            .map(|_| ())
            .ok_or_else(invalid),
        Field::MigrationDirection => OptimGeneticMigration::parse(&text(value)?)
            .map(|_| ())
            .ok_or_else(invalid),
        Field::StallTest => OptimGeneticStall::parse(&text(value)?)
            .map(|_| ())
            .ok_or_else(invalid),
        Field::Display => OptimDisplay::parse(&text(value)?)
            .filter(|v| OptimSolver::Ga.accepts_display(*v))
            .map(|_| ())
            .ok_or_else(invalid),
        Field::FunValCheck => OptimSwitch::parse(&text(value)?)
            .map(|_| ())
            .ok_or_else(invalid),
        Field::OutputFcn => common::output_callback::validate_for_builtin(
            owner.name(),
            value,
            common::output_callback::CallbackForm::Handles,
        ),
        Field::PlotFcn => common::output_callback::validate_for_builtin(
            owner.name(),
            value,
            common::output_callback::CallbackForm::HandlesOrNames,
        ),
        Field::UseVectorized => switch(owner, value).map(|_| ()),
        Field::UseParallel => {
            common::quasi_newton_controls::parallel_mode(owner.name(), value).map(|_| ())
        }
        Field::MaxTime | Field::MaxStallTime => {
            let limit = common::global_time::parse(owner.policy(), value)?;
            if field == Field::MaxStallTime && limit.is_zero() {
                Err(invalid())
            } else {
                Ok(())
            }
        }
        Field::EliteCount
        | Field::MaxIter
        | Field::MaxStallIterations
        | Field::MigrationInterval
        | Field::DisplayInterval
        | Field::PlotInterval => {
            let n = common::option_usize_value(owner.name(), Some(value), field.name(), 0)?;
            if n == 0
                && matches!(
                    field,
                    Field::MigrationInterval | Field::DisplayInterval | Field::PlotInterval
                )
            {
                Err(invalid())
            } else {
                Ok(())
            }
        }
        Field::PopulationSize => {
            if matches!(value,Value::Tensor(values) if values.len()>1) {
                return Ok(());
            }
            if common::option_usize_value(owner.name(), Some(value), field.name(), 0)? > 0 {
                Ok(())
            } else {
                Err(invalid())
            }
        }
        Field::FunctionTolerance
        | Field::TolCon
        | Field::CrossoverFraction
        | Field::MigrationFraction
        | Field::InitialPenalty
        | Field::PenaltyFactor
        | Field::ObjectiveLimit => {
            let value = real(owner, value)?;
            let valid = match field {
                Field::FunctionTolerance | Field::TolCon => value.is_finite() && value >= 0.,
                Field::CrossoverFraction | Field::MigrationFraction => {
                    value.is_finite() && (0.0..=1.0).contains(&value)
                }
                Field::InitialPenalty => value.is_finite() && value >= 1.,
                Field::PenaltyFactor => value.is_finite() && value > 1.,
                Field::ObjectiveLimit => !value.is_nan(),
                _ => unreachable!(),
            };
            if valid {
                Ok(())
            } else {
                Err(invalid())
            }
        }
        Field::InitialPopulationRange => Ok(()), // Cooperative value scan follows metadata admission.
        Field::CreationFcn => Operator::parse_for(owner, value, OptimGeneticCreation::parse)
            .and_then(|operator| match operator.kind() {
                Some(OptimGeneticCreation::NonlinearFeasible) => {
                    super::nonlinear_creation::validate(owner, operator.extra())
                }
                _ => operator.argument_count(0, 0),
            }),
        Field::CrossoverFcn => Operator::parse_for(owner, value, OptimGeneticCrossover::parse)
            .and_then(|operator| match operator.kind() {
                Some(OptimGeneticCrossover::Intermediate | OptimGeneticCrossover::Heuristic) => {
                    operator.argument_count(0, 1)
                }
                _ => operator.argument_count(0, 0),
            }),
        Field::MutationFcn => Operator::parse_for(owner, value, OptimGeneticMutation::parse)
            .and_then(|operator| match operator.kind() {
                Some(OptimGeneticMutation::Gaussian) => operator.argument_count(0, 2),
                Some(OptimGeneticMutation::Uniform) => operator.argument_count(0, 1),
                _ => operator.argument_count(0, 0),
            }),
        Field::SelectionFcn => Operator::parse_for(owner, value, OptimGeneticSelection::parse)
            .and_then(|operator| match operator.kind() {
                Some(OptimGeneticSelection::Tournament) => operator.argument_count(0, 1),
                _ => operator.argument_count(0, 0),
            }),
        Field::FitnessScalingFcn => Operator::parse_for(owner, value, OptimGeneticScaling::parse)
            .and_then(|operator| match operator.kind() {
                Some(OptimGeneticScaling::Top | OptimGeneticScaling::ShiftLinear) => {
                    operator.argument_count(0, 1)
                }
                _ => operator.argument_count(0, 0),
            }),
        Field::HybridFcn => super::super::hybrid::validate(owner, value),
        _ => Err(owner.invalid("option is not part of ga")),
    }
}

/// Long array controls are checked on the original cooperative budget, not in
/// the constructor's synchronous grammar pass or by source-controlled padding.
pub(in crate::builtins::math::optim) async fn validate_cooperative(
    owner: Owner,
    field: Field,
    value: &Value,
    cooperation: &mut common::ScalarCooperation,
) -> BuiltinResult<()> {
    validate(owner, field, value)?;
    if super::super::inputs::raw::unspecified(value) {
        return Ok(());
    }
    super::parameters::validate(owner, field, value, cooperation).await?;
    match (field, value) {
        (Field::PopulationSize, Value::Tensor(values)) => {
            for index in 0..values.len() {
                let scalar = if let Some(storage) = values.integer_storage() {
                    Value::Int(storage.value_at(index).expect("native count"))
                } else {
                    Value::Num(crate::builtins::common::tensor::tensor_value_f64(
                        values, index,
                    ))
                };
                if common::option_usize_value(owner.name(), Some(&scalar), field.name(), 0)? == 0 {
                    return Err(owner.invalid("every population count must be positive"));
                }
                cooperation.completed(1).await?;
            }
        }
        (Field::InitialPopulationRange, Value::Tensor(values)) => {
            if values.shape().len() != 2 || values.shape()[0] != 2 {
                return Err(owner.invalid("initial range requires two rows"));
            }
            for column in 0..values.shape()[1] {
                let a = crate::builtins::common::tensor::tensor_value_f64(values, 2 * column);
                let b = crate::builtins::common::tensor::tensor_value_f64(values, 2 * column + 1);
                if !a.is_finite() || !b.is_finite() || a > b {
                    return Err(owner.invalid("initial ranges must be finite ordered intervals"));
                }
                cooperation.completed(1).await?;
            }
        }
        (Field::InitialPopulationRange, _) => {
            return Err(owner.invalid("initial range requires a two-row numeric matrix"))
        }
        _ => {}
    }
    Ok(())
}
