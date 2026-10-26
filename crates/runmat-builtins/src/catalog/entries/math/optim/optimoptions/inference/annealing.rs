//! Static facts match the selected annealing control owner.
use super::{argument_error, text, Field};
use runmat_types::*;
pub(super) fn metadata(field: Field, value: &ValueFact) -> bool {
    if matches!(value.kind, ValueKindFact::Unknown) || value.shape.element_count() == Some(0) {
        return true;
    }
    let real = matches!(value.kind,ValueKindFact::Numeric(v)if v.domain==NumericDomain::Real)
        && value.storage != StorageFact::Sparse;
    match field {
        Field::AcceptanceFcn | Field::AnnealingFcn | Field::TemperatureFcn => {
            text(value) || matches!(value.kind, ValueKindFact::Callable(_))
        }
        Field::DataType | Field::Display => text(value),
        Field::HybridInterval => {
            text(value) || real && value.shape.element_count().is_none_or(|n| n == 1)
        }
        Field::InitialTemperature => {
            real && value.shape.known_dims().is_none_or(|d| {
                d.len() <= 2 && d.iter().filter(|v| v.is_some_and(|v| v != 1)).count() <= 1
            })
        }
        Field::OutputFcn | Field::PlotFcn => {
            fn callable(v: &ValueFact, plot: bool) -> bool {
                matches!(v.kind, ValueKindFact::Callable(_) | ValueKindFact::Unknown)
                    || plot && super::text(v)
                    || matches!(&v.kind,ValueKindFact::Cell(c)if c.elements.first().is_some_and(|v|callable(v,plot)))
            }
            callable(value, field == Field::PlotFcn)
                || matches!(&value.kind,ValueKindFact::Cell(c)if c.elements.iter().all(|v|callable(v,field==Field::PlotFcn)))
        }
        Field::HybridFcn => super::swarm::metadata(field, value),
        Field::MaxTime => super::swarm::metadata(field, value),
        _ => real && value.shape.element_count().is_none_or(|n| n == 1),
    }
}
pub(super) fn inspect(
    request: &CallRequest,
    field: Field,
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !metadata(field, value) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SA-OPTION-TYPE",
            "annealing option has invalid documented class or geometry",
            index,
        ));
    }
    if let Some(name) = request.literals.literal_string_at(index) {
        let valid = match field {
            Field::AnnealingFcn => !name.trim().is_empty(),
            Field::AcceptanceFcn => crate::OptimAcceptance::parse(&name).is_some(),
            Field::TemperatureFcn => crate::OptimTemperature::parse(&name).is_some(),
            Field::DataType => crate::OptimAnnealData::parse(&name).is_some(),
            Field::Display => crate::OptimDisplay::parse(&name)
                .is_some_and(|mode| crate::OptimSolver::Simulannealbnd.accepts_display(mode)),
            Field::HybridInterval => matches!(name.to_ascii_lowercase().as_str(), "never" | "end"),
            _ => true,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-SA-OPTION-VALUE",
                "unknown annealing selector value",
                index,
            ));
        }
    }
    if let Some(v) = request.literals.numeric_at(index) {
        let valid = match field {
            Field::ObjectiveLimit => !v.is_nan(),
            Field::MaxIter => v == f64::INFINITY || v.is_finite() && v >= 0.0 && v.fract() == 0.0,
            Field::MaxFunEvals | Field::MaxStallIterations | Field::ReannealInterval => {
                v.is_finite() && v >= 0.0 && v.fract() == 0.0
            }
            Field::DisplayInterval | Field::PlotInterval | Field::HybridInterval => {
                v.is_finite() && v >= 1.0 && v.fract() == 0.0
            }
            Field::MaxTime => !v.is_nan() && v >= 0.0,
            Field::FunctionTolerance | Field::InitialTemperature => v.is_finite() && v >= 0.0,
            _ => true,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-SA-OPTION-RANGE",
                "annealing option is outside its documented range",
                index,
            ));
        }
    }
}
pub(super) fn normalized(field: Field, source: Option<&ValueFact>) -> ValueFact {
    if let Some(value) = source {
        let mut value = value.clone();
        if matches!(value.kind, ValueKindFact::Numeric(_)) {
            value.residency = ResidencyFact::Host;
        }
        return value;
    }
    match field {
        Field::DataType
        | Field::Display
        | Field::AcceptanceFcn
        | Field::AnnealingFcn
        | Field::TemperatureFcn
        | Field::HybridInterval => ValueFact::scalar(ValueKindFact::String),
        Field::OutputFcn
        | Field::PlotFcn
        | Field::HybridFcn
        | Field::MaxFunEvals
        | Field::MaxStallIterations => {
            super::facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
        }
        _ => super::facts::double(ShapeFact::Scalar, StorageFact::Scalar),
    }
}
