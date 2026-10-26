//! Facts follow the one surrogate control owner; this does not register a
//! solver entry before its executable lifecycle is available.
use super::{argument_error, facts, text, Field};
use runmat_types::*;
fn real(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Numeric(v) if v.domain == NumericDomain::Real && v.class != NumericClass::Single)
}
fn matrix(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown)
        || matches!(value.kind, ValueKindFact::Numeric(v) if v.domain == NumericDomain::Real && v.class == NumericClass::Double)
            && value.shape.known_dims().is_none_or(|d| d.len() <= 2)
}
fn initial(value: &ValueFact) -> bool {
    let ValueKindFact::Struct(fields) = &value.kind else {
        return matrix(value);
    };
    if fields.fields_complete && !fields.fields.contains_key("X") {
        return false;
    }
    fields
        .fields
        .iter()
        .all(|(name, value)| match name.as_str() {
            "X" | "Ineq" => matrix(value),
            "Fval" => {
                matrix(value)
                    && value
                        .shape
                        .known_dims()
                        .is_none_or(|d| d.iter().filter(|n| n.is_some_and(|n| n != 1)).count() <= 1)
            }
            _ => false,
        })
}
pub(super) fn metadata(field: Field, value: &ValueFact) -> bool {
    if matches!(value.kind, ValueKindFact::Unknown) {
        return true;
    }
    if real(value) && value.shape.element_count() == Some(0) && value.storage != StorageFact::Sparse
    {
        return true;
    }
    match field {
        Field::InitialPoints => initial(value),
        Field::CheckpointFile | Field::Display => text(value),
        Field::OutputFcn | Field::PlotFcn => {
            let one = |v: &ValueFact| {
                matches!(v.kind, ValueKindFact::Callable(_) | ValueKindFact::Unknown) || text(v)
            };
            one(value) || matches!(&value.kind,ValueKindFact::Cell(c) if c.elements.iter().all(one))
        }
        Field::UseParallel | Field::UseVectorized => {
            text(value)
                || (real(value) || matches!(value.kind, ValueKindFact::Logical))
                    && value.shape.element_count().is_none_or(|n| n == 1)
        }
        Field::MaxTime => {
            matches!(value.kind, ValueKindFact::Object(_)) && super::swarm::metadata(field, value)
                || real(value)
                    && value.storage != StorageFact::Sparse
                    && value.shape.element_count().is_none_or(|n| n == 1)
        }
        _ => {
            real(value)
                && value.storage != StorageFact::Sparse
                && value.shape.element_count().is_none_or(|n| n == 1)
        }
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
            "RM-CATALOG-SURROGATE-OPTION-TYPE",
            "surrogate option has an unsupported class or geometry",
            index,
        ));
    }
    if let Some(name) = request.literals.literal_string_at(index) {
        let valid = match field {
            Field::Display => crate::OptimDisplay::parse(&name)
                .is_some_and(|mode| crate::OptimSolver::Surrogateopt.accepts_display(mode)),
            Field::UseParallel => {
                matches!(name.to_ascii_lowercase().as_str(), "off" | "on" | "auto")
            }
            Field::UseVectorized => crate::OptimSwitch::parse(&name).is_some(),
            _ => true,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-SURROGATE-OPTION-VALUE",
                "unknown surrogate selector value",
                index,
            ));
        }
    }
    if let Some(value) = request.literals.numeric_at(index) {
        let valid = match field {
            Field::BatchUpdateInterval | Field::MinSurrogatePoints => {
                value.is_finite() && value >= 1.0 && value.fract() == 0.0
            }
            Field::MaxFunEvals => value.is_finite() && value >= 0.0 && value.fract() == 0.0,
            Field::TolCon | Field::MinSampleDistance => value.is_finite() && value >= 0.0,
            Field::MaxTime => !value.is_nan() && value >= 0.0,
            Field::ObjectiveLimit => !value.is_nan(),
            Field::UseParallel | Field::UseVectorized => value == 0.0 || value == 1.0,
            _ => true,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-SURROGATE-OPTION-RANGE",
                "surrogate option is outside its supported range",
                index,
            ));
        }
    }
}
pub(super) fn normalized(field: Field, source: Option<&ValueFact>) -> ValueFact {
    if let Some(value) = source {
        let mut value = value.clone();
        if field != Field::InitialPoints
            && matches!(
                value.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            )
        {
            value.residency = ResidencyFact::Host;
        }
        return value;
    }
    match field {
        Field::Display | Field::CheckpointFile | Field::PlotFcn | Field::UseParallel => {
            ValueFact::scalar(ValueKindFact::String)
        }
        Field::UseVectorized => ValueFact::scalar(ValueKindFact::Logical),
        Field::InitialPoints
        | Field::OutputFcn
        | Field::MaxFunEvals
        | Field::MinSurrogatePoints => {
            facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
        }
        _ => facts::double(ShapeFact::Scalar, StorageFact::Scalar),
    }
}
