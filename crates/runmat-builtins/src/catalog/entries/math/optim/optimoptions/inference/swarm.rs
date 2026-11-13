//! Facts for the selected particle-swarm control profile, not scalar aliases.
use super::{argument_error, text, Field, OptimSolver};
use runmat_types::*;
pub(super) fn metadata(field: Field, value: &ValueFact) -> bool {
    if matches!(value.kind, ValueKindFact::Unknown) || value.shape.element_count() == Some(0) {
        return true;
    }
    match field {
        Field::CreationFcn => text(value) || matches!(value.kind, ValueKindFact::Callable(_)),
        Field::HybridFcn => {
            text(value)
                || matches!(value.kind, ValueKindFact::Callable(_))
                || matches!(&value.kind,ValueKindFact::Cell(cell)if cell.elements.len()==2
                &&(text(&cell.elements[0])||matches!(cell.elements[0].kind,ValueKindFact::Callable(_)))
                &&(matches!(cell.elements[1].kind,ValueKindFact::Struct(_)|ValueKindFact::Unknown)
                    ||cell.elements[1].shape.element_count()==Some(0)))
        }
        Field::OutputFcn | Field::PlotFcn => {
            let callable = |v: &ValueFact| {
                matches!(v.kind, ValueKindFact::Callable(_) | ValueKindFact::Unknown)
                    || field == Field::PlotFcn && text(v)
            };
            callable(value)
                || matches!(&value.kind,ValueKindFact::Cell(cell)if cell.elements.iter().all(callable))
        }
        Field::InitialPoints => match &value.kind {
            ValueKindFact::Struct(points) => {
                (!points.fields_complete || points.fields.contains_key("X"))
                    && points
                        .fields
                        .iter()
                        .all(|(name, v)| matches!(name.as_str(), "X" | "Fval") && double_matrix(v))
            }
            _ => double_matrix(value),
        },
        Field::InitialSwarmSpan | Field::InertiaRange => {
            real(value)
                && value.shape.known_dims().is_none_or(|dims| {
                    dims.len() <= 2
                        && dims.iter().filter(|d| d.is_some_and(|n| n != 1)).count() <= 1
                })
        }
        Field::MaxTime | Field::MaxStallTime => {
            if let ValueKindFact::Object(object) = &value.kind {
                return object
                    .runtime_class
                    .as_ref()
                    .is_none_or(|class| class.is(runmat_types::standard::DURATION))
                    && value.shape.element_count().is_none_or(|n| n == 1);
            }
            real(value) && value.shape.element_count().is_none_or(|n| n == 1)
        }
        Field::UseParallel | Field::UseVectorized => {
            text(value)
                || matches!(value.kind, ValueKindFact::Logical)
                    && value.shape.element_count().is_none_or(|n| n == 1)
                || real(value) && value.shape.element_count().is_none_or(|n| n == 1)
        }
        Field::Display | Field::FunValCheck => text(value),
        _ => real(value) && value.shape.element_count().is_none_or(|n| n == 1),
    }
}
fn real(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown)
        || matches!(value.kind,ValueKindFact::Numeric(n)if n.domain==NumericDomain::Real)
            && value.storage != StorageFact::Sparse
}
fn double_matrix(value: &ValueFact) -> bool {
    (matches!(value.kind, ValueKindFact::Unknown)
        || matches!(value.kind,ValueKindFact::Numeric(n)if n.class==NumericClass::Double&&n.domain==NumericDomain::Real))
        && value.storage != StorageFact::Sparse
        && value.shape.known_dims().is_none_or(|dims| dims.len() <= 2)
}
pub(super) fn inspect(
    solver: Option<OptimSolver>,
    request: &CallRequest,
    field: Field,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> bool {
    if solver != Some(OptimSolver::Particleswarm) {
        return false;
    }
    let mut invalid = false;
    if let Some(value) = request.literals.numeric_at(index) {
        invalid = match field {
            Field::MaxTime | Field::MaxStallTime => value.is_nan() || value < 0.0,
            Field::ObjectiveLimit => value.is_nan(),
            Field::MaxIter => !value.is_finite() || value < 0.0 || value.fract() != 0.0,
            Field::DisplayInterval | Field::MaxStallIterations => {
                !value.is_finite() || value <= 0.0 || value.fract() != 0.0
            }
            Field::SwarmSize => !value.is_finite() || value < 2.0 || value.fract() != 0.0,
            Field::MinNeighborsFraction => !value.is_finite() || !(0.0..=1.0).contains(&value),
            Field::FunctionTolerance => !value.is_finite() || value < 0.0,
            Field::InitialSwarmSpan => !value.is_finite() || value <= 0.0,
            Field::SelfAdjustmentWeight | Field::SocialAdjustmentWeight => !value.is_finite(),
            Field::UseParallel | Field::UseVectorized => value != 0.0 && value != 1.0,
            _ => false,
        };
    }
    if let Some(text) = request.literals.literal_string_at(index) {
        invalid |= match field {
            Field::CreationFcn => crate::OptimSwarmCreation::parse(&text).is_none(),
            Field::HybridFcn => crate::OptimHybridSolver::parse(&text).is_none(),
            Field::Display => crate::OptimDisplay::parse(&text)
                .is_none_or(|mode| !OptimSolver::Particleswarm.accepts_display(mode)),
            Field::FunValCheck | Field::UseVectorized => crate::OptimSwitch::parse(&text).is_none(),
            Field::UseParallel => !matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "off" | "on" | "auto"
            ),
            Field::PlotFcn => text != "pswplotbestf",
            _ => false,
        };
    }
    if invalid {
        diagnostics.push(argument_error(
            "RM-CATALOG-SWARM-OPTION",
            "control is outside its particle-swarm domain",
            index,
        ));
    }
    true
}
