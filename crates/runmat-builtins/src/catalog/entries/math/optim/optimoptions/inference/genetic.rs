//! GA constructor facts describe retained options, not executed populations.
use super::{argument_error, text, Field};
use crate::*;
use runmat_types::*;
fn real(v: &ValueFact) -> bool {
    matches!(v.kind, ValueKindFact::Unknown)
        || matches!(v.kind,ValueKindFact::Numeric(n)if n.domain==NumericDomain::Real)
            && v.storage != StorageFact::Sparse
}
fn vector(v: &ValueFact) -> bool {
    v.shape
        .known_dims()
        .is_none_or(|d| d.len() <= 2 && d.iter().filter(|n| n.is_some_and(|n| n != 1)).count() <= 1)
}
pub(super) fn metadata(field: Field, v: &ValueFact) -> bool {
    if matches!(v.kind, ValueKindFact::Unknown) || v.shape.element_count() == Some(0) {
        return true;
    }
    match field {
        // Activation/representation belongs to the later actual solver.
        Field::InitialPopulationMatrix | Field::InitialScoresMatrix => true,
        Field::PopulationType
        | Field::NonlinearConstraintAlgorithm
        | Field::MigrationDirection
        | Field::StallTest
        | Field::Display
        | Field::FunValCheck => text(v),
        Field::CreationFcn
        | Field::CrossoverFcn
        | Field::MutationFcn
        | Field::SelectionFcn
        | Field::FitnessScalingFcn => {
            text(v)
                || matches!(v.kind, ValueKindFact::Callable(_))
                || matches!(&v.kind,ValueKindFact::Cell(c)if c.elements.first().is_none_or(|first|text(first)||matches!(first.kind,ValueKindFact::Callable(_)|ValueKindFact::Unknown)))
        }
        Field::OutputFcn | Field::PlotFcn => super::swarm::metadata(field, v),
        Field::HybridFcn => super::swarm::metadata(field, v),
        Field::MaxTime | Field::MaxStallTime => super::swarm::metadata(field, v),
        Field::PopulationSize => real(v) && vector(v),
        Field::InitialPopulationRange => {
            real(v)
                && v.shape
                    .known_dims()
                    .is_none_or(|d| d.len() == 2 && d[0].is_none_or(|n| n == 2))
        }
        Field::UseVectorized | Field::UseParallel => {
            text(v)
                || matches!(v.kind, ValueKindFact::Logical)
                    && v.shape.element_count().is_none_or(|n| n == 1)
                || real(v) && v.shape.element_count().is_none_or(|n| n == 1)
        }
        _ => real(v) && v.shape.element_count().is_none_or(|n| n == 1),
    }
}
pub(super) fn inspect(
    r: &CallRequest,
    field: Field,
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !metadata(field, value) {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-OPTION-TYPE",
            "GA option has an unsupported class or geometry",
            index,
        ));
    }
    if let Some(name) = r.literals.literal_string_at(index) {
        let valid = match field {
            Field::PopulationType => OptimGeneticPopulation::parse(&name).is_some(),
            Field::NonlinearConstraintAlgorithm => OptimGeneticConstraint::parse(&name).is_some(),
            Field::MigrationDirection => OptimGeneticMigration::parse(&name).is_some(),
            Field::StallTest => OptimGeneticStall::parse(&name).is_some(),
            Field::Display => {
                OptimDisplay::parse(&name).is_some_and(|mode| OptimSolver::Ga.accepts_display(mode))
            }
            Field::FunValCheck => OptimSwitch::parse(&name).is_some(),
            Field::CreationFcn => OptimGeneticCreation::parse(&name).is_some(),
            Field::CrossoverFcn => OptimGeneticCrossover::parse(&name).is_some(),
            Field::MutationFcn => OptimGeneticMutation::parse(&name).is_some(),
            Field::SelectionFcn => OptimGeneticSelection::parse(&name).is_some(),
            Field::FitnessScalingFcn => OptimGeneticScaling::parse(&name).is_some(),
            Field::HybridFcn => OptimHybridSolver::parse(&name).is_some(),
            Field::UseParallel => {
                matches!(name.to_ascii_lowercase().as_str(), "off" | "auto" | "on")
            }
            Field::UseVectorized => OptimSwitch::parse(&name).is_some(),
            _ => true,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-OPTION-SELECTOR",
                "unrecognized GA selector",
                index,
            ));
        }
    }
    if let Some(v) = r.literals.numeric_at(index) {
        let valid = match field {
            Field::MaxIter | Field::EliteCount | Field::MaxStallIterations => {
                v.is_finite() && v >= 0. && v.fract() == 0.
            }
            Field::PopulationSize
            | Field::DisplayInterval
            | Field::PlotInterval
            | Field::MigrationInterval => v.is_finite() && v >= 1. && v.fract() == 0.,
            Field::FunctionTolerance | Field::TolCon => v.is_finite() && v >= 0.,
            Field::CrossoverFraction | Field::MigrationFraction => {
                v.is_finite() && (0.0..=1.0).contains(&v)
            }
            Field::InitialPenalty => v.is_finite() && v >= 1.,
            Field::PenaltyFactor => v.is_finite() && v > 1.,
            Field::MaxTime => !v.is_nan() && v >= 0.,
            Field::MaxStallTime => !v.is_nan() && v > 0.,
            Field::ObjectiveLimit => !v.is_nan(),
            Field::UseVectorized | Field::UseParallel => v == 0. || v == 1.,
            _ => true,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-OPTION-RANGE",
                "GA option is outside its documented range",
                index,
            ));
        }
    }
}
pub(super) fn normalized(field: Field, source: Option<&ValueFact>) -> ValueFact {
    if let Some(value) = source {
        let mut value = value.clone();
        if !matches!(
            field,
            Field::InitialPopulationMatrix | Field::InitialScoresMatrix
        ) && matches!(
            value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ) {
            value.residency = ResidencyFact::Host;
        }
        value.alias = AliasFact::Unknown;
        return value;
    }
    match field {
        Field::MaxIter
        | Field::PopulationSize
        | Field::EliteCount
        | Field::InitialPopulationMatrix
        | Field::InitialScoresMatrix
        | Field::InitialPopulationRange
        | Field::CreationFcn
        | Field::CrossoverFcn
        | Field::MutationFcn
        | Field::SelectionFcn
        | Field::NonlinearConstraintAlgorithm
        | Field::HybridFcn
        | Field::OutputFcn
        | Field::PlotFcn => {
            super::facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
        }
        Field::PopulationType
        | Field::StallTest
        | Field::FitnessScalingFcn
        | Field::MigrationDirection
        | Field::FunValCheck
        | Field::UseParallel
        | Field::Display => ValueFact::scalar(ValueKindFact::String),
        Field::UseVectorized => ValueFact::scalar(ValueKindFact::Logical),
        _ => super::facts::double(ShapeFact::Scalar, StorageFact::Scalar),
    }
}
