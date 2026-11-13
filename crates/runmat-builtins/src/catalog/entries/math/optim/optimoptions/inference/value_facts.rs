//! Input and normalized output facts for typed optimizer option values.
use super::{facts, Field, OptimSolver};
use runmat_types::*;

pub(super) fn value_metadata(field: Field, value: &ValueFact) -> bool {
    if matches!(value.kind, ValueKindFact::Unknown) {
        return true;
    }
    if value.shape.element_count() == Some(0)
        && matches!(value.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
        && matches!(value.storage, StorageFact::Dense | StorageFact::Unknown)
    {
        return true;
    }
    if matches!(
        field,
        Field::Diagnostics
            | Field::FunValCheck
            | Field::Display
            | Field::Algorithm
            | Field::FiniteDifferenceType
            | Field::MeritFunction
            | Field::HessianFcn
            | Field::SubproblemAlgorithm
    ) {
        return text(value);
    }
    if matches!(field, Field::FiniteDifferenceStepSize | Field::TypicalX) {
        return matches!(
            value.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                domain: NumericDomain::Real
            })
        ) && value.storage != StorageFact::Sparse
            && value
                .shape
                .known_dims()
                .is_none_or(|dims| dims.iter().filter(|d| d.is_some_and(|n| n != 1)).count() <= 1);
    }
    if field == Field::HessianApproximation {
        return text(value)
            || matches!(&value.kind,ValueKindFact::Cell(cell) if cell.elements.len()==2 && text(&cell.elements[0]) && cell.elements[1].shape.element_count().is_none_or(|n|n==1));
    }
    if matches!(field, Field::UseParallel | Field::UseVectorized) && text(value) {
        return true;
    }
    if matches!(field, Field::OutputFcn | Field::PlotFcn) {
        return match &value.kind {
            ValueKindFact::Callable(_) => true,
            ValueKindFact::Cell(cell) => cell.elements.iter().all(|element| {
                matches!(
                    element.kind,
                    ValueKindFact::Callable(_) | ValueKindFact::Unknown
                )
            }),
            ValueKindFact::Numeric(numeric) => {
                numeric.domain == NumericDomain::Real
                    && value.shape.element_count() == Some(0)
                    && matches!(value.storage, StorageFact::Dense | StorageFact::Unknown)
            }
            _ => false,
        };
    }
    if matches!(
        field,
        Field::HessianMultiplyFcn | Field::JacobianMultiplyFcn
    ) {
        return matches!(value.kind, ValueKindFact::Callable(_));
    }
    if field == Field::HessPattern {
        let real = matches!(value.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
            || matches!(value.kind, ValueKindFact::Logical);
        return real
            && value.shape.known_dims().is_none_or(|dims| {
                dims.len() == 2 && (dims[0].is_none() || dims[1].is_none() || dims[0] == dims[1])
            });
    }
    if matches!(
        field,
        Field::SpecifyObjectiveGradient | Field::SpecifyConstraintGradient
    ) && text(value)
    {
        return true;
    }
    value.shape.element_count().is_none_or(|count| count == 1)
        && value.storage != StorageFact::Sparse
        && match value.kind {
            ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Real,
            ValueKindFact::Logical => {
                matches!(
                    field,
                    Field::SpecifyObjectiveGradient
                        | Field::SpecifyConstraintGradient
                        | Field::UseParallel
                ) || value.storage != StorageFact::Scalar
            }
            _ => false,
        }
}

pub(super) fn normalized(
    field: Field,
    solver: Option<OptimSolver>,
    source: Option<&ValueFact>,
) -> ValueFact {
    if solver == Some(OptimSolver::Quadprog) && field == Field::MaxTime {
        return source
            .cloned()
            .map(|mut value| {
                if matches!(value.kind, ValueKindFact::Numeric(_)) {
                    value.residency = ResidencyFact::Host;
                }
                value
            })
            .unwrap_or_else(|| facts::double(ShapeFact::Scalar, StorageFact::Scalar));
    }
    if solver == Some(OptimSolver::Ga) {
        return super::genetic::normalized(field, source);
    }
    if solver.is_none()
        && matches!(
            field,
            Field::PopulationType
                | Field::PopulationSize
                | Field::EliteCount
                | Field::CrossoverFraction
                | Field::InitialPopulationMatrix
                | Field::InitialScoresMatrix
                | Field::InitialPopulationRange
                | Field::CrossoverFcn
                | Field::MutationFcn
                | Field::SelectionFcn
                | Field::FitnessScalingFcn
                | Field::NonlinearConstraintAlgorithm
                | Field::MigrationDirection
                | Field::MigrationFraction
                | Field::MigrationInterval
                | Field::StallTest
        )
    {
        return source
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    }
    if solver == Some(OptimSolver::Patternsearch) {
        return super::pattern::normalized(field, source);
    }
    if solver == Some(OptimSolver::Surrogateopt) {
        return super::surrogate::normalized(field, source);
    }
    if solver.is_none()
        && OptimSolver::Simulannealbnd.accepts(field)
        && !OptimSolver::Generic.accepts(field)
    {
        return super::annealing::normalized(field, source);
    }
    if solver == Some(OptimSolver::Simulannealbnd) {
        return super::annealing::normalized(field, source);
    }
    if solver == Some(OptimSolver::Particleswarm) {
        if let Some(value) = source {
            let mut value = value.clone();
            if !matches!(
                value.kind,
                ValueKindFact::Struct(_)
                    | ValueKindFact::Object(_)
                    | ValueKindFact::Callable(_)
                    | ValueKindFact::Cell(_)
            ) {
                value.residency = ResidencyFact::Host;
            }
            return value;
        }
        return match field {
            Field::CreationFcn | Field::Display | Field::FunValCheck | Field::UseParallel => {
                ValueFact::scalar(ValueKindFact::String)
            }
            Field::UseVectorized => ValueFact::scalar(ValueKindFact::Logical),
            Field::InertiaRange => {
                facts::double(ShapeFact::from(vec![Some(1), Some(2)]), StorageFact::Dense)
            }
            Field::HybridFcn
            | Field::InitialPoints
            | Field::OutputFcn
            | Field::PlotFcn
            | Field::SwarmSize
            | Field::MaxIter => {
                facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
            }
            _ => facts::double(ShapeFact::Scalar, StorageFact::Scalar),
        };
    }
    if let Some(source) = source.filter(|source| source.shape.element_count() == Some(0)) {
        return source.clone();
    }
    match field {
        Field::CreationFcn
        | Field::HybridFcn
        | Field::InitialPoints
        | Field::InitialSwarmSpan
        | Field::InertiaRange
        | Field::UseVectorized
        | Field::MaxStallTime => source
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        Field::FiniteDifferenceStepSize | Field::TypicalX => source
            .map(|source| {
                let mut fact = source.clone();
                fact.residency = ResidencyFact::Host;
                fact
            })
            .unwrap_or_else(|| {
                facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
            }),
        Field::HessianApproximation | Field::HessUpdate => source
            .cloned()
            .unwrap_or_else(|| ValueFact::scalar(ValueKindFact::String)),
        Field::UseParallel => source
            .map(|source| {
                let mut fact = source.clone();
                fact.residency = ResidencyFact::Host;
                fact
            })
            .unwrap_or_else(|| ValueFact::scalar(ValueKindFact::Logical)),
        Field::OutputFcn
        | Field::PlotFcn
        | Field::HessianMultiplyFcn
        | Field::JacobianMultiplyFcn => source.cloned().unwrap_or_else(|| {
            facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
        }),
        Field::Diagnostics
        | Field::FunValCheck
        | Field::Display
        | Field::Algorithm
        | Field::FiniteDifferenceType
        | Field::MeritFunction => ValueFact::scalar(ValueKindFact::String),
        Field::HessianFcn => source
            .map(|value| {
                if matches!(value.kind, ValueKindFact::Unknown) {
                    value.clone()
                } else {
                    ValueFact::scalar(ValueKindFact::String)
                }
            })
            .unwrap_or_else(|| {
                facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
            }),
        Field::SubproblemAlgorithm => source
            .map(|value| {
                if matches!(value.kind, ValueKindFact::Unknown) {
                    value.clone()
                } else {
                    ValueFact::scalar(ValueKindFact::String)
                }
            })
            .unwrap_or_else(|| ValueFact::scalar(ValueKindFact::String)),
        Field::HessPattern => source
            .map(|value| {
                let mut fact = value.clone();
                fact.residency = ResidencyFact::Host;
                fact
            })
            .unwrap_or_else(|| {
                facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
            }),
        Field::SpecifyObjectiveGradient | Field::SpecifyConstraintGradient => {
            ValueFact::scalar(ValueKindFact::Logical)
        }
        Field::MaxPCGIter | Field::MaxSQPIter | Field::RelLineSrchBnd if source.is_none() => {
            facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense)
        }
        Field::MaxSQPIter
        | Field::RelLineSrchBndDuration
        | Field::MaxIter
        | Field::AbsoluteMaxObjectiveCount
        | Field::EqualityGoalCount
        | Field::MaxFunEvals
        | Field::MaxPCGIter
        | Field::PrecondBandWidth
        | Field::MaxNodes
        | Field::MaxFeasiblePoints => match source.map(|value| &value.kind) {
            Some(ValueKindFact::Numeric(numeric)) if numeric.class.integer_class().is_some() => {
                ValueFact::scalar(ValueKindFact::Numeric(*numeric))
            }
            Some(ValueKindFact::Unknown) => ValueFact::scalar(ValueKindFact::Unknown),
            _ => facts::double(ShapeFact::Scalar, StorageFact::Scalar),
        },
        _ => facts::double(ShapeFact::Scalar, StorageFact::Scalar),
    }
}

pub(super) fn text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Character => value.shape.known_dims().is_none_or(|dims| {
            dims.len() == 1
                || dims
                    .first()
                    .is_some_and(|rows| rows.is_none_or(|rows| rows == 1))
        }),
        _ => false,
    }
}

pub(super) fn solver_value_metadata(
    field: Field,
    solver: Option<OptimSolver>,
    value: &ValueFact,
) -> bool {
    if solver == Some(OptimSolver::Quadprog) && field == Field::MaxTime {
        return super::swarm::metadata(field, value);
    }
    if solver == Some(OptimSolver::Surrogateopt) {
        return super::surrogate::metadata(field, value);
    }
    if solver == Some(OptimSolver::Ga) {
        return super::genetic::metadata(field, value);
    }
    if solver == Some(OptimSolver::Patternsearch) {
        return super::pattern::metadata(field, value);
    }
    if solver.is_none() {
        return value_metadata(field, value)
            || OptimSolver::Ga.accepts(field) && super::genetic::metadata(field, value)
            || OptimSolver::Simulannealbnd.accepts(field)
                && super::annealing::metadata(field, value)
            || OptimSolver::Particleswarm.accepts(field) && super::swarm::metadata(field, value);
    }
    if solver == Some(OptimSolver::Simulannealbnd) {
        return super::annealing::metadata(field, value);
    }
    if solver == Some(OptimSolver::Particleswarm) {
        return super::swarm::metadata(field, value);
    }
    if solver == Some(OptimSolver::Intlinprog) && matches!(field, Field::OutputFcn | Field::PlotFcn)
    {
        super::super::super::milp_observers::callback_metadata(value)
    } else if solver == Some(OptimSolver::Fseminf) && field == Field::PlotFcn {
        match &value.kind {
            ValueKindFact::String | ValueKindFact::Character => text(value),
            ValueKindFact::Cell(cell) => cell
                .elements
                .iter()
                .all(|v| text(v) || matches!(v.kind, ValueKindFact::Callable(_))),
            _ => value_metadata(field, value),
        }
    } else {
        value_metadata(field, value)
    }
}

/// One final typed public-key projection; existing profiles retain exact names.
pub(super) fn public_names(
    solver: Option<OptimSolver>,
    fields: &mut std::collections::BTreeMap<String, ValueFact>,
) {
    let Some(solver) = solver else { return };
    for field in Field::ALL {
        let public = field.name_for_solver(solver);
        if public != field.name() {
            if let Some(value) = fields.remove(field.name()) {
                fields.insert(public.into(), value);
            }
        }
    }
}
