use super::OptimOptionField as Field;
pub(super) const fn accepts(field: Field) -> bool {
    matches!(
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
            | Field::InitialPenalty
            | Field::PenaltyFactor
            | Field::PlotInterval
            | Field::Display
            | Field::FunctionTolerance
            | Field::MaxIter
            | Field::FunValCheck
            | Field::MaxTime
            | Field::ObjectiveLimit
            | Field::OutputFcn
            | Field::PlotFcn
            | Field::UseParallel
            | Field::CreationFcn
            | Field::HybridFcn
            | Field::DisplayInterval
            | Field::MaxStallIterations
            | Field::MaxStallTime
            | Field::UseVectorized
            | Field::TolCon
    )
}
