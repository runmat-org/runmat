use super::*;

impl FeaResourcePlanningInput {
    pub const fn topology_digest(&self) -> Digest {
        match &self.demand {
            FeaResourceDemand::AcousticFrequencyAssembly(demand) => demand.topology_digest,
            // Accuracy is scoped to the semantic numerical result, not a mesh operation.
            FeaResourceDemand::AccuracyEvaluation(demand) => demand.source_result.digest(),
            FeaResourceDemand::ResultPublication(demand) => demand.source_result.digest(),
            FeaResourceDemand::ThermalSolve(demand) => demand.topology_digest,
            FeaResourceDemand::AcousticHarmonic(demand) => demand.topology_digest,
            FeaResourceDemand::ElectricalStatic(demand) => demand.topology_digest,
            FeaResourceDemand::Magnetostatic(demand) => demand.topology_digest,
            FeaResourceDemand::MaxwellHarmonic(demand) => demand.topology_digest,
            FeaResourceDemand::AcousticResponse(demand) => demand.topology_digest,
            FeaResourceDemand::MaxwellFieldRecovery(demand) => demand.topology_digest,
            FeaResourceDemand::AcousticFieldRecovery(demand) => demand.topology_digest,
            FeaResourceDemand::FluidFieldRecovery(demand) => demand.topology_digest,
            FeaResourceDemand::FluidSteadySolve(demand) => demand.projection.topology_digest,
            FeaResourceDemand::FluidTransientBlock(demand) => demand.projection.topology_digest,
            FeaResourceDemand::ElectricalFieldRecovery(demand) => demand.topology_digest,
            FeaResourceDemand::MagneticFieldRecovery(demand)
            | FeaResourceDemand::StatisticalFieldRecovery(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralFieldRecovery(demand) => demand.topology_digest,
            FeaResourceDemand::NonlinearFieldRecovery(demand) => demand.topology_digest,
            FeaResourceDemand::SemanticStatePublication(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralLinearPipeline(demand) => demand.operator.topology_digest,
            FeaResourceDemand::StructuralNonlinearRecovery(demand) => {
                demand.operator.topology_digest
            }
            FeaResourceDemand::StructuralNonlinearPipeline(demand) => {
                demand.operator.topology_digest
            }
            FeaResourceDemand::StructuralModalExtraction(demand) => demand.operator.topology_digest,
            FeaResourceDemand::StructuralBuckling(demand) => demand.operator.topology_digest,
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => {
                demand.operator.topology_digest
            }
            FeaResourceDemand::StructuralObservationPartition(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralObservationReduction(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralObservationHistoryCheckpoint(demand) => {
                demand.topology_digest
            }
            FeaResourceDemand::ObservationHistoryFinalization(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralObservationAssembly(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralFrequencyResultAssembly(demand) => demand.scope_digest,
            FeaResourceDemand::StructuralTransientScopePreparation(demand) => {
                demand.operator.topology_digest
            }
            FeaResourceDemand::StructuralTransientBlock(demand) => demand.operator.topology_digest,
            FeaResourceDemand::StructuralTransientFieldRecovery(demand) => {
                demand.operator.topology_digest
            }
            FeaResourceDemand::StructuralTransientResultAssembly(demand) => demand.scope_digest,
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                demand.operator.topology_digest
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(demand) => {
                demand.scope_digest
            }
            FeaResourceDemand::StructuralRandomResponseInterval(demand) => {
                demand.operator.topology_digest
            }
            FeaResourceDemand::StructuralRandomResponseAssembly(demand) => demand.scope_digest,
        }
    }

    pub const fn observations(&self) -> Option<&ObservationResourceFacts> {
        match &self.demand {
            FeaResourceDemand::MaxwellFieldRecovery(_) => None,
            FeaResourceDemand::AcousticFieldRecovery(_) => None,
            FeaResourceDemand::FluidFieldRecovery(_) => None,
            FeaResourceDemand::FluidSteadySolve(_) | FeaResourceDemand::FluidTransientBlock(_) => {
                None
            }
            FeaResourceDemand::ElectricalFieldRecovery(_) => None,
            FeaResourceDemand::MagneticFieldRecovery(_)
            | FeaResourceDemand::StatisticalFieldRecovery(_) => None,
            FeaResourceDemand::StructuralFieldRecovery(_) => None,
            FeaResourceDemand::NonlinearFieldRecovery(_) => None,
            FeaResourceDemand::AccuracyEvaluation(_) => None,
            FeaResourceDemand::ResultPublication(_) => None,
            FeaResourceDemand::ThermalSolve(_) => None,
            FeaResourceDemand::AcousticHarmonic(_) => None,
            FeaResourceDemand::ElectricalStatic(_) => None,
            FeaResourceDemand::Magnetostatic(_) => None,
            FeaResourceDemand::MaxwellHarmonic(_) => None,
            FeaResourceDemand::AcousticResponse(_) => None,
            FeaResourceDemand::SemanticStatePublication(_) => None,
            FeaResourceDemand::StructuralLinearPipeline(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralNonlinearRecovery(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralNonlinearPipeline(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralModalExtraction(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralBuckling(demand) => Some(&demand.operator.observations),
            FeaResourceDemand::StructuralTransientScopePreparation(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralTransientBlock(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralTransientFieldRecovery(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralObservationPartition(_)
            | FeaResourceDemand::StructuralObservationReduction(_)
            | FeaResourceDemand::StructuralObservationHistoryCheckpoint(_)
            | FeaResourceDemand::ObservationHistoryFinalization(_)
            | FeaResourceDemand::StructuralObservationAssembly(_)
            | FeaResourceDemand::StructuralFrequencyResultAssembly(_)
            | FeaResourceDemand::AcousticFrequencyAssembly(_)
            | FeaResourceDemand::StructuralTransientResultAssembly(_) => None,
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(_) => None,
            FeaResourceDemand::StructuralRandomResponseInterval(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralRandomResponseAssembly(_) => None,
        }
    }

    pub const fn partitioning(&self) -> &PartitioningPolicy {
        match &self.demand {
            FeaResourceDemand::MaxwellFieldRecovery(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::AcousticFieldRecovery(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::FluidFieldRecovery(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::FluidSteadySolve(_) | FeaResourceDemand::FluidTransientBlock(_) => {
                &PartitioningPolicy::Monolithic
            }
            FeaResourceDemand::ElectricalFieldRecovery(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::MagneticFieldRecovery(_)
            | FeaResourceDemand::StatisticalFieldRecovery(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::StructuralFieldRecovery(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::NonlinearFieldRecovery(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::AccuracyEvaluation(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::ResultPublication(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::ThermalSolve(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::AcousticHarmonic(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::ElectricalStatic(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::Magnetostatic(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::MaxwellHarmonic(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::AcousticResponse(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::SemanticStatePublication(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::StructuralLinearPipeline(demand) => &demand.operator.partitioning,
            FeaResourceDemand::StructuralNonlinearRecovery(demand) => &demand.operator.partitioning,
            FeaResourceDemand::StructuralNonlinearPipeline(demand) => &demand.operator.partitioning,
            FeaResourceDemand::StructuralModalExtraction(demand) => &demand.operator.partitioning,
            FeaResourceDemand::StructuralBuckling(demand) => &demand.operator.partitioning,
            FeaResourceDemand::StructuralTransientScopePreparation(demand) => {
                &demand.operator.partitioning
            }
            FeaResourceDemand::StructuralTransientBlock(demand) => &demand.operator.partitioning,
            FeaResourceDemand::StructuralTransientFieldRecovery(demand) => {
                &demand.operator.partitioning
            }
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => {
                &demand.operator.partitioning
            }
            FeaResourceDemand::StructuralObservationPartition(_)
            | FeaResourceDemand::StructuralObservationReduction(_)
            | FeaResourceDemand::StructuralObservationHistoryCheckpoint(_)
            | FeaResourceDemand::ObservationHistoryFinalization(_)
            | FeaResourceDemand::StructuralObservationAssembly(_)
            | FeaResourceDemand::StructuralFrequencyResultAssembly(_)
            | FeaResourceDemand::AcousticFrequencyAssembly(_)
            | FeaResourceDemand::StructuralTransientResultAssembly(_) => {
                static MONOLITHIC: PartitioningPolicy = PartitioningPolicy::Monolithic;
                &MONOLITHIC
            }
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                &demand.operator.partitioning
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(_) => {
                static MONOLITHIC: PartitioningPolicy = PartitioningPolicy::Monolithic;
                &MONOLITHIC
            }
            FeaResourceDemand::StructuralRandomResponseInterval(demand) => {
                &demand.operator.partitioning
            }
            FeaResourceDemand::StructuralRandomResponseAssembly(_) => {
                static MONOLITHIC: PartitioningPolicy = PartitioningPolicy::Monolithic;
                &MONOLITHIC
            }
        }
    }

    pub const fn structural_operator(&self) -> Option<&StructuralOperatorResourceDemand> {
        match &self.demand {
            FeaResourceDemand::MaxwellFieldRecovery(_) => None,
            FeaResourceDemand::AcousticFieldRecovery(_) => None,
            FeaResourceDemand::FluidFieldRecovery(_) => None,
            FeaResourceDemand::FluidSteadySolve(_) | FeaResourceDemand::FluidTransientBlock(_) => {
                None
            }
            FeaResourceDemand::ElectricalFieldRecovery(_) => None,
            FeaResourceDemand::MagneticFieldRecovery(_)
            | FeaResourceDemand::StatisticalFieldRecovery(_) => None,
            FeaResourceDemand::StructuralFieldRecovery(_) => None,
            FeaResourceDemand::NonlinearFieldRecovery(_) => None,
            FeaResourceDemand::AccuracyEvaluation(_) => None,
            FeaResourceDemand::ResultPublication(_) => None,
            FeaResourceDemand::ThermalSolve(_) => None,
            FeaResourceDemand::AcousticHarmonic(_) => None,
            FeaResourceDemand::ElectricalStatic(_) => None,
            FeaResourceDemand::Magnetostatic(_) => None,
            FeaResourceDemand::MaxwellHarmonic(_) => None,
            FeaResourceDemand::AcousticResponse(_) => None,
            FeaResourceDemand::SemanticStatePublication(_) => None,
            FeaResourceDemand::StructuralLinearPipeline(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralNonlinearRecovery(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralNonlinearPipeline(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralModalExtraction(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralBuckling(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralTransientScopePreparation(demand) => {
                Some(&demand.operator)
            }
            FeaResourceDemand::StructuralTransientBlock(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralTransientFieldRecovery(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralObservationPartition(_)
            | FeaResourceDemand::StructuralObservationReduction(_)
            | FeaResourceDemand::StructuralObservationHistoryCheckpoint(_)
            | FeaResourceDemand::ObservationHistoryFinalization(_)
            | FeaResourceDemand::StructuralObservationAssembly(_)
            | FeaResourceDemand::StructuralFrequencyResultAssembly(_)
            | FeaResourceDemand::AcousticFrequencyAssembly(_)
            | FeaResourceDemand::StructuralTransientResultAssembly(_) => None,
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                Some(&demand.operator)
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(_) => None,
            FeaResourceDemand::StructuralRandomResponseInterval(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralRandomResponseAssembly(_) => None,
        }
    }

    pub fn structural_operator_mut(&mut self) -> Option<&mut StructuralOperatorResourceDemand> {
        match &mut self.demand {
            FeaResourceDemand::MaxwellFieldRecovery(_) => None,
            FeaResourceDemand::AcousticFieldRecovery(_) => None,
            FeaResourceDemand::FluidFieldRecovery(_) => None,
            FeaResourceDemand::FluidSteadySolve(_) | FeaResourceDemand::FluidTransientBlock(_) => {
                None
            }
            FeaResourceDemand::ElectricalFieldRecovery(_) => None,
            FeaResourceDemand::MagneticFieldRecovery(_)
            | FeaResourceDemand::StatisticalFieldRecovery(_) => None,
            FeaResourceDemand::StructuralFieldRecovery(_) => None,
            FeaResourceDemand::NonlinearFieldRecovery(_) => None,
            FeaResourceDemand::AccuracyEvaluation(_) => None,
            FeaResourceDemand::ResultPublication(_) => None,
            FeaResourceDemand::ThermalSolve(_) => None,
            FeaResourceDemand::AcousticHarmonic(_) => None,
            FeaResourceDemand::ElectricalStatic(_) => None,
            FeaResourceDemand::Magnetostatic(_) => None,
            FeaResourceDemand::MaxwellHarmonic(_) => None,
            FeaResourceDemand::AcousticResponse(_) => None,
            FeaResourceDemand::SemanticStatePublication(_) => None,
            FeaResourceDemand::StructuralLinearPipeline(demand) => Some(&mut demand.operator),
            FeaResourceDemand::StructuralNonlinearRecovery(demand) => Some(&mut demand.operator),
            FeaResourceDemand::StructuralNonlinearPipeline(demand) => Some(&mut demand.operator),
            FeaResourceDemand::StructuralModalExtraction(demand) => Some(&mut demand.operator),
            FeaResourceDemand::StructuralBuckling(demand) => Some(&mut demand.operator),
            FeaResourceDemand::StructuralTransientScopePreparation(demand) => {
                Some(&mut demand.operator)
            }
            FeaResourceDemand::StructuralTransientBlock(demand) => Some(&mut demand.operator),
            FeaResourceDemand::StructuralTransientFieldRecovery(demand) => {
                Some(&mut demand.operator)
            }
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => {
                Some(&mut demand.operator)
            }
            FeaResourceDemand::StructuralObservationPartition(_)
            | FeaResourceDemand::StructuralObservationReduction(_)
            | FeaResourceDemand::StructuralObservationHistoryCheckpoint(_)
            | FeaResourceDemand::ObservationHistoryFinalization(_)
            | FeaResourceDemand::StructuralObservationAssembly(_)
            | FeaResourceDemand::StructuralFrequencyResultAssembly(_)
            | FeaResourceDemand::AcousticFrequencyAssembly(_)
            | FeaResourceDemand::StructuralTransientResultAssembly(_) => None,
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                Some(&mut demand.operator)
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(_) => None,
            FeaResourceDemand::StructuralRandomResponseInterval(demand) => {
                Some(&mut demand.operator)
            }
            FeaResourceDemand::StructuralRandomResponseAssembly(_) => None,
        }
    }
}
