use super::*;

impl FeaResourcePlanningInput {
    pub const fn topology_digest(&self) -> Digest {
        match &self.demand {
            // Accuracy is scoped to the semantic numerical result, not a mesh operation.
            FeaResourceDemand::AccuracyEvaluation(demand) => demand.source_result.digest(),
            FeaResourceDemand::ResultPublication(demand) => demand.source_result.digest(),
            FeaResourceDemand::ThermalSolve(demand) => demand.topology_digest,
            FeaResourceDemand::SemanticStatePublication(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralLinearPipeline(demand) => demand.operator.topology_digest,
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
            FeaResourceDemand::AccuracyEvaluation(_) => None,
            FeaResourceDemand::ResultPublication(_) => None,
            FeaResourceDemand::ThermalSolve(_) => None,
            FeaResourceDemand::SemanticStatePublication(_) => None,
            FeaResourceDemand::StructuralLinearPipeline(demand) => {
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
            FeaResourceDemand::AccuracyEvaluation(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::ResultPublication(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::ThermalSolve(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::SemanticStatePublication(_) => &PartitioningPolicy::Monolithic,
            FeaResourceDemand::StructuralLinearPipeline(demand) => &demand.operator.partitioning,
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
            FeaResourceDemand::AccuracyEvaluation(_) => None,
            FeaResourceDemand::ResultPublication(_) => None,
            FeaResourceDemand::ThermalSolve(_) => None,
            FeaResourceDemand::SemanticStatePublication(_) => None,
            FeaResourceDemand::StructuralLinearPipeline(demand) => Some(&demand.operator),
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
            FeaResourceDemand::AccuracyEvaluation(_) => None,
            FeaResourceDemand::ResultPublication(_) => None,
            FeaResourceDemand::ThermalSolve(_) => None,
            FeaResourceDemand::SemanticStatePublication(_) => None,
            FeaResourceDemand::StructuralLinearPipeline(demand) => Some(&mut demand.operator),
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
