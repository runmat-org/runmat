//! Closed numerical phase and demand families; shared across planning and execution.
use super::*;

/// Canonical numerical phase governed by one exact resource request.
///
/// Task request identities prevent a plan estimated for one frequency entry or observation task
/// from authorizing another. The linear-pipeline variant preserves the existing whole-step
/// envelope until that path is decomposed into independently scheduled phases.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "request")]
pub enum FeaResourcePhase {
    AcousticFrequencyAssembly(AnalysisWorkloadIdentity),
    FluidSteadySolve(AnalysisWorkloadIdentity),
    FluidTransientBlock(AnalysisWorkloadIdentity),
    FluidFieldRecovery(FieldRecoveryRequestIdentity),
    AcousticHarmonic(AnalysisWorkloadIdentity),
    ElectricalStatic(AnalysisWorkloadIdentity),
    Magnetostatic(AnalysisWorkloadIdentity),
    MaxwellHarmonic(AnalysisWorkloadIdentity),
    MagneticFieldRecovery(FieldRecoveryRequestIdentity),
    ElectricalFieldRecovery(FieldRecoveryRequestIdentity),
    /// Random-response or response-spectrum fields recovered for authored observations.
    StatisticalFieldRecovery(FieldRecoveryRequestIdentity),
    StructuralFieldRecovery(FieldRecoveryRequestIdentity),
    NonlinearFieldRecovery(FieldRecoveryRequestIdentity),
    MaxwellFieldRecovery(FieldRecoveryRequestIdentity),
    AcousticResponse(AnalysisWorkloadIdentity),
    AcousticFieldRecovery(FieldRecoveryRequestIdentity),
    ResultPublication(crate::execution::PublicationRequestIdentity),
    SemanticStatePublication(Digest),
    AccuracyEvaluation(AccuracyEvaluationRequestIdentity),
    ThermalSolve(AnalysisWorkloadIdentity),
    StructuralLinearPipeline,
    StructuralNonlinearPipeline,
    StructuralNonlinearRecovery,
    StructuralModalExtraction(StructuralModalWorkloadIdentity),
    StructuralBuckling(StructuralBucklingWorkloadIdentity),
    StructuralFrequencyEntry(StructuralFrequencyEntryRequestIdentity),
    StructuralFrequencyFieldRecovery(FieldRecoveryRequestIdentity),
    StructuralObservationPartition(ObservationPartitionEvaluationRequestIdentity),
    StructuralObservationReduction(ObservationPartitionReductionRequestIdentity),
    StructuralObservationHistoryCheckpoint(ObservationHistoryCheckpointRequestIdentity),
    ObservationHistoryFinalization(ObservationHistoryFinalizationRequestIdentity),
    StructuralObservationAssembly(ObservationAssemblyRequestIdentity),
    StructuralFrequencyResultAssembly(StructuralFrequencyResultAssemblyRequestIdentity),
    StructuralTransientScopePreparation(Digest),
    StructuralTransientBlock(StructuralTransientBlockRequestIdentity),
    StructuralTransientFieldRecovery(SolverResultIdentity),
    StructuralTransientResultAssembly(StructuralTransientResultAssemblyRequestIdentity),
    StructuralModalResponseSpectrumDirection(
        StructuralModalResponseSpectrumDirectionRequestIdentity,
    ),
    StructuralModalResponseSpectrumAssembly(StructuralModalResponseSpectrumAssemblyRequestIdentity),
    StructuralRandomResponseInterval(StructuralRandomResponseIntervalRequestIdentity),
    StructuralRandomResponseAssembly(StructuralRandomResponseAssemblyRequestIdentity),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "demand")]
pub enum FeaResourceDemand {
    AcousticFrequencyAssembly(AcousticFrequencyAssemblyResourceDemand),
    FluidSteadySolve(FluidSteadySolveResourceDemand),
    FluidTransientBlock(FluidTransientBlockResourceDemand),
    FluidFieldRecovery(FluidFieldRecoveryResourceDemand),
    AcousticHarmonic(AcousticHarmonicResourceDemand),
    ElectricalStatic(ElectricalStaticResourceDemand),
    Magnetostatic(MagnetostaticResourceDemand),
    MaxwellHarmonic(MaxwellHarmonicResourceDemand),
    MagneticFieldRecovery(MagneticFieldRecoveryResourceDemand),
    ElectricalFieldRecovery(ElectricalFieldRecoveryResourceDemand),
    /// Host-only native-copy recovery with the same accounting as magnetic native fields.
    StatisticalFieldRecovery(MagneticFieldRecoveryResourceDemand),
    StructuralFieldRecovery(StructuralFieldRecoveryResourceDemand),
    NonlinearFieldRecovery(NonlinearFieldRecoveryResourceDemand),
    MaxwellFieldRecovery(MaxwellFieldRecoveryResourceDemand),
    AcousticResponse(AcousticResponseResourceDemand),
    AcousticFieldRecovery(AcousticFieldRecoveryResourceDemand),
    ResultPublication(ResultPublicationResourceDemand),
    SemanticStatePublication(SemanticStatePublicationResourceDemand),
    AccuracyEvaluation(AccuracyEvaluationResourceDemand),
    ThermalSolve(ThermalSolveResourceDemand),
    StructuralLinearPipeline(StructuralLinearPipelineResourceDemand),
    StructuralNonlinearPipeline(StructuralNonlinearPipelineResourceDemand),
    StructuralNonlinearRecovery(StructuralNonlinearRecoveryResourceDemand),
    StructuralModalExtraction(StructuralModalResourceDemand),
    StructuralBuckling(StructuralBucklingResourceDemand),
    StructuralFrequencyEntry(StructuralFrequencyOperatorResourceDemand),
    StructuralFrequencyFieldRecovery(StructuralFrequencyOperatorResourceDemand),
    StructuralObservationPartition(ObservationPartitionResourceDemand),
    StructuralObservationReduction(ObservationReductionResourceDemand),
    StructuralObservationHistoryCheckpoint(ObservationHistoryCheckpointResourceDemand),
    ObservationHistoryFinalization(ObservationHistoryFinalizationResourceDemand),
    StructuralObservationAssembly(ObservationAssemblyResourceDemand),
    StructuralFrequencyResultAssembly(StructuralFrequencyResultAssemblyResourceDemand),
    StructuralTransientScopePreparation(StructuralTransientScopeResourceDemand),
    StructuralTransientBlock(StructuralTransientBlockResourceDemand),
    StructuralTransientFieldRecovery(StructuralTransientFieldRecoveryResourceDemand),
    StructuralTransientResultAssembly(StructuralTransientResultAssemblyResourceDemand),
    StructuralModalResponseSpectrumDirection(
        StructuralModalResponseSpectrumDirectionResourceDemand,
    ),
    StructuralModalResponseSpectrumAssembly(StructuralModalResponseSpectrumAssemblyResourceDemand),
    StructuralRandomResponseInterval(StructuralRandomResponseIntervalResourceDemand),
    StructuralRandomResponseAssembly(StructuralRandomResponseAssemblyResourceDemand),
}
