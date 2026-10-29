use super::*;

impl FeaResourceDemand {
    /// Returns the admitted semantic envelope for one output kind governed by this phase.
    /// Legacy whole-pipeline plans predate independently prepared output envelopes.
    pub const fn output_facts(
        &self,
        kind: AnalysisNumericalObjectKind,
    ) -> Option<&PreparedOutputResourceFacts> {
        match (self, kind) {
            (
                Self::FluidTransientBlock(demand),
                AnalysisNumericalObjectKind::FluidTransientCheckpoint,
            ) => Some(&demand.output),
            (
                Self::StructuralNonlinearPipeline(demand),
                AnalysisNumericalObjectKind::NumericalReceipt,
            ) => match &demand.block_transport {
                Some(block) => Some(&block.receipt_output),
                None => None,
            },
            (
                Self::StructuralNonlinearPipeline(demand),
                AnalysisNumericalObjectKind::NonlinearSolidBlockArtifact,
            ) => match &demand.block_transport {
                Some(block) => Some(&block.output),
                None => None,
            },
            (Self::FluidFieldRecovery(demand), AnalysisNumericalObjectKind::Field) => {
                Some(&demand.output)
            }
            (Self::FluidSteadySolve(demand), AnalysisNumericalObjectKind::SolverResult) => {
                Some(&demand.result_output)
            }
            (Self::FluidSteadySolve(demand), AnalysisNumericalObjectKind::NumericalReceipt) => {
                Some(&demand.receipt_output)
            }
            (Self::MaxwellFieldRecovery(demand), AnalysisNumericalObjectKind::Field) => {
                Some(&demand.output)
            }
            (Self::AcousticFieldRecovery(demand), AnalysisNumericalObjectKind::Field) => {
                Some(&demand.output)
            }
            (Self::AcousticResponse(demand), AnalysisNumericalObjectKind::SolverResult) => {
                Some(&demand.result_output)
            }
            (Self::AcousticResponse(demand), AnalysisNumericalObjectKind::NumericalReceipt) => {
                Some(&demand.receipt_output)
            }
            (
                Self::AcousticResponse(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (Self::MaxwellHarmonic(demand), AnalysisNumericalObjectKind::SolverResult) => {
                Some(&demand.result_output)
            }
            (Self::MaxwellHarmonic(demand), AnalysisNumericalObjectKind::NumericalReceipt) => {
                Some(&demand.receipt_output)
            }
            (
                Self::MaxwellHarmonic(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (Self::StructuralFieldRecovery(demand), AnalysisNumericalObjectKind::Field) => {
                Some(&demand.output)
            }
            (Self::NonlinearFieldRecovery(demand), AnalysisNumericalObjectKind::Field) => {
                Some(&demand.output)
            }
            (
                Self::MagneticFieldRecovery(demand) | Self::StatisticalFieldRecovery(demand),
                AnalysisNumericalObjectKind::Field,
            ) => Some(&demand.output),
            (Self::ElectricalFieldRecovery(demand), AnalysisNumericalObjectKind::Field) => {
                Some(&demand.output)
            }
            (Self::Magnetostatic(demand), AnalysisNumericalObjectKind::SolverResult) => {
                Some(&demand.result_output)
            }
            (Self::Magnetostatic(demand), AnalysisNumericalObjectKind::NumericalReceipt) => {
                Some(&demand.receipt_output)
            }
            (
                Self::Magnetostatic(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (Self::ElectricalStatic(demand), AnalysisNumericalObjectKind::SolverResult) => {
                Some(&demand.result_output)
            }
            (Self::ElectricalStatic(demand), AnalysisNumericalObjectKind::NumericalReceipt) => {
                Some(&demand.receipt_output)
            }
            (
                Self::ElectricalStatic(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (Self::AcousticHarmonic(demand), AnalysisNumericalObjectKind::SolverResult) => {
                Some(&demand.result_output)
            }
            (Self::AcousticHarmonic(demand), AnalysisNumericalObjectKind::NumericalReceipt) => {
                Some(&demand.receipt_output)
            }
            (
                Self::AcousticHarmonic(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (
                Self::ObservationHistoryFinalization(demand),
                AnalysisNumericalObjectKind::History,
            ) => Some(&demand.history_output),
            (
                Self::ObservationHistoryFinalization(demand),
                AnalysisNumericalObjectKind::SeriesReductionResult,
            ) => demand.series_output.as_ref(),
            (Self::ResultPublication(demand), AnalysisNumericalObjectKind::PublishedResult) => {
                Some(&demand.output)
            }
            (
                Self::StructuralNonlinearPipeline(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (
                Self::StructuralLinearPipeline(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (
                Self::StructuralModalExtraction(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (
                Self::StructuralModalResponseSpectrumAssembly(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (
                Self::StructuralRandomResponseAssembly(demand),
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ) => Some(&demand.state_profiles_output),
            (
                Self::SemanticStatePublication(demand),
                AnalysisNumericalObjectKind::SemanticState,
            ) => Some(&demand.output),
            (Self::AccuracyEvaluation(demand), AnalysisNumericalObjectKind::AccuracyReport) => {
                Some(&demand.output)
            }
            (Self::ThermalSolve(demand), AnalysisNumericalObjectKind::SolverResult) => {
                Some(&demand.result_output)
            }
            (Self::ThermalSolve(demand), AnalysisNumericalObjectKind::NumericalReceipt) => {
                Some(&demand.receipt_output)
            }
            (Self::ThermalSolve(demand), AnalysisNumericalObjectKind::StatePublicationProfiles) => {
                Some(&demand.state_profiles_output)
            }
            (
                Self::StructuralModalExtraction(demand),
                AnalysisNumericalObjectKind::SolverResult,
            ) => Some(&demand.result_output),
            (
                Self::StructuralModalExtraction(demand),
                AnalysisNumericalObjectKind::NumericalReceipt,
            ) => Some(&demand.receipt_output),
            (Self::StructuralBuckling(demand), AnalysisNumericalObjectKind::SolverResult) => {
                Some(&demand.result_output)
            }
            (Self::StructuralBuckling(demand), AnalysisNumericalObjectKind::NumericalReceipt) => {
                Some(&demand.receipt_output)
            }
            (
                Self::StructuralFrequencyEntry(demand),
                AnalysisNumericalObjectKind::StructuralFrequencyEntryResult,
            )
            | (
                Self::StructuralFrequencyFieldRecovery(demand),
                AnalysisNumericalObjectKind::Field,
            ) => Some(&demand.output),
            (
                Self::StructuralObservationPartition(demand),
                AnalysisNumericalObjectKind::ObservationPartitionContribution,
            ) => Some(&demand.output),
            (
                Self::StructuralObservationReduction(demand),
                AnalysisNumericalObjectKind::ObservationResult,
            ) => Some(&demand.output),
            (
                Self::StructuralObservationHistoryCheckpoint(demand),
                AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
            ) => Some(&demand.output),
            (
                Self::StructuralObservationAssembly(demand),
                AnalysisNumericalObjectKind::Observation,
            ) => Some(&demand.output),
            (
                Self::AcousticFrequencyAssembly(demand),
                AnalysisNumericalObjectKind::SolverResult,
            ) => Some(&demand.result_output),
            (
                Self::AcousticFrequencyAssembly(demand),
                AnalysisNumericalObjectKind::FrequencySamplePublication,
            ) => Some(&demand.publication_output),
            (
                Self::StructuralFrequencyResultAssembly(demand),
                AnalysisNumericalObjectKind::SolverResult,
            ) => Some(&demand.result_output),
            (
                Self::StructuralFrequencyResultAssembly(demand),
                AnalysisNumericalObjectKind::NumericalReceipt,
            ) => Some(&demand.receipt_output),
            (
                Self::StructuralTransientResultAssembly(demand),
                AnalysisNumericalObjectKind::SolverResult,
            ) => Some(&demand.result_output),
            (
                Self::StructuralTransientResultAssembly(demand),
                AnalysisNumericalObjectKind::NumericalReceipt,
            ) => Some(&demand.receipt_output),
            (
                Self::StructuralModalResponseSpectrumDirection(demand),
                AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult,
            ) => Some(&demand.output),
            (
                Self::StructuralTransientScopePreparation(demand),
                AnalysisNumericalObjectKind::StructuralTransientCheckpointScope,
            ) => Some(&demand.scope_output),
            (
                Self::StructuralTransientBlock(demand),
                AnalysisNumericalObjectKind::StructuralTransientSegment,
            ) => Some(&demand.segment_output),
            (
                Self::StructuralModalResponseSpectrumAssembly(demand),
                AnalysisNumericalObjectKind::SolverResult,
            ) => Some(&demand.result_output),
            (
                Self::StructuralModalResponseSpectrumAssembly(demand),
                AnalysisNumericalObjectKind::NumericalReceipt,
            ) => Some(&demand.receipt_output),
            (
                Self::StructuralModalResponseSpectrumAssembly(demand),
                AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery,
            ) => Some(&demand.field_recovery_output),
            (
                Self::StructuralRandomResponseInterval(demand),
                AnalysisNumericalObjectKind::StructuralRandomResponseIntervalResult,
            ) => Some(&demand.output),
            (
                Self::StructuralRandomResponseAssembly(demand),
                AnalysisNumericalObjectKind::SolverResult,
            ) => Some(&demand.result_output),
            (
                Self::StructuralRandomResponseAssembly(demand),
                AnalysisNumericalObjectKind::NumericalReceipt,
            ) => Some(&demand.receipt_output),
            _ => None,
        }
    }

    pub const fn governs_prepared_outputs(&self) -> bool {
        if let Self::StructuralNonlinearPipeline(demand) = self {
            return demand.block_transport.is_some();
        }
        !matches!(
            self,
            Self::StructuralLinearPipeline(_)
                | Self::StructuralNonlinearPipeline(_)
                | Self::StructuralNonlinearRecovery(_)
                | Self::StructuralTransientFieldRecovery(_)
        )
    }
}
