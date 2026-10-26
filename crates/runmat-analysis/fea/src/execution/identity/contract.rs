use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::AnalysisExecutionResult;

macro_rules! identity_type {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(Digest);

        impl $name {
            pub const fn digest(self) -> Digest {
                self.0
            }

            #[allow(dead_code)]
            pub(in crate::execution) fn derive<T: Serialize>(
                domain: &'static str,
                input: &T,
            ) -> AnalysisExecutionResult<Self> {
                Ok(Self(super::derive_digest(domain, input)?))
            }

            #[allow(dead_code)]
            pub(in crate::execution) fn derive_content<T: Serialize>(
                domain: &'static str,
                input: &T,
            ) -> AnalysisExecutionResult<Self> {
                Ok(Self(super::derive_content_digest(domain, input)?))
            }
        }
    };
}

identity_type!(StudyIdentity);
identity_type!(SweepIdentity);
identity_type!(SequenceIdentity);
identity_type!(StepIdentity);
identity_type!(StateIdentity);
identity_type!(StateTransitionIdentity);
identity_type!(ConstraintPlanIdentity);
identity_type!(GeneratorIdentity);
identity_type!(NumericalMaterializationIdentity);
identity_type!(SignalMaterializationIdentity);
identity_type!(SignalRealizationIdentity);
identity_type!(SolverExcitationIdentity);
identity_type!(ProblemIdentity);
identity_type!(DiscretizationIdentity);
identity_type!(OperatorIdentity);
identity_type!(NumericalPlanIdentity);
identity_type!(ExecutionPlanIdentity);
identity_type!(AnalysisWorkloadIdentity);
identity_type!(ThermalNumericalIntentIdentity);
identity_type!(AnalysisTaskIdentity);
identity_type!(AnalysisAttemptIdentity);
identity_type!(ObservationIdentity);
identity_type!(AccuracyIdentity);
identity_type!(AnalysisResultIdentity);
identity_type!(SolverResultIdentity);
identity_type!(NumericalReceiptIdentity);
identity_type!(FieldRecoveryRequestIdentity);
identity_type!(ObservationEvaluationRequestIdentity);
identity_type!(ObservationPartitionEvaluationRequestIdentity);
identity_type!(ObservationPartitionReductionRequestIdentity);
identity_type!(ObservationAssemblyRequestIdentity);
identity_type!(AccuracyEvaluationRequestIdentity);
identity_type!(SweepResultIdentity);
identity_type!(PublicationIdentity);
identity_type!(PublicationRequestIdentity);
identity_type!(ObservationHistoryCheckpointRequestIdentity);
identity_type!(StructuralTransientScheduleIdentity);
identity_type!(StructuralDofLayoutIdentity);
identity_type!(StructuralFormulationIdentity);
identity_type!(StructuralStiffnessOperatorIdentity);
identity_type!(StructuralMassOperatorIdentity);
identity_type!(StructuralDampingOperatorIdentity);
identity_type!(StructuralExcitationSetIdentity);
identity_type!(StructuralKinematicsIdentity);
identity_type!(StructuralTransientSolverOptionsIdentity);
identity_type!(StructuralTransientCheckpointIdentity);
identity_type!(StructuralTransientCheckpointScopeIdentity);
identity_type!(StructuralTransientBlockRequestIdentity);
identity_type!(StructuralTransientSegmentIdentity);
identity_type!(StructuralTransientSegmentSampleIdentity);
identity_type!(StructuralTransientPhysicalSampleIdentity);
identity_type!(StructuralTransientResultAssemblyRequestIdentity);
identity_type!(StructuralFrequencyScopeIdentity);
identity_type!(StructuralFrequencyProblemIdentity);
identity_type!(StructuralFrequencyWorkloadIdentity);
identity_type!(StructuralFrequencyPhysicalSampleIdentity);
identity_type!(StructuralFrequencyEntryRequestIdentity);
identity_type!(StructuralFrequencyEntryResultIdentity);
identity_type!(StructuralFrequencyResultAssemblyRequestIdentity);
identity_type!(StructuralFrequencyPublicationIdentity);

impl StructuralFrequencyPublicationIdentity {
    /// A publication closure owns a larger bounded metadata inventory than a compact task
    /// identity. Use its contract's limits while preserving the canonical identity encoding.
    pub(in crate::execution) fn derive_publication<T: Serialize>(
        input: &T,
        limits: runmat_canonical_codec::CanonicalLimits,
    ) -> AnalysisExecutionResult<Self> {
        Ok(Self(super::derive_digest_with_limits(
            "structural-frequency-publication",
            input,
            limits,
        )?))
    }
}
identity_type!(StructuralModalWorkloadIdentity);
identity_type!(StructuralModalResponseSpectrumWorkloadIdentity);
identity_type!(StructuralModalResponseSpectrumScopeIdentity);
identity_type!(StructuralModalResponseSpectrumDirectionRequestIdentity);
identity_type!(StructuralModalResponseSpectrumDirectionResultIdentity);
identity_type!(StructuralModalResponseSpectrumAssemblyRequestIdentity);
identity_type!(StructuralRandomResponseWorkloadIdentity);
identity_type!(StructuralRandomResponseScopeIdentity);
identity_type!(StructuralRandomResponseIntervalRequestIdentity);
identity_type!(StructuralRandomResponseIntervalResultIdentity);
identity_type!(StructuralRandomResponseAssemblyRequestIdentity);
identity_type!(StructuralBucklingWorkloadIdentity);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisWorkKind {
    SolveNonlinearSolidBlock,
    SolveFluidSteady,
    SolveFluidTransientBlock,
    SolveThermal,
    SolveAcousticHarmonic,
    AssembleAcousticFrequencyResult,
    SolveElectricalStatic,
    SolveAcousticResponse,
    SolveMagnetostatic,
    SolveMaxwellHarmonic,
    MaterializeSignal,
    CompileConstraints,
    AssembleOperator,
    Solve,
    ExtractStructuralModes,
    SolveStructuralModalResponseSpectrumDirection,
    AssembleStructuralModalResponseSpectrumResult,
    SolveStructuralRandomResponseIntervals,
    AssembleStructuralRandomResponseResult,
    SolveStructuralBucklingModes,
    RecoverField,
    PublishState,
    CheckpointObservationHistory,
    FinalizeObservationHistory,
    PrepareStructuralTransientCheckpointScope,
    SolveStructuralTransientBlock,
    AssembleStructuralTransientResult,
    SolveStructuralFrequencyEntry,
    AssembleStructuralFrequencyResult,
    EvaluateObservationPartition,
    ReduceObservationPartitions,
    AssembleObservations,
    EvaluateAccuracy,
    PublishResult,
}
