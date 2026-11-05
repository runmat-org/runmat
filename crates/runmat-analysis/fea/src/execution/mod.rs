//! Portable execution contracts owned by the analysis domain.
//!
//! This module identifies analysis work and numerical objects. Scheduling, placement, transport,
//! encryption, and object storage remain owned by RunMat's generic execution crates.

pub use crate::pipeline::thermal::{PreparedThermalPrimarySolve, PreparedThermalSolve};

mod accuracy;
mod acoustic;
pub use acoustic::{
    acoustic_native_field_definitions, assemble_acoustic_frequency_result,
    AcousticFrequencyAssemblyOutput, AcousticFrequencyAssemblyRequest, AcousticFrequencyResult,
    AcousticFrequencyWorkload, AcousticHarmonicResultSummary, AcousticHarmonicSolverResult,
    AcousticHarmonicWorkload, AcousticNumericalReceipt, AcousticNumericalReceiptArtifact,
    ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES,
};
mod accuracy_source;
mod acoustic_response;
pub(crate) use acoustic_response::acoustic_response_pressure_definition;
pub use acoustic_response::{
    AcousticResponseDiagnostics, AcousticResponseNumericalReceipt,
    AcousticResponseNumericalReceiptArtifact, AcousticResponseSolver, AcousticResponseSolverResult,
    AcousticResponseWorkload, ACOUSTIC_PRESSURE_FIELD_ID,
    ACOUSTIC_RESPONSE_RECEIPT_MAX_ENCODED_BYTES,
};
mod buckling_result;
mod buckling_workload;
mod electrical_static;
mod electrothermal;
pub use electrothermal::{
    bind_element_heat_source, bind_retained_joule_heat, conduction_operating_temperatures,
    conductivity_at, electrical_part, electrothermal_conduction_study, element_temperatures,
    execute_coupled_electrothermal, execute_steady_electrothermal_from_retained,
    is_electrothermal_study, solve_electrothermal_heat_study, thermal_part,
    CoupledElectrothermalCompletion, ElectrothermalCouplingControls,
    ElectrothermalElectricalSample, ElectrothermalOperatingPoint, ElectrothermalSweep,
    ElementHeatSourceSample, RetainedJouleHeatSource, RetainedJouleHeatTransfer,
    SteadyElectrothermalCompletion,
};
mod magnetic_static;
mod maxwell_harmonic;
pub use magnetic_static::{
    magnetic_native_field_definition_inventory, magnetic_native_field_definitions,
    MagneticNumericalReceipt, MagneticNumericalReceiptArtifact, MagnetostaticSolverResult,
    MagnetostaticWorkload, MAGNETIC_RECEIPT_MAX_ENCODED_BYTES,
};
pub use maxwell_harmonic::{
    maxwell_native_field_definition_inventory, maxwell_native_field_definitions,
    MaxwellHarmonicSolverResult, MaxwellHarmonicWorkload, MaxwellNumericalReceipt,
    MaxwellNumericalReceiptArtifact, MAXWELL_NATIVE_FIELD_COUNT, MAXWELL_RECEIPT_MAX_ENCODED_BYTES,
};
mod error;
pub use electrical_static::{
    electrical_native_field_definitions, ElectricalNumericalReceipt,
    ElectricalNumericalReceiptArtifact, ElectricalStaticSolverResult, ElectricalStaticWorkload,
    ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES,
};
mod field;
mod field_plan;
mod field_topology;
mod fluid;
#[cfg(test)]
pub(crate) use fluid::tests::authored as fluid_test_workload;
pub use fluid::{
    fluid_native_field_definition_inventory, fluid_native_field_definitions,
    fluid_receipt_encoding_profile, FluidCoupling, FluidIntervalEvidence, FluidNumericalReceipt,
    FluidNumericalReceiptArtifact, FluidResultOnSource, FluidRetainedOriginal,
    FluidSolverProvenance, FluidSteadySolverResult, FluidSteadyWorkload, FluidTransientBlock,
    FluidTransientBlockReservation, FluidTransientBlockResult, FluidTransientBlockWorkload,
    FluidTransientCheckpoint, FluidTransientSchedule, FluidTransientSolverResult,
    FluidTransientWorkload, PreparedFluidTransient, PreparedFluidTransientContinuation,
    FLUID_RECEIPT_MAX_ENCODED_BYTES,
};
mod frequency_assembly;
mod frequency_balance;
mod frequency_entry;
mod frequency_publication;
mod frequency_sample_publication;
pub use frequency_sample_publication::{
    FrequencySampleNumericalEntry, FrequencySamplePublicationAuthority,
    FrequencySamplePublicationIdentity, FrequencySamplePublicationManifest,
    FrequencySamplePublicationSample,
};
mod frequency_receipt;
mod frequency_result;
mod frequency_sample;
mod frequency_scope;
mod frequency_workload;
mod identity;
mod initial_field;
pub(crate) use initial_field::{
    validate_initial_field_declarations, validate_initial_field_inventory,
    validate_initial_field_selection,
};
pub use initial_field::{AdmittedInitialField, InitialFieldBinding};
mod modal_result;
mod modal_workload;
mod object;
mod observation;
mod observation_assembly;
mod observation_history_checkpoint;
mod observation_history_finalization;
mod observation_partition;
mod observation_source;
mod publication;
mod random_response_interval;
mod statistical_fields;
mod random_response_result;
mod random_response_scope;
mod random_response_workload;
mod response_spectrum_entry;
mod response_spectrum_result;
mod response_spectrum_scope;
mod response_spectrum_workload;
mod signal;
mod solve;
mod state;
pub(crate) mod structural_identity;
mod study_workloads;
mod sweep;
mod thermal_receipt;
mod thermal_result;
pub use thermal_receipt::{
    ThermalNumericalReceipt, ThermalNumericalReceiptArtifact, THERMAL_RECEIPT_MAX_ENCODED_BYTES,
};
mod thermal_result_manifest;
pub use thermal_result_manifest::{ThermalResultEncodingProfile, ThermalSolverResultManifest};
mod thermal_schedule;
pub use thermal_schedule::ThermalSolveSchedule;
mod thermal_workload;
pub use thermal_workload::ThermalWorkload;
pub(crate) use thermal_workload::LIMITS as THERMAL_WORKLOAD_LIMITS;
mod transient_block;
mod transient_checkpoint;
mod transient_coordinates;
mod transient_result;
mod transient_schedule;
mod transient_segment;
mod transient_template;
mod transient_workload;

pub use accuracy::{AccuracyEvaluationRequest, AccuracySourceKind};
pub use accuracy_source::AccuracySourceReceipt;
pub use buckling_result::{
    StructuralBucklingMode, StructuralBucklingNumericalReceipt, StructuralBucklingSolverResult,
    StructuralBucklingSolverResultInput,
};
pub use buckling_workload::{
    StructuralBucklingSolverPolicy, StructuralBucklingTemplate, StructuralBucklingWorkload,
    StructuralBucklingWorkloadInput,
};
pub use error::{AnalysisExecutionError, AnalysisExecutionResult};
pub use field::{FieldRecoveryRequest, FieldRecoverySource};
pub use field_plan::{FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequirement};
pub use field_topology::FieldTopologyIdentity;
pub use frequency_assembly::{
    StructuralFrequencyAssemblyEntry, StructuralFrequencyAssemblyOutput,
    StructuralFrequencyResultAssemblyRequest,
};
pub use frequency_balance::{ComplexStructuralBalance, StructuralComplexVector};
pub use frequency_entry::{
    StructuralFrequencyEntryRequest, StructuralFrequencyEntryResult,
    StructuralFrequencyEntryResultManifest,
};
pub use frequency_publication::{
    StructuralFrequencyPublicationHistory, StructuralFrequencyPublicationInput,
    StructuralFrequencyPublicationManifest, StructuralFrequencyPublicationSample,
};
pub use frequency_receipt::{
    ModalTruncationBalance,
    StructuralFrequencyEntrySolverEvidence, StructuralFrequencyNumericalReceipt,
    StructuralFrequencySolutionSpace, StructuralFrequencySolverEvidenceInput,
};
pub use frequency_result::{
    StructuralFrequencyResultEntry, StructuralFrequencySolverResult,
    StructuralFrequencySolverResultManifest,
};
pub use frequency_sample::StructuralFrequencyPhysicalSample;
pub use frequency_scope::{
    StructuralFrequencyScope, StructuralFrequencyScopeInput, StructuralPhasorConvention,
};
pub use frequency_workload::{
    StructuralFrequencySolverPlan, StructuralFrequencyStateInputs, StructuralFrequencyWorkload,
    StructuralFrequencyWorkloadInput, StructuralModalFrequencyTemplate,
    StructuralPrestressedFrequencySolverTemplate, StructuralPrestressedFrequencyTemplate,
};
pub use identity::{
    AccuracyEvaluationRequestIdentity, AccuracyIdentity, AnalysisAttemptIdentity,
    AnalysisResultIdentity, AnalysisTaskIdentity, AnalysisWorkKind, AnalysisWorkloadIdentity,
    ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
    FieldRecoveryRequestIdentity, GeneratorIdentity, NumericalMaterializationIdentity,
    NumericalPlanIdentity, NumericalReceiptIdentity, ObservationAssemblyRequestIdentity,
    ObservationEvaluationRequestIdentity, ObservationHistoryCheckpointRequestIdentity,
    ObservationIdentity, ObservationPartitionEvaluationRequestIdentity,
    ObservationPartitionReductionRequestIdentity, OperatorIdentity, ProblemIdentity,
    PublicationIdentity, PublicationRequestIdentity, SequenceIdentity,
    SignalMaterializationIdentity, SignalRealizationIdentity, SolverExcitationIdentity,
    SolverResultIdentity, StateIdentity, StateTransitionIdentity, StepIdentity,
    StructuralBucklingWorkloadIdentity, StructuralDampingOperatorIdentity,
    StructuralDofLayoutIdentity, StructuralExcitationSetIdentity, StructuralFormulationIdentity,
    StructuralFrequencyEntryRequestIdentity, StructuralFrequencyEntryResultIdentity,
    StructuralFrequencyPhysicalSampleIdentity, StructuralFrequencyProblemIdentity,
    StructuralFrequencyPublicationIdentity, StructuralFrequencyResultAssemblyRequestIdentity,
    StructuralFrequencyScopeIdentity, StructuralFrequencyWorkloadIdentity,
    StructuralKinematicsIdentity, StructuralMassOperatorIdentity,
    StructuralModalResponseSpectrumAssemblyRequestIdentity,
    StructuralModalResponseSpectrumDirectionRequestIdentity,
    StructuralModalResponseSpectrumDirectionResultIdentity,
    StructuralModalResponseSpectrumScopeIdentity, StructuralModalResponseSpectrumWorkloadIdentity,
    StructuralModalWorkloadIdentity, StructuralRandomResponseAssemblyRequestIdentity,
    StructuralRandomResponseIntervalRequestIdentity,
    StructuralRandomResponseIntervalResultIdentity, StructuralRandomResponseScopeIdentity,
    StructuralRandomResponseWorkloadIdentity, StructuralStiffnessOperatorIdentity,
    StructuralTransientBlockRequestIdentity, StructuralTransientCheckpointIdentity,
    StructuralTransientCheckpointScopeIdentity, StructuralTransientPhysicalSampleIdentity,
    StructuralTransientResultAssemblyRequestIdentity, StructuralTransientScheduleIdentity,
    StructuralTransientSegmentIdentity, StructuralTransientSegmentSampleIdentity,
    StructuralTransientSolverOptionsIdentity, StudyIdentity, SweepIdentity, SweepResultIdentity,
    ThermalNumericalIntentIdentity,
};
pub use modal_result::{StructuralModalNumericalReceipt, StructuralModalSolverResult};
pub use modal_workload::{
    StructuralModalWorkload, StructuralModalWorkloadInput, StructuralPrestressedModalTemplate,
};
pub use object::{
    chunk_numerical_payload, chunk_partitioned_numerical_payload, AnalysisChunkDescriptor,
    AnalysisNumericalObjectKind, AnalysisObjectManifest, AnalysisPayloadBuilder,
    AnalysisPayloadPartition, AnalysisPayloadPartitionDescriptor, AnalysisPayloadPartitionPlan,
    AnalysisPayloadPlan, AnalysisTransportProfile, AnalysisTransportProfiler,
    ChunkedAnalysisPayload, ANALYSIS_CHUNK_MEDIA_TYPE, ANALYSIS_CHUNK_VALUE_SCHEMA,
    ANALYSIS_MANIFEST_MAX_ENCODED_BYTES, ANALYSIS_MANIFEST_MEDIA_TYPE,
    ANALYSIS_MANIFEST_VALUE_SCHEMA, ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES,
    DEFAULT_ANALYSIS_CHUNK_BYTES,
};
pub use observation::ObservationEvaluationRequest;
pub use observation_assembly::ObservationAssemblyRequest;
pub use observation_history_checkpoint::ObservationHistoryCheckpointRequest;
pub use observation_history_finalization::{
    FinalizedObservationHistory, ObservationHistoryFinalizationRequest,
    ObservationHistoryFinalizationRequestIdentity, ObservationHistorySourceContext,
};
pub(crate) use observation_partition::validate_published_observation_source;
pub use observation_partition::{
    ObservationPartitionEvaluationRequest, ObservationPartitionReductionRequest,
};
pub use observation_source::StructuralObservationSource;
pub use publication::ResultPublicationRequest;
pub use random_response_interval::{
    StructuralRandomResponseIntervalRequest, StructuralRandomResponseIntervalResult,
};
pub use random_response_result::{
    StructuralRandomResponseAssemblyRequest, StructuralRandomResponseNumericalReceipt,
    StructuralRandomResponseSolverResult, StructuralRandomStatistics,
};
pub use random_response_scope::{
    StructuralRandomResponseScope, StructuralRandomResponseScopeInput,
};
pub use random_response_workload::{
    StructuralRandomResponseTemplate, StructuralRandomResponseWorkload,
    StructuralRandomResponseWorkloadInput,
};
pub use response_spectrum_entry::{
    StructuralModalResponseSpectrumDirectionRequest, StructuralModalResponseSpectrumDirectionResult,
};
pub use response_spectrum_result::{
    StructuralModalResponseSpectrumAssemblyRequest,
    StructuralModalResponseSpectrumNumericalReceipt, StructuralModalResponseSpectrumSolverResult,
    StructuralResponseEnvelope,
};
pub use response_spectrum_scope::{
    StructuralModalResponseSpectrumScope, StructuralModalResponseSpectrumScopeInput,
};
pub use response_spectrum_workload::{
    StructuralModalResponseSpectrumTemplate, StructuralModalResponseSpectrumWorkload,
    StructuralModalResponseSpectrumWorkloadInput,
};
pub use signal::{
    CanonicalSignalContent, CanonicalSignalKnot, CanonicalSignalRealization,
    CanonicalSignalRealizationInput, CanonicalSignalValues, SignalInputAdmission,
};
pub use solve::{
    StructuralLinearNumericalReceipt, StructuralLinearNumericalReceiptArtifact,
    StructuralLinearStateInput, StructuralLinearWorkload, StructuralLinearWorkloadResult,
};
pub use state::{
    fluid_state_publication_profile, fluid_state_publication_requests,
    fluid_state_publication_workspace_bytes, nonlinear_solid_state_publication_requests,
    semantic_state_retained_storage_bytes, AcousticPhasorConvention, BoundThermalStateTransfer,
    CanonicalAcousticStatePayload, CanonicalElectricalStaticStatePayload,
    CanonicalFluidStatePayload, CanonicalFluidStatePayloadContents, CanonicalMaxwellStatePayload,
    CanonicalNonlinearSolidStatePayload, CanonicalSemanticState, CanonicalSemanticStateInput,
    CanonicalSemanticStatePayload, CanonicalStateTransition, CanonicalStateTransitionInput,
    CanonicalThermalStatePayload, SemanticStatePublicationProfileSource,
    SemanticStatePublicationRequest, SemanticStatePublicationSource, SolvedTemperatureField,
    StatePublicationProfileEntry, StatePublicationProfiles, STATE_PUBLICATION_PROFILES_MAX_ENTRIES,
};
pub use statistical_fields::{
    random_response_statistical_fields, response_spectrum_statistical_fields,
    statistical_field_definitions, statistical_field_request, statistical_recovered_fields,
    StatisticalFieldValues, StatisticalProcedure,
};
mod structural_state_profiles;
pub use structural_state_profiles::{
    structural_state_publication_profiles_encoding_profile, structural_state_publication_requests,
};
pub use study_workloads::{
    StructuralPrestressPrefix, StructuralStepWorkload, StructuralStudyWorkloads,
    StructuralTransientBlockPlan,
};
pub use sweep::{
    ComposedSweepResult, SweepCompositionPolicy, SweepDefinition, SweepEntryDisposition,
    SweepEntryOutcome,
};
pub use transient_block::StructuralTransientBlockRequest;
pub use transient_checkpoint::{
    StructuralTransientBalanceAccumulator, StructuralTransientCheckpoint,
    StructuralTransientCheckpointInput, StructuralTransientCheckpointScope,
    StructuralTransientContinuationPoint, StructuralTransientContinuationState,
    StructuralTransientCoordinateSpace, StructuralTransientNumericalState,
    StructuralTransientObservationCursor, StructuralTransientSolverCounters,
};
pub use transient_coordinates::{ModalProjectionEvidence, ModalTransientCoordinateState};
pub use transient_result::{
    StructuralTransientBlockReceipt, StructuralTransientNumericalReceipt,
    StructuralTransientResultAssemblyRequest, StructuralTransientResultEntry,
    StructuralTransientSolverResult, StructuralTransientSolverResultManifest,
};
pub use transient_schedule::{
    StructuralTransientContinuationApproach, StructuralTransientIntegrator,
    StructuralTransientPrecision, StructuralTransientSchedule, StructuralTransientScheduleArrival,
    StructuralTransientSchedulePoint,
};
pub use transient_segment::{
    StructuralTransientBlockSolverEvidence, StructuralTransientPhysicalSample,
    StructuralTransientPhysicalSampleInput, StructuralTransientSampleStep,
    StructuralTransientSegment, StructuralTransientSegmentEntry,
    StructuralTransientSegmentManifest, StructuralTransientSegmentSample,
    StructuralTransientSegmentSampleInput,
};
pub use transient_template::StructuralModalTransientTemplate;
pub(crate) use transient_workload::StructuralTransientCheckpointScopeInput;
pub use transient_workload::{StructuralTransientSolverPlan, StructuralTransientWorkload};

#[cfg(test)]
mod tests;

mod nonlinear_solid;
pub use nonlinear_solid::{
    nonlinear_native_field_definitions, NonlinearSolidBlockArtifact, NonlinearSolidBlockReference,
    NonlinearSolidBlockRequest, NonlinearSolidWorkload,
};
pub use nonlinear_solid::{
    NonlinearSolidNumericalReceipt, NonlinearSolidNumericalReceiptArtifact,
    NONLINEAR_SOLID_RECEIPT_MAX_ENCODED_BYTES,
};
