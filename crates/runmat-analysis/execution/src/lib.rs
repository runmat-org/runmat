//! Narrow bridge from analysis-owned logical contracts to shared RunMat execution services.

mod accuracy_evaluation;
mod accuracy_source_context;
pub mod assessment_program;
pub use accuracy_source_context::validate_accuracy_source_context;
mod accuracy_resource;
mod buckling_object_access;
mod buckling_task;
mod callable;
mod error;
mod field_recovery;
mod fluid_field_recovery;
mod fluid_phase_plan_memo;
mod fluid_state_publication;
pub use fluid_state_publication::{
    build_prepared_fluid_state_publication_resource_plan, prepare_fluid_state_publication,
    require_fluid_state_source_decode_budget, FluidStatePublicationPreparation,
};
mod fluid_source_admission;
mod fluid_steady_plan;
mod fluid_steady_result;
mod fluid_steady_task;
mod fluid_transient_plan;
mod fluid_transient_result;
mod fluid_transient_task;
mod phase_accounting;
pub use fluid_transient_plan::{
    decode_fluid_transient_task_source, prepare_fluid_transient_inputs,
    project_fluid_transient_checkpoint_output, rederive_fluid_transient_resource_plan,
    require_fluid_transient_source_closure, DecodedFluidTransientTaskSource,
    PreparedFluidTransientInputs,
};
pub use fluid_transient_result::{
    import_fluid_transient_checkpoint, ImportedFluidTransientCheckpoint,
};
pub use fluid_transient_task::{
    prepare_fluid_transient_task, FluidTransientTaskPreparation, PreparedFluidTransientTask,
};
pub use phase_accounting::validate_analysis_phase_accounting;
mod phase_execution;
pub use phase_execution::AnalysisPhaseExecutionEvidence;
mod nonlinear_block;
mod nonlinear_study;
mod study_closure;
pub use nonlinear_block::{
    import_nonlinear_solid_block, prepare_nonlinear_solid_block_task,
    rederive_nonlinear_solid_block_resource_plan, require_nonlinear_solid_source_decode_budget,
    ImportedNonlinearSolidBlock, NonlinearSolidBlockPreparation, PreparedNonlinearSolidBlockTask,
};
pub use nonlinear_study::{
    import_nonlinear_solid_study_completion, import_nonlinear_solid_study_root,
    prepare_nonlinear_solid_study_root, CompletedNonlinearSolidStudy,
    NonlinearSolidStudyCoordinator, NonlinearSolidStudyPhase, NonlinearSolidStudyPreparation,
    NonlinearSolidStudyResultClosure, NonlinearSolidStudyResultPhase,
    OriginalNonlinearSolidBlockProjection, PreparedNonlinearSolidStudyRoot,
    DEFAULT_NONLINEAR_SOLID_ACCEPTED_BLOCK_LIMIT,
};
mod frequency_field_recovery;
mod frequency_object_access;
mod frequency_sample_publication;
pub use frequency_sample_publication::{
    decode_prepared_frequency_sample_publication, prepare_frequency_sample_publication_objects,
};
mod frequency_task;
mod geometry_materialization;
mod graph;
mod host;
mod input_materialization;
mod input_plan;
mod materialized_signal;
mod modal_object_access;
mod modal_task;
pub mod named_callable;
mod object;
mod observation_assembly;
mod observation_evaluation;
mod observation_field_partitions;
mod observation_history_access;
mod observation_history_checkpoint;
mod observation_history_finalization;
mod observation_history_task;
mod observation_partition;
pub use observation_partition::require_observation_partition_source_decode_budget;
mod observation_source;
mod portable_source_admission;
pub use portable_source_admission::decode_admitted_resource_plan;
mod portable_task_support;
pub use fluid_field_recovery::{
    build_prepared_fluid_field_recovery_resource_plan, decode_fluid_recovery_source,
    import_fluid_recovered_fields, prepare_fluid_field_recovery,
    require_fluid_recovery_source_decode_budget, FluidFieldRecoveryPreparation,
    PreparedFluidFieldRecovery,
};
pub use fluid_source_admission::{
    require_fluid_source_decode_budget, require_fluid_transient_source_decode_budget,
};
pub use fluid_steady_plan::{
    prepare_fluid_steady_inputs, rederive_fluid_steady_resource_plan, PreparedFluidSteadyInputs,
};
pub use fluid_steady_result::{import_fluid_steady_result, ImportedFluidSteadyResult};
pub use fluid_steady_task::{
    prepare_fluid_steady_task, FluidSteadyTaskPreparation, PreparedFluidSteadyTask,
};
mod random_response_object_access;
mod random_response_task;
mod resource_usage;
pub use accuracy_resource::{
    build_prepared_accuracy_resource_plan, require_accuracy_source_decode_budget,
    validate_admitted_accuracy_source,
};
mod response;
mod response_spectrum_object_access;
mod response_spectrum_task;
mod result_publication;
mod result_publication_resource;
mod result_publication_usage;
pub use result_publication_resource::build_prepared_result_publication_resource_plan;
pub use result_publication_usage::complete_result_publication_resource_usage;
pub use runmat_analysis_fea::results::publication::{
    validate_result_publication_resource_usage, validate_retained_result_publication_resource_usage,
};
mod semantic_state_object_access;
mod serial;
mod signal_deferred;
mod signal_input;
pub use signal_deferred::{FrozenDeferredSignal, ResolvedDeferredSignalMaterialization};
mod signal_layout;
pub use signal_layout::CanonicalGeneratorValueLayout;
mod signal_materialization;
mod signal_program;
mod signal_topology;
pub use signal_topology::{SignalTopologyBinding, SignalTopologyResolution};
mod spectral_materialization;
mod state_publication;
mod state_transition;
mod structural;
mod structural_adaptation;
pub use structural_adaptation::{
    evaluate_original_structural_adaptation, StructuralAdaptationNativeIteration,
    StructuralAdaptationNativeIterationInput, StructuralAdaptationNativeSource,
};
mod structural_buckling_study;
mod structural_modal_frequency_study;
mod structural_modal_producer;
mod structural_modal_response_spectrum_study;
mod structural_modal_transient_study;
mod structural_prestress_producer;
pub use structural_prestress_producer::ThermalPrestressSource;
mod acoustic_frequency_assembly;
mod acoustic_harmonic_plan;
pub use acoustic_frequency_assembly::{
    decode_acoustic_frequency_assembly_entries, import_acoustic_frequency_assembly,
    prepare_acoustic_frequency_assembly_task, rederive_acoustic_frequency_assembly_resource_plan,
    validate_acoustic_frequency_assembly_entries, AcousticFrequencyAssemblyPreparation,
    ImportedAcousticFrequencyAssembly, PreparedAcousticFrequencyAssemblyTask,
};
mod acoustic_response_plan;
mod acoustic_response_result;
mod acoustic_response_state_profiles;
mod acoustic_response_study;
mod acoustic_response_task;
mod electrical_state_profiles;
mod electrical_static_plan;
mod electrical_static_result;
mod electrical_static_task;
mod electrical_study;
mod fluid_study;
mod magnetic_study;
mod maxwell_study;
pub use fluid_study::{
    import_fluid_study_completion, import_fluid_study_root, prepare_fluid_study_root,
    prepare_fluid_study_root_for, CompletedFluidStudy, FluidStudyCoordinator, FluidStudyPhase,
    FluidStudyPreparation, FluidStudyPreparationParts, FluidStudyResultClosure,
    FluidStudyResultPhase, PlannedFluidStudyPreparation, PreparedFluidStudyRoot,
    VerifiedFluidStudy,
};
mod sampled_history_observations;
pub use sampled_history_observations::{CompletedSampledObservations, SampledObservationTask};
mod sampled_study_observations;
pub use sampled_study_observations::history::{
    EntryHistoryContext, EntryHistoryPhase, EntryHistorySample,
};
mod statistical_study_observations;
pub use statistical_study_observations::{
    StatisticalObservationSource, StatisticalStudyObservations,
};
mod steady_study;
pub use magnetic_study::{
    import_magnetic_study_completion, import_magnetic_study_root, prepare_magnetic_study_root,
    CompletedMagneticStudy, MagneticStudyCoordinator, MagneticStudyPhase, MagneticStudyPreparation,
    MagneticStudyResultClosure, MagneticStudyResultPhase, PreparedMagneticStudyRoot,
};
pub use maxwell_study::{
    import_maxwell_study_completion, import_maxwell_study_root, prepare_maxwell_study_root,
    CompletedMaxwellStudy, MaxwellStudyCoordinator, MaxwellStudyPhase, MaxwellStudyPreparation,
    MaxwellStudyResultClosure, MaxwellStudyResultPhase, PreparedMaxwellStudyRoot,
};
mod electrical_field_recovery;
pub use electrical_field_recovery::{
    build_prepared_electrical_field_recovery_resource_plan, decode_electrical_recovery_source,
    import_electrical_recovered_fields, prepare_electrical_field_recovery,
    ElectricalFieldRecoveryPreparation, ElectricalRecoverySource, PreparedElectricalFieldRecovery,
};
mod magnetic_field_recovery;
mod statistical_field_recovery;
pub use statistical_field_recovery::{
    build_prepared_statistical_field_recovery_resource_plan, decode_statistical_recovery_source,
    import_statistical_recovered_fields, prepare_statistical_field_recovery, statistical_procedure,
    statistical_recovery_execution_plan, statistical_recovery_request, DecodedStatisticalSource,
    PreparedStatisticalFieldRecovery, StatisticalFieldRecoveryPreparation,
};
mod magnetic_state_profiles;
mod acoustic_field_recovery;
mod acoustic_response_field_recovery;
pub use acoustic_field_recovery::{
    build_prepared_acoustic_field_recovery_resource_plan, decode_acoustic_recovery_source,
    import_acoustic_recovered_fields, prepare_acoustic_field_recovery,
    AcousticFieldRecoveryPreparation, PreparedAcousticFieldRecovery,
};
pub use acoustic_response_field_recovery::{
    build_prepared_acoustic_response_field_recovery_resource_plan,
    decode_acoustic_response_recovery_source, import_acoustic_response_recovered_fields,
    prepare_acoustic_response_field_recovery, AcousticResponseFieldRecoveryPreparation,
    PreparedAcousticResponseFieldRecovery,
};
mod maxwell_field_recovery;
mod maxwell_harmonic_plan;
mod maxwell_harmonic_result;
mod maxwell_harmonic_task;
mod maxwell_source_admission;
mod maxwell_state_profiles;
pub use maxwell_field_recovery::{
    build_maxwell_recovery_metadata_resource_plan,
    build_prepared_maxwell_field_recovery_resource_plan, decode_maxwell_recovery_source,
    import_maxwell_recovered_fields, prepare_maxwell_field_recovery,
    MaxwellFieldRecoveryPreparation, PreparedMaxwellFieldRecovery,
};
pub use maxwell_harmonic_plan::{
    prepare_maxwell_harmonic_inputs, rederive_maxwell_harmonic_resource_plan,
    PreparedMaxwellHarmonicInputs,
};
pub use maxwell_harmonic_result::{import_maxwell_harmonic_result, ImportedMaxwellHarmonicResult};
pub use maxwell_harmonic_task::{
    prepare_maxwell_harmonic_task, MaxwellHarmonicTaskPreparation, PreparedMaxwellHarmonicTask,
};
pub use maxwell_source_admission::require_maxwell_source_decode_budget;
pub use maxwell_state_profiles::maxwell_state_publication_profiles;
mod magnetostatic_plan;
mod magnetostatic_result;
mod magnetostatic_task;
pub use acoustic_response_plan::{
    prepare_acoustic_response_inputs, rederive_acoustic_response_resource_plan,
    PreparedAcousticResponseInputs,
};
pub use acoustic_response_result::{
    import_acoustic_response_result, ImportedAcousticResponseResult,
};
pub use acoustic_response_state_profiles::acoustic_response_state_publication_profiles;
pub use acoustic_response_study::{
    import_acoustic_response_study_completion, import_acoustic_response_study_root,
    prepare_acoustic_response_study_root, AcousticResponseStudyCoordinator,
    AcousticResponseStudyPhase, AcousticResponseStudyPreparation,
    AcousticResponseStudyResultClosure, AcousticResponseStudyResultPhase,
    CompletedAcousticResponseStudy, PreparedAcousticResponseStudyRoot,
};
pub use acoustic_response_task::{
    prepare_acoustic_response_task, AcousticResponseTaskPreparation, PreparedAcousticResponseTask,
};
pub use electrical_state_profiles::electrical_state_publication_profiles;
pub use electrical_static_plan::{
    prepare_electrical_static_inputs, rederive_electrical_static_resource_plan,
    PreparedElectricalStaticInputs,
};
pub use electrical_static_result::{
    import_electrical_static_result, ImportedElectricalStaticResult,
};
pub use electrical_static_task::{
    prepare_electrical_static_task, ElectricalStaticTaskPreparation, PreparedElectricalStaticTask,
};
pub use electrical_study::{
    import_electrical_study_completion, import_electrical_study_root,
    prepare_electrical_study_root, CompletedElectricalStudy, ElectricalStudyCoordinator,
    ElectricalStudyPhase, ElectricalStudyPreparation, ElectricalStudyResultClosure,
    ElectricalStudyResultPhase, PreparedElectricalStudyRoot,
};
pub use magnetic_field_recovery::{
    build_magnetic_recovery_metadata_resource_plan,
    build_prepared_magnetic_field_recovery_resource_plan, decode_magnetic_recovery_source,
    import_magnetic_recovered_fields, prepare_magnetic_field_recovery,
    MagneticFieldRecoveryPreparation, PreparedMagneticFieldRecovery,
};
pub use magnetic_state_profiles::magnetic_state_publication_profiles;
pub use magnetostatic_plan::{
    prepare_magnetostatic_inputs, rederive_magnetostatic_resource_plan, PreparedMagnetostaticInputs,
};
pub use magnetostatic_result::{import_magnetostatic_result, ImportedMagnetostaticResult};
pub use magnetostatic_task::{
    prepare_magnetostatic_task, MagnetostaticTaskPreparation, PreparedMagnetostaticTask,
};
mod acoustic_harmonic_result;
mod acoustic_harmonic_task;
mod acoustic_state_profiles;
mod acoustic_study;
pub use acoustic_harmonic_result::{
    import_acoustic_harmonic_result, ImportedAcousticHarmonicResult,
};
pub use acoustic_state_profiles::acoustic_state_publication_profiles;
pub use acoustic_study::{
    import_acoustic_study_completion, import_acoustic_study_root, prepare_acoustic_study_root,
    AcousticStudyAssemblyClosure, AcousticStudyCoordinator, AcousticStudyEntryClosure,
    AcousticStudyEntryCoordinator, AcousticStudyPhase, AcousticStudyPreparation,
    AcousticStudyResultClosure, AcousticStudyResultPhase, CompletedAcousticStudy,
    CompletedAcousticStudyEntry, CompletedEntryHistories, PreparedAcousticStudyRoot,
};
mod structural_prestressed_frequency_study;
mod structural_prestressed_modal_frequency_study;
mod structural_prestressed_modal_study;
mod structural_random_response_study;
mod structural_sequence;
mod structural_transient_recovery;
mod structural_transient_sequence;
mod task;
mod thermal_execution;
mod thermal_plan;
pub use acoustic_harmonic_plan::{
    prepare_acoustic_harmonic_inputs, rederive_acoustic_harmonic_resource_plan,
    PreparedAcousticHarmonicInputs,
};
pub use acoustic_harmonic_task::{
    prepare_acoustic_harmonic_task, AcousticHarmonicTaskPreparation, PreparedAcousticHarmonicTask,
};
pub use thermal_plan::{
    prepare_thermal_inputs, rederive_thermal_resource_plan, validate_thermal_task_source,
    PreparedThermalInputs,
};
mod structural_state_profiles;
mod thermal_completion;
mod thermal_state_profiles;
mod thermal_study;
pub use structural_state_profiles::{
    structural_state_profiles_output_facts, structural_state_publication_profiles,
    StructuralStateProfileResult,
};
pub use thermal_study::{
    import_thermal_study_completion, import_thermal_study_root, prepare_thermal_study_root,
    CompletedThermalStudy, PreparedThermalStudyRoot, ThermalStudyCoordinator, ThermalStudyPhase,
    ThermalStudyPreparation, ThermalStudyResultClosure, ThermalStudyResultPhase, ThermalStudyScope,
};
mod thermal_task;
mod thermal_task_result;
pub use thermal_completion::{import_thermal_completion, ImportedThermalCompletion};
pub use thermal_state_profiles::thermal_state_publication_profiles;
pub use thermal_task::{prepare_thermal_task, PreparedThermalTask, ThermalTaskPreparation};
pub use thermal_task_result::{import_thermal_result, ImportedThermalResult};
mod thermal_result_access;
mod transient_block_task;
mod transient_object_access;
mod transient_result_access;
mod transient_result_task;
mod transient_scope_task;
mod transport;
mod workload;

pub use accuracy_evaluation::{
    import_accuracy, prepare_accuracy, AccuracyPreparation, ImportedAccuracy, PreparedAccuracy,
};
pub use buckling_object_access::{
    decode_prepared_structural_buckling_receipt, decode_prepared_structural_buckling_result,
    decode_prepared_structural_buckling_workload, prepare_structural_buckling_workload_objects,
};
pub use buckling_task::{
    import_structural_buckling_result, prepare_structural_buckling_task,
    ImportedStructuralBucklingResult, PreparedStructuralBucklingTask,
    StructuralBucklingTaskPreparation,
};
pub use callable::graph::admit_graph_callable_signal_generator;
pub use callable::{
    admit_callable_signal_generator, admit_named_random_spectrum_generator,
    admit_named_response_spectrum_generator, admit_named_signal_generator,
    AdmittedGeneratorInvocation, AdmittedNumericalGenerator, AdmittedSignalGenerator,
    FrozenGeneratorInputs, NumericalGeneratorKind,
};
pub use error::{AnalysisBridgeError, AnalysisBridgeResult, CoordinatorRetentionLimit};
pub use field_recovery::{
    build_prepared_structural_recovery_resource_plan,
    build_structural_recovery_metadata_resource_plan, decode_recovered_field_artifact,
    import_structural_recovered_fields, prepare_recovered_field_objects,
    prepare_structural_field_recovery, recovered_field_transport_reservation_bytes,
    require_structural_recovery_decode_budget, ImportedRecoveredFields,
    PreparedStructuralFieldRecovery, StructuralFieldRecoveryPreparation,
};
pub use frequency_field_recovery::{
    import_structural_frequency_recovered_fields, prepare_structural_frequency_field_recovery,
    PreparedStructuralFrequencyFieldRecovery, StructuralFrequencyFieldRecoveryPreparation,
};
pub use frequency_object_access::{
    decode_prepared_structural_frequency_entry_request,
    decode_prepared_structural_frequency_entry_result,
    decode_prepared_structural_frequency_numerical_receipt,
    decode_prepared_structural_frequency_result_assembly_request,
    decode_prepared_structural_frequency_solver_result,
    decode_prepared_structural_frequency_workload,
    prepare_structural_frequency_entry_request_objects,
    prepare_structural_frequency_entry_result_objects,
    prepare_structural_frequency_numerical_receipt_objects,
    prepare_structural_frequency_result_assembly_request_objects,
    prepare_structural_frequency_solver_result_objects,
    prepare_structural_frequency_workload_objects,
};
pub use frequency_task::{
    import_structural_frequency_entry, import_structural_frequency_result,
    prepare_structural_frequency_entry_task, prepare_structural_frequency_fanout,
    prepare_structural_frequency_result_task, ImportedStructuralFrequencyEntry,
    ImportedStructuralFrequencyResult, PreparedStructuralFrequencyEntryTask,
    PreparedStructuralFrequencyResultTask, StructuralFrequencyEntryPreparation,
    StructuralFrequencyFanoutPreparation, StructuralFrequencyResultPreparation,
};
pub use geometry_materialization::*;
pub use graph::{
    compile_analysis_sequence_graph, AnalysisSequenceExecutionGraph, AnalysisSequenceTask,
    PreparedAnalysisStepTask,
};
pub use host::{
    prepare_analysis_host_program, AnalysisHostProgramArtifact, AnalysisHostWorkload,
    PreparedAnalysisHostInvocation, ANALYSIS_HOST_ABI, ANALYSIS_HOST_EXECUTION_MODE,
    ANALYSIS_HOST_INVOCATION_MEDIA_TYPE, ANALYSIS_HOST_INVOCATION_VALUE_SCHEMA,
    ANALYSIS_HOST_TARGET_PROFILE,
};
pub use input_materialization::{InputMaterializationExpectation, PreparedInputMaterialization};
pub use input_plan::{InputMaterializationPlan, InputMaterializationPlanExpectation};
pub use materialized_signal::{
    canonical_tabulated_signal_content_digest, materialize_signal_for_solver,
};
pub use modal_object_access::{
    decode_prepared_structural_modal_receipt, decode_prepared_structural_modal_result,
    decode_prepared_structural_modal_workload, prepare_structural_modal_workload_objects,
};
pub use modal_task::{
    import_structural_modal_result, prepare_structural_modal_task, ImportedStructuralModalResult,
    PreparedStructuralModalTask, StructuralModalTaskPreparation,
};
pub use object::{
    analysis_object_references, import_analysis_objects, import_analysis_partition,
    prepare_analysis_objects, prepare_partitioned_analysis_objects, prepared_input_resource_facts,
    prepared_input_resource_facts_from_refs, prepared_input_resource_facts_without_resource_plan,
    projected_chunked_output_resource_facts, projected_output_resource_facts,
    projected_output_set_resource_facts, projected_transport_output_resource_facts,
    AnalysisPartitionImport, ImportedAnalysisPartition, PreparedAnalysisObjects,
};
pub use observation_assembly::{
    import_observation_assembly, prepare_observation_assembly, ObservationAssemblyPreparation,
    PreparedObservationAssembly,
};
pub use observation_evaluation::{
    import_observation_result, ImportedObservations, ObservationPreparation,
    ObservationPublishedSource,
};
pub use observation_history_access::{
    decode_prepared_observation_history_checkpoint, import_observation_history,
    import_observation_history_checkpoint, import_observation_history_sample,
    prepare_observation_history_checkpoint_objects, prepare_observation_history_objects,
    ImportedObservationHistory, ImportedObservationHistoryCheckpoint,
    ImportedObservationHistoryStorage, ObservationHistoryCheckpointImport,
    ObservationHistoryImport,
};
pub use observation_history_checkpoint::{
    ObservationHistoryCheckpointPublisher, PublishedObservationHistory,
    PublishedObservationHistoryCheckpoint,
};
pub use observation_history_finalization::{
    import_observation_history_finalization_task, prepare_observation_history_finalization_task,
    ImportedObservationHistoryFinalization, ObservationHistoryFinalizationPreparation,
    ObservationHistoryFinalizationSource, PreparedObservationHistoryFinalizationTask,
};
pub use observation_history_task::{
    import_observation_history_checkpoint_task, prepare_observation_history_checkpoint_task,
    ImportedObservationHistoryCheckpointTask, ObservationHistoryCheckpointPreparation,
    PreparedObservationHistoryCheckpointTask,
};
pub use observation_partition::{
    import_observation_partition_contribution, import_reduced_observation,
    prepare_observation_fanout, prepare_observation_reduction,
    ImportedObservationPartitionContribution, ImportedReducedObservation,
    ObservationReductionPreparation, PreparedObservationFanout, PreparedObservationPartition,
    PreparedObservationReduction,
};
pub use observation_source::ObservationWorkloadSource;
pub use random_response_object_access::{
    decode_prepared_structural_random_response_assembly_request,
    decode_prepared_structural_random_response_interval_request,
    decode_prepared_structural_random_response_interval_result,
    decode_prepared_structural_random_response_numerical_receipt,
    decode_prepared_structural_random_response_solver_result,
    decode_prepared_structural_random_response_workload,
    prepare_structural_random_response_assembly_request_objects,
    prepare_structural_random_response_interval_request_objects,
    prepare_structural_random_response_interval_result_objects,
    prepare_structural_random_response_numerical_receipt_objects,
    prepare_structural_random_response_workload_objects,
};
pub use random_response_task::{
    import_structural_random_response_interval, import_structural_random_response_result,
    prepare_structural_random_response_assembly, prepare_structural_random_response_fanout,
    ImportedStructuralRandomResponseInterval, ImportedStructuralRandomResponseResult,
    PreparedStructuralRandomResponseAssemblyTask, PreparedStructuralRandomResponseIntervalTask,
    StructuralRandomResponseAssemblyPreparation, StructuralRandomResponseFanoutPreparation,
};
pub use resource_usage::{
    extend_resource_usage, extend_structural_resource_usage, prepare_planned_resource_graph_usage,
    prepare_structural_resource_usage, validate_observation_resource_usage,
    validate_structural_resource_usage, PlannedResourceUsageAttempt, PlannedResourceUsageGraph,
    PreparedResourceUsage,
};
pub use response::{
    import_analysis_output_set, AnalysisHostResponse, AnalysisOutputRoot, AnalysisOutputSet,
    ImportedAnalysisOutputSet, ANALYSIS_OUTPUT_SET_MEDIA_TYPE, ANALYSIS_OUTPUT_SET_VALUE_SCHEMA,
};
pub use response_spectrum_object_access::{
    decode_prepared_structural_modal_response_spectrum_assembly_request,
    decode_prepared_structural_modal_response_spectrum_direction_request,
    decode_prepared_structural_modal_response_spectrum_direction_result,
    decode_prepared_structural_modal_response_spectrum_field_recovery,
    decode_prepared_structural_modal_response_spectrum_numerical_receipt,
    decode_prepared_structural_modal_response_spectrum_solver_result,
    decode_prepared_structural_modal_response_spectrum_workload,
    prepare_structural_modal_response_spectrum_assembly_request_objects,
    prepare_structural_modal_response_spectrum_direction_request_objects,
    prepare_structural_modal_response_spectrum_direction_result_objects,
    prepare_structural_modal_response_spectrum_field_recovery_objects,
    prepare_structural_modal_response_spectrum_numerical_receipt_objects,
    prepare_structural_modal_response_spectrum_workload_objects,
};
pub use response_spectrum_task::{
    import_structural_modal_response_spectrum_direction,
    import_structural_modal_response_spectrum_result,
    prepare_structural_modal_response_spectrum_assembly,
    prepare_structural_modal_response_spectrum_fanout,
    ImportedStructuralModalResponseSpectrumDirection,
    ImportedStructuralModalResponseSpectrumResult,
    PreparedStructuralModalResponseSpectrumAssemblyTask,
    PreparedStructuralModalResponseSpectrumDirectionTask,
    StructuralModalResponseSpectrumAssemblyPreparation,
    StructuralModalResponseSpectrumFanoutPreparation,
};
pub use result_publication::{
    import_published_result, prepare_result_publication, ImportedPublishedResult,
    PreparedResultPublication, ResultPublicationPreparation,
};
pub use runmat_analysis_fea::execution::SignalInputAdmission;
pub use runmat_analysis_fea::execution::{StructuralStudyWorkloads, StructuralTransientBlockPlan};
pub use semantic_state_object_access::{
    decode_prepared_semantic_state, prepare_semantic_state_objects,
};
pub use serial::{
    execute_reserved_serial_analysis, execute_serial_analysis,
    execute_serial_analysis_host_response, execute_serial_analysis_host_response_with_capacity,
    AnalysisKernel, AnalysisKernelFuture, AnalysisKernelInvocation, AnalysisKernelOutput,
    AnalysisKernelOutputPartition, AnalysisKernelPayload, AnalysisOutputPolicy, AsyncCapacityOwner,
    CompletedAnalysisWorkload, ReservedAnalysisKernelInvocation,
};
pub use signal_input::{validate_signal_input_payload, ResolvedSignalArtifact};
pub use signal_materialization::{
    canonicalize_signal_generator_outputs, plan_generator_chunks, plan_signal_generator_chunks,
    validate_generator_chunks, validate_signal_generator_chunks, GeneratorChunk,
    GeneratorChunkOutput, PlannedGenerator,
};
pub use signal_program::{
    build_generator_task_submission, generator_chunk_output_from_success, generator_task_id,
    prepare_generator_descriptor, prepare_generator_program, prepare_signal_materialization,
    GeneratorExecutionContext, GeneratorProgramAdmission, GeneratorProgramAdmissionRef,
    PreparedGeneratorChunk, PreparedGeneratorProgram, PreparedSignalMaterialization,
    SignalMaterializationExpectation, GENERATOR_EXECUTION_MODE, GENERATOR_TARGET_PROFILE,
};
pub use spectral_materialization::{
    canonical_materialized_random_spectrum_bytes, canonical_materialized_response_spectrum_bytes,
    decode_materialized_random_spectrum, decode_materialized_response_spectrum,
    materialize_random_spectrum, materialize_response_spectrum,
    plan_random_spectrum_generator_chunks, plan_response_spectrum_generator_chunks,
    prepare_numerical_materialization, NumericalGeneratorSource,
    NumericalMaterializationExpectation, PreparedNumericalMaterialization,
};
pub use state_publication::{
    import_semantic_state, import_semantic_state_completion, prepare_semantic_state_publication,
    ImportedSemanticState, PreparedSemanticStatePublication, SemanticStatePublicationPreparation,
};
mod nonlinear_state_publication;
mod state_publication_resource;
pub use nonlinear_state_publication::{
    build_prepared_nonlinear_state_publication_resource_plan, decode_nonlinear_state_publication,
    nonlinear_state_host_input, prepare_nonlinear_solid_state_publication,
    require_nonlinear_state_decode_budget, NonlinearSolidStatePublicationPreparation,
};
pub use state_publication_resource::build_prepared_state_publication_resource_plan;
pub use state_transition::bind_upstream_state_transition;
pub use structural::{
    import_structural_linear_completion, import_structural_linear_result,
    prepare_structural_linear_program, prepare_structural_sequence_step,
    ImportedStructuralLinearResult, PreparedStructuralLinearProgram,
    StructuralLinearProgramPreparation,
};
pub use structural_buckling_study::{
    StructuralBucklingStudyCoordinator, StructuralBucklingStudyPreparation,
};
pub use structural_modal_frequency_study::{
    PreparedStructuralModalFrequency, StructuralModalFrequencyStudyCoordinator,
    StructuralModalFrequencyStudyPreparation,
};
pub use structural_modal_response_spectrum_study::{
    StructuralModalResponseSpectrumStudyCoordinator,
    StructuralModalResponseSpectrumStudyPreparation,
};
pub use structural_modal_transient_study::{
    StructuralModalTransientStudyCoordinator, StructuralModalTransientStudyPreparation,
};
pub use structural_prestressed_frequency_study::{
    PreparedStructuralPrestressedFrequency, StructuralPrestressedFrequencyStudyCoordinator,
    StructuralPrestressedFrequencyStudyPreparation,
};
pub use structural_prestressed_modal_frequency_study::StructuralPrestressedModalFrequencyStudyCoordinator;
pub use structural_prestressed_modal_study::{
    StructuralPrestressedModalStudyCoordinator, StructuralPrestressedModalStudyPreparation,
};
pub use structural_random_response_study::{
    StructuralRandomResponseStudyCoordinator, StructuralRandomResponseStudyPreparation,
};
pub use structural_sequence::{
    import_structural_sequence_completion, validate_structural_sequence_workloads,
    CompletedStructuralSequence, CompletedStructuralSequenceState, CompletedStructuralSequenceStep,
    PreparedStructuralSolveTask, PreparedStructuralSolveWave, PreparedStructuralStateTask,
    PreparedStructuralStateWave, StructuralSequenceCoordinator, StructuralSequenceNode,
    StructuralSequencePreparation, StructuralSequenceResultClosure,
    StructuralSequenceSolveSelection, StructuralSequenceStateResourceAttempt,
};
pub use structural_transient_recovery::{
    StructuralTransientExecutionCheckpoint, StructuralTransientExecutionCheckpointIdentity,
    StructuralTransientRecoveryClosure, STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_MEDIA_TYPE,
    STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_VALUE_SCHEMA,
};
pub use structural_transient_sequence::{
    complete_structural_transient_sequence_preparation, prepare_structural_transient_sequence,
    StructuralTransientSequenceCoordinator, StructuralTransientSequencePlan,
    StructuralTransientSequencePreparation, StructuralTransientSequencePreparationInput,
};
pub use task::{
    build_analysis_task_submission, build_resource_planned_task_submission,
    build_resource_planned_task_submission_with_closure, AnalysisExecutionContext,
    PreparedAnalysisTaskClosure, DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS,
};
pub use thermal_execution::execute_thermal_solver_result_objects;
pub use thermal_result_access::{
    decode_prepared_thermal_manifest, decode_prepared_thermal_state, import_thermal_state,
    prepare_thermal_solver_result_objects, thermal_receipt_from_prepared_result,
    ThermalStateImport,
};
pub use transient_block_task::{
    import_structural_transient_block, prepare_structural_transient_block_task,
    ImportedStructuralTransientBlock, PreparedStructuralTransientBlockTask,
    StructuralTransientBlockPreparation,
};
pub use transient_object_access::{
    decode_prepared_structural_transient_block_request,
    decode_prepared_structural_transient_checkpoint, decode_prepared_structural_transient_scope,
    decode_prepared_structural_transient_segment, decode_prepared_structural_transient_workload,
    import_structural_transient_segment_sample, prepare_structural_transient_block_request_objects,
    prepare_structural_transient_checkpoint_objects, prepare_structural_transient_scope_objects,
    prepare_structural_transient_segment_objects, prepare_structural_transient_workload_objects,
    ImportedStructuralTransientSegmentSample, StructuralTransientSegmentSampleImport,
};
pub use transient_result_access::{
    decode_prepared_structural_transient_numerical_receipt,
    decode_prepared_structural_transient_result_assembly_request,
    decode_prepared_structural_transient_solver_result, import_structural_transient_result_sample,
    prepare_structural_transient_numerical_receipt_objects,
    prepare_structural_transient_result_assembly_request_objects,
    prepare_structural_transient_solver_result_objects, ImportedStructuralTransientResultSample,
    StructuralTransientResultSampleImport,
};
pub use transient_result_task::{
    build_prepared_structural_transient_result_resource_plan, import_structural_transient_result,
    prepare_structural_transient_result_task, ImportedStructuralTransientResult,
    PreparedStructuralTransientResultTask, StructuralTransientResultPreparation,
};
pub use transient_scope_task::{
    import_structural_transient_scope, prepare_structural_transient_scope_task,
    transient_scope_task_identity, ImportedStructuralTransientScope,
    PreparedStructuralTransientScopeTask, StructuralTransientScopeTaskPreparation,
};
pub use transport::{prepare_analysis_program, PreparedAnalysisProgram};
pub use workload::{
    analysis_worker_capabilities, AnalysisCapabilityRequirement, AnalysisWorkloadInput,
    AnalysisWorkloadRequest,
};

#[cfg(test)]
mod test_fixtures;
#[cfg(test)]
mod tests;

mod nonlinear_field_recovery;
pub use nonlinear_field_recovery::{
    build_nonlinear_recovery_metadata_resource_plan,
    build_prepared_nonlinear_recovery_resource_plan, decode_nonlinear_recovery_source,
    import_nonlinear_recovered_fields, prepare_nonlinear_field_recovery,
    require_nonlinear_recovery_decode_budget, NonlinearFieldRecoveryPreparation,
    PreparedNonlinearFieldRecovery,
};

mod serial_kernel;
pub use serial_kernel::{
    execute_serial_structural_sequence, execute_serial_structural_sequence_with_capacity,
    execute_state_publication_kernel, execute_structural_linear_kernel, SerialAnalysisHost,
    SerialAnalysisServices, SerialStructuralAnalysisKernel,
};

/// Source-bound artifact authority shared by checked-source execution hosts.
pub fn checked_fea_artifact_access(authority: &str) -> runmat_execution_artifact::ArtifactAccess {
    runmat_execution_artifact::ArtifactAccess {
        authorization_scope: authority.to_owned(),
        encryption_context: runmat_execution::Digest::sha256(
            [
                b"runmat-typed-fea-access/v1\0".as_slice(),
                authority.as_bytes(),
            ]
            .concat(),
        ),
    }
}
