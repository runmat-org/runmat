//! Narrow bridge from analysis-owned logical contracts to shared RunMat execution services.

mod accuracy_evaluation;
mod callable;
mod error;
mod field_recovery;
mod frequency_field_recovery;
mod frequency_object_access;
mod frequency_task;
mod graph;
mod host;
mod materialized_signal;
mod object;
mod observation_assembly;
mod observation_evaluation;
mod observation_field_partitions;
mod observation_history_access;
mod observation_history_checkpoint;
mod observation_history_task;
mod observation_partition;
mod observation_source;
mod resource_usage;
mod response;
mod result_publication;
mod semantic_state_object_access;
mod serial;
mod signal_input;
mod signal_materialization;
mod signal_program;
mod state_publication;
mod structural;
mod structural_sequence;
mod structural_transient_recovery;
mod structural_transient_sequence;
mod task;
mod transient_block_task;
mod transient_object_access;
mod transient_result_access;
mod transient_result_task;
mod transport;
mod workload;

pub use accuracy_evaluation::{
    import_structural_accuracy, prepare_structural_accuracy, ImportedAccuracy,
    PreparedStructuralAccuracy, StructuralAccuracyPreparation,
};
pub use callable::{
    admit_callable_signal_generator, admit_named_signal_generator, AdmittedSignalGenerator,
};
pub use error::{AnalysisBridgeError, AnalysisBridgeResult};
pub use field_recovery::{
    decode_recovered_field_artifact, import_structural_recovered_fields,
    prepare_structural_field_recovery, ImportedRecoveredFields, PreparedStructuralFieldRecovery,
    StructuralFieldRecoveryPreparation,
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
pub use materialized_signal::{
    canonical_tabulated_signal_content_digest, materialize_signal_for_solver,
};
pub use object::{
    analysis_object_references, import_analysis_objects, import_analysis_partition,
    prepare_analysis_objects, prepare_partitioned_analysis_objects, prepared_input_resource_facts,
    projected_output_resource_facts, projected_output_set_resource_facts, AnalysisPartitionImport,
    ImportedAnalysisPartition, PreparedAnalysisObjects,
};
pub use observation_assembly::{
    import_structural_observation_assembly, prepare_structural_observation_assembly,
    PreparedStructuralObservationAssembly, StructuralObservationAssemblyPreparation,
};
pub use observation_evaluation::{
    import_observation_result, ImportedObservations, StructuralObservationPreparation,
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
pub use observation_history_task::{
    import_observation_history_checkpoint_task, prepare_observation_history_checkpoint_task,
    ImportedObservationHistoryCheckpointTask, ObservationHistoryCheckpointPreparation,
    PreparedObservationHistoryCheckpointTask,
};
pub use observation_partition::{
    import_observation_partition_contribution, import_reduced_observation,
    prepare_structural_observation_fanout, prepare_structural_observation_reduction,
    ImportedObservationPartitionContribution, ImportedReducedObservation,
    PreparedStructuralObservationFanout, PreparedStructuralObservationPartition,
    PreparedStructuralObservationReduction, StructuralObservationReductionPreparation,
};
pub use observation_source::StructuralObservationWorkloadSource;
pub use resource_usage::{
    extend_structural_resource_usage, prepare_planned_resource_graph_usage,
    prepare_structural_resource_usage, validate_observation_resource_usage,
    validate_structural_resource_usage, PlannedResourceUsageAttempt, PlannedResourceUsageGraph,
    PreparedStructuralResourceUsage,
};
pub use response::{
    import_analysis_output_set, AnalysisHostResponse, AnalysisOutputRoot, AnalysisOutputSet,
    ImportedAnalysisOutputSet, ANALYSIS_OUTPUT_SET_MEDIA_TYPE, ANALYSIS_OUTPUT_SET_VALUE_SCHEMA,
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
    execute_serial_analysis, AnalysisKernel, AnalysisKernelInvocation, AnalysisKernelOutput,
    AnalysisKernelOutputPartition, CompletedAnalysisWorkload,
};
pub use signal_input::{validate_signal_input_payload, ResolvedSignalArtifact};
pub use signal_materialization::{
    canonicalize_signal_generator_outputs, plan_signal_generator_chunks,
    validate_signal_generator_chunks, PlannedSignalGenerator, SignalGeneratorChunk,
    SignalGeneratorChunkOutput,
};
pub use signal_program::{
    build_signal_generator_task_submission, prepare_signal_generator_program,
    prepare_signal_materialization, PreparedSignalGeneratorChunk, PreparedSignalGeneratorProgram,
    PreparedSignalMaterialization, SignalGeneratorExecutionContext,
    SignalMaterializationExpectation, SIGNAL_GENERATOR_EXECUTION_MODE,
    SIGNAL_GENERATOR_TARGET_PROFILE,
};
pub use state_publication::{
    import_semantic_state, import_semantic_state_completion, prepare_semantic_state_publication,
    ImportedSemanticState, PreparedSemanticStatePublication, SemanticStatePublicationPreparation,
};
pub use structural::{
    import_structural_linear_completion, import_structural_linear_result,
    prepare_structural_linear_program, prepare_structural_sequence_step,
    ImportedStructuralLinearResult, PreparedStructuralLinearProgram,
    StructuralLinearProgramPreparation,
};
pub use structural_sequence::{
    import_structural_sequence_completion, validate_structural_sequence_workloads,
    CompletedStructuralSequence, CompletedStructuralSequenceState, CompletedStructuralSequenceStep,
    PreparedStructuralSolveTask, PreparedStructuralSolveWave, PreparedStructuralStateTask,
    PreparedStructuralStateWave, StructuralSequenceCoordinator, StructuralSequenceNode,
    StructuralSequencePreparation, StructuralSequenceResultClosure,
};
pub use structural_transient_recovery::{
    StructuralTransientExecutionCheckpoint, StructuralTransientExecutionCheckpointIdentity,
    StructuralTransientRecoveryClosure, STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_MEDIA_TYPE,
    STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_VALUE_SCHEMA,
};
pub use structural_transient_sequence::{
    prepare_structural_transient_sequence, StructuralTransientSequenceCoordinator,
    StructuralTransientSequencePreparation, StructuralTransientSequencePreparationInput,
};
pub use task::{
    build_analysis_task_submission, build_structural_task_submission, AnalysisExecutionContext,
    DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS,
};
pub use transient_block_task::{
    import_structural_transient_block, prepare_structural_transient_block_task,
    ImportedStructuralTransientBlock, PreparedStructuralTransientBlockTask,
    StructuralTransientBlockPreparation,
};
pub use transient_object_access::{
    decode_prepared_structural_transient_block_request,
    decode_prepared_structural_transient_checkpoint, decode_prepared_structural_transient_segment,
    decode_prepared_structural_transient_workload, import_structural_transient_segment_sample,
    prepare_structural_transient_block_request_objects,
    prepare_structural_transient_checkpoint_objects, prepare_structural_transient_segment_objects,
    prepare_structural_transient_workload_objects, ImportedStructuralTransientSegmentSample,
    StructuralTransientSegmentSampleImport,
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
    import_structural_transient_result, prepare_structural_transient_result_task,
    ImportedStructuralTransientResult, PreparedStructuralTransientResultTask,
    StructuralTransientResultPreparation,
};
pub use transport::{prepare_analysis_program, PreparedAnalysisProgram};
pub use workload::{
    analysis_worker_capabilities, AnalysisCapabilityRequirement, AnalysisWorkloadInput,
    AnalysisWorkloadRequest,
};

#[cfg(test)]
mod tests;
