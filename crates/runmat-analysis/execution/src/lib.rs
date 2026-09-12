//! Narrow bridge from analysis-owned logical contracts to shared RunMat execution services.

mod accuracy_evaluation;
mod callable;
mod error;
mod field_recovery;
mod graph;
mod history_access;
mod history_checkpoint;
mod history_task;
mod host;
mod materialized_signal;
mod object;
mod observation_assembly;
mod observation_evaluation;
mod observation_field_partitions;
mod observation_partition;
mod resource_usage;
mod response;
mod result_publication;
mod serial;
mod signal_input;
mod signal_materialization;
mod signal_program;
mod state_publication;
mod structural;
mod structural_sequence;
mod task;
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
pub use graph::{
    compile_analysis_sequence_graph, AnalysisSequenceExecutionGraph, AnalysisSequenceTask,
    PreparedAnalysisStepTask,
};
pub use history_access::{
    decode_prepared_observation_history_checkpoint, import_observation_history,
    import_observation_history_checkpoint, import_observation_history_sample,
    prepare_observation_history_checkpoint_objects, prepare_observation_history_objects,
    ImportedObservationHistory, ImportedObservationHistoryCheckpoint,
    ImportedObservationHistoryStorage, ObservationHistoryCheckpointImport,
    ObservationHistoryImport,
};
pub use history_checkpoint::{
    ObservationHistoryCheckpointPublisher, PublishedObservationHistory,
    PublishedObservationHistoryCheckpoint,
};
pub use history_task::{
    import_history_checkpoint_task, prepare_history_checkpoint_task, HistoryCheckpointPreparation,
    ImportedHistoryCheckpointTask, PreparedHistoryCheckpointTask,
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
    prepare_analysis_objects, prepare_partitioned_analysis_objects, AnalysisPartitionImport,
    ImportedAnalysisPartition, PreparedAnalysisObjects,
};
pub use observation_assembly::{
    import_structural_observation_assembly, prepare_structural_observation_assembly,
    PreparedStructuralObservationAssembly, StructuralObservationAssemblyPreparation,
};
pub use observation_evaluation::{
    import_observation_result, ImportedObservations, StructuralObservationPreparation,
};
pub use observation_partition::{
    import_observation_partition_contribution, import_reduced_observation,
    prepare_structural_observation_fanout, prepare_structural_observation_reduction,
    ImportedObservationPartitionContribution, ImportedReducedObservation,
    PreparedStructuralObservationFanout, PreparedStructuralObservationPartition,
    PreparedStructuralObservationReduction, StructuralObservationReductionPreparation,
};
pub use resource_usage::{
    extend_structural_resource_usage, prepare_structural_resource_usage,
    validate_structural_resource_usage, PreparedStructuralResourceUsage,
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
pub use task::{
    build_analysis_task_submission, build_structural_task_submission, AnalysisExecutionContext,
    DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS,
};
pub use transport::{prepare_analysis_program, PreparedAnalysisProgram};
pub use workload::{
    analysis_worker_capabilities, AnalysisCapabilityRequirement, AnalysisWorkloadInput,
    AnalysisWorkloadRequest,
};

#[cfg(test)]
mod tests;
