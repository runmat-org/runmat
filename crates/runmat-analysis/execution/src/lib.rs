//! Narrow bridge from analysis-owned logical contracts to shared RunMat execution services.

mod callable;
mod error;
mod graph;
mod host;
mod materialized_signal;
mod object;
mod response;
mod serial;
mod signal_materialization;
mod signal_program;
mod structural;
mod task;
mod transport;
mod workload;

pub use callable::{admit_named_signal_generator, AdmittedSignalGenerator};
pub use error::{AnalysisBridgeError, AnalysisBridgeResult};
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
pub use materialized_signal::materialize_signal_for_solver;
pub use object::{import_analysis_objects, prepare_analysis_objects, PreparedAnalysisObjects};
pub use response::AnalysisHostResponse;
pub use serial::{
    execute_serial_analysis, AnalysisKernel, AnalysisKernelInvocation, AnalysisKernelOutput,
    CompletedAnalysisWorkload,
};
pub use signal_materialization::{
    canonicalize_signal_generator_outputs, plan_signal_generator_chunks, SignalGeneratorChunk,
    SignalGeneratorChunkOutput,
};
pub use signal_program::{
    build_signal_generator_task_submission, prepare_signal_generator_program,
    PreparedSignalGeneratorChunk, PreparedSignalGeneratorProgram, SignalGeneratorExecutionContext,
    SIGNAL_GENERATOR_EXECUTION_MODE, SIGNAL_GENERATOR_TARGET_PROFILE,
};
pub use structural::{
    import_structural_linear_result, prepare_structural_linear_program,
    PreparedStructuralLinearProgram, StructuralLinearProgramPreparation,
};
pub use task::{
    build_analysis_task_submission, AnalysisExecutionContext, DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS,
};
pub use transport::{prepare_analysis_program, PreparedAnalysisProgram};
pub use workload::{
    analysis_worker_capabilities, AnalysisCapabilityRequirement, AnalysisWorkloadInput,
    AnalysisWorkloadRequest,
};

#[cfg(test)]
mod tests;
