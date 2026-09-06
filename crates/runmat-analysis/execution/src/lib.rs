//! Narrow bridge from analysis-owned logical contracts to shared RunMat execution services.

mod error;
mod host;
mod object;
mod response;
mod serial;
mod structural;
mod task;
mod transport;
mod workload;

pub use error::{AnalysisBridgeError, AnalysisBridgeResult};
pub use host::{
    AnalysisHostWorkload, ANALYSIS_HOST_ABI, ANALYSIS_HOST_EXECUTION_MODE,
    ANALYSIS_HOST_TARGET_PROFILE,
};
pub use object::{import_analysis_objects, prepare_analysis_objects, PreparedAnalysisObjects};
pub use response::AnalysisHostResponse;
pub use serial::{
    execute_serial_analysis, AnalysisKernel, AnalysisKernelInvocation, AnalysisKernelOutput,
    CompletedAnalysisWorkload,
};
pub use structural::{
    import_structural_linear_result, prepare_structural_linear_program,
    PreparedStructuralLinearProgram, StructuralLinearProgramPreparation,
};
pub use task::{build_analysis_task_submission, AnalysisExecutionContext};
pub use transport::{prepare_analysis_program, PreparedAnalysisProgram};
pub use workload::{
    analysis_worker_capabilities, AnalysisCapabilityRequirement, AnalysisWorkloadInput,
    AnalysisWorkloadRequest,
};

#[cfg(test)]
mod tests;
