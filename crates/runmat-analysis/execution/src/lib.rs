//! Narrow bridge from analysis-owned logical contracts to shared RunMat execution services.

mod error;
mod host;
mod object;
mod task;
mod workload;

pub use error::{AnalysisBridgeError, AnalysisBridgeResult};
pub use host::{
    AnalysisHostWorkload, ANALYSIS_HOST_ABI, ANALYSIS_HOST_EXECUTION_MODE,
    ANALYSIS_HOST_TARGET_PROFILE,
};
pub use object::{prepare_analysis_objects, PreparedAnalysisObjects};
pub use task::{build_analysis_task_submission, AnalysisExecutionContext};
pub use workload::{
    analysis_worker_capabilities, AnalysisCapabilityRequirement, AnalysisWorkloadInput,
    AnalysisWorkloadRequest,
};

#[cfg(test)]
mod tests;
