//! Preserve typed admission failures without interpreting diagnostic text.
use runmat_analysis_execution::AnalysisBridgeError;
use runmat_analysis_fea::{contracts::FeaRunError, planning::resource::ResourcePlanningError};
use runmat_runtime::analysis::storage::AnalysisPublicationError;

pub(crate) fn limit(message: impl Into<String>) -> AnalysisPublicationError {
    AnalysisPublicationError::ResourceLimit {
        message: message.into(),
    }
}
fn resource(error: ResourcePlanningError) -> AnalysisPublicationError {
    match error {
        limited @ (ResourcePlanningError::Overflow(_)
        | ResourcePlanningError::AccuracyCeiling { .. }
        | ResourcePlanningError::Inventory(_)
        | ResourcePlanningError::MemoryBudget { .. }
        | ResourcePlanningError::ModalSubspaceCapacity { .. }
        | ResourcePlanningError::CanonicalWireCapacity { .. }
        | ResourcePlanningError::Capability(_)
        | ResourcePlanningError::Accelerator { .. }) => limit(limited.to_string()),
        invalid => invalid.to_string().into(),
    }
}
pub(crate) fn fea(error: FeaRunError) -> AnalysisPublicationError {
    match error {
        FeaRunError::Resource(error) => resource(error),
        rejected => rejected.to_string().into(),
    }
}
pub(crate) fn artifact(
    error: runmat_execution_artifact::ArtifactError,
) -> AnalysisPublicationError {
    match error {
        runmat_execution_artifact::ArtifactError::Limit(message) => limit(message),
        rejected => rejected.to_string().into(),
    }
}
pub(crate) fn bridge(error: AnalysisBridgeError) -> AnalysisPublicationError {
    match error {
        AnalysisBridgeError::RetentionCapacity(error) => limit(error.to_string()),
        AnalysisBridgeError::Codec(error) => error.into(),
        AnalysisBridgeError::Artifact(error) => artifact(error),
        AnalysisBridgeError::Analysis(
            runmat_analysis_fea::execution::AnalysisExecutionError::Limit(message),
        ) => limit(message),
        AnalysisBridgeError::Analysis(
            runmat_analysis_fea::execution::AnalysisExecutionError::Codec(error),
        ) => error.into(),
        AnalysisBridgeError::FeaRun(error) => fea(error),
        AnalysisBridgeError::Resource(error) => resource(error),
        rejected => rejected.to_string().into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_components_preserve_capacity_and_original_resource_cause() {
        let error = runmat_analysis_execution::CoordinatorRetentionLimit {
            cause: ResourcePlanningError::AccuracyCeiling {
                resource: "test coordinator",
                required: 21,
                maximum: 20,
            },
            decoded: 1,
            encoded: 2,
            coordinator: 3,
            transport: 4,
            caller: 5,
            import_workspace: 6,
        };
        assert!(matches!(
            &error.cause,
            ResourcePlanningError::AccuracyCeiling {
                required: 21,
                maximum: 20,
                ..
            }
        ));
        assert!(matches!(
            bridge(error.into()),
            AnalysisPublicationError::ResourceLimit { .. }
        ));
    }
}
