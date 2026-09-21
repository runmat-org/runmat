use anyhow::{Context, Result};
use runmat_analysis_core::AuthoredStudyInputBinding;
use runmat_analysis_language::ResolvedAnalysisStudy;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution_artifact::ArtifactAccess;

use super::source::NativeCheckedFeaSource;

pub(super) type PreparedTypedStudy = runmat_execution_runner_native::PreparedFeaStudy;

pub(super) async fn prepare_study(
    source: &NativeCheckedFeaSource,
    resolved: ResolvedAnalysisStudy,
    inputs: &[AuthoredStudyInputBinding],
    authority: &str,
    compute: &FeaComputeAllocation,
    _config: &RunMatRuntimeConfig,
) -> Result<PreparedTypedStudy> {
    runmat_execution_runner_native::prepare_checked_fea_study(
        &source.session_source,
        resolved,
        inputs,
        authority,
        &runmat_execution_runner_native::LocalFeaExecutionOptions {
            root: None,
            workers: None,
            backend: compute.backend,
            accelerators: compute.accelerators.clone(),
        },
    )
    .await
    .map_err(anyhow::Error::msg)
    .context("prepare checked FEA study")
}

#[derive(Clone, Debug)]
pub(super) struct FeaComputeAllocation {
    pub backend: runmat_analysis_fea::ComputeBackend,
    pub accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
}

pub(super) fn artifact_access(authority: &str) -> ArtifactAccess {
    runmat_execution_runner_native::checked_fea_artifact_access(authority)
}
