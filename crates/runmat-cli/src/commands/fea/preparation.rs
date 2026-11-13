use anyhow::{Context, Result};
use runmat_analysis_core::AuthoredStudyInputBinding;
use runmat_analysis_language::ResolvedAnalysisStudy;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution_artifact::ArtifactAccess;

use super::source::NativeCheckedFeaSource;

pub(super) type PreparedTypedStudy = runmat_execution_runner_native::PreparedFeaStudy;

/// Encode the one portable root that every authored family uses for remote submission.
pub(super) fn portable_root(
    root: runmat_execution_runner_native::StructuralStudyRootPreparation,
) -> Result<runmat_execution_runner_native::PreparedStructuralStudyRoot> {
    runmat_execution_runner_native::prepare_structural_study_root(root)
        .map_err(anyhow::Error::new)
        .context("encode the portable study root")
}

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
            accelerator_placement: compute.placement.clone(),
            ..Default::default()
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
    /// The admitted placement the published study records and settles.
    pub placement: runmat_analysis_fea::accelerator_admission::AcceleratorPlacement,
}

pub(super) fn artifact_access(authority: &str) -> ArtifactAccess {
    runmat_execution_runner_native::checked_fea_artifact_access(authority)
}
