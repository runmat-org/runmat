use anyhow::{bail, Context, Result};
use runmat_analysis_core::AuthoredStudyInputBinding;
use runmat_analysis_language::ResolvedAnalysisStudy;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution_artifact::ArtifactAccess;

use super::source::NativeCheckedFeaSource;

pub(super) type PreparedTypedStudy = runmat_execution_runner_native::PreparedFeaStudy;

pub(super) fn ensure_remote_structural(
    sequence: &runmat_analysis_core::AnalysisSequence,
) -> Result<()> {
    if sequence
        .steps
        .iter()
        .any(|step| step.family != runmat_analysis_core::PhysicsFamily::Structural)
    {
        bail!("RM.FEA.TYPED_EXECUTION.REMOTE_PHYSICS_UNSUPPORTED: authored remote execution requires a structural study; standalone thermal authored roots are not implemented");
    }
    Ok(())
}

pub(super) fn require_remote_root(
    root: runmat_execution_runner_native::PreparedFeaStudyRoot,
) -> Result<runmat_execution_runner_native::PreparedStructuralStudyRoot> {
    match root {
        runmat_execution_runner_native::PreparedFeaStudyRoot::Structural(root) => Ok(root),
        runmat_execution_runner_native::PreparedFeaStudyRoot::Electrical(_) => bail!("RM.FEA.TYPED_EXECUTION.REMOTE_PHYSICS_UNSUPPORTED: standalone electrical authored roots are not implemented for remote submission"),
        runmat_execution_runner_native::PreparedFeaStudyRoot::Magnetic(_) => bail!("RM.FEA.TYPED_EXECUTION.REMOTE_PHYSICS_UNSUPPORTED: standalone magnetic authored roots are not implemented for remote submission"),
        runmat_execution_runner_native::PreparedFeaStudyRoot::Acoustic(_) => bail!("RM.FEA.TYPED_EXECUTION.REMOTE_PHYSICS_UNSUPPORTED: standalone acoustic authored roots are not implemented for remote submission"),
        runmat_execution_runner_native::PreparedFeaStudyRoot::Thermal(_) => bail!("RM.FEA.TYPED_EXECUTION.REMOTE_PHYSICS_UNSUPPORTED: standalone thermal authored roots are not implemented for remote submission"),
    }
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
}

pub(super) fn artifact_access(authority: &str) -> ArtifactAccess {
    runmat_execution_runner_native::checked_fea_artifact_access(authority)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::{AnalysisSequence, ProcedureStep};

    #[test]
    fn remote_authored_thermal_is_rejected_before_preparation_or_submission() {
        for step in [
            ProcedureStep::thermal_steady("heat"),
            ProcedureStep::thermal_transient("heat"),
        ] {
            let error =
                ensure_remote_structural(&AnalysisSequence::from_steps("thermal", vec![step]))
                    .unwrap_err();
            assert!(error.to_string().contains("REMOTE_PHYSICS_UNSUPPORTED"));
        }
        assert!(ensure_remote_structural(&AnalysisSequence::from_steps(
            "structural",
            vec![ProcedureStep::structural_static("solid")]
        ))
        .is_ok());
    }
}
