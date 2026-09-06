use std::collections::BTreeSet;

use anyhow::{Context, Result};
use runmat_analysis_execution::{
    import_structural_linear_result, prepare_structural_linear_program,
    StructuralLinearProgramPreparation,
};
use runmat_analysis_fea::execution::{StructuralLinearWorkload, DEFAULT_ANALYSIS_CHUNK_BYTES};
use runmat_analysis_fea::FeaRunResult;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::{
    execute_analysis_program_request, FilesystemObjectStore, NativeAnalysisHostLimits,
    NativeAnalysisKernel,
};
use runmat_meshing_core::SolverMeshArtifact;
use runmat_runtime::analysis::{
    analysis_study_content_digest, AnalysisStudySpec, StructuralLinearExecutor,
};

pub(super) struct InlineStructuralExecutor {
    study_digest: runmat_execution::Digest,
    program_revision: ProgramRevision,
    object_root: tempfile::TempDir,
}

impl InlineStructuralExecutor {
    pub(super) fn new(study: &AnalysisStudySpec, config: &RunMatRuntimeConfig) -> Result<Self> {
        let study_digest = analysis_study_content_digest(study);
        let program_revision = ProgramRevision::new(
            study_digest,
            study_digest,
            runmat_core::program_environment(crate::diagnostics::parser_compat(
                config.language.compat,
            )),
        )?;
        Ok(Self {
            study_digest,
            program_revision,
            object_root: tempfile::tempdir().context("create inline analysis object store")?,
        })
    }

    fn access(&self) -> ArtifactAccess {
        ArtifactAccess {
            authorization_scope: format!("study:{}", self.study_digest),
            encryption_context: runmat_execution::Digest::sha256(
                [
                    b"runmat-inline-analysis-access/v1\0".as_slice(),
                    self.study_digest.bytes(),
                ]
                .concat(),
            ),
        }
    }
}

impl StructuralLinearExecutor for InlineStructuralExecutor {
    fn execute(
        &self,
        workload: &StructuralLinearWorkload,
        mesh: &SolverMeshArtifact,
    ) -> Result<FeaRunResult, String> {
        let limits = ObjectInventoryLimits::default();
        let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
            authored_study_digest: self.study_digest,
            frozen_project_digest: None,
            program_revision: self.program_revision.clone(),
            artifact_access: self.access(),
            workload: workload.clone(),
            mesh: mesh.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: limits,
            required_capabilities: BTreeSet::new(),
        })
        .map_err(|error| error.to_string())?;
        let mut store = FilesystemObjectStore::open(
            self.object_root.path().to_path_buf(),
            limits.max_object_bytes,
        )
        .map_err(|error| error.to_string())?;
        for input in &prepared.inputs {
            for object in &input.objects {
                store
                    .write_verified(object)
                    .map_err(|error| error.to_string())?;
            }
        }
        let response = execute_analysis_program_request(
            &prepared.program.request,
            &mut store,
            &NativeAnalysisKernel,
            NativeAnalysisHostLimits {
                chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                inventory: limits,
            },
        );
        import_structural_linear_result(&store, response, &prepared, limits)
            .map(|result| result.run)
            .map_err(|error| error.to_string())
    }
}
