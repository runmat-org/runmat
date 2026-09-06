use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use base64::Engine as _;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, ExecutionBundleBuilder, ProgramExecutionDescriptor, ProgramExecutionInputs,
    ProgramExecutionObject, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner_native::{
    import_exact_structural_study_result, prepare_exact_meshing_root,
    prepare_exact_structural_study_root, ExactMeshingRootPreparation,
    ExactStructuralStudyResultExpectation, ExactStructuralStudyRootPreparation,
    NativeMeshingContactDefinition, NativeMeshingPartitionPolicy, PreparedExactStructuralStudyRoot,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_runtime::analysis::{
    analysis_prepare_structural_study_intent_op, AnalysisStudyRunData, AnalysisStudySpec,
    PreparedStructuralStudyIntent,
};
use runmat_runtime::operations::{OperationContext, OperationEnvelope};

use crate::cli::RemoteExecutionTarget;
use crate::commands::job::secret::SavedOutputIntent;
use crate::commands::job::submit::{PreparedRemoteExecution, RemoteSubmissionOptions};

pub(super) async fn execute_study(
    source: &Path,
    spec: AnalysisStudySpec,
    context: OperationContext,
    target: RemoteExecutionTarget,
    cli: &crate::cli::Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<Option<OperationEnvelope<AnalysisStudyRunData>>> {
    let geometry_path = std::path::PathBuf::from(&spec.geometry.source.path);
    let geometry_name = geometry_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("FEA geometry file name is not valid UTF-8")?
        .to_owned();
    let study = analysis_prepare_structural_study_intent_op(&spec, context)
        .map_err(super::report_operation_error)?;
    validate_remote_study_backend(study.workload_template().backend)?;
    let geometry_bytes = std::fs::read(&geometry_path)
        .with_context(|| format!("read FEA geometry {}", geometry_path.display()))?;
    if Digest::sha256(&geometry_bytes) != study.geometry_source_digest() {
        anyhow::bail!("FEA geometry differs from its authored source SHA-256");
    }
    let continuation = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(study.continuation_bytes().map_err(anyhow::Error::msg)?);
    let resolved = crate::commands::package::resolve_for_source(source, cli)
        .await?
        .context("remote FEA requires a runmat.toml project containing the study document")?;
    let project_identity = resolved.resolved.frozen.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_identity.graph_digest.bytes()),
        Digest::from_bytes(*project_identity.source_revision.bytes()),
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let placeholder = prepare_root(
        &study,
        revision.clone(),
        "bundle-sizing-authority",
        &geometry_name,
        &geometry_bytes,
    )?;
    let mut bundle_builder =
        ExecutionBundleBuilder::native(&resolved.resolved.frozen, revision.clone())?
            .with_compiled_package_closure();
    for program in placeholder.required_programs()? {
        bundle_builder = bundle_builder.with_materialized_program(
            program.recipe,
            program.artifact.form,
            program.artifact.executable_bytes,
        );
    }
    let bundle = bundle_builder.build()?;
    let mut bundle_archive = Vec::new();
    write_bundle(&bundle, &mut bundle_archive, ArchiveLimits::default())?;
    let bundle_size = bundle_archive.len();
    let revision_for_scope = revision.clone();
    let committed = crate::commands::job::submit::submit_scoped(
        revision.clone(),
        bundle_size,
        |run_id| {
            let prepared = prepare_root(
                &study,
                revision_for_scope,
                run_id,
                &geometry_name,
                &geometry_bytes,
            )?;
            prepare_remote_execution(prepared, revision.clone(), bundle_archive)
        },
        RemoteSubmissionOptions {
            project: target.project,
            cluster: target.cluster,
            queue: target.queue,
            trust_identity: target.trust_identity,
            idempotency_key: target.idempotency_key,
            workers: target.workers.max(1),
            on_run_created: None,
            output_intent: SavedOutputIntent::StructuralStudy {
                continuation: continuation.clone(),
                revision: revision.clone(),
                materialization: crate::commands::job::secret::OutputMaterialization::Pending,
                result: None,
            },
        },
    )
    .await?;
    if target.detach {
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "run": committed,
                    "materialization": "deferred",
                    "output": "structural_study",
                }))?
            );
        } else {
            println!(
                "remote FEA job {} submitted ({})",
                committed.id, committed.state
            );
            println!("  detached; verified study results will materialize on job attach");
        }
        return Ok(None);
    }
    let (outputs, objects) =
        crate::commands::job::attach::await_downloaded_outputs(&committed.id).await?;
    let envelope = materialize_remote_outputs(
        &committed.id,
        outputs.response,
        objects,
        &continuation,
        revision,
    )?;
    crate::commands::job::secret::mark_structural_study_materialized(
        &committed.id,
        envelope.clone(),
    )?;
    Ok(Some(envelope))
}

pub(crate) fn materialize_remote_outputs(
    run_id: &str,
    response: runmat_execution_artifact::ProgramExecutionResponse,
    objects: Vec<ProgramExecutionObject>,
    continuation: &str,
    revision: ProgramRevision,
) -> Result<OperationEnvelope<AnalysisStudyRunData>> {
    let continuation = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(continuation)
        .context("saved structural study continuation is malformed")?;
    let study = PreparedStructuralStudyIntent::from_continuation_bytes(&continuation)
        .map_err(anyhow::Error::msg)?;
    let expectation = result_expectation(&study, revision, run_id)?;
    let cache = DownloadedResultCache(
        objects
            .into_iter()
            .map(|object| (object.reference.logical_digest, object.bytes))
            .collect(),
    );
    let result = import_exact_structural_study_result(
        &cache,
        response,
        &expectation,
        ObjectInventoryLimits::default(),
    )?;
    runmat_runtime::analysis::analysis_complete_structural_study_intent_op(
        study,
        runmat_runtime::meshing::MeshingExecutionOutput {
            artifact: result.mesh.artifact,
            evidence: result.mesh.evidence,
            boundary_region_ids: result.boundary_resolution.aliases,
        },
        result.analysis.run,
        runmat_runtime::analysis::StructuralStudyCompletionIdentity::new(run_id)
            .map_err(anyhow::Error::msg)?,
    )
    .map_err(super::report_operation_error)
}

fn prepare_root(
    study: &PreparedStructuralStudyIntent,
    revision: ProgramRevision,
    authority: &str,
    geometry_name: &str,
    geometry_bytes: &[u8],
) -> Result<PreparedExactStructuralStudyRoot> {
    let access = artifact_access(authority);
    let import_options = ExactCadImportOptions {
        source_units: study.geometry_units(),
        ..ExactCadImportOptions::default()
    };
    let meshing_request = authored_meshing_request(study)?;
    let contact_definitions = study
        .meshing_contacts()
        .iter()
        .map(|contact| NativeMeshingContactDefinition {
            side_a_face_selectors: contact.side_a_face_selectors.clone(),
            side_b_face_selectors: contact.side_b_face_selectors.clone(),
        })
        .collect();
    let partition = NativeMeshingPartitionPolicy::default();
    let exact_meshing = prepare_exact_meshing_root(ExactMeshingRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        source_name: geometry_name,
        source_bytes: geometry_bytes,
        import_options,
        contact_definitions,
        meshing_request,
        preferred_edges_per_partition: partition.preferred_edges_per_partition,
        preferred_faces_per_partition: partition.preferred_faces_per_partition,
        maximum_chunk_bytes: 4 * 1024 * 1024,
    })?;
    Ok(prepare_exact_structural_study_root(
        ExactStructuralStudyRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access,
            authored_study_digest: study.authored_study_digest(),
            frozen_project_digest: Some(*revision.graph_digest()),
            geometry_source_digest: study.geometry_source_digest(),
            workload_template: study.workload_template().clone(),
            exact_meshing,
        },
    )?)
}

fn prepare_remote_execution(
    prepared: PreparedExactStructuralStudyRoot,
    revision: ProgramRevision,
    bundle_archive: Vec<u8>,
) -> Result<PreparedRemoteExecution> {
    let objects = prepared.transport_objects()?;
    let request = prepared.into_request();
    let descriptor = serde_json::to_vec(&ProgramExecutionDescriptor {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: request.recipe,
        artifact: request.artifact,
        function: request.function,
        requested_outputs: request.requested_outputs,
    })?;
    let inputs = ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        arguments: request.arguments,
        objects: Vec::new(),
    };
    Ok(PreparedRemoteExecution {
        revision,
        bundle_archive,
        descriptor,
        inputs,
        objects,
    })
}

fn result_expectation(
    study: &PreparedStructuralStudyIntent,
    revision: ProgramRevision,
    authority: &str,
) -> Result<ExactStructuralStudyResultExpectation> {
    let expectation = ExactStructuralStudyResultExpectation {
        program_revision: revision.clone(),
        artifact_access: artifact_access(authority),
        authored_study_digest: study.authored_study_digest(),
        frozen_project_digest: Some(*revision.graph_digest()),
        geometry_source_digest: study.geometry_source_digest(),
        workload_template: study.workload_template().clone(),
        expected_meshing_request: authored_meshing_request(study)?,
    };
    expectation.validate()?;
    Ok(expectation)
}

fn authored_meshing_request(
    study: &PreparedStructuralStudyIntent,
) -> Result<runmat_meshing_core::MeshingRequest> {
    study
        .meshing_request()
        .cloned()
        .context("remote exact-CAD studies require an authored meshing request")
}

fn artifact_access(authority: &str) -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: authority.into(),
        encryption_context: Digest::sha256(
            [
                b"runmat-analysis-access/v1\0".as_slice(),
                authority.as_bytes(),
            ]
            .concat(),
        ),
    }
}

fn validate_remote_study_backend(backend: runmat_analysis_fea::ComputeBackend) -> Result<()> {
    match backend {
        runmat_analysis_fea::ComputeBackend::Cpu => Ok(()),
        runmat_analysis_fea::ComputeBackend::Gpu => anyhow::bail!(
            "remote structural studies do not yet advertise a typed accelerator allocation; use the CPU backend or local execution"
        ),
    }
}

struct DownloadedResultCache(BTreeMap<Digest, Vec<u8>>);

impl CacheImport for DownloadedResultCache {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        let Some(bytes) = self.0.get(&digest) else {
            return Ok(None);
        };
        if Digest::sha256(bytes) != digest {
            return Err(runmat_execution_artifact::ArtifactError::Identity(
                "downloaded structural result differs from its digest".into(),
            ));
        }
        Ok(Some(bytes.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::validate_remote_study_backend;

    #[test]
    fn remote_study_rejects_unallocated_accelerator_execution() {
        assert!(validate_remote_study_backend(runmat_analysis_fea::ComputeBackend::Cpu).is_ok());
        assert!(validate_remote_study_backend(runmat_analysis_fea::ComputeBackend::Gpu).is_err());
    }
}
