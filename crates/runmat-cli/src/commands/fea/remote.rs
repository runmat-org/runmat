use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use runmat_analysis_execution::{
    import_structural_linear_result, prepare_structural_linear_program,
    PreparedStructuralLinearProgram, StructuralLinearProgramPreparation,
};
use runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, ExecutionBundleBuilder, ProgramExecutionDescriptor,
    ProgramExecutionInputs, ProgramExecutionObject, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_runtime::analysis::{
    analysis_complete_structural_study_op, analysis_prepare_structural_study_op,
    analysis_study_content_digest, AnalysisStudyRunData, AnalysisStudySpec,
    PreparedStructuralStudy,
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
    let study = analysis_prepare_structural_study_op(&spec, context)
        .map_err(super::report_operation_error)?;
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
    let study_digest = analysis_study_content_digest(&spec);
    let placeholder = prepare_program(
        &study,
        revision.clone(),
        study_digest,
        "bundle-sizing-authority",
    )?;
    let bundle = ExecutionBundleBuilder::native(&resolved.resolved.frozen, revision.clone())?
        .with_compiled_package_closure()
        .with_materialized_program(
            placeholder.program.request.recipe.clone(),
            ExecutableForm::AnalysisWorkload,
            placeholder
                .program
                .request
                .artifact
                .executable_bytes
                .clone(),
        )
        .build()?;
    let mut bundle_archive = Vec::new();
    write_bundle(&bundle, &mut bundle_archive, ArchiveLimits::default())?;
    let bundle_size = bundle_archive.len();
    let revision_for_scope = revision.clone();
    let committed = crate::commands::job::submit::submit_scoped(
        revision.clone(),
        bundle_size,
        |run_id| {
            let prepared = prepare_program(&study, revision_for_scope, study_digest, run_id)?;
            prepare_remote_execution(prepared, revision.clone(), bundle_archive)
        },
        RemoteSubmissionOptions {
            project: target.project,
            cluster: target.cluster,
            queue: target.queue,
            trust_identity: target.trust_identity,
            idempotency_key: target.idempotency_key,
            workers: target.workers,
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
    let study = PreparedStructuralStudy::from_continuation_bytes(&continuation)
        .map_err(anyhow::Error::msg)?;
    let prepared = prepare_program(&study, revision, study.authored_study_digest(), run_id)?;
    let cache = DownloadedResultCache(
        objects
            .into_iter()
            .map(|object| (object.reference.logical_digest, object.bytes))
            .collect(),
    );
    let result = import_structural_linear_result(
        &cache,
        response,
        &prepared,
        ObjectInventoryLimits::default(),
    )?;
    analysis_complete_structural_study_op(study, result.run).map_err(super::report_operation_error)
}

fn prepare_program(
    study: &PreparedStructuralStudy,
    revision: ProgramRevision,
    study_digest: Digest,
    authority: &str,
) -> Result<PreparedStructuralLinearProgram> {
    Ok(prepare_structural_linear_program(
        StructuralLinearProgramPreparation {
            authored_study_digest: study_digest,
            frozen_project_digest: Some(*revision.graph_digest()),
            program_revision: revision,
            artifact_access: ArtifactAccess {
                authorization_scope: authority.into(),
                encryption_context: Digest::sha256(
                    [
                        b"runmat-analysis-access/v1\0".as_slice(),
                        authority.as_bytes(),
                    ]
                    .concat(),
                ),
            },
            workload: study.workload().clone(),
            mesh: study.solver_mesh().clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
        },
    )?)
}

fn prepare_remote_execution(
    prepared: PreparedStructuralLinearProgram,
    revision: ProgramRevision,
    bundle_archive: Vec<u8>,
) -> Result<PreparedRemoteExecution> {
    let request = prepared.program.request;
    let mut references = BTreeMap::new();
    for input in &prepared.inputs {
        for reference in
            input.references(&prepared.host.artifact_access, ValueRefKind::DriverObject)?
        {
            references.insert(reference.logical_digest, reference);
        }
    }
    let mut objects = Vec::new();
    for input in prepared.inputs {
        for object in input.objects {
            let reference = references
                .remove(&object.descriptor.digest)
                .context("structural program omitted an input object reference")?;
            let object = ProgramExecutionObject {
                reference,
                bytes: object.bytes,
            };
            object.validate()?;
            objects.push(object);
        }
    }
    if !references.is_empty() {
        bail!("structural program input inventory contains an unbound reference");
    }
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
