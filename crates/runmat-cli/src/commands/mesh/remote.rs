use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, ExecutionBundleBuilder, ProgramExecutionDescriptor, ProgramExecutionInputs,
    ProgramExecutionObject, ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_execution_runner_native::{
    prepare_exact_meshing_root, resolve_native_meshing_request, ExactMeshingRootPreparation,
    NativeMeshingPartitionPolicy,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::MeshingRequestSettings;
use runmat_meshing_execution::{import_exact_meshing_result, ExactMeshingDagRunResult};

use crate::cli::RemoteExecutionTarget;
use crate::commands::job::secret::{OutputMaterialization, SavedOutputIntent};
use crate::commands::job::submit::{PreparedRemoteExecution, RemoteSubmissionOptions};
use crate::presentation;

use super::{
    import_options, meshing_settings, persist_completed_mesh, print_result, validate_output_paths,
    ExistingOutputPolicy, MeshCommand, MeshOutputSummary,
};

pub(super) async fn execute(
    command: MeshCommand,
    target: RemoteExecutionTarget,
    cli: &crate::cli::Cli,
    _config: &runmat_config::runtime::RunMatRuntimeConfig,
) -> Result<()> {
    let worker_count = target.workers.max(1);
    let output = absolute_output_path(
        &command
            .output
            .clone()
            .unwrap_or_else(|| command.source.with_extension("solver-mesh.cbor")),
    )?;
    let evidence_path = absolute_output_path(
        &command
            .evidence
            .clone()
            .unwrap_or_else(|| command.source.with_extension("meshing-evidence.cbor")),
    )?;
    validate_output_paths(&output, &evidence_path, command.force)?;

    let source = std::fs::canonicalize(&command.source)
        .with_context(|| format!("resolve source CAD {}", command.source.display()))?;
    let source_bytes =
        std::fs::read(&source).with_context(|| format!("read source CAD {}", source.display()))?;
    let source_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .context("source CAD file name is not valid UTF-8")?
        .to_owned();
    let resolved = crate::commands::package::resolve_for_source(&source, cli)
        .await?
        .context("remote meshing requires a runmat.toml project containing the source CAD")?;
    let project_revision = resolved.resolved.frozen.revision();
    let project_policy = runmat_config::runtime::ClusterRuntimePolicy::from_project_file(
        &resolved.resolved.frozen.manifest_path,
    )?;
    let config = project_policy.runtime_config()?;
    let runtime_policy = runmat_core::cluster_settings::capture_project_policy(&project_policy)
        .map_err(anyhow::Error::msg)?;
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_revision.graph_digest.bytes()),
        Digest::from_bytes(*project_revision.source_revision.bytes()),
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let settings = meshing_settings(&command);
    let import_options = import_options(&command);
    let placeholder = prepare_root(
        revision.clone(),
        "bundle-sizing-authority",
        &source_name,
        &source_bytes,
        import_options.clone(),
        settings.clone(),
    )?;
    let mut builder = ExecutionBundleBuilder::native(&resolved.resolved.frozen, revision.clone())?
        .with_compiled_package_closure()
        .with_project_runtime_policy_object(runtime_policy)?;
    for program in placeholder.required_programs()? {
        builder = builder.with_materialized_program(
            program.recipe,
            program.artifact.form,
            program.artifact.executable_bytes,
        );
    }
    let mut bundle_archive = Vec::new();
    write_bundle(
        &builder.build()?,
        &mut bundle_archive,
        ArchiveLimits::default(),
    )?;
    let bundle_size = bundle_archive.len();
    let revision_for_scope = revision.clone();
    let committed = crate::commands::job::submit::submit_scoped(
        revision,
        bundle_size,
        move |run_id| {
            prepare_remote_root(
                revision_for_scope,
                run_id,
                &source_name,
                &source_bytes,
                import_options,
                settings,
                bundle_archive,
            )
        },
        RemoteSubmissionOptions {
            console: None,
            project: target.project,
            cluster: target.cluster,
            queue: target.queue,
            trust_identity: target.trust_identity,
            idempotency_key: target.idempotency_key,
            workers: worker_count,
            accelerator: None,
            on_run_created: None,
            output_intent: SavedOutputIntent::Mesh {
                solver_mesh_path: output.clone(),
                evidence_path: evidence_path.clone(),
                force: command.force,
                materialization: OutputMaterialization::Pending,
            },
        },
    )
    .await?;
    if target.detach {
        print_submission(&committed, &output, &evidence_path, command.json)?;
        return Ok(());
    }

    let (outputs, objects) =
        crate::commands::job::attach::await_downloaded_outputs(&committed.id).await?;
    let result = materialize_outputs(
        outputs.response,
        objects,
        &output,
        &evidence_path,
        command.force,
    )?;
    crate::commands::job::secret::mark_output_materialized(&committed.id)?;
    print_result(&command, &output, &evidence_path, &result)
}

fn absolute_output_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir()
            .context("resolve current directory for deferred mesh output")?
            .join(path))
    }
}

pub(crate) fn materialize_outputs(
    response: ProgramExecutionResponse,
    objects: Vec<ProgramExecutionObject>,
    output: &Path,
    evidence_path: &Path,
    force: bool,
) -> Result<MeshOutputSummary> {
    let (artifact, evidence) =
        decode_result(response, objects).map(|result| (result.artifact, result.evidence))?;
    persist_completed_mesh(
        output,
        evidence_path,
        artifact,
        evidence,
        ExistingOutputPolicy::ReuseIdentical {
            replace_different: force,
        },
    )
}

fn prepare_root<'a>(
    revision: ProgramRevision,
    authority: &str,
    source_name: &'a str,
    source_bytes: &'a [u8],
    import_options: ExactCadImportOptions,
    settings: MeshingRequestSettings,
) -> Result<runmat_execution_runner_native::PreparedExactMeshingRoot> {
    let meshing_request = resolve_native_meshing_request(&import_options, settings)?;
    Ok(prepare_exact_meshing_root(ExactMeshingRootPreparation {
        program_revision: revision,
        artifact_access: ArtifactAccess {
            authorization_scope: authority.into(),
            encryption_context: Digest::sha256(
                [b"runmat-mesh-access/v1\0".as_slice(), authority.as_bytes()].concat(),
            ),
        },
        source_name,
        source_bytes,
        import_options,
        contact_definitions: Vec::new(),
        meshing_request,
        preferred_edges_per_partition: NativeMeshingPartitionPolicy::default()
            .preferred_edges_per_partition,
        preferred_faces_per_partition: NativeMeshingPartitionPolicy::default()
            .preferred_faces_per_partition,
        maximum_chunk_bytes: usize::try_from(
            runmat_execution_artifact::MAX_PROGRAM_EXECUTION_OBJECT_BYTES,
        )?,
    })?)
}

fn prepare_remote_root(
    revision: ProgramRevision,
    run_id: &str,
    source_name: &str,
    source_bytes: &[u8],
    import_options: ExactCadImportOptions,
    settings: MeshingRequestSettings,
    bundle_archive: Vec<u8>,
) -> Result<PreparedRemoteExecution> {
    let prepared = prepare_root(
        revision.clone(),
        run_id,
        source_name,
        source_bytes,
        import_options,
        settings,
    )?;
    let request = prepared.request().clone();
    let references = request
        .arguments
        .iter()
        .filter_map(|argument| match argument {
            ValuePayload::Object(reference) => {
                Some((reference.logical_digest, reference.as_ref().clone()))
            }
            ValuePayload::Inline(_) | ValuePayload::Distributed(_) | ValuePayload::Composite(_) => {
                None
            }
        })
        .collect::<BTreeMap<_, _>>();
    let objects = prepared
        .objects()
        .iter()
        .map(|object| {
            let reference = references
                .get(&object.descriptor.digest)
                .context("exact-meshing root omitted an input object reference")?
                .clone();
            let object = ProgramExecutionObject {
                reference,
                bytes: object.bytes.clone(),
            };
            object.validate()?;
            Ok(object)
        })
        .collect::<Result<Vec<_>>>()?;
    if objects.len() != references.len() {
        bail!("exact-meshing root input inventory contains an unbound reference");
    }
    let descriptor = serde_json::to_vec(&ProgramExecutionDescriptor {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: request.recipe,
        artifact: request.artifact,
        callable: request.callable.clone(),
        requested_outputs: request.requested_outputs,
    })?;
    let inputs = ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        context: request.context,
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

fn decode_result(
    response: ProgramExecutionResponse,
    objects: Vec<ProgramExecutionObject>,
) -> Result<ExactMeshingDagRunResult> {
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        bail!("remote meshing did not return an externalized terminal result");
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        bail!("remote meshing returned an invalid terminal output");
    };
    let access = ArtifactAccess {
        authorization_scope: root.authorization_scope.clone(),
        encryption_context: root.encryption_context,
    };
    let cache = DownloadedResultCache(
        objects
            .into_iter()
            .map(|object| (object.reference.logical_digest, object.bytes))
            .collect(),
    );
    import_exact_meshing_result(
        &cache,
        root.as_ref().clone(),
        result_objects,
        access,
        ObjectInventoryLimits::default(),
    )
    .map_err(anyhow::Error::new)
}

struct DownloadedResultCache(BTreeMap<Digest, Vec<u8>>);

impl CacheImport for DownloadedResultCache {
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> runmat_execution_artifact::ArtifactResult<()>,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        runmat_execution_artifact::cache::copy_verified_bounded_with_admission(
            digest,
            self.0.get(&digest).map(Vec::as_slice),
            maximum_bytes,
            admit,
        )
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded_with_admission(digest, maximum_bytes, &mut |_| Ok(()))
    }
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        let Some(bytes) = self.0.get(&digest) else {
            return Ok(None);
        };
        if Digest::sha256(bytes) != digest {
            return Err(runmat_execution_artifact::ArtifactError::Identity(
                "downloaded result object differs from its digest".into(),
            ));
        }
        Ok(Some(bytes.clone()))
    }
}

fn print_submission(
    run: &runmat_server_client::execution::RunResponse,
    output: &Path,
    evidence_path: &Path,
    json: bool,
) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "run": run,
                "solverMeshArtifactPath": output,
                "meshingEvidenceArtifactPath": evidence_path,
                "materialization": "deferred",
            }))?
        );
    } else {
        println!(
            "{} {} ({})",
            presentation::stdout().success("Mesh job"),
            run.id,
            run.state
        );
        println!("  detached; verified mesh and evidence will materialize on job attach");
        println!("  mesh: {}", output.display());
        println!("  evidence: {}", evidence_path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use runmat_execution::value::{ValuePayload, ValueRefKind};
    use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::ProgramExecutionDescriptor;

    use super::{persist_completed_mesh, prepare_remote_root, ExistingOutputPolicy};

    fn revision() -> ProgramRevision {
        ProgramRevision::new(
            Digest::sha256(b"mesh-cli-graph"),
            Digest::sha256(b"mesh-cli-source"),
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256(b"mesh-cli-runtime"),
                Digest::sha256(b"mesh-cli-catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn remote_root_is_bound_only_after_run_authority_is_known() {
        let first = prepare_remote_root(
            revision(),
            "run-first",
            "part.step",
            b"source-cad-bytes",
            runmat_geometry_io::ExactCadImportOptions::default(),
            runmat_meshing_core::MeshingRequestSettings::default(),
            b"immutable-bundle".to_vec(),
        )
        .unwrap();
        let second = prepare_remote_root(
            revision(),
            "run-second",
            "part.step",
            b"source-cad-bytes",
            runmat_geometry_io::ExactCadImportOptions::default(),
            runmat_meshing_core::MeshingRequestSettings::default(),
            b"immutable-bundle".to_vec(),
        )
        .unwrap();

        assert_eq!(first.bundle_archive, second.bundle_archive);
        assert_eq!(first.descriptor, second.descriptor);
        assert_eq!(first.objects.len(), second.objects.len());
        assert!(first.objects.iter().all(|object| {
            object.reference.kind == ValueRefKind::DriverObject
                && object.reference.authorization_scope == "run-first"
                && first.inputs.arguments.iter().any(|argument| {
                    matches!(argument, ValuePayload::Object(reference) if reference.as_ref() == &object.reference)
                })
        }));
        assert!(first
            .objects
            .iter()
            .zip(&second.objects)
            .all(|(left, right)| {
                left.bytes == right.bytes
                    && left.reference.logical_digest == right.reference.logical_digest
                    && left.reference.authorization_scope != right.reference.authorization_scope
            }));
        let descriptor: ProgramExecutionDescriptor =
            serde_json::from_slice(&first.descriptor).unwrap();
        descriptor.validate().unwrap();
    }

    #[test]
    fn deferred_materialization_is_idempotent_and_never_clobbers_silently() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("mesh.cbor");
        let evidence_path = directory.path().join("evidence.cbor");
        let artifact = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
            runmat_meshing_core::ElementOrder::Tet4,
        );
        let evidence =
            runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&artifact);

        persist_completed_mesh(
            &output,
            &evidence_path,
            artifact.clone(),
            evidence.clone(),
            ExistingOutputPolicy::ReuseIdentical {
                replace_different: false,
            },
        )
        .unwrap();
        let first_mesh = std::fs::read(&output).unwrap();
        let first_evidence = std::fs::read(&evidence_path).unwrap();

        persist_completed_mesh(
            &output,
            &evidence_path,
            artifact.clone(),
            evidence.clone(),
            ExistingOutputPolicy::ReuseIdentical {
                replace_different: false,
            },
        )
        .unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), first_mesh);
        assert_eq!(std::fs::read(&evidence_path).unwrap(), first_evidence);

        std::fs::write(&output, b"different result").unwrap();
        assert!(persist_completed_mesh(
            &output,
            &evidence_path,
            artifact.clone(),
            evidence.clone(),
            ExistingOutputPolicy::ReuseIdentical {
                replace_different: false
            }
        )
        .is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"different result");
        assert_eq!(std::fs::read(&evidence_path).unwrap(), first_evidence);

        persist_completed_mesh(
            &output,
            &evidence_path,
            artifact.clone(),
            evidence.clone(),
            ExistingOutputPolicy::ReuseIdentical {
                replace_different: true,
            },
        )
        .unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), first_mesh);
        assert_eq!(std::fs::read(&evidence_path).unwrap(), first_evidence);
    }
}
