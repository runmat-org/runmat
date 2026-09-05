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
    ProgramExecutionObject, ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner_native::{
    mesh_source_with_native_execution, native_meshing_execution_provider,
    prepare_exact_meshing_root, resolve_native_meshing_request, ExactMeshingRootPreparation,
    NativeMeshingPartitionPolicy, NativeMeshingSourceRequest,
};
use runmat_geometry_core::UnitSystem;
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{
    CanonicalMeshingContract, ElementOrder, MeshingEvidence, MeshingRequestSettings,
    SolverMeshArtifact,
};
use runmat_meshing_execution::{import_exact_meshing_result, ExactMeshingDagRunResult};
use runmat_runtime::meshing::MeshingExecutionProvider;

use crate::cli::{
    ExecutionTargetArgs, MeshElementOrderArg, MeshSourceUnitsArg, RemoteExecutionTarget,
};
use crate::commands::job::submit::{PreparedRemoteExecution, RemoteSubmissionOptions};
use crate::presentation;

pub struct MeshCommand {
    pub source: PathBuf,
    pub source_units: MeshSourceUnitsArg,
    pub output: Option<PathBuf>,
    pub evidence: Option<PathBuf>,
    pub target_size_m: f64,
    pub maximum_deviation_m: f64,
    pub element_order: MeshElementOrderArg,
    pub maximum_elements: u64,
    pub deterministic_seed: u64,
    pub force: bool,
    pub json: bool,
}

pub fn execute(command: MeshCommand) -> Result<()> {
    execute_local(command)
}

pub async fn execute_with_target(
    command: MeshCommand,
    execution: ExecutionTargetArgs,
    cli: &crate::cli::Cli,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
) -> Result<()> {
    let target = execution.resolve()?;
    match target {
        crate::cli::ExecutionTarget::Local => execute_local(command),
        crate::cli::ExecutionTarget::Remote(target) => {
            execute_remote(command, target, cli, config).await
        }
    }
}

fn execute_local(command: MeshCommand) -> Result<()> {
    let output = command
        .output
        .clone()
        .unwrap_or_else(|| command.source.with_extension("solver-mesh.cbor"));
    let evidence = command
        .evidence
        .clone()
        .unwrap_or_else(|| command.source.with_extension("meshing-evidence.cbor"));
    validate_output_paths(&output, &evidence, command.force)?;

    let settings = meshing_settings(&command);
    let generated = mesh_source_with_native_execution(NativeMeshingSourceRequest {
        source: command.source.clone(),
        import_options: import_options(&command),
        contact_definitions: Vec::new(),
        expected_source_sha256: None,
        settings,
        partition_policy: NativeMeshingPartitionPolicy::default(),
    })
    .map_err(anyhow::Error::new)?;
    let result = &generated.dag_result;

    let artifact_bytes = result
        .artifact
        .canonical_encode()
        .context("failed to encode solver mesh")?;
    let evidence_bytes = result
        .evidence
        .canonical_encode()
        .context("failed to encode meshing evidence")?;
    runmat_runtime::meshing::write_canonical_artifact_pair(
        &output,
        &artifact_bytes,
        &evidence,
        &evidence_bytes,
        command.force,
    )
    .map_err(anyhow::Error::msg)?;
    verify_persisted_outputs(&output, &evidence, &result.artifact, &result.evidence)?;

    print_result(&command, &output, &evidence, result)
}

async fn execute_remote(
    command: MeshCommand,
    target: RemoteExecutionTarget,
    cli: &crate::cli::Cli,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
) -> Result<()> {
    let worker_count = target.workers.max(1);
    if target.detach && (command.output.is_some() || command.evidence.is_some() || command.force) {
        bail!("detached meshing does not materialize local output paths; omit --output, --evidence, and --force");
    }
    let output = command
        .output
        .clone()
        .unwrap_or_else(|| command.source.with_extension("solver-mesh.cbor"));
    let evidence_path = command
        .evidence
        .clone()
        .unwrap_or_else(|| command.source.with_extension("meshing-evidence.cbor"));
    if !target.detach {
        validate_output_paths(&output, &evidence_path, command.force)?;
    }
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
        .with_compiled_package_closure();
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
            project: target.project,
            cluster: target.cluster,
            queue: target.queue,
            trust_identity: target.trust_identity,
            idempotency_key: target.idempotency_key,
            workers: worker_count,
            on_run_created: None,
        },
    )
    .await?;
    if target.detach {
        print_remote_submission(&committed, command.json)?;
        return Ok(());
    }
    let (outputs, objects) =
        crate::commands::job::attach::await_downloaded_outputs(&committed.id).await?;
    let result = decode_remote_result(outputs.response, objects)?;
    let artifact_bytes = result.artifact.canonical_encode()?;
    let evidence_bytes = result.evidence.canonical_encode()?;
    runmat_runtime::meshing::write_canonical_artifact_pair(
        &output,
        &artifact_bytes,
        &evidence_path,
        &evidence_bytes,
        command.force,
    )
    .map_err(anyhow::Error::msg)?;
    verify_persisted_outputs(&output, &evidence_path, &result.artifact, &result.evidence)?;
    print_result(&command, &output, &evidence_path, &result)
}

fn meshing_settings(command: &MeshCommand) -> MeshingRequestSettings {
    let mut settings = MeshingRequestSettings {
        element_order: match command.element_order {
            MeshElementOrderArg::Tet4 => ElementOrder::Tet4,
            MeshElementOrderArg::Tet10 => ElementOrder::Tet10,
        },
        deterministic_seed: command.deterministic_seed,
        target_edge_length_m: command.target_size_m,
        maximum_chordal_deviation_m: command.maximum_deviation_m,
        ..MeshingRequestSettings::default()
    };
    settings.resources.maximum_elements = command.maximum_elements;
    settings
}

fn import_options(command: &MeshCommand) -> ExactCadImportOptions {
    ExactCadImportOptions {
        source_units: match command.source_units {
            MeshSourceUnitsArg::Meter => UnitSystem::Meter,
            MeshSourceUnitsArg::Millimeter => UnitSystem::Millimeter,
            MeshSourceUnitsArg::Inch => UnitSystem::Inch,
        },
        ..ExactCadImportOptions::default()
    }
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
            ValuePayload::Inline(_) => None,
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

fn decode_remote_result(
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

fn print_remote_submission(
    run: &runmat_server_client::execution::RunResponse,
    json: bool,
) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(run)?);
    } else {
        println!(
            "{} {} ({})",
            presentation::stdout().success("Mesh job"),
            run.id,
            run.state
        );
        println!("  detached; no local mesh or evidence file was materialized");
    }
    Ok(())
}

fn print_result(
    command: &MeshCommand,
    output: &Path,
    evidence: &Path,
    result: &ExactMeshingDagRunResult,
) -> Result<()> {
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "solver_mesh_artifact_path": output,
                "meshing_evidence_artifact_path": evidence,
                "canonical_digest": hex_digest(result.artifact.canonical_digest.bytes()),
                "node_count": result.artifact.topology.nodes.len(),
                "element_count": result.artifact.topology.volume_elements.len(),
                "boundary_face_count": result.artifact.topology.boundary_faces.len(),
                "resource_usage": result.evidence.resources,
                "stages": result.evidence.stages,
            }))?
        );
    } else {
        let styles = presentation::stdout();
        println!("{}", styles.success("Solver mesh generated"));
        println!(
            "  {} {}",
            styles.label("mesh:"),
            styles.path(output.display())
        );
        println!(
            "  {} {}",
            styles.label("evidence:"),
            styles.path(evidence.display())
        );
        println!(
            "  {} {} nodes, {} tetrahedra, {} boundary faces",
            styles.label("topology:"),
            result.artifact.topology.nodes.len(),
            result.artifact.topology.volume_elements.len(),
            result.artifact.topology.boundary_faces.len()
        );
    }
    Ok(())
}

pub(crate) fn meshing_execution_provider() -> MeshingExecutionProvider {
    native_meshing_execution_provider()
}

fn validate_output_paths(output: &Path, evidence: &Path, force: bool) -> Result<()> {
    if output == evidence {
        anyhow::bail!("solver mesh and meshing evidence require distinct output paths");
    }
    if !force {
        for path in [output, evidence] {
            if path.exists() {
                anyhow::bail!(
                    "output {} already exists; use --force to replace it",
                    path.display()
                );
            }
        }
    }
    Ok(())
}

fn verify_persisted_outputs(
    output: &Path,
    evidence_path: &Path,
    expected_artifact: &SolverMeshArtifact,
    expected_evidence: &MeshingEvidence,
) -> Result<()> {
    let artifact_bytes = std::fs::read(output).with_context(|| {
        format!(
            "failed to verify persisted solver mesh {}",
            output.display()
        )
    })?;
    let evidence_bytes = std::fs::read(evidence_path).with_context(|| {
        format!(
            "failed to verify persisted meshing evidence {}",
            evidence_path.display()
        )
    })?;
    let artifact = SolverMeshArtifact::canonical_decode(&artifact_bytes)
        .context("persisted solver mesh failed canonical decoding")?;
    let evidence = MeshingEvidence::canonical_decode(&evidence_bytes)
        .context("persisted meshing evidence failed canonical decoding")?;
    evidence
        .validate(&artifact)
        .context("persisted meshing outputs failed cross-validation")?;
    if &artifact != expected_artifact || &evidence != expected_evidence {
        anyhow::bail!("persisted meshing outputs differ from the completed canonical result");
    }
    Ok(())
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use runmat_execution::value::{ValuePayload, ValueRefKind};
    use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::ProgramExecutionDescriptor;

    use super::{prepare_remote_root, validate_output_paths};

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
    fn output_paths_must_be_distinct_and_do_not_clobber_by_default() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("mesh.cbor");
        let evidence = directory.path().join("evidence.cbor");
        assert!(validate_output_paths(&output, &output, false).is_err());

        runmat_runtime::meshing::write_canonical_artifact_pair(
            &output,
            b"first mesh",
            &evidence,
            b"first evidence",
            false,
        )
        .unwrap();
        assert!(validate_output_paths(&output, &evidence, false).is_err());
        assert!(runmat_runtime::meshing::write_canonical_artifact_pair(
            &output,
            b"second mesh",
            &evidence,
            b"second evidence",
            false,
        )
        .is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"first mesh");
        assert_eq!(std::fs::read(&evidence).unwrap(), b"first evidence");
    }

    #[test]
    fn force_replaces_an_existing_output_atomically() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("mesh.cbor");
        let evidence = directory.path().join("evidence.cbor");
        runmat_runtime::meshing::write_canonical_artifact_pair(
            &output,
            b"first mesh",
            &evidence,
            b"first evidence",
            false,
        )
        .unwrap();
        runmat_runtime::meshing::write_canonical_artifact_pair(
            &output,
            b"second mesh",
            &evidence,
            b"second evidence",
            true,
        )
        .unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), b"second mesh");
        assert_eq!(std::fs::read(&evidence).unwrap(), b"second evidence");
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
}
