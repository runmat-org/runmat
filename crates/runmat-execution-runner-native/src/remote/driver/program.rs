//! Admit one encrypted remote program object before choosing a worker route.
//! Direct V7 keeps its separate bundle/input objects; Graph V10 carries its
//! complete selected-unit authority in the program object itself.

use runmat_execution::Digest;
use runmat_execution_artifact::archive::{read_bundle, ArchiveLimits};
use runmat_execution_artifact::encryption::{
    decode_encrypted_run_object, open_program_chunk_manifest, ProgramManifestAuthority,
    MAX_PROGRAM_CHUNK_MANIFEST_BYTES,
};
use runmat_execution_artifact::encryption::{EncryptionPurpose, RunKeyMaterial};
use runmat_execution_artifact::{
    admit_remote_program_object_bytes, GraphExecutionRequest, ProgramExecutionInputs,
    ProgramExecutionObjectLocation, ProgramExecutionRequest, RemoteProgramObject,
};
use runmat_execution_transport_native::control::{
    DriverArtifactDownload, DriverControlPlane, SignedArtifactDownloader,
};
use std::sync::Arc;

use super::super::program_staging::{download_and_stage, StagedGraphProgram};
use super::{open_download, protocol, required_artifact};
use crate::{materialized_project::MaterializedProject, NativeExecutionResult};

pub(in crate::remote) enum AdmittedRemoteProgram {
    Direct(Box<AdmittedDirectProgram>),
    Graph(Box<AdmittedGraphProgram>),
}

pub(in crate::remote) struct AdmittedDirectProgram {
    pub request: ProgramExecutionRequest,
    pub bundle_archive: Vec<u8>,
    pub bundle: Arc<runmat_execution_artifact::ExecutionBundle>,
    pub materialized: Option<MaterializedProject>,
    /// Independent encrypted input objects named by the admitted inputs.
    pub input_locations: Vec<ProgramExecutionObjectLocation>,
}

pub(in crate::remote) struct AdmittedGraphProgram {
    pub request: GraphExecutionRequest,
    pub encoded_bytes: usize,
    /// Digest of the exact authenticated Program object bytes.
    pub program_digest: Digest,
    pub _staged: Option<StagedGraphProgram>,
    pub transfer: Option<super::super::pool::SignedGraphTransfer>,
}

impl AdmittedRemoteProgram {
    /// Exact program identity bound into replacement-driver checkpoints.
    pub(in crate::remote) fn checkpoint_identity(&self) -> NativeExecutionResult<Digest> {
        match self {
            Self::Direct(direct) => direct.request.canonical_digest().map_err(protocol),
            Self::Graph(graph) => Ok(graph.program_digest),
        }
    }

    /// Closed Direct host forms replay deterministically; Graph and arbitrary
    /// language programs may have external effects once started.
    pub(in crate::remote) fn execution_start_phase(
        &self,
    ) -> super::super::checkpoint::RemoteDriverCheckpointPhase {
        match self {
            Self::Direct(direct) => {
                super::super::checkpoint::execution_start_phase(&direct.request)
            }
            Self::Graph(_) => {
                super::super::checkpoint::RemoteDriverCheckpointPhase::EffectsMayHaveStarted
            }
        }
    }

    pub(in crate::remote) fn input_locations(&self) -> &[ProgramExecutionObjectLocation] {
        match self {
            Self::Direct(direct) => &direct.input_locations,
            Self::Graph(_) => &[],
        }
    }
}

pub(super) async fn admit(
    control: &dyn DriverControlPlane,
    run_key: &RunKeyMaterial,
    artifacts: &[DriverArtifactDownload],
    run_id: &str,
    project_revision: &str,
    desired_worker_count: u32,
    maximum_program_bytes: u64,
) -> NativeExecutionResult<AdmittedRemoteProgram> {
    if let Some(manifest) = artifacts
        .iter()
        .find(|artifact| artifact.kind == "program-manifest")
    {
        if artifacts
            .iter()
            .filter(|artifact| artifact.kind == "program-manifest")
            .count()
            != 1
        {
            return Err(protocol("Graph Program has multiple signed manifests"));
        }
        return admit_chunked_graph(
            control,
            run_key,
            artifacts,
            manifest,
            run_id,
            project_revision,
            maximum_program_bytes,
        )
        .await;
    }
    let program = required_artifact(artifacts, "program")?;
    let program_bytes = open_download(
        control,
        run_key,
        program,
        run_id,
        EncryptionPurpose::Program,
    )
    .await?;
    admit_remote_program_object_bytes(&program_bytes).map_err(protocol)?;
    let program: RemoteProgramObject = serde_json::from_slice(&program_bytes).map_err(protocol)?;
    match program {
        RemoteProgramObject::Direct(descriptor) => {
            let bundle = required_artifact(artifacts, "bundle")?;
            let input = required_artifact(artifacts, "input")?;
            let bundle_archive =
                open_download(control, run_key, bundle, run_id, EncryptionPurpose::Bundle).await?;
            let bundle = read_bundle(bundle_archive.as_slice(), ArchiveLimits::default())
                .map_err(protocol)?;
            let inputs: ProgramExecutionInputs = serde_json::from_slice(
                &open_download(control, run_key, input, run_id, EncryptionPurpose::Input).await?,
            )
            .map_err(protocol)?;
            inputs.validate().map_err(protocol)?;
            if inputs
                .objects
                .iter()
                .any(|object| object.reference.authorization_scope != run_id)
            {
                return Err(protocol(
                    "program input object is outside the admitted run authority",
                ));
            }
            bundle.validate().map_err(protocol)?;
            runmat_core::cluster_settings::decode_project_policy(&bundle).map_err(protocol)?
                .ok_or_else(|| protocol("cluster bundle has no project-only runtime policy; rebuild for this target"))?;
            if bundle.manifest.portable_environment != descriptor.environment.entries()
                || bundle.manifest.program_revision != descriptor.recipe.program_revision
                || !bundle
                    .manifest
                    .recipes
                    .iter()
                    .any(|value| value == &descriptor.recipe)
                || !bundle
                    .manifest
                    .artifacts
                    .iter()
                    .any(|value| value == &descriptor.artifact)
                || super::super::crypto::project_revision_identity(
                    &bundle.manifest.program_revision,
                ) != project_revision
            {
                return Err(protocol(
                    "bundle, program, and admitted project revision do not converge",
                ));
            }
            let input_locations = inputs.objects.clone();
            let request =
                ProgramExecutionRequest::from_parts(*descriptor, inputs).map_err(protocol)?;
            let materialized = (desired_worker_count == 0 && bundle.requires_materialization())
                .then(|| MaterializedProject::from_bundle(&bundle))
                .transpose()?;
            Ok(AdmittedRemoteProgram::Direct(Box::new(
                AdmittedDirectProgram {
                    request,
                    bundle_archive,
                    bundle: Arc::new(bundle),
                    materialized,
                    input_locations,
                },
            )))
        }
        RemoteProgramObject::Graph(request) => {
            runmat_core::cluster_settings::validate_graph_policy(&request).map_err(protocol)?;
            if request.assignment.is_some() {
                return Err(protocol(
                    "graph Program object must be unassigned before remote scheduling",
                ));
            }
            if artifacts.iter().any(|artifact| {
                matches!(artifact.kind.as_str(), "bundle" | "input" | "program-chunk")
            }) {
                return Err(protocol(
                    "graph program cannot borrow direct bundle or input objects",
                ));
            }
            if super::super::crypto::project_revision_identity(&request.recipe.program_revision)
                != project_revision
            {
                return Err(protocol(
                    "graph primary revision differs from remote project admission",
                ));
            }
            Ok(AdmittedRemoteProgram::Graph(Box::new(
                AdmittedGraphProgram {
                    request: *request,
                    encoded_bytes: program_bytes.len(),
                    program_digest: Digest::sha256(&program_bytes),
                    _staged: None,
                    transfer: None,
                },
            )))
        }
    }
}

async fn admit_chunked_graph(
    control: &dyn DriverControlPlane,
    run_key: &RunKeyMaterial,
    artifacts: &[DriverArtifactDownload],
    manifest_download: &DriverArtifactDownload,
    run_id: &str,
    project_revision: &str,
    maximum_program_bytes: u64,
) -> NativeExecutionResult<AdmittedRemoteProgram> {
    if artifacts
        .iter()
        .any(|artifact| matches!(artifact.kind.as_str(), "program" | "bundle" | "input"))
    {
        return Err(protocol(
            "chunked Graph Program cannot borrow Direct objects",
        ));
    }
    if manifest_download.ciphertext_size_bytes
        > runmat_execution_artifact::encryption::MAX_PROGRAM_CHUNK_MANIFEST_ENCODED_OBJECT_BYTES
            as u64
    {
        return Err(protocol("encrypted Graph Program manifest is oversized"));
    }
    let encoded = control
        .download(manifest_download)
        .await
        .map_err(protocol)?;
    if encoded.len() as u64 != manifest_download.ciphertext_size_bytes
        || super::super::crypto::ciphertext_digest(&encoded) != manifest_download.ciphertext_digest
    {
        return Err(protocol("signed Graph Program manifest download changed"));
    }
    let encrypted = decode_encrypted_run_object(&encoded, MAX_PROGRAM_CHUNK_MANIFEST_BYTES + 16)
        .map_err(protocol)?;
    let authority = ProgramManifestAuthority {
        run_identity: run_id.into(),
        key_epoch: 1,
    };
    let opened =
        open_program_chunk_manifest(run_key, &authority, &encrypted, maximum_program_bytes)
            .map_err(protocol)?;
    let downloader = SignedArtifactDownloader::new();
    let ordered_downloads = opened
        .manifest()
        .chunks
        .iter()
        .map(|chunk| {
            artifacts
                .iter()
                .find(|artifact| {
                    artifact.kind == "program-chunk" && artifact.artifact_id == chunk.artifact_id
                })
                .cloned()
                .ok_or_else(|| protocol("Graph Program chunk has no signed download"))
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let staged = download_and_stage(
        &downloader,
        run_key,
        opened,
        artifacts,
        maximum_program_bytes,
    )
    .await?;
    let encoded_bytes = usize::try_from(staged.length())
        .map_err(|_| protocol("Graph Program length exceeds process ABI"))?;
    let (request, staged) = tokio::task::spawn_blocking({
        move || -> NativeExecutionResult<(GraphExecutionRequest, StagedGraphProgram)> {
            let file = std::fs::File::open(staged.private_path()).map_err(protocol)?;
            let program: RemoteProgramObject = serde_json::from_reader(file).map_err(protocol)?;
            match program {
                RemoteProgramObject::Graph(request) => {
                    request.validate().map_err(protocol)?;
                    runmat_core::cluster_settings::validate_graph_policy(&request)
                        .map_err(protocol)?;
                    if request.assignment.is_some() {
                        return Err(protocol(
                            "staged Graph Program must be unassigned before scheduling",
                        ));
                    }
                    Ok((*request, staged))
                }
                RemoteProgramObject::Direct(_) => Err(protocol(
                    "chunked Graph Program contains a Direct descriptor",
                )),
            }
        }
    })
    .await
    .map_err(protocol)??;
    if super::super::crypto::project_revision_identity(&request.recipe.program_revision)
        != project_revision
    {
        return Err(protocol(
            "graph primary revision differs from remote project admission",
        ));
    }
    let program_digest = staged.digest();
    let program_length = staged.length();
    Ok(AdmittedRemoteProgram::Graph(Box::new(
        AdmittedGraphProgram {
            request,
            encoded_bytes,
            program_digest,
            _staged: Some(staged),
            transfer: Some(super::super::pool::SignedGraphTransfer {
                encrypted_manifest: Arc::from(encoded),
                ordered_downloads: Arc::from(ordered_downloads),
                maximum_program_bytes,
                program_digest,
                program_length,
            }),
        },
    )))
}
