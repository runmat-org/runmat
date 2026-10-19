use anyhow::Result;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{
    ArtifactAccess, ProgramExecutionDescriptor, ProgramExecutionInputs, ProgramExecutionObject,
    ProgramExecutionRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_execution_runner_native::{PreparedStructuralStudyRoot, PreparedStructuralSweepRoot};
use runmat_runtime::analysis::PreparedStructuralStudyIntent;

use crate::commands::job::submit::PreparedRemoteExecution;

pub(crate) fn prepare_typed_remote_execution(
    prepared: PreparedStructuralStudyRoot,
    revision: ProgramRevision,
    bundle_archive: Vec<u8>,
) -> Result<PreparedRemoteExecution> {
    let objects = prepared.transport_objects()?;
    let request = prepared.into_request();
    prepare_remote_execution(request, objects, revision, bundle_archive)
}

pub(crate) fn prepare_typed_remote_sweep_execution(
    prepared: PreparedStructuralSweepRoot,
    revision: ProgramRevision,
    bundle_archive: Vec<u8>,
) -> Result<PreparedRemoteExecution> {
    let objects = prepared.transport_objects();
    let request = prepared.into_request();
    prepare_remote_execution(request, objects, revision, bundle_archive)
}

fn prepare_remote_execution(
    request: ProgramExecutionRequest,
    objects: Vec<ProgramExecutionObject>,
    revision: ProgramRevision,
    bundle_archive: Vec<u8>,
) -> Result<PreparedRemoteExecution> {
    let descriptor = serde_json::to_vec(&ProgramExecutionDescriptor {
        console: None,
        environment: request.environment,
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

pub(super) fn authored_meshing_request(
    study: &PreparedStructuralStudyIntent,
) -> Result<runmat_meshing_core::MeshingRequest> {
    study
        .meshing_request()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("exact-CAD studies require an authored meshing request"))
}

pub(super) fn artifact_access(authority: &str) -> ArtifactAccess {
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
