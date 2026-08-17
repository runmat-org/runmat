//! Bounded inert program form consumed only by a meshing-capable execution host.

use runmat_execution::value::{ValuePayload, ValueRef};
use runmat_execution::{OutputContract, ProgramRevision};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest, ProgramTarget,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingRequest, MeshingStageIdentity, MeshingWorkloadRequest,
};

use crate::task::validate_inputs;
use crate::{MeshingArtifactAccess, MeshingExecutionError, MeshingExecutionResult};

pub const MESHING_HOST_WORKLOAD_SCHEMA_VERSION: u16 = 2;
pub const MESHING_HOST_EXECUTION_MODE: &str = "meshing";
pub const MESHING_HOST_TARGET_PROFILE: &str = "portable-meshing-host-v2";
const HOST_ENTRYPOINT: &str = "meshing_workload";
const HOST_PREFIX: &[u8] = b"runmat-meshing-host-workload/v2\0";
const MAX_HOST_BYTES: usize = 64 * 1024 * 1024;
const MAX_COMPONENT_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct MeshingHostWorkload {
    pub schema_version: u16,
    pub workload: MeshingWorkloadRequest,
    pub stage_identity: MeshingStageIdentity,
    pub resolved_request: MeshingRequest,
    pub artifact_access: MeshingArtifactAccess,
}

impl MeshingHostWorkload {
    pub fn new(
        workload: MeshingWorkloadRequest,
        stage_identity: MeshingStageIdentity,
        resolved_request: MeshingRequest,
        artifact_access: MeshingArtifactAccess,
    ) -> MeshingExecutionResult<Self> {
        let host = Self {
            schema_version: MESHING_HOST_WORKLOAD_SCHEMA_VERSION,
            workload,
            stage_identity,
            resolved_request,
            artifact_access,
        };
        host.validate()?;
        Ok(host)
    }

    pub fn validate(&self) -> MeshingExecutionResult<()> {
        if self.schema_version != MESHING_HOST_WORKLOAD_SCHEMA_VERSION {
            return Err(MeshingExecutionError::Invalid(
                "unsupported meshing host workload schema".into(),
            ));
        }
        self.workload.validate()?;
        self.stage_identity.validate()?;
        self.resolved_request.validate()?;
        self.artifact_access.validate()?;
        if self.workload.stage != self.stage_identity.stage
            || self.workload.stage_identity_digest != self.stage_identity.canonical_digest()?
            || self.stage_identity.resolved_request_digest
                != self.resolved_request.canonical_digest()?
            || self.workload.inputs != self.stage_identity.prerequisites
        {
            return Err(MeshingExecutionError::Invalid(
                "meshing host workload identity does not converge".into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> MeshingExecutionResult<Vec<u8>> {
        self.validate()?;
        let workload = self.workload.canonical_encode()?;
        let identity = self.stage_identity.canonical_encode()?;
        let request = self.resolved_request.canonical_encode()?;
        let access = encode_access(&self.artifact_access)?;
        let mut bytes = Vec::with_capacity(
            HOST_PREFIX.len()
                + 2
                + 4 * 4
                + workload.len()
                + identity.len()
                + request.len()
                + access.len(),
        );
        bytes.extend_from_slice(HOST_PREFIX);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        for component in [&workload, &identity, &request, &access] {
            if component.len() > MAX_COMPONENT_BYTES {
                return Err(MeshingExecutionError::Invalid(
                    "meshing host workload component exceeds its bound".into(),
                ));
            }
            bytes.extend_from_slice(&(component.len() as u32).to_be_bytes());
            bytes.extend_from_slice(component);
        }
        if bytes.len() > MAX_HOST_BYTES {
            return Err(MeshingExecutionError::Invalid(
                "meshing host workload exceeds its byte bound".into(),
            ));
        }
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> MeshingExecutionResult<Self> {
        if bytes.len() > MAX_HOST_BYTES {
            return Err(MeshingExecutionError::Invalid(
                "meshing host workload exceeds its byte bound".into(),
            ));
        }
        let mut decoder = HostDecoder::new(bytes)?;
        let schema_version = decoder.u16()?;
        let workload = MeshingWorkloadRequest::canonical_decode(decoder.component()?)?;
        let stage_identity = MeshingStageIdentity::canonical_decode(decoder.component()?)?;
        let resolved_request = MeshingRequest::canonical_decode(decoder.component()?)?;
        let artifact_access = decode_access(decoder.component()?)?;
        decoder.finish()?;
        let host = Self {
            schema_version,
            workload,
            stage_identity,
            resolved_request,
            artifact_access,
        };
        host.validate()?;
        if host.canonical_bytes()? != bytes {
            return Err(MeshingExecutionError::Invalid(
                "meshing host workload is not canonical".into(),
            ));
        }
        Ok(host)
    }

    pub fn program_request(
        &self,
        program_revision: ProgramRevision,
        input_roots: &[ValueRef],
    ) -> MeshingExecutionResult<ProgramExecutionRequest> {
        self.validate()?;
        validate_inputs(&self.workload, input_roots, &self.artifact_access)?;
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision,
            entrypoint: HOST_ENTRYPOINT.into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: MESHING_HOST_EXECUTION_MODE.into(),
            target: ProgramTarget::portable(MESHING_HOST_TARGET_PROFILE),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::MeshingWorkload,
            self.canonical_bytes()?,
        )?;
        let request = ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe,
            artifact,
            function: 0,
            arguments: input_roots
                .iter()
                .cloned()
                .map(|root| ValuePayload::Object(Box::new(root)))
                .collect(),
            requested_outputs: 1,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn from_program_request(request: &ProgramExecutionRequest) -> MeshingExecutionResult<Self> {
        request.validate_for_portable_host()?;
        if request.artifact.form != ExecutableForm::MeshingWorkload
            || request.recipe.execution_mode != MESHING_HOST_EXECUTION_MODE
            || request.recipe.target != ProgramTarget::portable(MESHING_HOST_TARGET_PROFILE)
        {
            return Err(MeshingExecutionError::Invalid(
                "program request is not a meshing host workload".into(),
            ));
        }
        let host = Self::from_canonical_bytes(&request.artifact.executable_bytes)?;
        let roots = request
            .arguments
            .iter()
            .map(|value| match value {
                ValuePayload::Object(root) => Ok((**root).clone()),
                ValuePayload::Inline(_) => Err(MeshingExecutionError::Invalid(
                    "meshing host inputs must be externalized root manifests".into(),
                )),
            })
            .collect::<MeshingExecutionResult<Vec<_>>>()?;
        validate_inputs(&host.workload, &roots, &host.artifact_access)?;
        Ok(host)
    }
}

fn encode_access(access: &MeshingArtifactAccess) -> MeshingExecutionResult<Vec<u8>> {
    access.validate()?;
    let scope = access.authorization_scope.as_bytes();
    let mut bytes = Vec::with_capacity(4 + scope.len() + 32);
    bytes.extend_from_slice(&(scope.len() as u32).to_be_bytes());
    bytes.extend_from_slice(scope);
    bytes.extend_from_slice(access.encryption_context.bytes());
    Ok(bytes)
}

fn decode_access(bytes: &[u8]) -> MeshingExecutionResult<MeshingArtifactAccess> {
    let length = bytes
        .get(..4)
        .ok_or_else(truncated)?
        .try_into()
        .map(u32::from_be_bytes)
        .map_err(|_| truncated())? as usize;
    let end = 4_usize.checked_add(length).ok_or_else(truncated)?;
    let digest_end = end.checked_add(32).ok_or_else(truncated)?;
    let scope =
        std::str::from_utf8(bytes.get(4..end).ok_or_else(truncated)?).map_err(|_| truncated())?;
    let digest: [u8; 32] = bytes
        .get(end..digest_end)
        .ok_or_else(truncated)?
        .try_into()
        .map_err(|_| truncated())?;
    if digest_end != bytes.len() {
        return Err(truncated());
    }
    let access = MeshingArtifactAccess {
        authorization_scope: scope.into(),
        encryption_context: runmat_execution::Digest::from_bytes(digest),
    };
    access.validate()?;
    Ok(access)
}

struct HostDecoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> HostDecoder<'a> {
    fn new(bytes: &'a [u8]) -> MeshingExecutionResult<Self> {
        if !bytes.starts_with(HOST_PREFIX) {
            return Err(MeshingExecutionError::Invalid(
                "meshing host workload domain is invalid".into(),
            ));
        }
        Ok(Self {
            bytes,
            position: HOST_PREFIX.len(),
        })
    }

    fn u16(&mut self) -> MeshingExecutionResult<u16> {
        let bytes: [u8; 2] = self.take(2)?.try_into().map_err(|_| truncated())?;
        Ok(u16::from_be_bytes(bytes))
    }

    fn component(&mut self) -> MeshingExecutionResult<&'a [u8]> {
        let bytes: [u8; 4] = self.take(4)?.try_into().map_err(|_| truncated())?;
        let length = u32::from_be_bytes(bytes) as usize;
        if length > MAX_COMPONENT_BYTES {
            return Err(MeshingExecutionError::Invalid(
                "meshing host workload component exceeds its bound".into(),
            ));
        }
        self.take(length)
    }

    fn take(&mut self, length: usize) -> MeshingExecutionResult<&'a [u8]> {
        let end = self.position.checked_add(length).ok_or_else(truncated)?;
        let value = self.bytes.get(self.position..end).ok_or_else(truncated)?;
        self.position = end;
        Ok(value)
    }

    fn finish(self) -> MeshingExecutionResult<()> {
        if self.position != self.bytes.len() {
            return Err(MeshingExecutionError::Invalid(
                "meshing host workload contains trailing bytes".into(),
            ));
        }
        Ok(())
    }
}

fn truncated() -> MeshingExecutionError {
    MeshingExecutionError::Invalid("meshing host workload is truncated or malformed".into())
}
