//! Bounded inert program form consumed only by a meshing-capable execution host.

use runmat_execution::value::{ValuePayload, ValueRef};
use runmat_execution::{OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact, ProgramBuildRecipe,
    ProgramExecutionRequest, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_geometry_core::{
    decode_geometry_document, encode_geometry_document, GeometryDocument, GeometryModel,
};
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingCapabilityRequirement, MeshingExecutionCeilings,
    MeshingInputKind, MeshingRequest, MeshingResourceBudget, MeshingStageIdentity,
    MeshingWorkloadRequest,
};

use crate::task::validate_inputs;
use crate::{ArtifactAccess, MeshingExecutionError, MeshingExecutionResult};

pub const MESHING_HOST_WORKLOAD_SCHEMA_VERSION: u16 = 3;
pub const MESHING_HOST_ABI: &str = "host-v2";
pub const MESHING_HOST_EXECUTION_MODE: &str = "meshing";
pub const MESHING_HOST_TARGET_PROFILE: &str = "portable-meshing-host-v2";
const HOST_ENTRYPOINT: &str = "meshing_workload";
const HOST_PREFIX: &[u8] = b"runmat-meshing-host-workload/v2\0";
const HOST_PROGRAM_BYTES: &[u8] = b"runmat-meshing-stage-host\0";
const MAX_HOST_BYTES: usize = 64 * 1024 * 1024;
const MAX_COMPONENT_BYTES: usize = 32 * 1024 * 1024;
pub const MESHING_HOST_INVOCATION_MEDIA_TYPE: &str =
    "application/vnd.runmat.meshing-host-invocation+cbor";
pub const MESHING_HOST_INVOCATION_VALUE_SCHEMA: &str = "runmat.meshing.host-invocation";

#[derive(Clone, Debug, PartialEq)]
pub struct MeshingHostWorkload {
    pub schema_version: u16,
    pub workload: MeshingWorkloadRequest,
    pub stage_identity: MeshingStageIdentity,
    pub resolved_request: MeshingRequest,
    /// Execution ceilings for this run; outside the request identity.
    pub ceilings: MeshingExecutionCeilings,
    pub artifact_access: ArtifactAccess,
    pub geometry_document: Option<GeometryDocument>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedMeshingHostInvocation {
    request: ProgramExecutionRequest,
    descriptor: LogicalObject,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshingHostProgram {
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
}

pub fn prepare_meshing_host_program(
    program_revision: ProgramRevision,
) -> MeshingExecutionResult<MeshingHostProgram> {
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision,
        entrypoint: HOST_ENTRYPOINT.into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: MESHING_HOST_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(MESHING_HOST_TARGET_PROFILE),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::MeshingWorkload,
        HOST_PROGRAM_BYTES.to_vec(),
    )?;
    Ok(MeshingHostProgram { recipe, artifact })
}

impl PreparedMeshingHostInvocation {
    pub const fn request(&self) -> &ProgramExecutionRequest {
        &self.request
    }

    pub const fn descriptor(&self) -> &LogicalObject {
        &self.descriptor
    }

    pub fn write_to(&self, store: &mut impl CacheExport) -> MeshingExecutionResult<()> {
        store.write_verified(&self.descriptor)?;
        Ok(())
    }

    pub fn into_request(self) -> ProgramExecutionRequest {
        self.request
    }
}

impl MeshingHostWorkload {
    pub fn new(
        workload: MeshingWorkloadRequest,
        stage_identity: MeshingStageIdentity,
        resolved_request: MeshingRequest,
        ceilings: MeshingExecutionCeilings,
        artifact_access: ArtifactAccess,
        geometry_document: Option<GeometryDocument>,
    ) -> MeshingExecutionResult<Self> {
        let host = Self {
            schema_version: MESHING_HOST_WORKLOAD_SCHEMA_VERSION,
            workload,
            stage_identity,
            resolved_request,
            ceilings,
            artifact_access,
            geometry_document,
        };
        host.validate()?;
        Ok(host)
    }

    /// The hard budget this host executes under: the request's size limits and the run's
    /// execution ceilings.
    pub fn resources(&self) -> MeshingResourceBudget {
        MeshingResourceBudget::new(self.resolved_request.limits, self.ceilings)
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
        self.ceilings
            .validate()
            .map_err(|error| MeshingExecutionError::Invalid(error.to_string()))?;
        self.artifact_access.validate()?;
        self.validate_geometry_document()?;
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
        let ceilings = encode_ceilings(&self.ceilings);
        let access = encode_access(&self.artifact_access)?;
        let geometry_document = self
            .geometry_document
            .as_ref()
            .map(encode_geometry_document)
            .transpose()?
            .unwrap_or_default();
        let mut bytes = Vec::with_capacity(
            HOST_PREFIX.len()
                + 2
                + 6 * 4
                + workload.len()
                + identity.len()
                + request.len()
                + ceilings.len()
                + access.len()
                + geometry_document.len(),
        );
        bytes.extend_from_slice(HOST_PREFIX);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        for component in [
            &workload,
            &identity,
            &request,
            &ceilings,
            &access,
            &geometry_document,
        ] {
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
        let ceilings = decode_ceilings(decoder.component()?)?;
        let artifact_access = decode_access(decoder.component()?)?;
        let document_bytes = decoder.component()?;
        let geometry_document = if document_bytes.is_empty() {
            None
        } else {
            Some(decode_geometry_document(document_bytes)?)
        };
        decoder.finish()?;
        let host = Self {
            schema_version,
            workload,
            stage_identity,
            resolved_request,
            ceilings,
            artifact_access,
            geometry_document,
        };
        host.validate()?;
        if host.canonical_bytes()? != bytes {
            return Err(MeshingExecutionError::Invalid(
                "meshing host workload is not canonical".into(),
            ));
        }
        Ok(host)
    }

    fn validate_geometry_document(&self) -> MeshingExecutionResult<()> {
        let geometry_inputs = self
            .workload
            .inputs
            .iter()
            .filter(|input| {
                matches!(
                    input.kind,
                    MeshingInputKind::ExactGeometry | MeshingInputKind::FacetedGeometry
                )
            })
            .collect::<Vec<_>>();
        match (geometry_inputs.as_slice(), &self.geometry_document) {
            ([], None) => Ok(()),
            ([input], Some(document)) => {
                document.validate()?;
                let exact_capabilities = self
                    .workload
                    .required_capabilities
                    .iter()
                    .filter_map(|capability| match capability {
                        MeshingCapabilityRequirement::ExactCadKernel { abi } => Some(abi),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let capability_matches = match (&input.kind, &document.model) {
                    (MeshingInputKind::ExactGeometry, GeometryModel::ExactBRep { model }) => {
                        exact_capabilities == [&model.kernel_abi]
                    }
                    (MeshingInputKind::FacetedGeometry, GeometryModel::FacetedSolid { .. }) => {
                        exact_capabilities.is_empty()
                    }
                    _ => false,
                };
                if !document.display_tessellations.is_empty()
                    || document.primary_artifact().digest.bytes() != input.digest.bytes()
                    || document.source.content_digest.bytes()
                        != self.stage_identity.geometry.source_digest.bytes()
                    || document.revision.revision != self.stage_identity.geometry.geometry_revision
                    || document.revision.persistent_mapping_version
                        != self.stage_identity.geometry.persistent_mapping_version
                    || document.tolerance != self.resolved_request.tolerance
                    || !capability_matches
                {
                    return Err(MeshingExecutionError::Identity(
                        "geometry document does not bind the task identity, kind, and capability",
                    ));
                }
                Ok(())
            }
            _ => Err(MeshingExecutionError::Invalid(
                "host workload requires exactly one document for exactly one geometry input".into(),
            )),
        }
    }

    /// Materializes this run-scoped invocation beside its semantic inputs and returns a request
    /// for the single immutable meshing host program.
    ///
    /// Dynamic stage identity never enters the executable artifact. This is required for remote
    /// bundles, where every authorized executable is frozen before downstream stage roots exist.
    pub fn prepare_program_request(
        &self,
        program_revision: ProgramRevision,
        input_roots: &[ValueRef],
    ) -> MeshingExecutionResult<PreparedMeshingHostInvocation> {
        self.validate()?;
        validate_inputs(&self.workload, input_roots, &self.artifact_access)?;
        let invocation = crate::object_support::logical_object(
            "meshing/host-invocations",
            ObjectNamespace::InputValue,
            MESHING_HOST_INVOCATION_MEDIA_TYPE,
            self.canonical_bytes()?,
        )?;
        let invocation_root = self.artifact_access.reference(
            &invocation,
            runmat_execution::value::ValueRefKind::DriverObject,
            MESHING_HOST_INVOCATION_VALUE_SCHEMA,
        )?;
        let MeshingHostProgram { recipe, artifact } =
            prepare_meshing_host_program(program_revision)?;
        let mut arguments = Vec::with_capacity(input_roots.len() + 1);
        arguments.push(ValuePayload::Object(Box::new(invocation_root)));
        arguments.extend(
            input_roots
                .iter()
                .cloned()
                .map(|root| ValuePayload::Object(Box::new(root))),
        );
        let request = ProgramExecutionRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe,
            artifact,
            callable: runmat_execution::ProgramCallable::semantic(
                runmat_execution::ProgramFunctionId(0),
                None,
            ),
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments,
            requested_outputs: 1,
        };
        request.validate()?;
        Ok(PreparedMeshingHostInvocation {
            request,
            descriptor: invocation,
        })
    }

    pub fn from_program_request<S: CacheImport>(
        request: &ProgramExecutionRequest,
        store: &S,
    ) -> MeshingExecutionResult<Self> {
        request.validate_for_portable_host()?;
        if request.artifact.form != ExecutableForm::MeshingWorkload
            || request.recipe.execution_mode != MESHING_HOST_EXECUTION_MODE
            || request.recipe.target != ProgramTarget::portable(MESHING_HOST_TARGET_PROFILE)
            || request.artifact.executable_bytes != HOST_PROGRAM_BYTES
            || !request.recipe.compile_options.is_empty()
        {
            return Err(MeshingExecutionError::Invalid(
                "program request is not a meshing host workload".into(),
            ));
        }
        let (invocation, inputs) = request.arguments.split_first().ok_or_else(|| {
            MeshingExecutionError::Invalid(
                "meshing host request is missing its invocation descriptor".into(),
            )
        })?;
        let ValuePayload::Object(invocation_root) = invocation else {
            return Err(MeshingExecutionError::Invalid(
                "meshing host invocation must remain externalized".into(),
            ));
        };
        if invocation_root.kind != runmat_execution::value::ValueRefKind::DriverObject
            || invocation_root.media_type != MESHING_HOST_INVOCATION_MEDIA_TYPE
            || invocation_root.value_schema != MESHING_HOST_INVOCATION_VALUE_SCHEMA
            || invocation_root.encoded_length == 0
            || invocation_root.encoded_length > MAX_HOST_BYTES as u64
        {
            return Err(MeshingExecutionError::Invalid(
                "meshing host invocation reference is malformed".into(),
            ));
        }
        let bytes = store.read_verified(invocation_root.logical_digest)?.ok_or(
            MeshingExecutionError::MissingObject(invocation_root.logical_digest),
        )?;
        if bytes.len() as u64 != invocation_root.encoded_length
            || runmat_execution::Digest::sha256(&bytes) != invocation_root.logical_digest
        {
            return Err(MeshingExecutionError::Identity(
                "meshing host invocation bytes differ from their reference",
            ));
        }
        let host = Self::from_canonical_bytes(&bytes)?;
        host.artifact_access.validate_reference(
            invocation_root,
            runmat_execution::value::ValueRefKind::DriverObject,
            MESHING_HOST_INVOCATION_MEDIA_TYPE,
            MESHING_HOST_INVOCATION_VALUE_SCHEMA,
        )?;
        let roots = inputs
            .iter()
            .map(|value| match value {
                ValuePayload::Object(root) => Ok((**root).clone()),
                ValuePayload::Inline(_)
                | ValuePayload::Distributed(_)
                | ValuePayload::Composite(_) => Err(MeshingExecutionError::Invalid(
                    "meshing host inputs must be externalized root manifests".into(),
                )),
            })
            .collect::<MeshingExecutionResult<Vec<_>>>()?;
        validate_inputs(&host.workload, &roots, &host.artifact_access)?;
        Ok(host)
    }

    pub fn invocation_root(request: &ProgramExecutionRequest) -> MeshingExecutionResult<&ValueRef> {
        match request.arguments.first() {
            Some(ValuePayload::Object(root)) => Ok(root),
            _ => Err(MeshingExecutionError::Invalid(
                "meshing host request is missing its externalized invocation".into(),
            )),
        }
    }

    pub fn semantic_input_roots(
        request: &ProgramExecutionRequest,
    ) -> MeshingExecutionResult<Vec<ValueRef>> {
        request
            .arguments
            .get(1..)
            .ok_or_else(|| {
                MeshingExecutionError::Invalid(
                    "meshing host request is missing its invocation descriptor".into(),
                )
            })?
            .iter()
            .map(|value| match value {
                ValuePayload::Object(root) => Ok((**root).clone()),
                _ => Err(MeshingExecutionError::Invalid(
                    "meshing host inputs must be externalized root manifests".into(),
                )),
            })
            .collect()
    }
}

fn encode_access(access: &ArtifactAccess) -> MeshingExecutionResult<Vec<u8>> {
    access.validate()?;
    let scope = access.authorization_scope.as_bytes();
    let mut bytes = Vec::with_capacity(4 + scope.len() + 32);
    bytes.extend_from_slice(&(scope.len() as u32).to_be_bytes());
    bytes.extend_from_slice(scope);
    bytes.extend_from_slice(access.encryption_context.bytes());
    Ok(bytes)
}

fn decode_access(bytes: &[u8]) -> MeshingExecutionResult<ArtifactAccess> {
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
    let access = ArtifactAccess {
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

fn encode_ceilings(ceilings: &MeshingExecutionCeilings) -> Vec<u8> {
    [
        ceilings.maximum_memory_bytes,
        ceilings.maximum_scratch_bytes,
        ceilings.maximum_wall_time_ms,
        ceilings.maximum_artifact_bytes,
        ceilings.maximum_search_work,
        ceilings.maximum_iterations,
    ]
    .into_iter()
    .flat_map(u64::to_be_bytes)
    .collect()
}

fn decode_ceilings(bytes: &[u8]) -> MeshingExecutionResult<MeshingExecutionCeilings> {
    if bytes.len() != 48 {
        return Err(MeshingExecutionError::Invalid(
            "meshing host execution ceilings are malformed".into(),
        ));
    }
    let value = |index: usize| {
        u64::from_be_bytes(
            bytes[index * 8..index * 8 + 8]
                .try_into()
                .expect("eight bytes"),
        )
    };
    Ok(MeshingExecutionCeilings {
        maximum_memory_bytes: value(0),
        maximum_scratch_bytes: value(1),
        maximum_wall_time_ms: value(2),
        maximum_artifact_bytes: value(3),
        maximum_search_work: value(4),
        maximum_iterations: value(5),
    })
}
