//! Portable, content-addressed root contract for exact CAD admission and meshing.

use std::collections::{BTreeMap, BTreeSet};

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionRequest, ProgramExecutionResponse, ProgramTarget,
    MAX_PROGRAM_EXECUTION_ARGUMENTS, MAX_PROGRAM_EXECUTION_OBJECT_BYTES,
    MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_geometry_core::{GeometryDocument, PersistentEntityId, PersistentEntityKind};
use runmat_geometry_io::{ExactCadImportOptions, GeometryImportContext};
use runmat_meshing_core::{MeshingRequest, PlatformBuildIdentity};
use runmat_meshing_execution::{
    prepare_exact_geometry_result, prepare_meshing_host_program, MeshingHostProgram,
};
use serde::{Deserialize, Serialize};

use crate::exact_geometry_admission::same_requested_tolerance;
use crate::meshing_platform::native_meshing_platform;
use crate::{NativeExecutionError, NativeExecutionResult, NativeMeshingContactDefinition};

pub const EXACT_MESHING_ROOT_EXECUTION_MODE: &str = "exact-meshing-root";
pub const EXACT_MESHING_ROOT_TARGET_PROFILE: &str = "portable-exact-meshing-root";
const ROOT_PROGRAM_BYTES: &[u8] = b"runmat-exact-meshing-root\0";
const ROOT_PREFIX: &[u8] = b"runmat-exact-meshing-root/v1\0";
const ROOT_DOMAIN: &str = "exact-meshing-root";
const SOURCE_PREFIX: &[u8] = b"runmat-source-cad-manifest/v1\0";
const SOURCE_DOMAIN: &str = "source-cad-manifest";
const ADMISSION_PREFIX: &[u8] = b"runmat-exact-geometry-admission/v1\0";
const ADMISSION_DOMAIN: &str = "exact-geometry-admission";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(16 * 1024 * 1024, 1_000_000, 65_536, 64);
const ROOT_MEDIA_TYPE: &str = "application/vnd.runmat.exact-meshing-root+cbor";
const ROOT_VALUE_SCHEMA: &str = "runmat.meshing.exact-root";
const SOURCE_MANIFEST_MEDIA_TYPE: &str = "application/vnd.runmat.source-cad-manifest+cbor";
const SOURCE_MANIFEST_VALUE_SCHEMA: &str = "runmat.geometry.source-cad-manifest";
const SOURCE_CHUNK_MEDIA_TYPE: &str = "application/vnd.runmat.source-cad-chunk";
const SOURCE_CHUNK_VALUE_SCHEMA: &str = "runmat.geometry.source-cad-chunk";
pub(crate) const ADMISSION_MEDIA_TYPE: &str =
    "application/vnd.runmat.exact-geometry-admission+cbor";
pub(crate) const ADMISSION_VALUE_SCHEMA: &str = "runmat.geometry.exact-admission";
const MAX_SOURCE_NAME_BYTES: usize = 4096;
const MAX_CONTACTS: usize = 1_000_000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactMeshingRootRequest {
    pub schema_version: u16,
    pub source_name: String,
    pub source_manifest_digest: Digest,
    pub source_manifest_encoded_length: u64,
    pub import_options: ExactCadImportOptions,
    pub contact_definitions: Vec<NativeMeshingContactDefinition>,
    pub meshing_request: MeshingRequest,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
}

impl ExactMeshingRootRequest {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        if self.schema_version != 1
            || self.source_name.is_empty()
            || self.source_name.len() > MAX_SOURCE_NAME_BYTES
            || self.source_name.chars().any(char::is_control)
            || self
                .source_manifest_digest
                .bytes()
                .iter()
                .all(|byte| *byte == 0)
            || self.source_manifest_encoded_length == 0
            || self.preferred_edges_per_partition == 0
            || self.preferred_faces_per_partition == 0
            || self.contact_definitions.len() > MAX_CONTACTS
        {
            return Err(protocol("exact meshing root request is invalid"));
        }
        self.import_options.validate().map_err(protocol)?;
        self.meshing_request.validate().map_err(protocol)?;
        for contact in &self.contact_definitions {
            if contact.side_a_face_selectors.is_empty()
                || contact.side_b_face_selectors.is_empty()
                || contact
                    .side_a_face_selectors
                    .iter()
                    .chain(&contact.side_b_face_selectors)
                    .any(|selector| {
                        selector.is_empty()
                            || selector.len() > 4096
                            || selector.chars().any(char::is_control)
                    })
            {
                return Err(protocol("exact meshing contact definition is invalid"));
            }
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> NativeExecutionResult<Vec<u8>> {
        self.validate()?;
        encode_contract(ROOT_PREFIX, ROOT_DOMAIN, self, CONTRACT_LIMITS).map_err(protocol)
    }

    fn decode(bytes: &[u8]) -> NativeExecutionResult<Self> {
        let request =
            decode_contract(ROOT_PREFIX, ROOT_DOMAIN, bytes, CONTRACT_LIMITS).map_err(protocol)?;
        Self::validate(&request)?;
        Ok(request)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCadChunk {
    index: u32,
    digest: Digest,
    encoded_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCadManifest {
    schema_version: u16,
    source_digest: Digest,
    encoded_length: u64,
    chunks: Vec<SourceCadChunk>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedExactMeshingRoot {
    request: ProgramExecutionRequest,
    objects: Vec<LogicalObject>,
    meshing_request: MeshingRequest,
}

pub struct ExactMeshingRootPreparation<'a> {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub source_name: &'a str,
    pub source_bytes: &'a [u8],
    pub import_options: ExactCadImportOptions,
    pub contact_definitions: Vec<NativeMeshingContactDefinition>,
    pub meshing_request: MeshingRequest,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
    pub maximum_chunk_bytes: usize,
}

impl PreparedExactMeshingRoot {
    pub const fn request(&self) -> &ProgramExecutionRequest {
        &self.request
    }

    pub fn objects(&self) -> &[LogicalObject] {
        &self.objects
    }

    pub fn meshing_request(&self) -> &MeshingRequest {
        &self.meshing_request
    }

    /// Returns every immutable program that must be bundled before this root can run.
    pub fn required_programs(&self) -> NativeExecutionResult<Vec<MeshingHostProgram>> {
        Ok(vec![
            MeshingHostProgram {
                recipe: self.request.recipe.clone(),
                artifact: self.request.artifact.clone(),
            },
            prepare_meshing_host_program(self.request.recipe.program_revision.clone())
                .map_err(protocol)?,
        ])
    }

    pub fn write_to(&self, store: &mut impl CacheExport) -> NativeExecutionResult<()> {
        for object in &self.objects {
            store.write_verified(object).map_err(protocol)?;
        }
        Ok(())
    }

    pub fn into_request(self) -> ProgramExecutionRequest {
        self.request
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedExactMeshingRoot {
    pub request: ExactMeshingRootRequest,
    pub source_bytes: Vec<u8>,
    pub artifact_access: ArtifactAccess,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExactGeometryAdmission {
    pub schema_version: u16,
    pub document: GeometryDocument,
    pub geometry_root: ValueRef,
    pub source_edge_ids: BTreeMap<u64, PersistentEntityId>,
    pub source_face_ids: BTreeMap<u64, PersistentEntityId>,
    pub source_region_ids: BTreeMap<u64, PersistentEntityId>,
    pub platform: PlatformBuildIdentity,
}

impl ExactGeometryAdmission {
    fn validate(&self) -> NativeExecutionResult<()> {
        self.document.validate().map_err(protocol)?;
        self.platform.validate().map_err(protocol)?;
        if self.schema_version != 1
            || self.geometry_root.kind != ValueRefKind::ResultObject
            || self.document.primary_artifact().digest.bytes()
                != self.geometry_root.logical_digest.bytes()
            || self.document.primary_artifact().encoded_length != self.geometry_root.encoded_length
            || self.document.source.kernel_version != self.platform.exact_kernel_abi
        {
            return Err(protocol("exact geometry admission contract is invalid"));
        }
        validate_source_entity_order(&self.source_edge_ids, PersistentEntityKind::Edge, "edge")?;
        validate_source_entity_order(&self.source_face_ids, PersistentEntityKind::Face, "face")?;
        validate_source_entity_order(
            &self.source_region_ids,
            PersistentEntityKind::Region,
            "region",
        )?;
        Ok(())
    }
}

fn validate_source_entity_order(
    source_ids: &BTreeMap<u64, PersistentEntityId>,
    expected_kind: PersistentEntityKind,
    label: &str,
) -> NativeExecutionResult<()> {
    let mut identities = BTreeSet::new();
    for (expected_index, (source_index, identity)) in source_ids.iter().enumerate() {
        identity.validate().map_err(protocol)?;
        if *source_index != expected_index as u64
            || identity.kind != expected_kind
            || !identities.insert(identity)
        {
            return Err(protocol(format!(
                "exact geometry admission source-{label} map is not a contiguous {label} bijection"
            )));
        }
    }
    Ok(())
}

pub(crate) fn execute_exact_meshing_root_admission<S>(
    program: &ProgramExecutionRequest,
    store: &mut S,
    cancellation: std::sync::Arc<std::sync::atomic::AtomicBool>,
    limits: crate::NativeMeshingHostLimits,
) -> ProgramExecutionResponse
where
    S: CacheImport + CacheExport,
{
    let response = (|| -> NativeExecutionResult<ProgramExecutionResponse> {
        limits.validate()?;
        let decoded = decode_exact_meshing_root(program, store)?;
        let prepared = crate::prepare_exact_geometry_admission(
            &decoded.request.source_name,
            &decoded.source_bytes,
            &decoded.request.import_options,
            &decoded.request.contact_definitions,
            &GeometryImportContext::with_cancellation(cancellation),
            limits.inventory,
        )
        .map_err(protocol)?;
        if !same_requested_tolerance(
            &decoded.request.meshing_request.tolerance,
            &prepared.document().tolerance,
        ) {
            return Err(protocol(
                "meshing request tolerance differs from exact geometry import policy",
            ));
        }
        let document = prepared.document().clone();
        let source_edge_ids = prepared.source_edge_ids().clone();
        let source_face_ids = prepared.source_face_ids().clone();
        let source_region_ids = prepared.source_region_ids().clone();
        let geometry_objects = prepared.into_objects();
        for object in &geometry_objects.objects {
            store.write_verified(object).map_err(protocol)?;
        }
        let geometry = prepare_exact_geometry_result(
            geometry_objects,
            decoded.artifact_access.clone(),
            limits.inventory,
        )
        .map_err(protocol)?;
        let admission = ExactGeometryAdmission {
            schema_version: 1,
            platform: native_meshing_platform(None, document.source.kernel_version.clone()),
            document,
            geometry_root: geometry.root_input().clone(),
            source_edge_ids,
            source_face_ids,
            source_region_ids,
        };
        let admission_object = logical_object(
            ObjectNamespace::ResultValue,
            "geometry/exact-admissions",
            ADMISSION_MEDIA_TYPE,
            encode_admission(&admission)?,
        )?;
        store.write_verified(&admission_object).map_err(protocol)?;
        let admission_reference = decoded
            .artifact_access
            .reference(
                &admission_object,
                ValueRefKind::ResultObject,
                ADMISSION_VALUE_SCHEMA,
            )
            .map_err(protocol)?;
        let mut result_objects = geometry.input_objects().to_vec();
        result_objects.push(admission_reference.clone());
        let response = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(admission_reference))],
            result_objects,
        };
        response.validate_against(program).map_err(protocol)?;
        Ok(response)
    })();
    response.unwrap_or_else(|error| ProgramExecutionResponse::Failure {
        message: error.to_string(),
    })
}

pub fn prepare_exact_meshing_root(
    preparation: ExactMeshingRootPreparation<'_>,
) -> NativeExecutionResult<PreparedExactMeshingRoot> {
    let ExactMeshingRootPreparation {
        program_revision,
        artifact_access,
        source_name,
        source_bytes,
        import_options,
        contact_definitions,
        meshing_request,
        preferred_edges_per_partition,
        preferred_faces_per_partition,
        maximum_chunk_bytes,
    } = preparation;
    artifact_access.validate().map_err(protocol)?;
    if source_bytes.is_empty()
        || maximum_chunk_bytes == 0
        || u64::try_from(maximum_chunk_bytes).map_err(protocol)?
            > MAX_PROGRAM_EXECUTION_OBJECT_BYTES
        || u64::try_from(source_bytes.len()).map_err(protocol)?
            > MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES
    {
        return Err(protocol(
            "source CAD is empty or exceeds its root object byte bounds",
        ));
    }
    if source_bytes.len().div_ceil(maximum_chunk_bytes)
        > MAX_PROGRAM_EXECUTION_ARGUMENTS.saturating_sub(2)
    {
        return Err(protocol("source CAD exceeds the root object count bound"));
    }
    let mut objects = Vec::new();
    let mut chunks = Vec::new();
    for (index, bytes) in source_bytes.chunks(maximum_chunk_bytes).enumerate() {
        let object = logical_object(
            ObjectNamespace::InputValue,
            "geometry/source/chunks",
            SOURCE_CHUNK_MEDIA_TYPE,
            bytes.to_vec(),
        )?;
        chunks.push(SourceCadChunk {
            index: u32::try_from(index).map_err(protocol)?,
            digest: object.descriptor.digest,
            encoded_length: object.descriptor.encoded_length,
        });
        objects.push(object);
    }
    let manifest = SourceCadManifest {
        schema_version: 1,
        source_digest: Digest::sha256(source_bytes),
        encoded_length: u64::try_from(source_bytes.len()).map_err(protocol)?,
        chunks,
    };
    let manifest_bytes = encode_contract(SOURCE_PREFIX, SOURCE_DOMAIN, &manifest, CONTRACT_LIMITS)
        .map_err(protocol)?;
    let manifest_object = logical_object(
        ObjectNamespace::InputValue,
        "geometry/source/manifests",
        SOURCE_MANIFEST_MEDIA_TYPE,
        manifest_bytes,
    )?;
    let root = ExactMeshingRootRequest {
        schema_version: 1,
        source_name: source_name.into(),
        source_manifest_digest: manifest_object.descriptor.digest,
        source_manifest_encoded_length: manifest_object.descriptor.encoded_length,
        import_options,
        contact_definitions,
        meshing_request,
        preferred_edges_per_partition,
        preferred_faces_per_partition,
    };
    let root_object = logical_object(
        ObjectNamespace::InputValue,
        "meshing/exact-roots",
        ROOT_MEDIA_TYPE,
        root.canonical_bytes()?,
    )?;
    let root_reference = artifact_access
        .reference(&root_object, ValueRefKind::DriverObject, ROOT_VALUE_SCHEMA)
        .map_err(protocol)?;
    let manifest_reference = artifact_access
        .reference(
            &manifest_object,
            ValueRefKind::DriverObject,
            SOURCE_MANIFEST_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let mut arguments = vec![
        ValuePayload::Object(Box::new(root_reference)),
        ValuePayload::Object(Box::new(manifest_reference)),
    ];
    for object in &objects {
        arguments.push(ValuePayload::Object(Box::new(
            artifact_access
                .reference(
                    object,
                    ValueRefKind::DriverObject,
                    SOURCE_CHUNK_VALUE_SCHEMA,
                )
                .map_err(protocol)?,
        )));
    }
    objects.push(manifest_object);
    objects.push(root_object);
    validate_inventory(
        &objects,
        ObjectInventoryLimits {
            max_objects: MAX_PROGRAM_EXECUTION_ARGUMENTS,
            max_object_bytes: MAX_PROGRAM_EXECUTION_OBJECT_BYTES,
            max_total_bytes: MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES,
        },
    )
    .map_err(protocol)?;
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision,
        entrypoint: "meshing_workload".into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: EXACT_MESHING_ROOT_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(EXACT_MESHING_ROOT_TARGET_PROFILE),
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
        ROOT_PROGRAM_BYTES.to_vec(),
    )
    .map_err(protocol)?;
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
    request.validate().map_err(protocol)?;
    Ok(PreparedExactMeshingRoot {
        request,
        objects,
        meshing_request: root.meshing_request,
    })
}

pub fn decode_exact_meshing_root(
    program: &ProgramExecutionRequest,
    store: &impl CacheImport,
) -> NativeExecutionResult<DecodedExactMeshingRoot> {
    program.validate_for_portable_host().map_err(protocol)?;
    if program.artifact.form != ExecutableForm::MeshingWorkload
        || program.recipe.execution_mode != EXACT_MESHING_ROOT_EXECUTION_MODE
        || program.recipe.target != ProgramTarget::portable(EXACT_MESHING_ROOT_TARGET_PROFILE)
        || program.artifact.executable_bytes != ROOT_PROGRAM_BYTES
        || !program.recipe.compile_options.is_empty()
    {
        return Err(protocol("program is not an exact meshing root workload"));
    }
    let roots = program
        .arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(reference) => Ok(reference.as_ref()),
            _ => Err(protocol("exact meshing root inputs must be objects")),
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let [root_reference, manifest_reference, chunk_references @ ..] = roots.as_slice() else {
        return Err(protocol("exact meshing root is missing source objects"));
    };
    let artifact_access = ArtifactAccess {
        authorization_scope: root_reference.authorization_scope.clone(),
        encryption_context: root_reference.encryption_context,
    };
    artifact_access
        .validate_reference(
            root_reference,
            ValueRefKind::DriverObject,
            ROOT_MEDIA_TYPE,
            ROOT_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let root_bytes = read(store, root_reference)?;
    let request = ExactMeshingRootRequest::decode(&root_bytes)?;
    artifact_access
        .validate_reference(
            manifest_reference,
            ValueRefKind::DriverObject,
            SOURCE_MANIFEST_MEDIA_TYPE,
            SOURCE_MANIFEST_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    if manifest_reference.logical_digest != request.source_manifest_digest
        || manifest_reference.encoded_length != request.source_manifest_encoded_length
    {
        return Err(protocol(
            "source CAD manifest differs from the root request",
        ));
    }
    let manifest: SourceCadManifest = decode_contract(
        SOURCE_PREFIX,
        SOURCE_DOMAIN,
        &read(store, manifest_reference)?,
        CONTRACT_LIMITS,
    )
    .map_err(protocol)?;
    if manifest.schema_version != 1
        || manifest.encoded_length == 0
        || manifest.encoded_length > MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES
        || manifest.source_digest.bytes().iter().all(|byte| *byte == 0)
        || manifest.chunks.is_empty()
        || manifest.chunks.len() != chunk_references.len()
    {
        return Err(protocol(
            "source CAD manifest has an invalid chunk inventory",
        ));
    }
    let declared_source_bytes = manifest.chunks.iter().try_fold(0_u64, |total, chunk| {
        if chunk.encoded_length == 0 || chunk.encoded_length > MAX_PROGRAM_EXECUTION_OBJECT_BYTES {
            return Err(protocol("source CAD chunk exceeds its byte bound"));
        }
        total
            .checked_add(chunk.encoded_length)
            .ok_or_else(|| protocol("source CAD chunk inventory overflowed"))
    })?;
    if declared_source_bytes != manifest.encoded_length {
        return Err(protocol(
            "source CAD manifest length differs from its chunk inventory",
        ));
    }
    let mut source_bytes =
        Vec::with_capacity(usize::try_from(manifest.encoded_length).map_err(protocol)?);
    for (index, (chunk, reference)) in manifest.chunks.iter().zip(chunk_references).enumerate() {
        if chunk.index != index as u32
            || chunk.digest != reference.logical_digest
            || chunk.encoded_length != reference.encoded_length
        {
            return Err(protocol("source CAD chunk differs from its manifest"));
        }
        artifact_access
            .validate_reference(
                reference,
                ValueRefKind::DriverObject,
                SOURCE_CHUNK_MEDIA_TYPE,
                SOURCE_CHUNK_VALUE_SCHEMA,
            )
            .map_err(protocol)?;
        source_bytes.extend_from_slice(&read(store, reference)?);
    }
    if source_bytes.len() as u64 != manifest.encoded_length
        || Digest::sha256(&source_bytes) != manifest.source_digest
    {
        return Err(protocol("source CAD closure failed identity validation"));
    }
    Ok(DecodedExactMeshingRoot {
        request,
        source_bytes,
        artifact_access,
    })
}

pub(crate) fn encode_admission(
    admission: &ExactGeometryAdmission,
) -> NativeExecutionResult<Vec<u8>> {
    admission.validate()?;
    encode_contract(
        ADMISSION_PREFIX,
        ADMISSION_DOMAIN,
        admission,
        CONTRACT_LIMITS,
    )
    .map_err(protocol)
}

pub(crate) fn decode_admission(bytes: &[u8]) -> NativeExecutionResult<ExactGeometryAdmission> {
    let admission = decode_contract(ADMISSION_PREFIX, ADMISSION_DOMAIN, bytes, CONTRACT_LIMITS)
        .map_err(protocol)?;
    ExactGeometryAdmission::validate(&admission)?;
    Ok(admission)
}

fn logical_object(
    namespace: ObjectNamespace,
    prefix: &str,
    media_type: &str,
    bytes: Vec<u8>,
) -> NativeExecutionResult<LogicalObject> {
    let digest = Digest::sha256(&bytes);
    LogicalObject::new(namespace, format!("{prefix}/{digest}"), media_type, bytes).map_err(protocol)
}

fn read(store: &impl CacheImport, reference: &ValueRef) -> NativeExecutionResult<Vec<u8>> {
    let bytes = store
        .read_verified(reference.logical_digest)
        .map_err(protocol)?
        .ok_or_else(|| {
            protocol(format!(
                "object {} is unavailable",
                reference.logical_digest
            ))
        })?;
    if bytes.len() as u64 != reference.encoded_length
        || Digest::sha256(&bytes) != reference.logical_digest
    {
        return Err(protocol("object bytes differ from their reference"));
    }
    Ok(bytes)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use runmat_execution::ProgramRevision;
    use runmat_execution_artifact::cache::FilesystemObjectStore;
    use runmat_geometry_core::GeometryTolerancePolicy;
    use runmat_meshing_core::{resolve_meshing_request, MeshingRequestSettings};

    use super::*;

    fn face(source: &str, occurrence: &str) -> PersistentEntityId {
        PersistentEntityId::new(
            PersistentEntityKind::Face,
            source.into(),
            vec!["root".into(), occurrence.into()],
        )
    }

    fn revision() -> ProgramRevision {
        ProgramRevision::new(
            Digest::sha256(b"exact-root-graph"),
            Digest::sha256(b"exact-root-source"),
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        )
        .unwrap()
    }

    fn access() -> ArtifactAccess {
        ArtifactAccess {
            authorization_scope: "exact-root-run".into(),
            encryption_context: Digest::sha256(b"exact-root-encryption"),
        }
    }

    fn meshing_request() -> MeshingRequest {
        resolve_meshing_request(
            GeometryTolerancePolicy {
                source_tolerance_m: runmat_types::physical::Length::new(0.0),
                absolute_floor_m: runmat_types::physical::Length::new(1.0e-12),
                model_relative_term: 1.0e-12,
                requested_deviation_m: runmat_types::physical::Length::new(1.0e-4),
                maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
            },
            MeshingRequestSettings::default(),
        )
        .unwrap()
    }

    fn prepared() -> (
        tempfile::TempDir,
        FilesystemObjectStore,
        PreparedExactMeshingRoot,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let mut store =
            FilesystemObjectStore::open(directory.path().join("objects"), 1 << 20).unwrap();
        let prepared = prepare_exact_meshing_root(ExactMeshingRootPreparation {
            program_revision: revision(),
            artifact_access: access(),
            source_name: "part.step",
            source_bytes: b"0123456789abcdef",
            import_options: ExactCadImportOptions::default(),
            contact_definitions: Vec::new(),
            meshing_request: meshing_request(),
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
            maximum_chunk_bytes: 5,
        })
        .unwrap();
        prepared.write_to(&mut store).unwrap();
        (directory, store, prepared)
    }

    #[test]
    fn exact_root_round_trips_a_chunked_source_and_fixed_program() {
        let (_directory, store, prepared) = prepared();
        let decoded = decode_exact_meshing_root(prepared.request(), &store).unwrap();
        assert_eq!(decoded.source_bytes, b"0123456789abcdef");
        assert_eq!(decoded.request.source_name, "part.step");
        assert_eq!(decoded.artifact_access, access());
        assert_eq!(
            prepared.request().artifact.executable_bytes,
            ROOT_PROGRAM_BYTES
        );
        let programs = prepared.required_programs().unwrap();
        assert_eq!(programs.len(), 2);
        assert_ne!(programs[0].artifact.id, programs[1].artifact.id);
        assert!(programs
            .iter()
            .all(|program| program.recipe.program_revision == revision()));
    }

    #[test]
    fn exact_root_rejects_reordered_chunks_and_mixed_authority() {
        let (_directory, store, prepared) = prepared();
        let mut reordered = prepared.request().clone();
        reordered.arguments.swap(2, 3);
        assert!(decode_exact_meshing_root(&reordered, &store).is_err());

        let mut mixed = prepared.request().clone();
        let ValuePayload::Object(reference) = &mut mixed.arguments[2] else {
            panic!("chunk was not externalized")
        };
        reference.authorization_scope = "different-run".into();
        assert!(decode_exact_meshing_root(&mixed, &store).is_err());
    }

    #[test]
    fn exact_root_enforces_transport_chunk_bounds() {
        let error = prepare_exact_meshing_root(ExactMeshingRootPreparation {
            program_revision: revision(),
            artifact_access: access(),
            source_name: "part.step",
            source_bytes: b"source",
            import_options: ExactCadImportOptions::default(),
            contact_definitions: Vec::new(),
            meshing_request: meshing_request(),
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
            maximum_chunk_bytes: usize::try_from(MAX_PROGRAM_EXECUTION_OBJECT_BYTES).unwrap() + 1,
        })
        .unwrap_err();
        assert!(error.to_string().contains("byte bounds"));
    }

    #[test]
    fn source_entity_order_requires_contiguous_unique_typed_occurrences() {
        let valid = BTreeMap::from([(0, face("shared", "a")), (1, face("shared", "b"))]);
        validate_source_entity_order(&valid, PersistentEntityKind::Face, "face").unwrap();

        let sparse = BTreeMap::from([(0, face("a", "a")), (2, face("b", "b"))]);
        assert!(validate_source_entity_order(&sparse, PersistentEntityKind::Face, "face").is_err());

        let duplicate = BTreeMap::from([(0, face("same", "a")), (1, face("same", "a"))]);
        assert!(
            validate_source_entity_order(&duplicate, PersistentEntityKind::Face, "face").is_err()
        );

        let mut non_face = face("solid", "a");
        non_face.kind = PersistentEntityKind::Solid;
        assert!(validate_source_entity_order(
            &BTreeMap::from([(0, non_face)]),
            PersistentEntityKind::Face,
            "face"
        )
        .is_err());

        let region = PersistentEntityId::new(
            PersistentEntityKind::Region,
            "solid".into(),
            vec!["root".into()],
        );
        validate_source_entity_order(
            &BTreeMap::from([(0, region)]),
            PersistentEntityKind::Region,
            "region",
        )
        .unwrap();
    }

    #[cfg(feature = "occt-native")]
    #[test]
    fn worker_admits_raw_cad_as_a_reconstructible_exact_result() {
        use runmat_execution_artifact::object::ObjectInventoryLimits;
        use runmat_meshing_core::MeshingChunkPolicy;
        use runmat_meshing_execution::import_exact_geometry_input;

        const BOX: &[u8] = include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep");
        let import_options = ExactCadImportOptions {
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            source_units_authority: runmat_geometry_io::SourceUnitsAuthority::Authored,
            ..Default::default()
        };

        let directory = tempfile::tempdir().unwrap();
        let mut store =
            FilesystemObjectStore::open(directory.path().join("objects"), 512 << 20).unwrap();
        let prepared = prepare_exact_meshing_root(ExactMeshingRootPreparation {
            program_revision: revision(),
            artifact_access: access(),
            source_name: "box.brep",
            source_bytes: BOX,
            import_options,
            contact_definitions: Vec::new(),
            meshing_request: meshing_request(),
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
            maximum_chunk_bytes: 64 * 1024,
        })
        .unwrap();
        prepared.write_to(&mut store).unwrap();
        let request = prepared.into_request();
        let limits = crate::NativeMeshingHostLimits {
            chunk_policy: MeshingChunkPolicy {
                maximum_chunk_bytes: 1 << 20,
                maximum_records_per_chunk: 4096,
                maximum_total_encoded_bytes: 512 << 20,
            },
            inventory: ObjectInventoryLimits::default(),
            max_message_bytes: 64 << 20,
        };
        let response = execute_exact_meshing_root_admission(
            &request,
            &mut store,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            limits,
        );
        response.validate_against(&request).unwrap();
        let ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } = response
        else {
            panic!("raw CAD admission did not externalize exact geometry: {response:?}")
        };
        let [ValuePayload::Object(admission_root)] = outputs.as_slice() else {
            panic!("raw CAD admission returned an invalid root")
        };
        let admission = decode_admission(&read(&store, admission_root).unwrap()).unwrap();
        assert!(result_objects.contains(&admission.geometry_root));
        assert_eq!(
            admission.platform.exact_kernel_abi,
            admission.document.source.kernel_version
        );
        let source_region_ids = admission.source_region_ids.clone();
        let geometry = import_exact_geometry_input(
            &store,
            admission.document,
            &admission.geometry_root,
            access(),
            limits.inventory,
        )
        .unwrap();
        assert!(!geometry.geometry_objects().topology.regions.is_empty());
        assert_eq!(
            source_region_ids.values().cloned().collect::<BTreeSet<_>>(),
            geometry
                .geometry_objects()
                .topology
                .regions
                .iter()
                .map(|region| region.id.clone())
                .collect::<BTreeSet<_>>()
        );
    }

    #[cfg(feature = "occt-native")]
    #[tokio::test(flavor = "current_thread")]
    async fn raw_cad_reaches_a_verified_terminal_mesh_over_encrypted_quic() {
        use std::collections::BTreeSet;
        use std::net::{Ipv4Addr, SocketAddr};
        use std::sync::Arc;
        use std::time::Duration;

        use rcgen::{generate_simple_self_signed, CertifiedKey};
        use runmat_execution::resource::ResourceInventory;
        use runmat_execution::value::ValuePayload;
        use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
        use runmat_execution_artifact::object::ObjectInventoryLimits;
        use runmat_execution_artifact::ExecutionBundleBuilder;
        use runmat_execution_runner::{PoolSpec, WorkerSpec};
        use runmat_execution_transport_native::control::ResourceRequest;
        use runmat_execution_transport_native::frame::FrameLimits;
        use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
        use runmat_geometry_io::{import_exact_cad, GeometryFormat};
        use runmat_meshing_execution::exact_meshing_worker_capabilities;

        use crate::{
            native_meshing_kernel_dispatcher, run_remote_meshing_worker_quic,
            QuicRemoteWorkerChannel, RemoteMeshingWorkerQuicRequest, RemotePoolDriver,
            RemoteWorkerChannelConfig, RemoteWorkerQuicRequest,
        };

        const BOX: &[u8] = include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep");
        let import_options = ExactCadImportOptions {
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            source_units_authority: runmat_geometry_io::SourceUnitsAuthority::Authored,
            ..Default::default()
        };
        let project_root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(project_root.path().join("src")).unwrap();
        std::fs::write(
            project_root.path().join("runmat.toml"),
            "[package]\nname = \"remote-exact-root\"\n[sources]\nroots = [\"src\"]\n",
        )
        .unwrap();
        let project = runmat_package::build_frozen_project(
            &project_root.path().join("runmat.toml"),
            BTreeSet::new(),
        )
        .unwrap();
        let identity = project.revision();
        let revision = ProgramRevision::new(
            Digest::from_bytes(*identity.graph_digest.bytes()),
            Digest::from_bytes(*identity.source_revision.bytes()),
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        )
        .unwrap();
        let access = ArtifactAccess {
            authorization_scope: "remote-exact-root-run".into(),
            encryption_context: Digest::sha256(b"remote-exact-root-encryption"),
        };
        let mut request = meshing_request();
        request.metric.global_metric =
            runmat_meshing_core::MetricTensor3::isotropic_length_m(10.0).unwrap();
        request.quality.curve.maximum_chordal_deviation_m =
            runmat_types::physical::Length::new(0.1);
        request.quality.curve.maximum_tangent_change_degrees = 180.0;
        request.quality.curve.minimum_metric_edge_length = 0.01;
        request.quality.curve.maximum_metric_edge_length = 10.0;
        request.quality.surface.minimum_metric_angle_degrees = 0.1;
        request.quality.surface.maximum_physical_aspect_ratio = 1_000.0;
        request.quality.surface.maximum_chordal_deviation_m =
            runmat_types::physical::Length::new(0.1);
        request.quality.surface.maximum_normal_deviation_degrees = 180.0;
        request.quality.volume.maximum_metric_edge_length = 2.0;
        request.quality.volume.maximum_radius_edge_ratio = 10.0;
        request.quality.volume.minimum_scaled_jacobian = 0.01;
        request.limits.maximum_nodes = 10_000;
        request.limits.maximum_elements = 10_000;
        let prepared = prepare_exact_meshing_root(ExactMeshingRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access.clone(),
            source_name: "box.brep",
            source_bytes: BOX,
            import_options: import_options.clone(),
            contact_definitions: Vec::new(),
            meshing_request: request.clone(),
            preferred_edges_per_partition: 3,
            preferred_faces_per_partition: 2,
            maximum_chunk_bytes: 1024,
        })
        .unwrap();
        let mut builder = ExecutionBundleBuilder::native(&project, revision.clone())
            .unwrap()
            .with_project_runtime_policy_object(crate::remote::runtime_settings::fixture_policy(
                &revision,
            ))
            .unwrap()
            .with_compiled_package_closure();
        for program in prepared.required_programs().unwrap() {
            builder = builder.with_materialized_program(
                program.recipe,
                ExecutableForm::MeshingWorkload,
                program.artifact.executable_bytes,
            );
        }
        let mut bundle = Vec::new();
        write_bundle(
            &builder.build().unwrap(),
            &mut bundle,
            ArchiveLimits::default(),
        )
        .unwrap();

        let imported = import_exact_cad(
            "box.brep",
            BOX,
            GeometryFormat::Brep,
            &import_options,
            &GeometryImportContext::new(),
        )
        .unwrap();
        let document = imported.geometry_document().unwrap();
        let cohort = crate::meshing_platform::native_meshing_capability_cohort();
        let capabilities =
            exact_meshing_worker_capabilities(&document, &request, Some(&cohort)).unwrap();
        let resources = ResourceRequest {
            cpu_millicores: 1_000,
            memory_bytes: 1 << 30,
            scratch_bytes: 1 << 30,
            accelerators: Vec::new(),
            maximum_wall_millis: 180_000,
        };
        let inventory = ResourceInventory {
            cpu_millicores: 1_000,
            memory_bytes: resources.memory_bytes,
            scratch_bytes: resources.scratch_bytes,
            accelerators: Vec::new(),
            capabilities,
        };
        let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"remote-exact-root-scope"]);
        let pool_id = runmat_execution::PoolId::derive(&[b"remote-exact-root-pool"]);
        let worker = WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: runmat_execution::identity::WorkerId::derive(&[b"remote-exact-root-worker"]),
            pool_id,
            resources: inventory.clone(),
        };
        let CertifiedKey { cert, signing_key } =
            generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
        let certificate_der = cert.der().to_vec();
        let listener = QuicOverlayListener::bind(
            SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            vec![certificate_der.clone()],
            signing_key.serialize_der(),
            FrameLimits::default(),
        )
        .unwrap();
        let authority = listener.local_addr().unwrap();
        let run_key =
            runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([37; 32]).unwrap();
        let server = run_remote_meshing_worker_quic(RemoteMeshingWorkerQuicRequest {
            worker: RemoteWorkerQuicRequest {
                listener,
                run_identity: access.authorization_scope.clone(),
                worker: worker.clone(),
                driver_fence: 47,
                session_id: [29; 16],
                run_key: run_key.clone(),
                limits: FrameLimits::default(),
            },
            kernel: Arc::new(native_meshing_kernel_dispatcher()),
            meshing_limits: crate::NativeMeshingHostLimits::default(),
        });
        let client = async {
            let channel = QuicRemoteWorkerChannel::connect(
                RemoteWorkerChannelConfig {
                    run_identity: access.authorization_scope.clone(),
                    node_identity: "remote-exact-root-node".into(),
                    worker: worker.clone(),
                    driver_fence: 47,
                    session_id: [29; 16],
                    run_key,
                    limits: FrameLimits::default(),
                },
                &PinnedQuicEndpoint {
                    authority,
                    server_name: "runmat.execution".into(),
                    certificate_der,
                },
            )
            .await
            .unwrap();
            let pool = RemotePoolDriver::new_with_value_scope(
                scope_id,
                PoolSpec {
                    id: pool_id,
                    min_workers: 1,
                    max_workers: 1,
                    max_in_flight: 1,
                    resource_limit: inventory,
                },
                47,
                bundle,
                access.authorization_scope.clone(),
            )
            .unwrap();
            pool.add_worker(channel.clone()).await.unwrap();
            for argument in &prepared.request().arguments {
                let ValuePayload::Object(reference) = argument else {
                    panic!("root input was not externalized")
                };
                let object = prepared
                    .objects()
                    .iter()
                    .find(|object| object.descriptor.digest == reference.logical_digest)
                    .unwrap();
                pool.register_execution_object(reference.as_ref().clone(), object.bytes.clone())
                    .unwrap();
            }
            let outcome = tokio::time::timeout(
                Duration::from_secs(180),
                crate::remote::exact_meshing::execute(
                    Arc::clone(&pool),
                    prepared.into_request(),
                    access.authorization_scope.clone(),
                    resources,
                ),
            )
            .await
            .expect("remote exact root timed out")
            .unwrap();
            let Ok(result) = outcome else {
                panic!("remote exact root did not complete")
            };
            let ProgramExecutionResponse::ExternalizedSuccess {
                outputs,
                result_objects,
            } = result.response
            else {
                panic!("remote exact root did not publish a terminal mesh")
            };
            let [ValuePayload::Object(root)] = outputs.as_slice() else {
                panic!("remote exact root returned an invalid terminal root")
            };
            assert!(result_objects.contains(root));
            let terminal = runmat_meshing_execution::import_exact_meshing_result(
                pool.as_ref(),
                root.as_ref().clone(),
                result_objects,
                access,
                ObjectInventoryLimits::default(),
            )
            .unwrap();
            assert!(!terminal.artifact.topology.volume_elements.is_empty());
            pool.remove_worker(worker.id, false).await.unwrap();
        };
        let (server_result, ()) = tokio::join!(server, client);
        server_result.unwrap();
    }
}
