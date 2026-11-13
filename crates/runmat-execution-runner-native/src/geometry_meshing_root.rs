//! Driver-owned composition of typed geometry materialization and exact meshing.

use runmat_analysis_execution::{
    decode_geometry_materialization_program, PreparedGeometryMaterializationProgram,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::{OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionRequest, ProgramTarget,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_meshing_core::MeshingRequest;
use runmat_meshing_execution::prepare_meshing_host_program;
use serde::{Deserialize, Serialize};

use crate::{NativeExecutionError, NativeExecutionResult};

pub const GEOMETRY_MESHING_ROOT_EXECUTION_MODE: &str = "geometry-meshing-root";
pub const GEOMETRY_MESHING_ROOT_TARGET_PROFILE: &str = "portable-geometry-meshing-root";
const ROOT_PROGRAM_BYTES: &[u8] = b"runmat-geometry-meshing-root\0";
const ROOT_PREFIX: &[u8] = b"runmat-geometry-meshing-root\0";
const ROOT_DOMAIN: &str = "geometry-meshing-root";
const ROOT_MEDIA_TYPE: &str = "application/vnd.runmat.geometry-meshing-root+cbor";
const ROOT_VALUE_SCHEMA: &str = "runmat.meshing.geometry-root";
const CONTRACT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(65_536, 64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryMeshingRootRequest {
    pub schema_version: u16,
    pub geometry: ProgramExecutionDescriptor,
    pub geometry_argument_count: u32,
    pub meshing_request: MeshingRequest,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
}

impl GeometryMeshingRootRequest {
    fn validate(&self) -> NativeExecutionResult<()> {
        self.geometry
            .validate_for_portable_host()
            .map_err(protocol)?;
        self.meshing_request.validate().map_err(protocol)?;
        if self.schema_version != 1
            || self.geometry.artifact.form != ExecutableForm::GeometryWorkload
            || self.geometry.recipe.execution_mode
                != runmat_analysis_execution::GEOMETRY_MATERIALIZATION_EXECUTION_MODE
            || self.geometry.requested_outputs != 1
            || self.geometry_argument_count == 0
            || self.preferred_edges_per_partition == 0
            || self.preferred_faces_per_partition == 0
        {
            return Err(protocol("geometry meshing root request is invalid"));
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

pub struct GeometryMeshingRootPreparation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub geometry: PreparedGeometryMaterializationProgram,
    /// Immutable source objects owned by this root. Derived/generated references may instead name
    /// already-published objects in the caller's execution store.
    pub geometry_source_objects: Vec<LogicalObject>,
    pub meshing_request: MeshingRequest,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedGeometryMeshingRoot {
    request: ProgramExecutionRequest,
    objects: Vec<LogicalObject>,
    geometry_request: runmat_analysis_execution::GeometryMaterializationRequest,
    meshing_request: MeshingRequest,
    required_programs: Vec<RequiredGeometryMeshingProgram>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequiredGeometryMeshingProgram {
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
}

impl PreparedGeometryMeshingRoot {
    pub const fn request(&self) -> &ProgramExecutionRequest {
        &self.request
    }

    pub fn objects(&self) -> &[LogicalObject] {
        &self.objects
    }

    pub const fn meshing_request(&self) -> &MeshingRequest {
        &self.meshing_request
    }

    pub const fn geometry_request(
        &self,
    ) -> &runmat_analysis_execution::GeometryMaterializationRequest {
        &self.geometry_request
    }

    pub fn required_programs(&self) -> &[RequiredGeometryMeshingProgram] {
        &self.required_programs
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
pub struct DecodedGeometryMeshingRoot {
    pub request: GeometryMeshingRootRequest,
    pub geometry: ProgramExecutionRequest,
    pub artifact_access: ArtifactAccess,
}

pub fn prepare_geometry_meshing_root(
    preparation: GeometryMeshingRootPreparation,
) -> NativeExecutionResult<PreparedGeometryMeshingRoot> {
    let GeometryMeshingRootPreparation {
        program_revision,
        artifact_access,
        geometry,
        geometry_source_objects,
        meshing_request,
        preferred_edges_per_partition,
        preferred_faces_per_partition,
    } = preparation;
    let geometry_request = geometry.request.clone();
    artifact_access.validate().map_err(protocol)?;
    geometry.request.validate().map_err(protocol)?;
    geometry
        .program
        .validate_for_portable_host()
        .map_err(protocol)?;
    meshing_request.validate().map_err(protocol)?;
    if geometry.program.recipe.program_revision != program_revision {
        return Err(protocol(
            "geometry and meshing root program revisions differ",
        ));
    }
    for argument in &geometry.program.arguments {
        let ValuePayload::Object(reference) = argument else {
            return Err(protocol("geometry root arguments must be externalized"));
        };
        if reference.authorization_scope != artifact_access.authorization_scope
            || reference.encryption_context != artifact_access.encryption_context
        {
            return Err(protocol(
                "geometry and meshing root artifact authorities differ",
            ));
        }
    }
    let root = GeometryMeshingRootRequest {
        schema_version: 1,
        geometry: descriptor_from_request(&geometry.program),
        geometry_argument_count: u32::try_from(geometry.program.arguments.len())
            .map_err(protocol)?,
        meshing_request: meshing_request.clone(),
        preferred_edges_per_partition,
        preferred_faces_per_partition,
    };
    let root_object = LogicalObject::new(
        ObjectNamespace::InputValue,
        "meshing/geometry-roots",
        ROOT_MEDIA_TYPE,
        root.canonical_bytes()?,
    )
    .map_err(protocol)?;
    let root_reference = artifact_access
        .reference(&root_object, ValueRefKind::DriverObject, ROOT_VALUE_SCHEMA)
        .map_err(protocol)?;
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: program_revision.clone(),
        entrypoint: "meshing_workload".into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: GEOMETRY_MESHING_ROOT_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(GEOMETRY_MESHING_ROOT_TARGET_PROFILE),
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
    let required_programs = vec![
        RequiredGeometryMeshingProgram {
            recipe: recipe.clone(),
            artifact: artifact.clone(),
        },
        RequiredGeometryMeshingProgram {
            recipe: geometry.program.recipe.clone(),
            artifact: geometry.program.artifact.clone(),
        },
        {
            let program =
                prepare_meshing_host_program(program_revision.clone()).map_err(protocol)?;
            RequiredGeometryMeshingProgram {
                recipe: program.recipe,
                artifact: program.artifact,
            }
        },
    ];
    let mut arguments = vec![ValuePayload::Object(Box::new(root_reference))];
    arguments.extend(geometry.program.arguments);
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
    request.validate_for_portable_host().map_err(protocol)?;
    let mut objects = geometry_source_objects;
    objects.push(geometry.descriptor);
    objects.push(root_object);
    Ok(PreparedGeometryMeshingRoot {
        request,
        objects,
        geometry_request,
        meshing_request,
        required_programs,
    })
}

pub fn decode_geometry_meshing_root(
    program: &ProgramExecutionRequest,
    store: &impl CacheImport,
) -> NativeExecutionResult<DecodedGeometryMeshingRoot> {
    program.validate_for_portable_host().map_err(protocol)?;
    if program.artifact.form != ExecutableForm::MeshingWorkload
        || program.recipe.execution_mode != GEOMETRY_MESHING_ROOT_EXECUTION_MODE
        || program.recipe.target != ProgramTarget::portable(GEOMETRY_MESHING_ROOT_TARGET_PROFILE)
        || program.artifact.executable_bytes != ROOT_PROGRAM_BYTES
        || !program.recipe.compile_options.is_empty()
    {
        return Err(protocol("program is not a geometry meshing root workload"));
    }
    let ValuePayload::Object(root_reference) = program
        .arguments
        .first()
        .ok_or_else(|| protocol("geometry meshing root is missing its descriptor"))?
    else {
        return Err(protocol(
            "geometry meshing root descriptor must be externalized",
        ));
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
    let bytes = store
        .read_verified(root_reference.logical_digest)
        .map_err(protocol)?
        .ok_or_else(|| protocol("geometry meshing root descriptor is unavailable"))?;
    if bytes.len() as u64 != root_reference.encoded_length {
        return Err(protocol(
            "geometry meshing root descriptor length differs from its reference",
        ));
    }
    let request = GeometryMeshingRootRequest::decode(&bytes)?;
    if request.geometry.recipe.program_revision != program.recipe.program_revision {
        return Err(protocol(
            "geometry and meshing root program revisions differ",
        ));
    }
    let argument_count = usize::try_from(request.geometry_argument_count).map_err(protocol)?;
    let geometry_arguments = program
        .arguments
        .get(1..)
        .filter(|arguments| arguments.len() == argument_count)
        .ok_or_else(|| protocol("geometry meshing component argument count differs"))?
        .to_vec();
    let geometry = ProgramExecutionRequest {
        console: request.geometry.console,
        environment: request.geometry.environment.clone(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: request.geometry.recipe.clone(),
        artifact: request.geometry.artifact.clone(),
        callable: request.geometry.callable.clone(),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: geometry_arguments,
        requested_outputs: request.geometry.requested_outputs,
    };
    decode_geometry_materialization_program(&geometry, store, &artifact_access)
        .map_err(protocol)?;
    Ok(DecodedGeometryMeshingRoot {
        request,
        geometry,
        artifact_access,
    })
}

fn descriptor_from_request(request: &ProgramExecutionRequest) -> ProgramExecutionDescriptor {
    ProgramExecutionDescriptor {
        console: request.console,
        environment: request.environment.clone(),
        schema_version: request.schema_version,
        recipe: request.recipe.clone(),
        artifact: request.artifact.clone(),
        callable: request.callable.clone(),
        requested_outputs: request.requested_outputs,
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(all(test, feature = "occt-native"))]
#[path = "geometry_meshing_root_tests.rs"]
mod tests;
