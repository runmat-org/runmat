use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionRequest, ProgramTarget,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};

use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, GeometryMaterializationRequest,
    GENERATED_GEOMETRY_PRODUCT_MEDIA_TYPE, GENERATED_GEOMETRY_PRODUCT_VALUE_SCHEMA,
};

pub const GEOMETRY_MATERIALIZATION_EXECUTION_MODE: &str = "geometry-materialization";
pub const GEOMETRY_MATERIALIZATION_TARGET_PROFILE: &str = "portable-geometry-host";
pub const GEOMETRY_MATERIALIZATION_REQUEST_MEDIA_TYPE: &str =
    "application/vnd.runmat.geometry-materialization-request+cbor";
pub const GEOMETRY_MATERIALIZATION_REQUEST_VALUE_SCHEMA: &str =
    "runmat.geometry.materialization-request";
const HOST_ENTRYPOINT: &str = "geometry_workload";
const HOST_PROGRAM_BYTES: &[u8] = b"runmat-geometry-materialization-host\0";

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedGeometryMaterializationProgram {
    pub request: GeometryMaterializationRequest,
    pub program: ProgramExecutionRequest,
    pub descriptor: LogicalObject,
}

pub fn prepare_geometry_materialization_program(
    request: GeometryMaterializationRequest,
    revision: ProgramRevision,
    access: &ArtifactAccess,
    source_roots: &[ValueRef],
) -> AnalysisBridgeResult<PreparedGeometryMaterializationProgram> {
    request.validate()?;
    validate_source_roots(&request, source_roots, access)?;
    let descriptor = LogicalObject::new(
        ObjectNamespace::InputValue,
        "geometry/materialization/request",
        GEOMETRY_MATERIALIZATION_REQUEST_MEDIA_TYPE,
        request.canonical_bytes()?,
    )?;
    let descriptor_root = access.reference(
        &descriptor,
        ValueRefKind::DriverObject,
        GEOMETRY_MATERIALIZATION_REQUEST_VALUE_SCHEMA,
    )?;
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: HOST_ENTRYPOINT.into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: GEOMETRY_MATERIALIZATION_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(GEOMETRY_MATERIALIZATION_TARGET_PROFILE),
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
        ExecutableForm::GeometryWorkload,
        HOST_PROGRAM_BYTES.to_vec(),
    )?;
    let mut arguments = Vec::with_capacity(source_roots.len() + 1);
    arguments.push(ValuePayload::Object(Box::new(descriptor_root)));
    arguments.extend(
        source_roots
            .iter()
            .cloned()
            .map(|root| ValuePayload::Object(Box::new(root))),
    );
    let program = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: runmat_execution::ProgramCallable::semantic(
            runmat_execution::ProgramFunctionId(0),
            Some(HOST_ENTRYPOINT.into()),
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments,
        requested_outputs: 1,
    };
    program.validate_for_portable_host()?;
    Ok(PreparedGeometryMaterializationProgram {
        request,
        program,
        descriptor,
    })
}

impl PreparedGeometryMaterializationProgram {
    pub fn write_to(&self, store: &mut impl CacheExport) -> AnalysisBridgeResult<()> {
        store.write_verified(&self.descriptor)?;
        Ok(())
    }
}

pub fn decode_geometry_materialization_program(
    program: &ProgramExecutionRequest,
    store: &impl CacheImport,
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<(GeometryMaterializationRequest, Vec<ValueRef>)> {
    program.validate_for_portable_host()?;
    if program.artifact.form != ExecutableForm::GeometryWorkload
        || program.recipe.execution_mode != GEOMETRY_MATERIALIZATION_EXECUTION_MODE
        || program.recipe.target != ProgramTarget::portable(GEOMETRY_MATERIALIZATION_TARGET_PROFILE)
        || program.artifact.executable_bytes != HOST_PROGRAM_BYTES
        || !program.recipe.compile_options.is_empty()
    {
        return Err(AnalysisBridgeError::Invalid(
            "program is not a geometry materialization workload".into(),
        ));
    }
    let (descriptor, sources) = program.arguments.split_first().ok_or_else(|| {
        AnalysisBridgeError::Invalid("geometry workload has no request descriptor".into())
    })?;
    let ValuePayload::Object(descriptor) = descriptor else {
        return Err(AnalysisBridgeError::Invalid(
            "geometry workload descriptor must be externalized".into(),
        ));
    };
    access.validate_reference(
        descriptor,
        ValueRefKind::DriverObject,
        GEOMETRY_MATERIALIZATION_REQUEST_MEDIA_TYPE,
        GEOMETRY_MATERIALIZATION_REQUEST_VALUE_SCHEMA,
    )?;
    let bytes = store
        .read_verified(descriptor.logical_digest)?
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("missing geometry workload descriptor".into())
        })?;
    if bytes.len() as u64 != descriptor.encoded_length {
        return Err(AnalysisBridgeError::Identity(
            "geometry workload descriptor length differs from its reference",
        ));
    }
    let request = GeometryMaterializationRequest::from_canonical_bytes(&bytes)?;
    let sources = sources
        .iter()
        .map(|value| match value {
            ValuePayload::Object(reference) => Ok((**reference).clone()),
            _ => Err(AnalysisBridgeError::Invalid(
                "geometry workload source must be externalized".into(),
            )),
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    validate_source_roots(&request, &sources, access)?;
    Ok((request, sources))
}

fn validate_source_roots(
    request: &GeometryMaterializationRequest,
    roots: &[ValueRef],
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<()> {
    use crate::GeometryMaterializationSource as Source;
    let expected = match &request.source {
        Source::Constructed => 0,
        Source::Imported { chunks, .. } => chunks.len(),
        Source::Derived { .. } => roots.len(),
        Source::Generated { .. } => 1,
    };
    if roots.len() != expected
        || matches!(request.source, Source::Derived { .. }) && roots.is_empty()
    {
        return Err(AnalysisBridgeError::Invalid(
            "geometry source object inventory differs from its request".into(),
        ));
    }
    for root in roots {
        access.validate()?;
        let valid_kind = match &request.source {
            Source::Derived { .. } => matches!(
                root.kind,
                ValueRefKind::DriverObject | ValueRefKind::ResultObject
            ),
            Source::Imported { .. } => root.kind == ValueRefKind::DriverObject,
            Source::Constructed => false,
            Source::Generated { receipt } => {
                root.kind == ValueRefKind::ResultObject
                    && root.media_type == GENERATED_GEOMETRY_PRODUCT_MEDIA_TYPE
                    && root.value_schema == GENERATED_GEOMETRY_PRODUCT_VALUE_SCHEMA
                    && root.logical_digest == receipt.product_digest
                    && root.encoded_length == receipt.product_encoded_length
            }
        };
        if !valid_kind
            || root.authorization_scope != access.authorization_scope
            || root.encryption_context != access.encryption_context
            || root.resident_fence.is_some()
        {
            return Err(AnalysisBridgeError::Identity(
                "geometry source reference is outside its artifact access authority",
            ));
        }
    }
    let unique = roots
        .iter()
        .map(|root| root.logical_digest)
        .collect::<std::collections::BTreeSet<_>>();
    if unique.len() != roots.len() && !matches!(request.source, Source::Imported { .. }) {
        return Err(AnalysisBridgeError::Invalid(
            "geometry source closure contains duplicate objects".into(),
        ));
    }
    let total_bytes = roots.iter().try_fold(0_u64, |total, root| {
        if root.encoded_length > request.resources.maximum_object_bytes {
            return Err(AnalysisBridgeError::Invalid(
                "geometry source object exceeds its admitted object bound".into(),
            ));
        }
        total.checked_add(root.encoded_length).ok_or_else(|| {
            AnalysisBridgeError::Invalid("geometry source inventory size overflowed".into())
        })
    })?;
    if total_bytes > request.resources.maximum_artifact_bytes {
        return Err(AnalysisBridgeError::Invalid(
            "geometry source closure exceeds its admitted artifact bound".into(),
        ));
    }
    if let Source::Imported { chunks, .. } = &request.source {
        if chunks.iter().zip(roots).any(|(chunk, root)| {
            root.logical_digest != chunk.digest || root.encoded_length != chunk.encoded_length
        }) {
            return Err(AnalysisBridgeError::Identity(
                "imported geometry chunks differ from their source manifest",
            ));
        }
    }
    if let Source::Derived { root, .. } = &request.source {
        if !roots.iter().any(|candidate| {
            candidate.logical_digest.bytes() == root.digest.bytes()
                && candidate.encoded_length == root.encoded_length
        }) {
            return Err(AnalysisBridgeError::Identity(
                "derived geometry source closure omits its manifest root",
            ));
        }
    }
    Ok(())
}
