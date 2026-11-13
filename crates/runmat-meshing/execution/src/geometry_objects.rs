use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{ArtifactError, LogicalObject, ObjectDescriptor, ObjectNamespace};
use runmat_geometry_core::{
    build_exact_geometry_closure, ExactBRepTopology, ExactEvaluatorRegistry, ExactGeometryManifest,
    GeometryDocument, GeometryHealingReport, GeometryModel, GeometryObjectRef,
    EXACT_BREP_MEDIA_TYPE, EXACT_EVALUATOR_MEDIA_TYPE, EXACT_TOPOLOGY_MEDIA_TYPE,
    GEOMETRY_HEALING_MEDIA_TYPE, GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
    KERNEL_REPRESENTATION_MEDIA_TYPE,
};

use crate::object_support::{
    add_inventory_bytes, admit_import_storage, enforce_object_length, import_slots, logical_object,
    read_exact,
};
use crate::{ArtifactAccess, MeshingExecutionError, MeshingExecutionResult};

const EXACT_MANIFEST_SCHEMA: &str = "runmat.geometry.exact-manifest.v2";
const EXACT_TOPOLOGY_SCHEMA: &str = "runmat.geometry.exact-topology.v2";
const EXACT_EVALUATORS_SCHEMA: &str = "runmat.geometry.exact-evaluators.v2";
const KERNEL_REPRESENTATION_SCHEMA: &str = "runmat.geometry.kernel-representation.v2";
const HEALING_REPORT_SCHEMA: &str = "runmat.geometry.healing-report.v2";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactGeometryObjectRoot {
    pub digest: Digest,
    pub encoded_length: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedExactGeometryObjects {
    pub document: GeometryDocument,
    pub manifest: ExactGeometryManifest,
    pub topology: ExactBRepTopology,
    pub evaluators: ExactEvaluatorRegistry,
    pub healing_report: Option<GeometryHealingReport>,
    pub root: ObjectDescriptor,
    pub objects: Vec<LogicalObject>,
}

impl PreparedExactGeometryObjects {
    pub fn root_reference(&self) -> ExactGeometryObjectRoot {
        ExactGeometryObjectRoot {
            digest: self.root.digest,
            encoded_length: self.root.encoded_length,
        }
    }

    pub fn revalidate(&self, limits: ObjectInventoryLimits) -> MeshingExecutionResult<()> {
        verification::revalidate(self, limits)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedExactGeometryInput {
    geometry_objects: PreparedExactGeometryObjects,
    root_input: ValueRef,
    input_objects: Vec<ValueRef>,
}

impl PreparedExactGeometryInput {
    /// Transfer the verified closure without retaining a second encoded inventory.
    pub fn into_geometry_objects(self) -> PreparedExactGeometryObjects {
        self.geometry_objects
    }

    pub const fn geometry_objects(&self) -> &PreparedExactGeometryObjects {
        &self.geometry_objects
    }

    pub const fn root_input(&self) -> &ValueRef {
        &self.root_input
    }

    pub fn input_objects(&self) -> &[ValueRef] {
        &self.input_objects
    }
}

pub fn prepare_exact_geometry_input(
    geometry_objects: PreparedExactGeometryObjects,
    access: ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedExactGeometryInput> {
    prepare_exact_geometry_value(geometry_objects, access, ValueRefKind::DriverObject, limits)
}

/// Projects a worker-admitted exact geometry closure as a reusable execution result.
pub fn prepare_exact_geometry_result(
    geometry_objects: PreparedExactGeometryObjects,
    access: ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedExactGeometryInput> {
    prepare_exact_geometry_value(geometry_objects, access, ValueRefKind::ResultObject, limits)
}

fn prepare_exact_geometry_value(
    geometry_objects: PreparedExactGeometryObjects,
    access: ArtifactAccess,
    kind: ValueRefKind,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedExactGeometryInput> {
    access.validate()?;
    geometry_objects.revalidate(limits)?;
    project_exact_geometry_value(geometry_objects, access, kind, None)
}

// Private projection is reached only after public revalidation or raw import.
fn project_exact_geometry_value(
    geometry_objects: PreparedExactGeometryObjects,
    access: ArtifactAccess,
    kind: ValueRefKind,
    control: Option<&dyn runmat_geometry_core::GeometryCanonicalControl>,
) -> MeshingExecutionResult<PreparedExactGeometryInput> {
    admit_import_storage(
        control,
        import_slots::<ValueRef>(geometry_objects.objects.len())?,
    )?;
    let mut input_objects = Vec::with_capacity(geometry_objects.objects.len());
    for object in &geometry_objects.objects {
        let schema = exact_object_schema(object, &geometry_objects.root)?;
        admit_import_storage(
            control,
            access.reference_text_bytes(&object.descriptor.media_type, schema)?,
        )?;
        input_objects.push(exact_object_reference(
            object,
            &geometry_objects.root,
            &access,
            kind,
        )?);
    }
    admit_import_storage(
        control,
        access.reference_text_bytes(EXACT_BREP_MEDIA_TYPE, EXACT_MANIFEST_SCHEMA)?,
    )?;
    let root_input = input_objects
        .iter()
        .find(|reference| reference.logical_digest == geometry_objects.root.digest)
        .cloned()
        .ok_or(MeshingExecutionError::Identity(
            "prepared exact geometry has no root input reference",
        ))?;
    Ok(PreparedExactGeometryInput {
        geometry_objects,
        root_input,
        input_objects,
    })
}

pub fn import_exact_geometry_input(
    source: &impl CacheImport,
    document: GeometryDocument,
    root: &ValueRef,
    access: ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedExactGeometryInput> {
    import_exact_geometry_input_controlled(source, document, root, access, limits, None)
}
pub(crate) fn import_exact_geometry_input_controlled(
    source: &impl CacheImport,
    document: GeometryDocument,
    root: &ValueRef,
    access: ArtifactAccess,
    limits: ObjectInventoryLimits,
    control: Option<&dyn runmat_geometry_core::GeometryCanonicalControl>,
) -> MeshingExecutionResult<PreparedExactGeometryInput> {
    access.validate()?;
    validate_exact_root(root, &access)?;
    let geometry_objects = import::import_exact_geometry_objects_controlled(
        source,
        document,
        ExactGeometryObjectRoot {
            digest: root.logical_digest,
            encoded_length: root.encoded_length,
        },
        limits,
        control,
    )?;
    let prepared = project_exact_geometry_value(geometry_objects, access, root.kind, control)?;
    if prepared.root_input != *root {
        return Err(MeshingExecutionError::Identity(
            "imported exact geometry root differs from its execution reference",
        ));
    }
    Ok(prepared)
}

pub fn prepare_exact_geometry_objects(
    document: GeometryDocument,
    topology: ExactBRepTopology,
    evaluators: ExactEvaluatorRegistry,
    kernel_representation: Option<Vec<u8>>,
    healing_report: Option<GeometryHealingReport>,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedExactGeometryObjects> {
    let encoded = build_exact_geometry_closure(
        document,
        &topology,
        &evaluators,
        kernel_representation.as_deref(),
        healing_report.as_ref(),
    )?;
    let topology_object = geometry_object(
        "geometry/canonical/topology",
        EXACT_TOPOLOGY_MEDIA_TYPE,
        encoded.topology_bytes,
    )?;
    let evaluator_object = geometry_object(
        "geometry/canonical/evaluators",
        EXACT_EVALUATOR_MEDIA_TYPE,
        encoded.evaluator_bytes,
    )?;
    let kernel_representation_object = kernel_representation
        .map(|bytes| {
            geometry_object(
                "geometry/canonical/kernel",
                KERNEL_REPRESENTATION_MEDIA_TYPE,
                bytes,
            )
        })
        .transpose()?;
    let healing_object = healing_report
        .as_ref()
        .zip(encoded.healing_bytes)
        .map(|(_, bytes)| {
            geometry_object(
                "geometry/canonical/healing",
                GEOMETRY_HEALING_MEDIA_TYPE,
                bytes,
            )
        })
        .transpose()?;
    let manifest_object = geometry_object(
        "geometry/canonical/manifests",
        EXACT_BREP_MEDIA_TYPE,
        encoded.manifest_bytes,
    )?;
    validate_geometry_reference(&topology_object, &encoded.manifest.topology)?;
    validate_geometry_reference(&evaluator_object, &encoded.manifest.evaluators)?;
    if let (Some(object), Some(reference)) = (
        &kernel_representation_object,
        &encoded.manifest.kernel_representation,
    ) {
        validate_geometry_reference(object, reference)?;
    }
    if let (Some(object), Some(reference)) = (&healing_object, &encoded.manifest.healing_report) {
        validate_geometry_reference(object, reference)?;
    }
    validate_geometry_reference(&manifest_object, encoded.document.primary_artifact())?;

    let mut objects = Vec::with_capacity(
        3 + usize::from(kernel_representation_object.is_some())
            + usize::from(healing_object.is_some()),
    );
    objects.push(topology_object);
    objects.push(evaluator_object);
    if let Some(object) = kernel_representation_object {
        objects.push(object);
    }
    if let Some(object) = healing_object {
        objects.push(object);
    }
    let root = manifest_object.descriptor.clone();
    objects.push(manifest_object);
    validate_inventory(&objects, limits)?;
    Ok(PreparedExactGeometryObjects {
        document: encoded.document,
        manifest: encoded.manifest,
        topology,
        evaluators,
        healing_report,
        root,
        objects,
    })
}

pub use import::import_exact_geometry_objects;
mod import;

pub(crate) fn geometry_object(
    logical_prefix: &str,
    media_type: &str,
    bytes: Vec<u8>,
) -> MeshingExecutionResult<LogicalObject> {
    logical_object(
        logical_prefix,
        ObjectNamespace::InputValue,
        media_type,
        bytes,
    )
}

pub(crate) fn validate_geometry_reference(
    object: &LogicalObject,
    reference: &GeometryObjectRef,
) -> MeshingExecutionResult<()> {
    if object.descriptor.digest.bytes() != reference.digest.bytes()
        || object.descriptor.encoded_length != reference.encoded_length
        || object.descriptor.media_type != reference.media_type
        || reference.schema_version != GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION
    {
        return Err(MeshingExecutionError::Identity(
            "execution object differs from its geometry-owned reference",
        ));
    }
    Ok(())
}

fn exact_object_schema(
    object: &LogicalObject,
    root: &ObjectDescriptor,
) -> MeshingExecutionResult<&'static str> {
    let value_schema = if object.descriptor.digest == root.digest {
        EXACT_MANIFEST_SCHEMA
    } else {
        match object.descriptor.media_type.as_str() {
            EXACT_TOPOLOGY_MEDIA_TYPE => EXACT_TOPOLOGY_SCHEMA,
            EXACT_EVALUATOR_MEDIA_TYPE => EXACT_EVALUATORS_SCHEMA,
            KERNEL_REPRESENTATION_MEDIA_TYPE => KERNEL_REPRESENTATION_SCHEMA,
            GEOMETRY_HEALING_MEDIA_TYPE => HEALING_REPORT_SCHEMA,
            _ => {
                return Err(MeshingExecutionError::Identity(
                    "exact geometry object has an unknown component media type",
                ));
            }
        }
    };
    Ok(value_schema)
}
fn exact_object_reference(
    object: &LogicalObject,
    root: &ObjectDescriptor,
    access: &ArtifactAccess,
    kind: ValueRefKind,
) -> MeshingExecutionResult<ValueRef> {
    let value_schema = exact_object_schema(object, root)?;
    let reference = access.reference_descriptor(&object.descriptor, kind, value_schema)?;
    reference
        .validate(runmat_execution::value::ValueLimits::default())
        .map_err(|_| {
            MeshingExecutionError::Identity("invalid exact geometry execution reference")
        })?;
    Ok(reference)
}

fn validate_exact_root(root: &ValueRef, access: &ArtifactAccess) -> MeshingExecutionResult<()> {
    if !matches!(
        root.kind,
        ValueRefKind::DriverObject | ValueRefKind::ResultObject
    ) {
        return Err(MeshingExecutionError::Identity(
            "exact geometry root has a nonportable execution kind",
        ));
    }
    access.validate_reference(
        root,
        root.kind,
        EXACT_BREP_MEDIA_TYPE,
        EXACT_MANIFEST_SCHEMA,
    )?;
    Ok(())
}

fn read_component(
    source: &impl CacheImport,
    class: &str,
    reference: &GeometryObjectRef,
    total_bytes: &mut u64,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<Vec<u8>> {
    enforce_object_length(class, reference.encoded_length, limits)?;
    add_inventory_bytes(
        "exact geometry",
        total_bytes,
        reference.encoded_length,
        limits,
    )?;
    read_exact(
        source,
        Digest::from_bytes(*reference.digest.bytes()),
        reference.encoded_length,
    )
}

mod components;
mod verification;
