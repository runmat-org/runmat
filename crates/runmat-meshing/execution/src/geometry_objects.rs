use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{ArtifactError, LogicalObject, ObjectDescriptor, ObjectNamespace};
use runmat_geometry_core::{
    admit_exact_geometry_closure, decode_exact_evaluators, decode_exact_topology,
    decode_geometry_healing_report, encode_exact_evaluators, encode_exact_topology,
    encode_geometry_healing_report, ExactBRepTopology, ExactEvaluatorRegistry,
    ExactGeometryManifest, GeometryDigest, GeometryDocument, GeometryHealingReport, GeometryModel,
    GeometryObjectRef, EXACT_BREP_MEDIA_TYPE, EXACT_EVALUATOR_MEDIA_TYPE,
    EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION, EXACT_TOPOLOGY_MEDIA_TYPE, GEOMETRY_HEALING_MEDIA_TYPE,
    GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
};

use crate::object_support::{
    add_inventory_bytes, enforce_object_length, logical_object, read_exact,
};
use crate::{MeshingExecutionError, MeshingExecutionResult};

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
}

pub fn prepare_exact_geometry_objects(
    mut document: GeometryDocument,
    topology: ExactBRepTopology,
    evaluators: ExactEvaluatorRegistry,
    healing_report: Option<GeometryHealingReport>,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedExactGeometryObjects> {
    let GeometryModel::ExactBRep { model } = &document.model else {
        return Err(MeshingExecutionError::Invalid(
            "faceted geometry cannot be packaged as an exact closure".into(),
        ));
    };
    let topology_bytes = encode_exact_topology(&topology, model)?;
    let evaluator_bytes = encode_exact_evaluators(&evaluators, &topology, model)?;
    let topology_object = geometry_object(
        "geometry/canonical/topology",
        EXACT_TOPOLOGY_MEDIA_TYPE,
        topology_bytes,
    )?;
    let evaluator_object = geometry_object(
        "geometry/canonical/evaluators",
        EXACT_EVALUATOR_MEDIA_TYPE,
        evaluator_bytes,
    )?;
    let healing_object = healing_report
        .as_ref()
        .map(|report| {
            geometry_object(
                "geometry/canonical/healing",
                GEOMETRY_HEALING_MEDIA_TYPE,
                encode_geometry_healing_report(report)?,
            )
        })
        .transpose()?;
    let manifest = ExactGeometryManifest {
        schema_version: EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION,
        source_digest: document.source.content_digest,
        revision: document.revision.clone(),
        kernel_abi: model.kernel_abi.clone(),
        topology: geometry_reference(&topology_object),
        evaluators: geometry_reference(&evaluator_object),
        healing_report: healing_object.as_ref().map(geometry_reference),
    };
    let manifest_object = geometry_object(
        "geometry/canonical/manifests",
        EXACT_BREP_MEDIA_TYPE,
        manifest.canonical_encode()?,
    )?;
    let GeometryModel::ExactBRep { model } = &mut document.model else {
        unreachable!()
    };
    model.artifact = geometry_reference(&manifest_object);
    document.validate()?;

    let mut objects = Vec::with_capacity(if healing_object.is_some() { 4 } else { 3 });
    objects.push(topology_object);
    objects.push(evaluator_object);
    if let Some(object) = healing_object {
        objects.push(object);
    }
    let root = manifest_object.descriptor.clone();
    objects.push(manifest_object);
    validate_inventory(&objects, limits)?;
    admit_exact_geometry_closure(
        &document,
        &objects.last().expect("manifest object").bytes,
        &objects[0].bytes,
        &objects[1].bytes,
        healing_report.as_ref().map(|_| objects[2].bytes.as_slice()),
    )?;
    Ok(PreparedExactGeometryObjects {
        document,
        manifest,
        topology,
        evaluators,
        healing_report,
        root,
        objects,
    })
}

pub fn import_exact_geometry_objects(
    source: &impl CacheImport,
    document: GeometryDocument,
    root: ExactGeometryObjectRoot,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedExactGeometryObjects> {
    document.validate()?;
    if document.primary_artifact().digest.bytes() != root.digest.bytes()
        || document.primary_artifact().encoded_length != root.encoded_length
    {
        return Err(MeshingExecutionError::Identity(
            "exact geometry root differs from the document artifact",
        ));
    }
    enforce_object_length("exact geometry manifest", root.encoded_length, limits)?;
    let manifest_bytes = read_exact(source, root.digest, root.encoded_length)?;
    let manifest = ExactGeometryManifest::canonical_decode(&manifest_bytes)?;
    let object_count = if manifest.healing_report.is_some() {
        4
    } else {
        3
    };
    if object_count > limits.max_objects {
        return Err(ArtifactError::Limit("too many exact geometry objects".into()).into());
    }
    let mut total_bytes = root.encoded_length;
    let topology_bytes = read_component(
        source,
        "exact geometry topology",
        &manifest.topology,
        &mut total_bytes,
        limits,
    )?;
    let evaluator_bytes = read_component(
        source,
        "exact geometry evaluators",
        &manifest.evaluators,
        &mut total_bytes,
        limits,
    )?;
    let healing_bytes = manifest
        .healing_report
        .as_ref()
        .map(|reference| {
            read_component(
                source,
                "geometry healing report",
                reference,
                &mut total_bytes,
                limits,
            )
        })
        .transpose()?;
    let GeometryModel::ExactBRep { model } = &document.model else {
        return Err(MeshingExecutionError::Invalid(
            "faceted geometry cannot import an exact closure".into(),
        ));
    };
    let topology = decode_exact_topology(&topology_bytes, model)?;
    let evaluators = decode_exact_evaluators(&evaluator_bytes, &topology, model)?;
    let healing_report = healing_bytes
        .as_deref()
        .map(decode_geometry_healing_report)
        .transpose()?;
    let prepared =
        prepare_exact_geometry_objects(document, topology, evaluators, healing_report, limits)?;
    if prepared.root.digest != root.digest || prepared.root.encoded_length != root.encoded_length {
        return Err(MeshingExecutionError::Identity(
            "imported exact geometry manifest differs from requested root",
        ));
    }
    Ok(prepared)
}

fn geometry_object(
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

fn geometry_reference(object: &LogicalObject) -> GeometryObjectRef {
    GeometryObjectRef {
        digest: GeometryDigest::from_bytes(*object.descriptor.digest.bytes()),
        encoded_length: object.descriptor.encoded_length,
        media_type: object.descriptor.media_type.clone(),
        schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
    }
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
