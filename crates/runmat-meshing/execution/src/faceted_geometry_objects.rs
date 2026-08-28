use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{LogicalObject, ObjectDescriptor};
use runmat_geometry_core::{
    admit_faceted_solid, build_faceted_solid_closure, FacetedSolid, GeometryDocument,
    GeometryModel, FACETED_SOLID_MEDIA_TYPE,
};

use crate::geometry_objects::{geometry_object, validate_geometry_reference};
use crate::object_support::{
    enforce_object_length, input_object_reference, read_exact, validate_input_root,
};
use crate::{MeshingArtifactAccess, MeshingExecutionError, MeshingExecutionResult};

const FACETED_SOLID_SCHEMA: &str = "runmat.geometry.faceted-solid.v2";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FacetedGeometryObjectRoot {
    pub digest: Digest,
    pub encoded_length: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedFacetedGeometryObjects {
    pub document: GeometryDocument,
    pub solid: FacetedSolid,
    pub root: ObjectDescriptor,
    pub objects: Vec<LogicalObject>,
}

impl PreparedFacetedGeometryObjects {
    pub fn root_reference(&self) -> FacetedGeometryObjectRoot {
        FacetedGeometryObjectRoot {
            digest: self.root.digest,
            encoded_length: self.root.encoded_length,
        }
    }

    pub fn revalidate(&self, limits: ObjectInventoryLimits) -> MeshingExecutionResult<()> {
        let rebuilt =
            prepare_faceted_geometry_objects(self.document.clone(), self.solid.clone(), limits)?;
        if rebuilt != *self {
            return Err(MeshingExecutionError::Identity(
                "prepared faceted geometry is not its canonical object closure",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedFacetedGeometryInput {
    geometry_objects: PreparedFacetedGeometryObjects,
    root_input: ValueRef,
    input_objects: Vec<ValueRef>,
}

impl PreparedFacetedGeometryInput {
    pub const fn geometry_objects(&self) -> &PreparedFacetedGeometryObjects {
        &self.geometry_objects
    }

    pub const fn root_input(&self) -> &ValueRef {
        &self.root_input
    }

    pub fn input_objects(&self) -> &[ValueRef] {
        &self.input_objects
    }
}

pub fn prepare_faceted_geometry_input(
    geometry_objects: PreparedFacetedGeometryObjects,
    access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedFacetedGeometryInput> {
    access.validate()?;
    geometry_objects.revalidate(limits)?;
    let object = geometry_objects
        .objects
        .first()
        .ok_or(MeshingExecutionError::Identity(
            "prepared faceted geometry has no root object",
        ))?;
    let root_input = input_object_reference(
        object,
        &access,
        FACETED_SOLID_SCHEMA,
        "invalid faceted geometry input reference",
    )?;
    Ok(PreparedFacetedGeometryInput {
        geometry_objects,
        input_objects: vec![root_input.clone()],
        root_input,
    })
}

pub fn import_faceted_geometry_input(
    source: &impl CacheImport,
    document: GeometryDocument,
    root: &ValueRef,
    access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedFacetedGeometryInput> {
    access.validate()?;
    validate_input_root(
        root,
        &access,
        FACETED_SOLID_MEDIA_TYPE,
        FACETED_SOLID_SCHEMA,
        "faceted geometry root is outside input artifact authority",
    )?;
    let geometry_objects = import_faceted_geometry_objects(
        source,
        document,
        FacetedGeometryObjectRoot {
            digest: root.logical_digest,
            encoded_length: root.encoded_length,
        },
        limits,
    )?;
    let prepared = prepare_faceted_geometry_input(geometry_objects, access, limits)?;
    if prepared.root_input != *root {
        return Err(MeshingExecutionError::Identity(
            "imported faceted geometry root differs from its execution reference",
        ));
    }
    Ok(prepared)
}

pub fn prepare_faceted_geometry_objects(
    document: GeometryDocument,
    solid: FacetedSolid,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedFacetedGeometryObjects> {
    let encoded = build_faceted_solid_closure(document, solid)?;
    let object = geometry_object(
        "geometry/canonical/faceted-solids",
        FACETED_SOLID_MEDIA_TYPE,
        encoded.solid_bytes,
    )?;
    validate_geometry_reference(&object, encoded.document.primary_artifact())?;
    let root = object.descriptor.clone();
    let objects = vec![object];
    validate_inventory(&objects, limits)?;
    Ok(PreparedFacetedGeometryObjects {
        document: encoded.document,
        solid: encoded.solid,
        root,
        objects,
    })
}

pub fn import_faceted_geometry_objects(
    source: &impl CacheImport,
    document: GeometryDocument,
    root: FacetedGeometryObjectRoot,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedFacetedGeometryObjects> {
    document.validate()?;
    if !matches!(document.model, GeometryModel::FacetedSolid { .. }) {
        return Err(MeshingExecutionError::Invalid(
            "exact geometry cannot import a faceted closure".into(),
        ));
    }
    if document.primary_artifact().digest.bytes() != root.digest.bytes()
        || document.primary_artifact().encoded_length != root.encoded_length
    {
        return Err(MeshingExecutionError::Identity(
            "faceted geometry root differs from the document artifact",
        ));
    }
    enforce_object_length("faceted geometry", root.encoded_length, limits)?;
    if limits.max_objects == 0 || root.encoded_length > limits.max_total_bytes {
        return Err(runmat_execution_artifact::ArtifactError::Limit(
            "faceted geometry object inventory exceeds its limit".into(),
        )
        .into());
    }
    let bytes = read_exact(source, root.digest, root.encoded_length)?;
    let solid = admit_faceted_solid(&document, &bytes)?;
    let prepared = prepare_faceted_geometry_objects(document, solid, limits)?;
    if prepared.root.digest != root.digest || prepared.root.encoded_length != root.encoded_length {
        return Err(MeshingExecutionError::Identity(
            "imported faceted geometry differs from requested root",
        ));
    }
    Ok(prepared)
}
