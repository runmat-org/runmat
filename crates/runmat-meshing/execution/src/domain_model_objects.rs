use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{LogicalObject, ObjectDescriptor, ObjectNamespace};
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingDomainModel, MESHING_DOMAIN_MODEL_MEDIA_TYPE,
};

use crate::object_support::{
    enforce_object_length, input_object_reference, logical_object, read_exact, validate_input_root,
};
use crate::{MeshingArtifactAccess, MeshingExecutionError, MeshingExecutionResult};

pub(crate) const DOMAIN_MODEL_VALUE_SCHEMA: &str = "runmat.meshing.domain-model.v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DomainModelObjectRoot {
    pub digest: Digest,
    pub encoded_length: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedDomainModelObjects {
    pub model: MeshingDomainModel,
    pub root: ObjectDescriptor,
    pub objects: Vec<LogicalObject>,
}

impl PreparedDomainModelObjects {
    pub fn root_reference(&self) -> DomainModelObjectRoot {
        DomainModelObjectRoot {
            digest: self.root.digest,
            encoded_length: self.root.encoded_length,
        }
    }

    pub fn revalidate(&self, limits: ObjectInventoryLimits) -> MeshingExecutionResult<()> {
        if prepare_domain_model_objects(self.model.clone(), limits)? != *self {
            return Err(MeshingExecutionError::Identity(
                "prepared domain model is not its canonical object closure",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedDomainModelInput {
    domain_model_objects: PreparedDomainModelObjects,
    root_input: ValueRef,
    input_objects: Vec<ValueRef>,
}

impl PreparedDomainModelInput {
    pub const fn domain_model_objects(&self) -> &PreparedDomainModelObjects {
        &self.domain_model_objects
    }

    pub const fn root_input(&self) -> &ValueRef {
        &self.root_input
    }

    pub fn input_objects(&self) -> &[ValueRef] {
        &self.input_objects
    }
}

pub fn prepare_domain_model_input(
    domain_model_objects: PreparedDomainModelObjects,
    access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedDomainModelInput> {
    access.validate()?;
    domain_model_objects.revalidate(limits)?;
    let object = domain_model_objects
        .objects
        .first()
        .ok_or(MeshingExecutionError::Identity(
            "prepared domain model has no root object",
        ))?;
    let root_input = input_object_reference(
        object,
        &access,
        DOMAIN_MODEL_VALUE_SCHEMA,
        "invalid domain model input reference",
    )?;
    Ok(PreparedDomainModelInput {
        domain_model_objects,
        input_objects: vec![root_input.clone()],
        root_input,
    })
}

pub fn import_domain_model_input(
    source: &impl CacheImport,
    root: &ValueRef,
    access: MeshingArtifactAccess,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedDomainModelInput> {
    access.validate()?;
    validate_input_root(
        root,
        &access,
        MESHING_DOMAIN_MODEL_MEDIA_TYPE,
        DOMAIN_MODEL_VALUE_SCHEMA,
        "domain model root is outside input artifact authority",
    )?;
    let objects = import_domain_model_objects(
        source,
        DomainModelObjectRoot {
            digest: root.logical_digest,
            encoded_length: root.encoded_length,
        },
        limits,
    )?;
    let prepared = prepare_domain_model_input(objects, access, limits)?;
    if prepared.root_input != *root {
        return Err(MeshingExecutionError::Identity(
            "imported domain model root differs from its execution reference",
        ));
    }
    Ok(prepared)
}

pub fn prepare_domain_model_objects(
    model: MeshingDomainModel,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedDomainModelObjects> {
    let bytes = model.canonical_encode()?;
    let object = logical_object(
        "meshing/domain-models",
        ObjectNamespace::InputValue,
        MESHING_DOMAIN_MODEL_MEDIA_TYPE,
        bytes,
    )?;
    let root = object.descriptor.clone();
    let objects = vec![object];
    validate_inventory(&objects, limits)?;
    Ok(PreparedDomainModelObjects {
        model,
        root,
        objects,
    })
}

pub fn import_domain_model_objects(
    source: &impl CacheImport,
    root: DomainModelObjectRoot,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedDomainModelObjects> {
    enforce_object_length("domain model", root.encoded_length, limits)?;
    if limits.max_objects == 0 || root.encoded_length > limits.max_total_bytes {
        return Err(runmat_execution_artifact::ArtifactError::Limit(
            "domain model object inventory exceeds its limit".into(),
        )
        .into());
    }
    let bytes = read_exact(source, root.digest, root.encoded_length)?;
    let model = MeshingDomainModel::canonical_decode(&bytes)?;
    let prepared = prepare_domain_model_objects(model, limits)?;
    if prepared.root.digest != root.digest || prepared.root.encoded_length != root.encoded_length {
        return Err(MeshingExecutionError::Identity(
            "imported domain model differs from requested root",
        ));
    }
    Ok(prepared)
}
