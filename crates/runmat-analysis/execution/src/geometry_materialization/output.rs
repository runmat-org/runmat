use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{ArtifactAccess, LogicalObject, ObjectNamespace};

use crate::{AnalysisBridgeError, AnalysisBridgeResult, GeometryMaterializationOutput};

pub const GEOMETRY_MATERIALIZATION_OUTPUT_MEDIA_TYPE: &str =
    "application/vnd.runmat.geometry-materialization-output+cbor";
pub const GEOMETRY_MATERIALIZATION_OUTPUT_VALUE_SCHEMA: &str =
    "runmat.geometry.materialization-output";

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedGeometryMaterializationOutput {
    pub output: GeometryMaterializationOutput,
    pub object: LogicalObject,
    pub root: ValueRef,
}

pub fn prepare_geometry_materialization_output(
    output: GeometryMaterializationOutput,
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<PreparedGeometryMaterializationOutput> {
    output.validate()?;
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "geometry/materialization/output",
        GEOMETRY_MATERIALIZATION_OUTPUT_MEDIA_TYPE,
        output.canonical_bytes()?,
    )?;
    let root = access.reference(
        &object,
        ValueRefKind::ResultObject,
        GEOMETRY_MATERIALIZATION_OUTPUT_VALUE_SCHEMA,
    )?;
    Ok(PreparedGeometryMaterializationOutput {
        output,
        object,
        root,
    })
}

impl PreparedGeometryMaterializationOutput {
    pub fn write_to(&self, store: &mut impl CacheExport) -> AnalysisBridgeResult<()> {
        store.write_verified(&self.object)?;
        Ok(())
    }
}

pub fn import_geometry_materialization_output(
    store: &impl CacheImport,
    root: &ValueRef,
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<GeometryMaterializationOutput> {
    access.validate_reference(
        root,
        ValueRefKind::ResultObject,
        GEOMETRY_MATERIALIZATION_OUTPUT_MEDIA_TYPE,
        GEOMETRY_MATERIALIZATION_OUTPUT_VALUE_SCHEMA,
    )?;
    let bytes = store.read_verified(root.logical_digest)?.ok_or_else(|| {
        AnalysisBridgeError::Invalid("missing geometry materialization output".into())
    })?;
    if bytes.len() as u64 != root.encoded_length {
        return Err(AnalysisBridgeError::Identity(
            "geometry materialization output length differs from its reference",
        ));
    }
    GeometryMaterializationOutput::from_canonical_bytes(&bytes)
}
