use std::collections::BTreeMap;

use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::{ArtifactAccess, LogicalObject, ObjectNamespace};
use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryDefinition, GeometryDefinitionSource, GeometryDocument,
    GeometryObjectRef,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_geometry_ops::GeometryFeaturePlanLimits;

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

use super::{
    GeneratedGeometryExecutionReceipt, GeometryMaterializationRequest,
    GeometryMaterializationResources, GeometryMaterializationSource, GeometryProviderProfile,
    GeometrySourceChunk,
};

pub const GEOMETRY_SOURCE_MEDIA_TYPE: &str = "application/vnd.runmat.geometry-source";
pub const GEOMETRY_SOURCE_VALUE_SCHEMA: &str = "runmat.geometry.source-chunk";

/// Host inputs required by exactly one geometry materialization request.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedGeometrySource {
    pub source: GeometryMaterializationSource,
    pub objects: Vec<LogicalObject>,
    pub roots: Vec<ValueRef>,
}

impl PreparedGeometrySource {
    pub fn constructed() -> Self {
        Self {
            source: GeometryMaterializationSource::Constructed,
            objects: Vec::new(),
            roots: Vec::new(),
        }
    }

    pub fn imported(
        source_name: impl Into<String>,
        bytes: &[u8],
        maximum_chunk_bytes: u64,
        access: &ArtifactAccess,
    ) -> AnalysisBridgeResult<Self> {
        let source_name = source_name.into();
        if bytes.is_empty() || maximum_chunk_bytes == 0 {
            return Err(AnalysisBridgeError::Invalid(
                "imported geometry source and chunk bound must be non-empty".into(),
            ));
        }
        let chunk_size = usize::try_from(maximum_chunk_bytes).map_err(|_| {
            AnalysisBridgeError::Invalid(
                "geometry source chunk bound exceeds the host address space".into(),
            )
        })?;
        let mut objects = Vec::new();
        let mut roots = Vec::new();
        let mut chunks = Vec::new();
        for (ordinal, bytes) in bytes.chunks(chunk_size).enumerate() {
            let ordinal = u32::try_from(ordinal).map_err(|_| {
                AnalysisBridgeError::Invalid("geometry source has too many chunks".into())
            })?;
            let object = LogicalObject::new(
                ObjectNamespace::InputValue,
                format!("geometry/sources/{ordinal:08}"),
                GEOMETRY_SOURCE_MEDIA_TYPE,
                bytes.to_vec(),
            )?;
            let root = access.reference(
                &object,
                ValueRefKind::DriverObject,
                GEOMETRY_SOURCE_VALUE_SCHEMA,
            )?;
            chunks.push(GeometrySourceChunk {
                ordinal,
                digest: object.descriptor.digest,
                encoded_length: object.descriptor.encoded_length,
            });
            objects.push(object);
            roots.push(root);
        }
        Ok(Self {
            source: GeometryMaterializationSource::Imported {
                source_name,
                source_digest: Digest::sha256(bytes),
                encoded_length: bytes.len() as u64,
                chunks,
            },
            objects,
            roots,
        })
    }

    pub fn derived(
        definition: runmat_geometry_core::GeometryDefinitionRef,
        document: GeometryDocument,
        root: GeometryObjectRef,
        roots: Vec<ValueRef>,
    ) -> Self {
        Self {
            source: GeometryMaterializationSource::Derived {
                definition,
                document: Box::new(document),
                root,
            },
            objects: Vec::new(),
            roots,
        }
    }

    pub fn generated(receipt: GeneratedGeometryExecutionReceipt, product_root: ValueRef) -> Self {
        Self {
            source: GeometryMaterializationSource::Generated {
                receipt: Box::new(receipt),
            },
            objects: Vec::new(),
            roots: vec![product_root],
        }
    }

    pub fn write_to(&self, store: &mut impl CacheExport) -> AnalysisBridgeResult<()> {
        for object in &self.objects {
            store.write_verified(object)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeometryMaterializationPreparation {
    pub import_options: ExactCadImportOptions,
    pub plan_limits: GeometryFeaturePlanLimits,
    pub resources: GeometryMaterializationResources,
}

/// Combines source-neutral geometry intent with already admitted host inputs.
///
/// Derived and generated definitions reach this function only after their prerequisite task has
/// published the exact document or provider-neutral generated product named by `source`.
pub fn prepare_geometry_materialization_request(
    definition: GeometryDefinition,
    bindings: BTreeMap<String, GeometryAuthoringValue>,
    source: &PreparedGeometrySource,
    provider: GeometryProviderProfile,
    mut preparation: GeometryMaterializationPreparation,
) -> AnalysisBridgeResult<GeometryMaterializationRequest> {
    match &definition.source {
        GeometryDefinitionSource::Imported(imported) => {
            // `units` resolved at check (authored, or the file's own declaration) is authoritative.
            preparation.import_options.source_units = imported.authored_units;
            preparation.import_options.source_units_authority =
                runmat_geometry_io::SourceUnitsAuthority::Authored;
            preparation
                .import_options
                .analysis
                .healing
                .shared_interfaces = imported.shared_interfaces;
        }
        GeometryDefinitionSource::Constructed(_) | GeometryDefinitionSource::Derived(_) => {}
        GeometryDefinitionSource::Generated(_) => {}
        GeometryDefinitionSource::CrossSection(_) => {
            return Err(AnalysisBridgeError::Invalid(
                "a cross-section is meshed and solved by the transmission-line solver; no geometry kernel materializes it".into(),
            ))
        }
    }
    if source
        .roots
        .iter()
        .any(|root| root.encoded_length > preparation.resources.maximum_object_bytes)
    {
        return Err(AnalysisBridgeError::Invalid(
            "geometry source object exceeds its materialization resource bound".into(),
        ));
    }
    let request = GeometryMaterializationRequest {
        schema_version: 1,
        definition,
        bindings,
        source: source.source.clone(),
        import_options: preparation.import_options,
        provider,
        plan_limits: preparation.plan_limits,
        resources: preparation.resources,
    };
    request.validate()?;
    Ok(request)
}
