mod topology;

use runmat_geometry_core::{
    build_faceted_solid_closure, EncodedFacetedSolidClosure, FacetedSolidModel, GeometryDigest,
    GeometryDocument, GeometryHealingPolicy, GeometryModel, GeometryObjectRef,
    GeometryRevisionIdentity, GeometrySourceFormat, GeometrySourceIdentity,
    GeometryTolerancePolicy, UnitSystem, FACETED_SOLID_MEDIA_TYPE,
    GEOMETRY_DOCUMENT_SCHEMA_VERSION, GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

use crate::{
    detect_geometry_format,
    import::{
        import_geometry_with_context, GeometryImportBudgetPolicy, GeometryImportContext,
        GeometryImportError, GeometryImportOptions,
    },
    report::ImportReport,
    GeometryFormat,
};

const MAX_FACETED_TRIANGLES: u64 = u32::MAX as u64 / 3;

#[derive(Debug, Clone, PartialEq)]
pub struct FacetedSolidImportOptions {
    pub source_units: UnitSystem,
    pub revision: GeometryRevisionIdentity,
    pub source_tolerance_m: f64,
    pub absolute_tolerance_floor_m: f64,
    pub model_relative_tolerance: f64,
    pub requested_deviation_m: f64,
    pub max_triangles: u64,
}

impl Default for FacetedSolidImportOptions {
    fn default() -> Self {
        Self {
            source_units: UnitSystem::Meter,
            revision: GeometryRevisionIdentity {
                revision: 1,
                persistent_mapping_version: 1,
                parent_document_digest: None,
            },
            source_tolerance_m: 0.0,
            absolute_tolerance_floor_m: 1.0e-12,
            model_relative_tolerance: 1.0e-12,
            requested_deviation_m: 1.0e-4,
            max_triangles: 16_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportedFacetedSolid {
    pub closure: EncodedFacetedSolidClosure,
    pub report: ImportReport,
}

pub fn import_faceted_solid(
    source_name: &str,
    bytes: &[u8],
    options: &FacetedSolidImportOptions,
    context: &GeometryImportContext,
) -> Result<ImportedFacetedSolid, GeometryImportError> {
    context.check_cancelled()?;
    let format = detect_geometry_format(source_name, bytes);
    let source_format = source_format(format)?;
    let meters_per_source_unit = meters_per_unit(options.source_units)?;
    validate_options(options)?;
    let imported = import_geometry_with_context(
        source_name,
        bytes,
        GeometryImportOptions {
            max_triangles: Some(options.max_triangles),
            budget_policy: GeometryImportBudgetPolicy::Strict,
            units: options.source_units,
            tessellation_profile: Default::default(),
            relative_deflection: false,
        },
        context,
    )?;
    let solid = topology::canonical_faceted_solid(
        &imported.asset.surface_meshes,
        meters_per_source_unit,
        context,
    )?;
    let document = GeometryDocument {
        schema_version: GEOMETRY_DOCUMENT_SCHEMA_VERSION,
        source: GeometrySourceIdentity {
            content_digest: GeometryDigest::from_bytes(Sha256::digest(bytes).into()),
            format: source_format,
            importer_version: "runmat-faceted-import/1".into(),
            kernel_version: None,
            source_units: options.source_units,
            meters_per_source_unit,
        },
        revision: options.revision.clone(),
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: options.source_tolerance_m,
            absolute_floor_m: options.absolute_tolerance_floor_m,
            model_relative_term: options.model_relative_tolerance,
            requested_deviation_m: options.requested_deviation_m,
            maximum_healing_displacement_m: 0.0,
        },
        healing: GeometryHealingPolicy {
            algorithm_version: "none/1".into(),
            sew: false,
            repair_orientation: false,
            consolidate_duplicates: false,
            repair_tolerance_scale_gaps: false,
            simplify_short_edges_and_sliver_faces: false,
        },
        model: GeometryModel::FacetedSolid {
            model: FacetedSolidModel {
                artifact: GeometryObjectRef {
                    digest: GeometryDigest::from_bytes([1; 32]),
                    encoded_length: 1,
                    media_type: FACETED_SOLID_MEDIA_TYPE.into(),
                    schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
                },
                vertex_count: solid.vertices.len() as u64,
                triangle_count: solid.triangles.len() as u64,
                shell_count: solid.shells.len() as u64,
                is_watertight: true,
                is_oriented: true,
            },
        },
        display_tessellations: Vec::new(),
    };
    let closure = build_faceted_solid_closure(document, solid)
        .map_err(|error| GeometryImportError::InvalidGeometry(error.to_string()))?;
    context.check_cancelled()?;
    Ok(ImportedFacetedSolid {
        closure,
        report: imported.report,
    })
}

fn source_format(format: GeometryFormat) -> Result<GeometrySourceFormat, GeometryImportError> {
    match format {
        GeometryFormat::Stl => Ok(GeometrySourceFormat::Stl),
        GeometryFormat::Obj => Ok(GeometrySourceFormat::Obj),
        GeometryFormat::Ply => Ok(GeometrySourceFormat::Ply),
        GeometryFormat::Gltf => Ok(GeometrySourceFormat::Gltf),
        _ => Err(GeometryImportError::UnsupportedFormat),
    }
}

fn meters_per_unit(units: UnitSystem) -> Result<f64, GeometryImportError> {
    match units {
        UnitSystem::Meter => Ok(1.0),
        UnitSystem::Millimeter => Ok(0.001),
        UnitSystem::Inch => Ok(0.0254),
        UnitSystem::Unspecified => Err(GeometryImportError::InvalidOptions(
            "faceted solid source units must be explicit".into(),
        )),
    }
}

fn validate_options(options: &FacetedSolidImportOptions) -> Result<(), GeometryImportError> {
    if options.max_triangles == 0 || options.max_triangles > MAX_FACETED_TRIANGLES {
        return Err(GeometryImportError::InvalidOptions(
            "faceted solid triangle limit must fit the canonical u32 vertex index space".into(),
        ));
    }
    GeometryTolerancePolicy {
        source_tolerance_m: options.source_tolerance_m,
        absolute_floor_m: options.absolute_tolerance_floor_m,
        model_relative_term: options.model_relative_tolerance,
        requested_deviation_m: options.requested_deviation_m,
        maximum_healing_displacement_m: 0.0,
    }
    .validate()
    .map_err(|error| GeometryImportError::InvalidOptions(error.to_string()))?;
    if options.revision.revision == 0 || options.revision.persistent_mapping_version == 0 {
        return Err(GeometryImportError::InvalidOptions(
            "faceted solid revision values must be non-zero".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
