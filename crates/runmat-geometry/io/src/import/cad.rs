use std::collections::BTreeMap;

use runmat_geometry_core::{
    AssemblyNode, CadColorEvidence, CadLabelRef, CadPhysicalMaterialEvidence, CadRegionOwnership,
    CadSemanticKind, EntityIdRange, EntityKind, Region, RegionEntityMapping, SourceGeometryKind,
    SurfaceMesh,
};

use crate::{
    cad::{parse_step_summary, StepImportSummary},
    occt::{import_cad_topology, OcctCadFormat, OcctCadTopology},
    report::{ImportDiagnostic, ImportDiagnosticSeverity, ImportResult},
    sniff::GeometryFormat,
};

use super::{
    build_asset, build_result, capacity_guard, push_entity_range, push_mesh_count_diagnostics,
    BuildAssetInput, GeometryImportContext, GeometryImportError, GeometryImportOptions,
};

pub(super) fn import_cad(
    path: &str,
    bytes: &[u8],
    format: GeometryFormat,
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<ImportResult, GeometryImportError> {
    context.check_cancelled()?;
    let Some(occt_format) = OcctCadFormat::from_geometry_format(format) else {
        return Err(GeometryImportError::UnsupportedFormat);
    };

    let metadata = match occt_format {
        OcctCadFormat::Step => Some(parse_step_metadata(path, bytes, context)?),
        OcctCadFormat::Iges | OcctCadFormat::Brep => None,
    };

    if occt_format == OcctCadFormat::Step && !step_payload_has_topology_entities(bytes) {
        return build_step_metadata_result(path, metadata.expect("STEP metadata"), options);
    }

    context.check_cancelled()?;
    if let Some(topology) = import_cad_topology(path, bytes, occt_format, &options, context)? {
        context.check_cancelled()?;
        return build_topology_result(path, occt_format, topology, metadata, options);
    }

    match occt_format {
        OcctCadFormat::Step => {
            build_step_metadata_result(path, metadata.expect("STEP metadata"), options)
        }
        OcctCadFormat::Iges | OcctCadFormat::Brep => Err(GeometryImportError::BackendUnavailable(
            format!(
                "{} CAD topology import requires the occt-native feature on native hosts or the occt-wasm-host feature with a configured sidecar on wasm hosts",
                occt_format.as_str().to_uppercase()
            ),
        )),
    }
}

fn parse_step_metadata(
    path: &str,
    bytes: &[u8],
    context: &GeometryImportContext,
) -> Result<StepImportSummary, GeometryImportError> {
    context.check_cancelled()?;
    let text = std::str::from_utf8(bytes)
        .map_err(|_| GeometryImportError::ParseFailed("invalid UTF-8 STEP payload".to_string()))?;

    let summary = parse_step_summary(path, text).map_err(|reason| {
        GeometryImportError::ParseFailed(format!("STEP parse failed: {reason}"))
    })?;
    context.check_cancelled()?;
    Ok(summary)
}

fn step_payload_has_topology_entities(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return true;
    };
    let upper = text.to_ascii_uppercase();
    [
        "ADVANCED_FACE",
        "CLOSED_SHELL",
        "EDGE_CURVE",
        "FACE_BOUND",
        "MANIFOLD_SOLID_BREP",
        "POLY_LOOP",
        "VERTEX_POINT",
    ]
    .iter()
    .any(|entity| upper.contains(entity))
}

fn build_step_metadata_result(
    path: &str,
    summary: StepImportSummary,
    options: GeometryImportOptions,
) -> Result<ImportResult, GeometryImportError> {
    let mut diagnostics = summary.diagnostics.clone();
    diagnostics.push(ImportDiagnostic {
        code: "CAD_IMPORT_TOPOLOGY_BACKEND_UNAVAILABLE".to_string(),
        severity: ImportDiagnosticSeverity::Info,
        message: "STEP import used metadata-only fallback because no OCCT CAD backend is enabled"
            .to_string(),
    });

    let mut asset = build_asset(BuildAssetInput {
        path,
        importer_version: "step/v1",
        units: options.units,
        tessellation_profile: options.tessellation_profile.clone(),
        vertex_count: 0,
        element_count: 0,
        surface_meshes: Vec::new(),
        diagnostics: diagnostics.clone(),
    });
    asset.source_geometry.kind = summary.source_kind;
    asset.source_geometry.assembly = summary.assembly;
    asset.source_geometry.material_evidence = summary.material_evidence;
    asset.regions = summary.regions;

    Ok(build_result(asset, diagnostics))
}

pub(crate) fn build_topology_result(
    path: &str,
    format: OcctCadFormat,
    topology: OcctCadTopology,
    metadata: Option<StepImportSummary>,
    options: GeometryImportOptions,
) -> Result<ImportResult, GeometryImportError> {
    capacity_guard(topology.triangles.len() as u64, &options)?;

    let mut diagnostics = metadata
        .as_ref()
        .map(|summary| summary.diagnostics.clone())
        .unwrap_or_default();
    diagnostics.push(ImportDiagnostic {
        code: "CAD_IMPORT_BACKEND".to_string(),
        severity: ImportDiagnosticSeverity::Info,
        message: format!(
            "{} import used {} topology backend",
            topology.format_name.to_uppercase(),
            topology.backend
        ),
    });
    diagnostics.push(ImportDiagnostic {
        code: "CAD_IMPORT_FACE_COUNT".to_string(),
        severity: ImportDiagnosticSeverity::Info,
        message: format!(
            "{} import resolved {} topology faces",
            topology.format_name.to_uppercase(),
            topology.faces.len()
        ),
    });
    if topology.truncated {
        diagnostics.push(ImportDiagnostic {
            code: "CAD_IMPORT_TESSELLATION_TRUNCATED".to_string(),
            severity: ImportDiagnosticSeverity::Warning,
            message: match topology.triangle_budget {
                Some(limit) => format!(
                    "{} import returned a bounded preview mesh truncated at {limit} triangles",
                    topology.format_name.to_uppercase()
                ),
                None => format!(
                    "{} import returned a bounded preview mesh truncated by the CAD backend",
                    topology.format_name.to_uppercase()
                ),
            },
        });
    }
    for warning in &topology.warnings {
        diagnostics.push(ImportDiagnostic {
            code: "CAD_IMPORT_TOPOLOGY_WARNING".to_string(),
            severity: ImportDiagnosticSeverity::Warning,
            message: warning.clone(),
        });
    }
    push_mesh_count_diagnostics(
        &mut diagnostics,
        &topology.format_name.to_uppercase(),
        topology.vertices.len() as u64,
        topology.triangles.len() as u64,
    );

    let vertex_count = topology.vertices.len() as u64;
    let triangle_count = topology.triangles.len() as u64;
    let (regions, mappings) = topology_regions(&topology);
    let importer_version = format!("cad/occt/{}/v1", format.as_str());
    let mut asset = build_asset(BuildAssetInput {
        path,
        importer_version: &importer_version,
        units: options.units,
        tessellation_profile: options.tessellation_profile.clone(),
        vertex_count,
        element_count: triangle_count,
        surface_meshes: if topology.triangles.is_empty() {
            Vec::new()
        } else {
            vec![SurfaceMesh::new(
                "mesh_1",
                topology.vertices,
                topology.triangles,
            )]
        },
        diagnostics: diagnostics.clone(),
    });

    asset.source_geometry.kind = SourceGeometryKind::Cad;
    asset.source_geometry.assembly = topology
        .assembly
        .clone()
        .or_else(|| {
            metadata
                .as_ref()
                .and_then(|summary| summary.assembly.clone())
        })
        .or_else(|| Some(default_assembly(path)));
    asset.source_geometry.material_evidence = metadata
        .as_ref()
        .map(|summary| summary.material_evidence.clone())
        .unwrap_or_default();
    asset.regions = regions;
    asset.region_entity_mappings = mappings;

    Ok(build_result(asset, diagnostics))
}

fn topology_regions(topology: &OcctCadTopology) -> (Vec<Region>, Vec<RegionEntityMapping>) {
    let mut ranges_by_face = BTreeMap::<u64, Vec<EntityIdRange>>::new();
    for (triangle_id, face_id) in topology.triangle_face_ids.iter().enumerate() {
        push_entity_range(
            ranges_by_face.entry(*face_id).or_default(),
            triangle_id as u64,
        );
    }

    let mut regions = Vec::new();
    let mut mappings = Vec::new();
    let mut semantic_ranges = BTreeMap::<String, SemanticRegionAccumulator>::new();
    for face in &topology.faces {
        let Some(ranges) = ranges_by_face.get(&face.face_id) else {
            continue;
        };
        let region_id = face_region_id(face.face_id);
        regions.push(Region {
            region_id: region_id.clone(),
            name: face.name.clone(),
            tag: Some("occt_face".to_string()),
            cad_ownership: face.ownership.clone(),
        });
        mappings.push(RegionEntityMapping::new(
            region_id,
            "mesh_1",
            EntityKind::Face,
            ranges.clone(),
        ));
        if let Some(ownership) = &face.ownership {
            accumulate_semantic_regions(&mut semantic_ranges, ownership, ranges);
        }
    }

    for (_, accumulator) in semantic_ranges {
        regions.push(accumulator.region);
        mappings.push(RegionEntityMapping::new(
            accumulator.region_id,
            "mesh_1",
            EntityKind::Face,
            accumulator.ranges,
        ));
    }

    (regions, mappings)
}

struct SemanticRegionAccumulator {
    region_id: String,
    region: Region,
    ranges: Vec<EntityIdRange>,
}

fn accumulate_semantic_regions(
    semantic_ranges: &mut BTreeMap<String, SemanticRegionAccumulator>,
    ownership: &CadRegionOwnership,
    ranges: &[EntityIdRange],
) {
    for owner in &ownership.owner_path {
        if semantic_label_is_selectable(owner) {
            let key = format!("label:{}", owner.label_entry);
            let region_id = format!("cad_label_{}", stable_region_slug(&owner.label_entry));
            let region = Region {
                region_id: region_id.clone(),
                name: if owner.name.is_empty() {
                    owner.label_entry.clone()
                } else {
                    owner.name.clone()
                },
                tag: Some(format!("cad_{}", cad_kind_tag(owner.kind))),
                cad_ownership: Some(CadRegionOwnership {
                    face_id: None,
                    label: Some(owner.clone()),
                    owner_path: vec![owner.clone()],
                    layers: Vec::new(),
                    color: None,
                    material: None,
                }),
            };
            push_semantic_ranges(semantic_ranges, key, region_id, region, ranges);
        }
    }

    for layer in &ownership.layers {
        let Some(layer) = non_empty(layer) else {
            continue;
        };
        let key = format!("layer:{layer}");
        let region_id = format!("cad_layer_{}", stable_region_slug(layer));
        let region = Region {
            region_id: region_id.clone(),
            name: layer.to_string(),
            tag: Some("cad_layer".to_string()),
            cad_ownership: Some(CadRegionOwnership {
                face_id: None,
                label: Some(CadLabelRef {
                    label_entry: format!("layer:{layer}"),
                    name: layer.to_string(),
                    kind: CadSemanticKind::Layer,
                }),
                owner_path: Vec::new(),
                layers: vec![layer.to_string()],
                color: None,
                material: None,
            }),
        };
        push_semantic_ranges(semantic_ranges, key, region_id, region, ranges);
    }

    if let Some(color) = &ownership.color {
        if let Some(hex) = non_empty(&color.hex_rgba) {
            let key = format!("color:{hex}");
            let region_id = format!("cad_color_{}", stable_region_slug(hex));
            let region = Region {
                region_id: region_id.clone(),
                name: hex.to_string(),
                tag: Some("cad_color".to_string()),
                cad_ownership: Some(CadRegionOwnership {
                    face_id: None,
                    label: Some(CadLabelRef {
                        label_entry: format!("color:{hex}"),
                        name: hex.to_string(),
                        kind: CadSemanticKind::Color,
                    }),
                    owner_path: Vec::new(),
                    layers: Vec::new(),
                    color: Some(CadColorEvidence {
                        source: color.source.clone(),
                        color_type: color.color_type.clone(),
                        hex_rgba: color.hex_rgba.clone(),
                    }),
                    material: None,
                }),
            };
            push_semantic_ranges(semantic_ranges, key, region_id, region, ranges);
        }
    }

    if let Some(material) = &ownership.material {
        if let Some(name) = non_empty(&material.name) {
            let key = format!("material:{name}");
            let region_id = format!("cad_material_{}", stable_region_slug(name));
            let region = Region {
                region_id: region_id.clone(),
                name: name.to_string(),
                tag: Some("cad_material".to_string()),
                cad_ownership: Some(CadRegionOwnership {
                    face_id: None,
                    label: Some(CadLabelRef {
                        label_entry: if material.label_entry.is_empty() {
                            format!("material:{name}")
                        } else {
                            material.label_entry.clone()
                        },
                        name: name.to_string(),
                        kind: CadSemanticKind::Material,
                    }),
                    owner_path: Vec::new(),
                    layers: Vec::new(),
                    color: None,
                    material: Some(CadPhysicalMaterialEvidence {
                        label_entry: material.label_entry.clone(),
                        name: material.name.clone(),
                        description: material.description.clone(),
                        density: material.density.clone(),
                        density_name: material.density_name.clone(),
                        density_value_type: material.density_value_type.clone(),
                    }),
                }),
            };
            push_semantic_ranges(semantic_ranges, key, region_id, region, ranges);
        }
    }
}

fn push_semantic_ranges(
    semantic_ranges: &mut BTreeMap<String, SemanticRegionAccumulator>,
    key: String,
    region_id: String,
    region: Region,
    ranges: &[EntityIdRange],
) {
    let accumulator = semantic_ranges
        .entry(key)
        .or_insert_with(|| SemanticRegionAccumulator {
            region_id,
            region,
            ranges: Vec::new(),
        });
    for range in ranges {
        push_range(&mut accumulator.ranges, *range);
    }
}

fn push_range(ranges: &mut Vec<EntityIdRange>, range: EntityIdRange) {
    if range.count == 0 {
        return;
    }
    if let Some(last) = ranges.last_mut() {
        if last.end_exclusive() == Some(range.start) {
            last.count = last.count.saturating_add(range.count);
            return;
        }
    }
    ranges.push(range);
}

fn semantic_label_is_selectable(label: &CadLabelRef) -> bool {
    !label.name.trim().is_empty()
        && !matches!(
            label.kind,
            CadSemanticKind::Face | CadSemanticKind::Subshape | CadSemanticKind::Unknown
        )
}

fn cad_kind_tag(kind: CadSemanticKind) -> &'static str {
    match kind {
        CadSemanticKind::Assembly => "assembly",
        CadSemanticKind::Component => "component",
        CadSemanticKind::Reference => "reference",
        CadSemanticKind::Body => "body",
        CadSemanticKind::Compound => "compound",
        CadSemanticKind::Face => "face",
        CadSemanticKind::Subshape => "subshape",
        CadSemanticKind::Layer => "layer",
        CadSemanticKind::Color => "color",
        CadSemanticKind::Material => "material",
        CadSemanticKind::Shape => "shape",
        CadSemanticKind::Unknown => "unknown",
    }
}

fn non_empty(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn stable_region_slug(value: &str) -> String {
    let mut slug = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
        } else if !slug.ends_with('_') {
            slug.push('_');
        }
    }
    let slug = slug.trim_matches('_').to_string();
    if slug.is_empty() {
        "unnamed".to_string()
    } else {
        slug
    }
}

fn face_region_id(face_id: u64) -> String {
    format!("face_{:06}", face_id + 1)
}

fn default_assembly(path: &str) -> AssemblyNode {
    AssemblyNode {
        node_id: "assembly_root".to_string(),
        label: path
            .rsplit('/')
            .next()
            .unwrap_or(path)
            .trim_end_matches(".iges")
            .trim_end_matches(".igs")
            .trim_end_matches(".brep")
            .trim_end_matches(".brp")
            .to_string(),
        children: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occt::{OcctCadFace, OcctCadTopology};

    #[test]
    fn topology_regions_include_cad_semantic_ownership_regions() {
        let topology = OcctCadTopology {
            backend: "test".to_string(),
            format_name: "step".to_string(),
            truncated: false,
            triangle_budget: None,
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
            triangle_face_ids: vec![0],
            faces: vec![OcctCadFace {
                face_id: 0,
                name: "Mount Face".to_string(),
                ownership: Some(CadRegionOwnership {
                    face_id: Some(0),
                    label: Some(CadLabelRef {
                        label_entry: "0:1:1:7".to_string(),
                        name: "Mount Face".to_string(),
                        kind: CadSemanticKind::Face,
                    }),
                    owner_path: vec![
                        CadLabelRef {
                            label_entry: "0:1".to_string(),
                            name: "Assembly A".to_string(),
                            kind: CadSemanticKind::Assembly,
                        },
                        CadLabelRef {
                            label_entry: "0:1:1".to_string(),
                            name: "Bracket".to_string(),
                            kind: CadSemanticKind::Body,
                        },
                    ],
                    layers: vec!["Boundary Faces".to_string()],
                    color: Some(CadColorEvidence {
                        source: "occt_xcaf".to_string(),
                        color_type: "surface".to_string(),
                        hex_rgba: "#FFAA00FF".to_string(),
                    }),
                    material: Some(CadPhysicalMaterialEvidence {
                        label_entry: "0:5:1".to_string(),
                        name: "Aluminum 6061".to_string(),
                        description: Some("test material".to_string()),
                        density: Some("2.7".to_string()),
                        density_name: Some("density".to_string()),
                        density_value_type: Some("g/cm3".to_string()),
                    }),
                }),
            }],
            assembly: Some(AssemblyNode {
                node_id: "0:1".to_string(),
                label: "Assembly A".to_string(),
                children: Vec::new(),
            }),
            warnings: Vec::new(),
        };

        let (regions, mappings) = topology_regions(&topology);
        assert!(regions
            .iter()
            .any(|region| region.region_id == "face_000001" && region.cad_ownership.is_some()));
        assert!(regions
            .iter()
            .any(|region| region.region_id == "cad_label_0_1"
                && region.tag.as_deref() == Some("cad_assembly")));
        assert!(regions
            .iter()
            .any(|region| region.region_id == "cad_label_0_1_1"
                && region.name == "Bracket"
                && region.tag.as_deref() == Some("cad_body")));
        assert!(regions
            .iter()
            .any(|region| region.region_id == "cad_layer_boundary_faces"));
        assert!(regions
            .iter()
            .any(|region| region.region_id == "cad_color_ffaa00ff"));
        assert!(regions
            .iter()
            .any(|region| region.region_id == "cad_material_aluminum_6061"));
        assert!(mappings.iter().any(|mapping| {
            mapping.region_id == "cad_material_aluminum_6061" && mapping.entity_count() == 1
        }));
    }

    #[test]
    fn topology_result_preserves_truncated_preview_as_warning() {
        let topology = OcctCadTopology {
            backend: "test".to_string(),
            format_name: "step".to_string(),
            truncated: true,
            triangle_budget: Some(1),
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
            triangle_face_ids: vec![0],
            faces: vec![OcctCadFace {
                face_id: 0,
                name: "Face 1".to_string(),
                ownership: None,
            }],
            assembly: None,
            warnings: Vec::new(),
        };
        let options = GeometryImportOptions {
            max_triangles: Some(1),
            budget_policy: crate::GeometryImportBudgetPolicy::Truncate,
            units: runmat_geometry_core::UnitSystem::Meter,
            tessellation_profile: Default::default(),
            relative_deflection: true,
        };

        let result =
            build_topology_result("/part.step", OcctCadFormat::Step, topology, None, options)
                .expect("truncated preview topology should import");

        assert_eq!(result.asset.surface_meshes[0].triangles.len(), 1);
        assert!(result
            .asset
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "CAD_IMPORT_TESSELLATION_TRUNCATED"));
    }
}
