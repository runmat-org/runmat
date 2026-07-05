use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    CadCurveEvaluator, CadFaceEvaluator, CadSemanticKind, EntityKind, GeometryAsset,
    SourceGeometryKind,
};

use super::CadTopologySource;

pub(super) fn cad_topology_source(geometry: &GeometryAsset) -> CadTopologySource {
    if geometry.source_geometry.kind != SourceGeometryKind::Cad {
        return CadTopologySource::MeshFallback;
    }
    if geometry
        .regions
        .iter()
        .any(|region| region.cad_ownership.is_some())
    {
        CadTopologySource::SemanticCad
    } else {
        CadTopologySource::GenericCadMesh
    }
}

pub(super) fn semantic_face_regions(geometry: &GeometryAsset) -> BTreeSet<String> {
    geometry
        .regions
        .iter()
        .filter(|region| {
            region.cad_ownership.as_ref().is_some_and(|ownership| {
                ownership.face_id.is_some()
                    || ownership
                        .label
                        .as_ref()
                        .is_some_and(|label| label.kind == CadSemanticKind::Face)
            })
        })
        .map(|region| region.region_id.clone())
        .collect()
}

pub(super) fn imported_face_ids_by_region(geometry: &GeometryAsset) -> BTreeMap<String, u64> {
    geometry
        .regions
        .iter()
        .filter_map(|region| {
            region
                .cad_ownership
                .as_ref()
                .and_then(|ownership| ownership.face_id)
                .map(|face_id| (region.region_id.clone(), face_id))
        })
        .collect()
}

pub(super) fn imported_curve_ids_by_region(geometry: &GeometryAsset) -> BTreeMap<String, u64> {
    geometry
        .regions
        .iter()
        .filter_map(|region| {
            region
                .cad_ownership
                .as_ref()
                .and_then(|ownership| ownership.curve_id)
                .map(|curve_id| (region.region_id.clone(), curve_id))
        })
        .collect()
}

pub(super) fn imported_face_id_for_regions(
    imported_face_ids_by_region: &BTreeMap<String, u64>,
    region_ids: &[String],
) -> Option<u64> {
    region_ids
        .iter()
        .find_map(|region_id| imported_face_ids_by_region.get(region_id).copied())
}

pub(super) fn imported_curve_id_for_regions(
    imported_curve_ids_by_region: &BTreeMap<String, u64>,
    region_ids: &[String],
) -> Option<u64> {
    region_ids
        .iter()
        .find_map(|region_id| imported_curve_ids_by_region.get(region_id).copied())
}

pub(super) fn evaluator_faces_by_imported_id(
    geometry: &GeometryAsset,
) -> BTreeMap<u64, &CadFaceEvaluator> {
    geometry
        .source_geometry
        .cad_evaluators
        .iter()
        .flat_map(|evaluator| {
            evaluator
                .faces
                .iter()
                .map(|face| (face.imported_face_id, face))
        })
        .collect()
}

pub(super) fn evaluator_curves_by_imported_id(
    geometry: &GeometryAsset,
) -> BTreeMap<u64, &CadCurveEvaluator> {
    geometry
        .source_geometry
        .cad_evaluators
        .iter()
        .flat_map(|evaluator| {
            evaluator
                .curves
                .iter()
                .map(|curve| (curve.imported_curve_id, curve))
        })
        .collect()
}

pub(super) fn face_regions_by_source_face(geometry: &GeometryAsset) -> BTreeMap<u32, Vec<String>> {
    let mut regions = BTreeMap::<u32, BTreeSet<String>>::new();
    for mapping in &geometry.region_entity_mappings {
        if !matches!(mapping.entity_kind, EntityKind::Face | EntityKind::Element) {
            continue;
        }
        for range in &mapping.ranges {
            for entity_id in range.start..range.start.saturating_add(range.count) {
                regions
                    .entry(entity_id as u32)
                    .or_default()
                    .insert(mapping.region_id.clone());
            }
        }
    }
    regions
        .into_iter()
        .map(|(face_id, region_ids)| (face_id, region_ids.into_iter().collect()))
        .collect()
}

pub(super) fn material_regions_by_source_face(
    geometry: &GeometryAsset,
) -> BTreeMap<u32, Vec<String>> {
    let material_region_ids = geometry
        .regions
        .iter()
        .filter(|region| region.has_material_role())
        .map(|region| region.region_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut regions = BTreeMap::<u32, BTreeSet<String>>::new();
    for mapping in &geometry.region_entity_mappings {
        if !matches!(mapping.entity_kind, EntityKind::Face | EntityKind::Element)
            || !material_region_ids.contains(mapping.region_id.as_str())
        {
            continue;
        }
        for range in &mapping.ranges {
            for entity_id in range.start..range.start.saturating_add(range.count) {
                regions
                    .entry(entity_id as u32)
                    .or_default()
                    .insert(mapping.region_id.clone());
            }
        }
    }
    regions
        .into_iter()
        .map(|(face_id, region_ids)| (face_id, region_ids.into_iter().collect()))
        .collect()
}

pub(super) fn edge_regions_by_source_edge(geometry: &GeometryAsset) -> BTreeMap<u32, Vec<String>> {
    let mut regions = BTreeMap::<u32, BTreeSet<String>>::new();
    for mapping in &geometry.region_entity_mappings {
        if !matches!(mapping.entity_kind, EntityKind::Edge) {
            continue;
        }
        for range in &mapping.ranges {
            for entity_id in range.start..range.start.saturating_add(range.count) {
                regions
                    .entry(entity_id as u32)
                    .or_default()
                    .insert(mapping.region_id.clone());
            }
        }
    }
    regions
        .into_iter()
        .map(|(edge_id, region_ids)| (edge_id, region_ids.into_iter().collect()))
        .collect()
}
