use runmat_geometry_core::{GeometryAsset, MeshKind};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingProfile {
    SurfaceOnly,
    AnalysisReady,
    AdaptiveRefine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshingOptions {
    pub profile: MeshingProfile,
    pub target_element_budget: usize,
}

impl Default for MeshingOptions {
    fn default() -> Self {
        Self {
            profile: MeshingProfile::AnalysisReady,
            target_element_budget: 250_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshConnectivityClass {
    SparseBand,
    SurfacePatch,
    VolumeCore,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementFamilyHint {
    Triangle,
    Quad,
    Tet,
    Hex,
    Mixed,
}

fn default_coordinate_span_m() -> [f64; 3] {
    [1.0, 0.0, 0.0]
}

fn default_coordinate_active_dimension_count() -> u8 {
    1
}

fn default_coordinate_characteristic_length_m() -> f64 {
    1.0
}

fn default_zero_u64() -> u64 {
    0
}

fn default_zero_f64() -> f64 {
    0.0
}

fn default_reference_element_coordinates_m() -> [[f64; 3]; 3] {
    [[0.0; 3]; 3]
}

fn default_element_topology_sample_edge_nodes() -> [[u32; 2]; 8] {
    [[0; 2]; 8]
}

fn default_element_topology_sample_node_coordinates_m() -> [[f64; 3]; 8] {
    [[0.0; 3]; 8]
}

fn default_element_topology_sample_element_edges() -> [[u32; 3]; 4] {
    [[0; 3]; 4]
}

fn default_element_topology_sample_element_orientations() -> [[i8; 3]; 4] {
    [[0; 3]; 4]
}

fn default_element_topology_sample_element_areas_m2() -> [f64; 4] {
    [0.0; 4]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedMeshDescriptor {
    pub prepared_mesh_id: String,
    pub source_mesh_id: String,
    pub kind: MeshKind,
    pub node_count: u64,
    pub element_count: u64,
    pub connectivity_class: MeshConnectivityClass,
    pub element_family_hint: ElementFamilyHint,
    pub region_span_hint: u32,
    #[serde(default = "default_coordinate_span_m")]
    pub coordinate_span_m: [f64; 3],
    #[serde(default = "default_coordinate_active_dimension_count")]
    pub coordinate_active_dimension_count: u8,
    #[serde(default = "default_coordinate_characteristic_length_m")]
    pub coordinate_characteristic_length_m: f64,
    #[serde(default = "default_zero_u64")]
    pub element_geometry_node_count: u64,
    #[serde(default = "default_zero_u64")]
    pub element_geometry_edge_count: u64,
    #[serde(default = "default_zero_f64")]
    pub mean_element_edge_length_m: f64,
    #[serde(default = "default_zero_f64")]
    pub mean_element_area_m2: f64,
    #[serde(default = "default_zero_f64")]
    pub element_geometry_coverage_ratio: f64,
    #[serde(default = "default_reference_element_coordinates_m")]
    pub reference_element_coordinates_m: [[f64; 3]; 3],
    #[serde(default = "default_zero_f64")]
    pub reference_element_area_m2: f64,
    #[serde(default = "default_zero_u64")]
    pub control_volume_cell_count: u64,
    #[serde(default = "default_zero_u64")]
    pub control_volume_face_count: u64,
    #[serde(default = "default_zero_u64")]
    pub control_volume_internal_face_count: u64,
    #[serde(default = "default_zero_u64")]
    pub control_volume_boundary_face_count: u64,
    #[serde(default = "default_zero_f64")]
    pub control_volume_connectivity_coverage_ratio: f64,
    #[serde(default = "default_zero_u64")]
    pub element_topology_sample_element_count: u64,
    #[serde(default = "default_zero_u64")]
    pub element_topology_sample_edge_count: u64,
    #[serde(default = "default_element_topology_sample_edge_nodes")]
    pub element_topology_sample_edge_nodes: [[u32; 2]; 8],
    #[serde(default = "default_element_topology_sample_node_coordinates_m")]
    pub element_topology_sample_node_coordinates_m: [[f64; 3]; 8],
    #[serde(default = "default_element_topology_sample_element_edges")]
    pub element_topology_sample_element_edges: [[u32; 3]; 4],
    #[serde(default = "default_element_topology_sample_element_orientations")]
    pub element_topology_sample_element_orientations: [[i8; 3]; 4],
    #[serde(default = "default_element_topology_sample_element_areas_m2")]
    pub element_topology_sample_element_areas_m2: [f64; 4],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionMeshMapping {
    pub region_id: String,
    pub source_mesh_ids: Vec<String>,
    pub prepared_mesh_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshingQualityReport {
    pub min_scaled_jacobian: f64,
    pub mean_aspect_ratio: f64,
    pub inverted_element_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshingProvenance {
    pub algorithm: String,
    pub profile: MeshingProfile,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshingPrepResult {
    pub schema_version: String,
    pub prepared_meshes: Vec<PreparedMeshDescriptor>,
    pub region_mappings: Vec<RegionMeshMapping>,
    pub quality: MeshingQualityReport,
    pub provenance: MeshingProvenance,
}

pub fn prepare_geometry_for_analysis(
    geometry: &GeometryAsset,
    options: MeshingOptions,
) -> Result<MeshingPrepResult, String> {
    if geometry.meshes.is_empty() {
        return Err("geometry has no meshes to prepare".to_string());
    }
    if options.target_element_budget == 0 {
        return Err("target_element_budget must be greater than zero".to_string());
    }

    let mut source_meshes = geometry.meshes.clone();
    source_meshes.sort_by(|a, b| a.mesh_id.cmp(&b.mesh_id));
    let per_mesh_budget =
        (options.target_element_budget / source_meshes.len().max(1)).max(1) as u64;

    let mut prepared_meshes = Vec::with_capacity(source_meshes.len());
    for mesh in source_meshes {
        let profile_scale = match options.profile {
            MeshingProfile::SurfaceOnly => 1.0,
            MeshingProfile::AnalysisReady => {
                if mesh.kind == MeshKind::Surface {
                    1.4
                } else {
                    1.1
                }
            }
            MeshingProfile::AdaptiveRefine => {
                if mesh.kind == MeshKind::Surface {
                    1.8
                } else {
                    1.35
                }
            }
        };
        let proposed = ((mesh.element_count as f64) * profile_scale).round() as u64;
        let min_refined_elements = match options.profile {
            MeshingProfile::AdaptiveRefine => mesh.element_count.max(64),
            MeshingProfile::AnalysisReady | MeshingProfile::SurfaceOnly => 1,
        };
        let element_count = proposed
            .max(min_refined_elements)
            .min(per_mesh_budget.max(mesh.element_count));
        let node_count = (mesh.vertex_count.max(3)).max(element_count / 2 + 2);
        let connectivity_class = match mesh.kind {
            MeshKind::Surface => {
                if element_count > 20_000 {
                    MeshConnectivityClass::SparseBand
                } else {
                    MeshConnectivityClass::SurfacePatch
                }
            }
            MeshKind::Volume => MeshConnectivityClass::VolumeCore,
        };
        let element_family_hint = match mesh.kind {
            MeshKind::Surface => {
                if element_count % 2 == 0 {
                    ElementFamilyHint::Triangle
                } else {
                    ElementFamilyHint::Quad
                }
            }
            MeshKind::Volume => {
                if element_count % 2 == 0 {
                    ElementFamilyHint::Tet
                } else {
                    ElementFamilyHint::Hex
                }
            }
        };
        let coordinate_span_m = mesh_coordinate_span_m(geometry, &mesh.mesh_id);
        let coordinate_active_dimension_count =
            coordinate_active_dimension_count(coordinate_span_m);
        let coordinate_characteristic_length_m = coordinate_characteristic_length_m(
            coordinate_span_m,
            coordinate_active_dimension_count,
            node_count,
        );
        let element_geometry = mesh_element_geometry_metrics(geometry, &mesh.mesh_id);
        let region_span_hint = (geometry.regions.len().max(1) as u32)
            .clamp(1, 64)
            .saturating_sub((prepared_meshes.len() as u32) % 2);
        prepared_meshes.push(PreparedMeshDescriptor {
            prepared_mesh_id: format!("prep_{}_{}", geometry.revision, mesh.mesh_id),
            source_mesh_id: mesh.mesh_id,
            kind: mesh.kind,
            node_count,
            element_count,
            connectivity_class,
            element_family_hint,
            region_span_hint,
            coordinate_span_m,
            coordinate_active_dimension_count,
            coordinate_characteristic_length_m,
            element_geometry_node_count: element_geometry.node_count,
            element_geometry_edge_count: element_geometry.edge_count,
            mean_element_edge_length_m: element_geometry.mean_edge_length_m,
            mean_element_area_m2: element_geometry.mean_area_m2,
            element_geometry_coverage_ratio: element_geometry.coverage_ratio,
            reference_element_coordinates_m: element_geometry.reference_coordinates_m,
            reference_element_area_m2: element_geometry.reference_area_m2,
            control_volume_cell_count: element_geometry.control_volume_cell_count,
            control_volume_face_count: element_geometry.control_volume_face_count,
            control_volume_internal_face_count: element_geometry.control_volume_internal_face_count,
            control_volume_boundary_face_count: element_geometry.control_volume_boundary_face_count,
            control_volume_connectivity_coverage_ratio: element_geometry.coverage_ratio,
            element_topology_sample_element_count: element_geometry
                .element_topology_sample
                .element_count,
            element_topology_sample_edge_count: element_geometry.element_topology_sample.edge_count,
            element_topology_sample_edge_nodes: element_geometry.element_topology_sample.edge_nodes,
            element_topology_sample_node_coordinates_m: element_geometry
                .element_topology_sample
                .node_coordinates_m,
            element_topology_sample_element_edges: element_geometry
                .element_topology_sample
                .element_edges,
            element_topology_sample_element_orientations: element_geometry
                .element_topology_sample
                .element_orientations,
            element_topology_sample_element_areas_m2: element_geometry
                .element_topology_sample
                .element_areas_m2,
        });
    }

    let mut prepared_by_source = BTreeMap::<String, String>::new();
    for prepared in &prepared_meshes {
        prepared_by_source.insert(
            prepared.source_mesh_id.clone(),
            prepared.prepared_mesh_id.clone(),
        );
    }

    let mut source_mesh_ids_by_region = BTreeMap::<String, Vec<String>>::new();
    for mapping in &geometry.region_entity_mappings {
        let entry = source_mesh_ids_by_region
            .entry(mapping.region_id.clone())
            .or_default();
        if !entry.iter().any(|mesh_id| mesh_id == &mapping.mesh_id) {
            entry.push(mapping.mesh_id.clone());
        }
    }
    for mesh_ids in source_mesh_ids_by_region.values_mut() {
        mesh_ids.sort();
    }

    let fallback_source_mesh_ids = prepared_meshes
        .iter()
        .map(|mesh| mesh.source_mesh_id.clone())
        .collect::<Vec<_>>();
    let fallback_prepared_mesh_ids = prepared_meshes
        .iter()
        .map(|mesh| mesh.prepared_mesh_id.clone())
        .collect::<Vec<_>>();

    let mut region_mappings = Vec::<RegionMeshMapping>::new();
    for region in &geometry.regions {
        let source_mesh_ids = source_mesh_ids_by_region
            .get(&region.region_id)
            .cloned()
            .filter(|mesh_ids| !mesh_ids.is_empty())
            .unwrap_or_else(|| fallback_source_mesh_ids.clone());
        let mut prepared_mesh_ids = source_mesh_ids
            .iter()
            .filter_map(|mesh_id| prepared_by_source.get(mesh_id).cloned())
            .collect::<Vec<_>>();
        if prepared_mesh_ids.is_empty() {
            prepared_mesh_ids = fallback_prepared_mesh_ids.clone();
        }
        region_mappings.push(RegionMeshMapping {
            region_id: region.region_id.clone(),
            source_mesh_ids,
            prepared_mesh_ids,
        });
    }
    if region_mappings.is_empty() {
        let source_mesh_ids = prepared_meshes
            .iter()
            .map(|mesh| mesh.source_mesh_id.clone())
            .collect::<Vec<_>>();
        let prepared_mesh_ids = prepared_meshes
            .iter()
            .map(|mesh| mesh.prepared_mesh_id.clone())
            .collect::<Vec<_>>();
        region_mappings.push(RegionMeshMapping {
            region_id: "region_default".to_string(),
            source_mesh_ids: source_mesh_ids.clone(),
            prepared_mesh_ids: prepared_mesh_ids.clone(),
        });
    }
    region_mappings.sort_by(|a, b| a.region_id.cmp(&b.region_id));

    let total_elements = prepared_meshes
        .iter()
        .map(|mesh| mesh.element_count)
        .sum::<u64>()
        .max(1);
    let total_nodes = prepared_meshes
        .iter()
        .map(|mesh| mesh.node_count)
        .sum::<u64>()
        .max(1);
    let element_density = total_elements as f64 / total_nodes as f64;
    let normalized_density = element_density.min(1.0);
    let (min_scaled_jacobian, mean_aspect_ratio) = match options.profile {
        MeshingProfile::SurfaceOnly => (
            (0.89 - 0.1 * normalized_density).max(0.5),
            1.4 + normalized_density,
        ),
        MeshingProfile::AnalysisReady => (
            (0.92 - 0.1 * normalized_density).max(0.5),
            1.2 + normalized_density,
        ),
        MeshingProfile::AdaptiveRefine => (
            (0.95 - 0.03 * normalized_density).clamp(0.5, 0.99),
            1.05 + 0.15 * normalized_density,
        ),
    };

    Ok(MeshingPrepResult {
        schema_version: "geometry-prep-for-analysis/v1".to_string(),
        prepared_meshes,
        region_mappings,
        quality: MeshingQualityReport {
            min_scaled_jacobian,
            mean_aspect_ratio,
            inverted_element_count: 0,
        },
        provenance: MeshingProvenance {
            algorithm: "deterministic_topology_seed/v1".to_string(),
            profile: options.profile,
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
        },
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct ElementGeometryMetrics {
    node_count: u64,
    edge_count: u64,
    mean_edge_length_m: f64,
    mean_area_m2: f64,
    coverage_ratio: f64,
    reference_coordinates_m: [[f64; 3]; 3],
    reference_area_m2: f64,
    control_volume_cell_count: u64,
    control_volume_face_count: u64,
    control_volume_internal_face_count: u64,
    control_volume_boundary_face_count: u64,
    element_topology_sample: ElementTopologySample,
}

#[derive(Debug, Clone, Copy, Default)]
struct ElementTopologySample {
    element_count: u64,
    edge_count: u64,
    edge_nodes: [[u32; 2]; 8],
    node_coordinates_m: [[f64; 3]; 8],
    element_edges: [[u32; 3]; 4],
    element_orientations: [[i8; 3]; 4],
    element_areas_m2: [f64; 4],
}

fn mesh_element_geometry_metrics(
    geometry: &GeometryAsset,
    mesh_id: &str,
) -> ElementGeometryMetrics {
    let Some(surface) = geometry
        .surface_meshes
        .iter()
        .find(|surface| surface.mesh_id == mesh_id)
    else {
        return ElementGeometryMetrics::default();
    };
    let Some(descriptor) = geometry.meshes.iter().find(|mesh| mesh.mesh_id == mesh_id) else {
        return ElementGeometryMetrics::default();
    };

    let mut referenced_nodes = BTreeSet::<u32>::new();
    let mut unique_edges = BTreeSet::<(u32, u32)>::new();
    let mut edge_length_sum = 0.0_f64;
    let mut edge_length_count = 0_u64;
    let mut area_sum = 0.0_f64;
    let mut valid_triangle_count = 0_u64;
    let mut edge_incidence = BTreeMap::<(u32, u32), u64>::new();
    let mut edge_indices = BTreeMap::<(u32, u32), u32>::new();
    let mut element_topology_sample = ElementTopologySample::default();
    let mut reference_coordinates_m = [[0.0_f64; 3]; 3];
    let mut reference_area_m2 = 0.0_f64;
    for triangle in &surface.triangles {
        let indices = [triangle[0], triangle[1], triangle[2]];
        let Some(vertices) = triangle_vertices(&surface.vertices, indices) else {
            continue;
        };
        valid_triangle_count += 1;
        let triangle_area = triangle_area_m2(vertices);
        if reference_area_m2 == 0.0 && triangle_area.is_finite() && triangle_area > 0.0 {
            reference_coordinates_m = vertices;
            reference_area_m2 = triangle_area;
        }
        for index in indices {
            referenced_nodes.insert(index);
        }
        for (index, vertex) in indices.into_iter().zip(vertices) {
            if (index as usize) < element_topology_sample.node_coordinates_m.len() {
                element_topology_sample.node_coordinates_m[index as usize] = vertex;
            }
        }
        for (left, right) in [
            (indices[0], indices[1]),
            (indices[1], indices[2]),
            (indices[2], indices[0]),
        ] {
            let edge = (left.min(right), left.max(right));
            unique_edges.insert(edge);
            *edge_incidence.entry(edge).or_insert(0) += 1;
            if !edge_indices.contains_key(&edge) && edge_indices.len() < 8 {
                let edge_index = edge_indices.len() as u32;
                edge_indices.insert(edge, edge_index);
                element_topology_sample.edge_nodes[edge_index as usize] = [edge.0, edge.1];
                element_topology_sample.edge_count = edge_indices.len() as u64;
            }
        }
        if (element_topology_sample.element_count as usize) < 4 {
            let element_index = element_topology_sample.element_count as usize;
            for (local_index, (left, right)) in [
                (indices[0], indices[1]),
                (indices[1], indices[2]),
                (indices[2], indices[0]),
            ]
            .into_iter()
            .enumerate()
            {
                let edge = (left.min(right), left.max(right));
                element_topology_sample.element_edges[element_index][local_index] =
                    *edge_indices.get(&edge).unwrap_or(&0);
                element_topology_sample.element_orientations[element_index][local_index] =
                    if left <= right { 1 } else { -1 };
            }
            element_topology_sample.element_areas_m2[element_index] = triangle_area;
            element_topology_sample.element_count += 1;
        }
        for (left, right) in [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0]),
        ] {
            edge_length_sum += distance_m(left, right);
            edge_length_count += 1;
        }
        area_sum += triangle_area;
    }

    let control_volume_internal_face_count =
        edge_incidence.values().filter(|count| **count > 1).count() as u64;
    let control_volume_boundary_face_count =
        edge_incidence.values().filter(|count| **count == 1).count() as u64;

    ElementGeometryMetrics {
        node_count: referenced_nodes.len() as u64,
        edge_count: unique_edges.len() as u64,
        mean_edge_length_m: if edge_length_count == 0 {
            0.0
        } else {
            edge_length_sum / edge_length_count as f64
        },
        mean_area_m2: if valid_triangle_count == 0 {
            0.0
        } else {
            area_sum / valid_triangle_count as f64
        },
        coverage_ratio: if descriptor.element_count == 0 {
            0.0
        } else {
            (valid_triangle_count as f64 / descriptor.element_count as f64).clamp(0.0, 1.0)
        },
        reference_coordinates_m,
        reference_area_m2,
        control_volume_cell_count: valid_triangle_count,
        control_volume_face_count: unique_edges.len() as u64,
        control_volume_internal_face_count,
        control_volume_boundary_face_count,
        element_topology_sample,
    }
}

fn triangle_vertices(vertices: &[[f64; 3]], indices: [u32; 3]) -> Option<[[f64; 3]; 3]> {
    let a = *vertices.get(indices[0] as usize)?;
    let b = *vertices.get(indices[1] as usize)?;
    let c = *vertices.get(indices[2] as usize)?;
    Some([a, b, c])
}

fn distance_m(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((right[0] - left[0]).powi(2) + (right[1] - left[1]).powi(2) + (right[2] - left[2]).powi(2))
        .sqrt()
}

fn triangle_area_m2(vertices: [[f64; 3]; 3]) -> f64 {
    let ab = [
        vertices[1][0] - vertices[0][0],
        vertices[1][1] - vertices[0][1],
        vertices[1][2] - vertices[0][2],
    ];
    let ac = [
        vertices[2][0] - vertices[0][0],
        vertices[2][1] - vertices[0][1],
        vertices[2][2] - vertices[0][2],
    ];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

fn mesh_coordinate_span_m(geometry: &GeometryAsset, mesh_id: &str) -> [f64; 3] {
    let Some(surface) = geometry
        .surface_meshes
        .iter()
        .find(|surface| surface.mesh_id == mesh_id)
    else {
        return default_coordinate_span_m();
    };
    let Some(first) = surface.vertices.first().copied() else {
        return default_coordinate_span_m();
    };
    let mut min = first;
    let mut max = first;
    for vertex in &surface.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    [
        finite_positive_or_default(max[0] - min[0], 0.0),
        finite_positive_or_default(max[1] - min[1], 0.0),
        finite_positive_or_default(max[2] - min[2], 0.0),
    ]
}

fn coordinate_active_dimension_count(span_m: [f64; 3]) -> u8 {
    span_m
        .iter()
        .filter(|span| span.is_finite() && **span > 1.0e-12)
        .count()
        .max(1) as u8
}

fn coordinate_characteristic_length_m(
    span_m: [f64; 3],
    active_dimension_count: u8,
    node_count: u64,
) -> f64 {
    let active_spans = span_m
        .into_iter()
        .filter(|span| span.is_finite() && *span > 1.0e-12)
        .collect::<Vec<_>>();
    if active_spans.is_empty() {
        return default_coordinate_characteristic_length_m();
    }
    let domain_measure = active_spans.iter().product::<f64>();
    let node_scale = (node_count.max(2) as f64).powf(1.0 / active_dimension_count.max(1) as f64);
    finite_positive_or_default(
        domain_measure.powf(1.0 / active_dimension_count as f64) / node_scale,
        1.0,
    )
}

fn finite_positive_or_default(value: f64, default: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        GeometrySource, MaterialEvidence, MeshDescriptor, Region, RegionEntityMapping,
        SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
    };

    fn sample_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_meshing_test".to_string(),
            source: GeometrySource {
                path: "/fixtures/cube.stl".to_string(),
                sha256: "mesh-test".to_string(),
                importer_version: "stl/v1".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Mesh,
                assembly: None,
                material_evidence: vec![MaterialEvidence {
                    source_key: "fixture".to_string(),
                    normalized_key: "fixture".to_string(),
                    value: "steel".to_string(),
                    confidence: runmat_geometry_core::MaterialEvidenceConfidence::High,
                    unit_basis: None,
                    assumptions: Vec::new(),
                }],
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 7,
            meshes: vec![MeshDescriptor {
                mesh_id: "mesh_a".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 120,
                element_count: 200,
            }],
            surface_meshes: Vec::new(),
            regions: vec![Region {
                region_id: "region_main".to_string(),
                name: "main".to_string(),
                tag: None,
                cad_ownership: None,
            }],
            region_entity_mappings: vec![RegionEntityMapping::all_faces(
                "region_main",
                "mesh_a",
                200,
            )],
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn meshing_prep_is_deterministic() {
        let geometry = sample_geometry();
        let first = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
            .expect("first meshing prep should work");
        let second = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
            .expect("second meshing prep should work");
        assert_eq!(first, second);
        let descriptor = first
            .prepared_meshes
            .first()
            .expect("prepared mesh descriptor should exist");
        assert!(descriptor.region_span_hint >= 1);
    }

    #[test]
    fn meshing_prep_carries_element_geometry_metrics() {
        let mut geometry = sample_geometry();
        geometry.meshes[0].vertex_count = 4;
        geometry.meshes[0].element_count = 2;
        geometry.surface_meshes = vec![SurfaceMesh::new(
            "mesh_a",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )];

        let prep = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
            .expect("meshing prep should work");
        let descriptor = prep
            .prepared_meshes
            .first()
            .expect("prepared mesh descriptor should exist");
        assert_eq!(descriptor.element_geometry_node_count, 4);
        assert_eq!(descriptor.element_geometry_edge_count, 5);
        assert!(descriptor.mean_element_edge_length_m > 1.0);
        assert!((descriptor.mean_element_area_m2 - 0.5).abs() < 1.0e-12);
        assert_eq!(descriptor.element_geometry_coverage_ratio, 1.0);
        assert_eq!(
            descriptor.reference_element_coordinates_m,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        );
        assert!((descriptor.reference_element_area_m2 - 0.5).abs() < 1.0e-12);
        assert_eq!(descriptor.control_volume_cell_count, 2);
        assert_eq!(descriptor.control_volume_face_count, 5);
        assert_eq!(descriptor.control_volume_internal_face_count, 1);
        assert_eq!(descriptor.control_volume_boundary_face_count, 4);
        assert_eq!(descriptor.control_volume_connectivity_coverage_ratio, 1.0);
        assert_eq!(descriptor.element_topology_sample_element_count, 2);
        assert_eq!(descriptor.element_topology_sample_edge_count, 5);
        assert_eq!(descriptor.element_topology_sample_edge_nodes[0], [0, 1]);
        assert_eq!(
            descriptor.element_topology_sample_node_coordinates_m[0],
            [0.0, 0.0, 0.0]
        );
        assert_eq!(
            descriptor.element_topology_sample_element_edges[0],
            [0, 1, 2]
        );
        assert_eq!(
            descriptor.element_topology_sample_element_orientations[0],
            [1, 1, -1]
        );
        assert!((descriptor.element_topology_sample_element_areas_m2[0] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn meshing_prep_validates_options() {
        let geometry = sample_geometry();
        let error = prepare_geometry_for_analysis(
            &geometry,
            MeshingOptions {
                profile: MeshingProfile::AnalysisReady,
                target_element_budget: 0,
            },
        )
        .expect_err("zero budget should fail");
        assert!(error.contains("target_element_budget"));
    }

    #[test]
    fn metadata_only_regions_map_to_prepared_meshes() {
        let mut geometry = sample_geometry();
        geometry.region_entity_mappings.clear();

        let prep = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
            .expect("meshing prep should work");
        let mapping = prep
            .region_mappings
            .iter()
            .find(|mapping| mapping.region_id == "region_main")
            .expect("region mapping should exist");
        assert_eq!(mapping.source_mesh_ids, vec!["mesh_a"]);
        assert_eq!(mapping.prepared_mesh_ids, vec!["prep_7_mesh_a"]);
    }

    #[test]
    fn adaptive_refine_profile_improves_quality_within_budget() {
        let geometry = sample_geometry();
        let analysis_ready = prepare_geometry_for_analysis(
            &geometry,
            MeshingOptions {
                profile: MeshingProfile::AnalysisReady,
                target_element_budget: 4_000,
            },
        )
        .expect("analysis-ready prep should work");
        let adaptive = prepare_geometry_for_analysis(
            &geometry,
            MeshingOptions {
                profile: MeshingProfile::AdaptiveRefine,
                target_element_budget: 4_000,
            },
        )
        .expect("adaptive-refine prep should work");

        let adaptive_total_elements = adaptive
            .prepared_meshes
            .iter()
            .map(|mesh| mesh.element_count)
            .sum::<u64>();
        assert!(adaptive_total_elements <= 4_000);
        assert!(adaptive.quality.min_scaled_jacobian >= analysis_ready.quality.min_scaled_jacobian);
        assert!(adaptive.quality.mean_aspect_ratio <= analysis_ready.quality.mean_aspect_ratio);
    }
}
