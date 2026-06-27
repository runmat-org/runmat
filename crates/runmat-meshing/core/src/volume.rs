use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    artifact::{
        ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode,
        AnalysisVolumeElement,
    },
    boundary::{BoundaryMeshInput, BoundaryMeshInputError},
    options::{MeshProfile, MeshTargetSize, VolumeMeshingOptions},
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance, SourceEntityKind},
    quality::{AnalysisMeshQualityReport, ElementQuality},
    sizing::{MeshSizingField, SizingSample},
    topology::{BoundaryElementKind, VolumeElementKind},
};

pub trait VolumeMesher {
    fn mesh(
        &self,
        input: &BoundaryMeshInput,
        options: &VolumeMeshingOptions,
    ) -> Result<AnalysisMeshArtifact, MeshingError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshingError {
    BoundaryInput(String),
    UnsupportedElementKind(VolumeElementKind),
    InvalidElementBudget,
    InvalidTargetSize,
    EmptyBoundaryRegions,
}

impl std::fmt::Display for MeshingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundaryInput(message) => write!(formatter, "invalid boundary mesh: {message}"),
            Self::UnsupportedElementKind(kind) => {
                write!(formatter, "unsupported volume element kind: {kind:?}")
            }
            Self::InvalidElementBudget => write!(formatter, "max_elements must be greater than 0"),
            Self::InvalidTargetSize => {
                write!(
                    formatter,
                    "target_size must be auto or a finite positive length"
                )
            }
            Self::EmptyBoundaryRegions => write!(formatter, "boundary mesh has no region ids"),
        }
    }
}

impl std::error::Error for MeshingError {}

impl From<BoundaryMeshInputError> for MeshingError {
    fn from(value: BoundaryMeshInputError) -> Self {
        Self::BoundaryInput(value.to_string())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StructuredTetMesher;

impl VolumeMesher for StructuredTetMesher {
    fn mesh(
        &self,
        input: &BoundaryMeshInput,
        options: &VolumeMeshingOptions,
    ) -> Result<AnalysisMeshArtifact, MeshingError> {
        self.mesh_with_sizing(input, options, None)
    }
}

impl StructuredTetMesher {
    pub fn mesh_with_sizing(
        &self,
        input: &BoundaryMeshInput,
        options: &VolumeMeshingOptions,
        sizing: Option<&MeshSizingField>,
    ) -> Result<AnalysisMeshArtifact, MeshingError> {
        if !matches!(options.element, VolumeElementKind::Tet4) {
            return Err(MeshingError::UnsupportedElementKind(options.element));
        }
        if options.max_elements == 0 {
            return Err(MeshingError::InvalidElementBudget);
        }
        if input.region_ids.is_empty() {
            return Err(MeshingError::EmptyBoundaryRegions);
        }

        let mut mesh_sizing = sizing.cloned().unwrap_or_default();
        append_geometry_focus_sizing_samples(input, options, &mut mesh_sizing);

        let divisions = grid_divisions(input, options, Some(&mesh_sizing))?;
        let nodes = grid_nodes(input, divisions);
        let node_id_at = |i: usize, j: usize, k: usize| -> u32 {
            (1 + i + (divisions + 1) * (j + (divisions + 1) * k)) as u32
        };

        let material_region_id = input
            .region_ids
            .first()
            .cloned()
            .unwrap_or_else(|| "region_default".to_string());
        let provenance = vec![MeshEntityProvenance {
            source_geometry_id: input.source_geometry_id.clone(),
            source_geometry_revision: input.source_geometry_revision,
            source_entity_kind: SourceEntityKind::Mesh,
            source_entity_id: input.mesh_id.clone(),
            region_ids: input.region_ids.clone(),
        }];

        let mut volume_elements = Vec::<AnalysisVolumeElement>::new();
        let mut element_quality = Vec::<ElementQuality>::new();
        for k in 0..divisions {
            for j in 0..divisions {
                for i in 0..divisions {
                    let cell_nodes = [
                        node_id_at(i, j, k),
                        node_id_at(i + 1, j, k),
                        node_id_at(i, j + 1, k),
                        node_id_at(i + 1, j + 1, k),
                        node_id_at(i, j, k + 1),
                        node_id_at(i + 1, j, k + 1),
                        node_id_at(i, j + 1, k + 1),
                        node_id_at(i + 1, j + 1, k + 1),
                    ];
                    for tet in [
                        [cell_nodes[0], cell_nodes[1], cell_nodes[3], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[3], cell_nodes[2], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[2], cell_nodes[6], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[6], cell_nodes[4], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[4], cell_nodes[5], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[5], cell_nodes[1], cell_nodes[7]],
                    ] {
                        let element_id = format!("tet_{}", volume_elements.len() + 1);
                        let oriented = orient_tet(tet, &nodes);
                        let volume_m3 = tet_volume(oriented, &nodes).abs();
                        let aspect_ratio = tet_aspect_ratio(oriented, &nodes);
                        element_quality.push(ElementQuality {
                            element_id: element_id.clone(),
                            scaled_jacobian: 1.0 / aspect_ratio.max(1.0),
                            aspect_ratio,
                            volume_m3,
                        });
                        volume_elements.push(AnalysisVolumeElement {
                            element_id,
                            kind: VolumeElementKind::Tet4,
                            node_ids: oriented.to_vec(),
                            material_region_id: material_region_id.clone(),
                            provenance: provenance.clone(),
                        });
                    }
                }
            }
        }

        let boundary_faces = grid_boundary_faces(input, divisions, &node_id_at);
        let quality = quality_report(element_quality);
        mesh_sizing.global_target_size_m = target_size_m(input, options, divisions);

        Ok(AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: format!("analysis_{}", input.mesh_id),
            nodes,
            volume_elements,
            boundary_faces,
            boundary_edges: Vec::new(),
            quality,
            sizing: mesh_sizing,
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "structured_bbox_tet/v1".to_string(),
                source_geometry_id: input.source_geometry_id.clone(),
                source_geometry_revision: input.source_geometry_revision,
                source_geometry_sha256: input.source_geometry_sha256.clone(),
            },
        })
    }
}

pub fn generate_analysis_mesh(
    geometry: &runmat_geometry_core::GeometryAsset,
    options: VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, MeshingError> {
    let input = BoundaryMeshInput::from_geometry(geometry)?;
    StructuredTetMesher.mesh(&input, &options)
}

pub fn generate_analysis_mesh_with_sizing(
    geometry: &runmat_geometry_core::GeometryAsset,
    options: VolumeMeshingOptions,
    sizing: &MeshSizingField,
) -> Result<AnalysisMeshArtifact, MeshingError> {
    let input = BoundaryMeshInput::from_geometry(geometry)?;
    StructuredTetMesher.mesh_with_sizing(&input, &options, Some(sizing))
}

fn grid_divisions(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
) -> Result<usize, MeshingError> {
    let max_by_budget = ((options.max_elements / 6).max(1) as f64)
        .cbrt()
        .floor()
        .max(1.0) as usize;
    let requested = match options.target_size {
        MeshTargetSize::Auto => match options.profile {
            MeshProfile::Coarse => 1,
            MeshProfile::AnalysisReady => 2,
            MeshProfile::Adaptive | MeshProfile::Fine => 3,
        },
        MeshTargetSize::LengthM(length_m) => {
            if !length_m.is_finite() || length_m <= 0.0 {
                return Err(MeshingError::InvalidTargetSize);
            }
            let max_span = (0..3)
                .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
                .fold(0.0_f64, f64::max);
            (max_span / length_m).ceil().max(1.0) as usize
        }
    };
    let requested = sizing
        .and_then(sizing_target_size_m)
        .map(|length_m| requested.max(divisions_for_target_size(input, length_m)))
        .unwrap_or(requested);
    Ok(requested.clamp(1, max_by_budget))
}

fn divisions_for_target_size(input: &BoundaryMeshInput, length_m: f64) -> usize {
    let max_span = (0..3)
        .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
        .fold(0.0_f64, f64::max);
    (max_span / length_m).ceil().max(1.0) as usize
}

fn sizing_target_size_m(sizing: &MeshSizingField) -> Option<f64> {
    sizing
        .samples
        .iter()
        .map(|sample| sample.target_size_m)
        .chain(sizing.min_size_m)
        .chain(sizing.global_target_size_m)
        .filter(|value| value.is_finite() && *value > 0.0)
        .reduce(f64::min)
}

fn append_geometry_focus_sizing_samples(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    sizing: &mut MeshSizingField,
) {
    if options.refinement.focus.curvature {
        sizing.samples.extend(curvature_sizing_samples(input));
    }
    if options.refinement.focus.small_features {
        sizing.samples.extend(small_feature_sizing_samples(input));
    }
}

fn curvature_sizing_samples(input: &BoundaryMeshInput) -> Vec<SizingSample> {
    let mut triangles_by_edge = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (triangle_index, triangle) in input.triangles.iter().enumerate() {
        for edge in triangle_edges(triangle.node_ids) {
            triangles_by_edge
                .entry(edge)
                .or_default()
                .push(triangle_index);
        }
    }

    triangles_by_edge
        .into_iter()
        .filter_map(|(edge, triangle_indices)| {
            if triangle_indices.len() != 2 {
                return None;
            }
            let left = input.triangles.get(triangle_indices[0])?;
            let right = input.triangles.get(triangle_indices[1])?;
            let left_normal = triangle_unit_normal(input, left.node_ids)?;
            let right_normal = triangle_unit_normal(input, right.node_ids)?;
            let normal_dot = dot(left_normal, right_normal).clamp(-1.0, 1.0);
            if 1.0 - normal_dot.abs() <= 0.05 {
                return None;
            }
            let left_vertex = *input.vertices.get(edge[0] as usize)?;
            let right_vertex = *input.vertices.get(edge[1] as usize)?;
            let edge_length = distance(left_vertex, right_vertex);
            (edge_length.is_finite() && edge_length > 0.0).then_some(SizingSample {
                position_m: midpoint(left_vertex, right_vertex),
                target_size_m: edge_length * 0.5,
                reason: Some("geometry.curvature".to_string()),
            })
        })
        .collect()
}

fn small_feature_sizing_samples(input: &BoundaryMeshInput) -> Vec<SizingSample> {
    let max_span = boundary_max_span(input);
    if !max_span.is_finite() || max_span <= 0.0 {
        return Vec::new();
    }
    let threshold = max_span * 0.35;
    input
        .triangles
        .iter()
        .filter_map(|triangle| {
            let vertices = triangle_vertices(input, triangle.node_ids)?;
            let min_edge = triangle_min_edge(vertices);
            if !min_edge.is_finite() || min_edge <= 0.0 || min_edge > threshold {
                return None;
            }
            Some(SizingSample {
                position_m: triangle_centroid(vertices),
                target_size_m: min_edge * 0.5,
                reason: Some("geometry.small_features".to_string()),
            })
        })
        .collect()
}

fn grid_nodes(input: &BoundaryMeshInput, divisions: usize) -> Vec<AnalysisMeshNode> {
    let mut nodes = Vec::with_capacity((divisions + 1).pow(3));
    for k in 0..=divisions {
        for j in 0..=divisions {
            for i in 0..=divisions {
                let t = [
                    i as f64 / divisions as f64,
                    j as f64 / divisions as f64,
                    k as f64 / divisions as f64,
                ];
                nodes.push(AnalysisMeshNode {
                    node_id: nodes.len() as u32 + 1,
                    coordinates_m: [
                        lerp(input.bounds_min_m[0], input.bounds_max_m[0], t[0]),
                        lerp(input.bounds_min_m[1], input.bounds_max_m[1], t[1]),
                        lerp(input.bounds_min_m[2], input.bounds_max_m[2], t[2]),
                    ],
                    provenance: Vec::new(),
                });
            }
        }
    }
    nodes
}

fn grid_boundary_faces(
    input: &BoundaryMeshInput,
    divisions: usize,
    node_id_at: &impl Fn(usize, usize, usize) -> u32,
) -> Vec<AnalysisBoundaryFace> {
    let regions_by_side = regions_by_boundary_side(input);
    let mut faces = Vec::new();
    for side in BoundarySide::ALL {
        for a in 0..divisions {
            for b in 0..divisions {
                let quad = match side {
                    BoundarySide::XMin => [
                        node_id_at(0, a, b),
                        node_id_at(0, a + 1, b),
                        node_id_at(0, a + 1, b + 1),
                        node_id_at(0, a, b + 1),
                    ],
                    BoundarySide::XMax => [
                        node_id_at(divisions, a, b),
                        node_id_at(divisions, a, b + 1),
                        node_id_at(divisions, a + 1, b + 1),
                        node_id_at(divisions, a + 1, b),
                    ],
                    BoundarySide::YMin => [
                        node_id_at(a, 0, b),
                        node_id_at(a, 0, b + 1),
                        node_id_at(a + 1, 0, b + 1),
                        node_id_at(a + 1, 0, b),
                    ],
                    BoundarySide::YMax => [
                        node_id_at(a, divisions, b),
                        node_id_at(a + 1, divisions, b),
                        node_id_at(a + 1, divisions, b + 1),
                        node_id_at(a, divisions, b + 1),
                    ],
                    BoundarySide::ZMin => [
                        node_id_at(a, b, 0),
                        node_id_at(a + 1, b, 0),
                        node_id_at(a + 1, b + 1, 0),
                        node_id_at(a, b + 1, 0),
                    ],
                    BoundarySide::ZMax => [
                        node_id_at(a, b, divisions),
                        node_id_at(a, b + 1, divisions),
                        node_id_at(a + 1, b + 1, divisions),
                        node_id_at(a + 1, b, divisions),
                    ],
                };
                let region_ids = regions_by_side
                    .get(&side)
                    .cloned()
                    .filter(|regions| !regions.is_empty())
                    .unwrap_or_else(|| input.region_ids.clone());
                for tri in [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]] {
                    faces.push(AnalysisBoundaryFace {
                        face_id: format!("bf_{}", faces.len() + 1),
                        kind: BoundaryElementKind::Tri3,
                        node_ids: tri.to_vec(),
                        adjacent_volume_element_ids: Vec::new(),
                        region_ids: region_ids.clone(),
                        provenance: vec![MeshEntityProvenance {
                            source_geometry_id: input.source_geometry_id.clone(),
                            source_geometry_revision: input.source_geometry_revision,
                            source_entity_kind: SourceEntityKind::Region,
                            source_entity_id: region_ids.join(","),
                            region_ids: region_ids.clone(),
                        }],
                    });
                }
            }
        }
    }
    faces
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BoundarySide {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

impl BoundarySide {
    const ALL: [Self; 6] = [
        Self::XMin,
        Self::XMax,
        Self::YMin,
        Self::YMax,
        Self::ZMin,
        Self::ZMax,
    ];
}

fn regions_by_boundary_side(input: &BoundaryMeshInput) -> BTreeMap<BoundarySide, Vec<String>> {
    let mut by_side = BTreeMap::<BoundarySide, BTreeSet<String>>::new();
    let tolerance = (0..3)
        .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
        .fold(0.0_f64, f64::max)
        * 1.0e-8;
    for triangle in &input.triangles {
        let centroid = triangle
            .node_ids
            .iter()
            .filter_map(|node_id| input.vertices.get(*node_id as usize))
            .fold([0.0; 3], |mut sum, vertex| {
                for axis in 0..3 {
                    sum[axis] += vertex[axis] / 3.0;
                }
                sum
            });
        let sides = [
            (BoundarySide::XMin, 0, input.bounds_min_m[0]),
            (BoundarySide::XMax, 0, input.bounds_max_m[0]),
            (BoundarySide::YMin, 1, input.bounds_min_m[1]),
            (BoundarySide::YMax, 1, input.bounds_max_m[1]),
            (BoundarySide::ZMin, 2, input.bounds_min_m[2]),
            (BoundarySide::ZMax, 2, input.bounds_max_m[2]),
        ];
        for (side, axis, plane) in sides {
            if (centroid[axis] - plane).abs() <= tolerance.max(1.0e-12) {
                by_side
                    .entry(side)
                    .or_default()
                    .extend(triangle.region_ids.iter().cloned());
            }
        }
    }
    by_side
        .into_iter()
        .map(|(side, regions)| (side, regions.into_iter().collect()))
        .collect()
}

fn target_size_m(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    divisions: usize,
) -> Option<f64> {
    match options.target_size {
        MeshTargetSize::LengthM(value) => Some(value),
        MeshTargetSize::Auto => {
            let max_span = (0..3)
                .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
                .fold(0.0_f64, f64::max);
            Some(max_span / divisions as f64)
        }
    }
}

fn quality_report(elements: Vec<ElementQuality>) -> AnalysisMeshQualityReport {
    let min_scaled_jacobian = elements
        .iter()
        .map(|element| element.scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let max_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .fold(0.0_f64, f64::max);
    let mean_aspect_ratio = if elements.is_empty() {
        0.0
    } else {
        elements
            .iter()
            .map(|element| element.aspect_ratio)
            .sum::<f64>()
            / elements.len() as f64
    };
    AnalysisMeshQualityReport {
        min_scaled_jacobian,
        mean_aspect_ratio,
        max_aspect_ratio,
        inverted_element_count: 0,
        elements,
    }
}

fn orient_tet(mut node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> [u32; 4] {
    if tet_volume(node_ids, nodes) < 0.0 {
        node_ids.swap(0, 1);
    }
    node_ids
}

fn tet_volume(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> f64 {
    let Some([a, b, c, d]) = tet_points(node_ids, nodes) else {
        return 0.0;
    };
    dot(sub(b, a), cross(sub(c, a), sub(d, a))) / 6.0
}

fn tet_aspect_ratio(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> f64 {
    let Some(points) = tet_points(node_ids, nodes) else {
        return f64::INFINITY;
    };
    let mut min_edge = f64::INFINITY;
    let mut max_edge = 0.0_f64;
    for (left, right) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        let length = norm(sub(points[left], points[right]));
        min_edge = min_edge.min(length);
        max_edge = max_edge.max(length);
    }
    max_edge / min_edge.max(f64::EPSILON)
}

fn tet_points(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> Option<[[f64; 3]; 4]> {
    Some([
        nodes
            .get(node_ids[0].checked_sub(1)? as usize)?
            .coordinates_m,
        nodes
            .get(node_ids[1].checked_sub(1)? as usize)?
            .coordinates_m,
        nodes
            .get(node_ids[2].checked_sub(1)? as usize)?
            .coordinates_m,
        nodes
            .get(node_ids[3].checked_sub(1)? as usize)?
            .coordinates_m,
    ])
}

fn lerp(left: f64, right: f64, t: f64) -> f64 {
    left + (right - left) * t
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    norm(sub(left, right))
}

fn midpoint(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        (left[0] + right[0]) * 0.5,
        (left[1] + right[1]) * 0.5,
        (left[2] + right[2]) * 0.5,
    ]
}

fn triangle_edges(triangle: [u32; 3]) -> [[u32; 2]; 3] {
    [
        sorted_edge(triangle[0], triangle[1]),
        sorted_edge(triangle[1], triangle[2]),
        sorted_edge(triangle[2], triangle[0]),
    ]
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    [left.min(right), left.max(right)]
}

fn triangle_vertices(input: &BoundaryMeshInput, node_ids: [u32; 3]) -> Option<[[f64; 3]; 3]> {
    Some([
        *input.vertices.get(node_ids[0] as usize)?,
        *input.vertices.get(node_ids[1] as usize)?,
        *input.vertices.get(node_ids[2] as usize)?,
    ])
}

fn triangle_unit_normal(input: &BoundaryMeshInput, node_ids: [u32; 3]) -> Option<[f64; 3]> {
    let [a, b, c] = triangle_vertices(input, node_ids)?;
    let normal = cross(sub(b, a), sub(c, a));
    let length = norm(normal);
    (length > 0.0).then_some([
        normal[0] / length,
        normal[1] / length,
        normal[2] / length,
    ])
}

fn triangle_min_edge(vertices: [[f64; 3]; 3]) -> f64 {
    distance(vertices[0], vertices[1])
        .min(distance(vertices[1], vertices[2]))
        .min(distance(vertices[2], vertices[0]))
}

fn triangle_centroid(vertices: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3.0,
        (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3.0,
        (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3.0,
    ]
}

fn boundary_max_span(input: &BoundaryMeshInput) -> f64 {
    (0..3)
        .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeshKindRequest, SizingSample, validate_analysis_mesh};
    use runmat_geometry_core::{
        GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region, RegionEntityMapping,
        SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
    };

    fn cube_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_tet_cube".to_string(),
            source: GeometrySource {
                path: "/fixtures/generic_cube.step".to_string(),
                sha256: "generic-cube".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 3,
            meshes: vec![MeshDescriptor {
                mesh_id: "cube_surface".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 8,
                element_count: 12,
            }],
            surface_meshes: vec![SurfaceMesh::new(
                "cube_surface",
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                vec![
                    [0, 2, 1],
                    [0, 3, 2],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ],
            )],
            regions: vec![
                Region {
                    region_id: "region_fixed".to_string(),
                    name: "fixed".to_string(),
                    tag: Some("fixed".to_string()),
                    cad_ownership: None,
                },
                Region {
                    region_id: "region_load".to_string(),
                    name: "load".to_string(),
                    tag: Some("load".to_string()),
                    cad_ownership: None,
                },
            ],
            region_entity_mappings: vec![
                RegionEntityMapping::all_faces("region_fixed", "cube_surface", 2),
                RegionEntityMapping::new(
                    "region_load",
                    "cube_surface",
                    runmat_geometry_core::EntityKind::Face,
                    vec![runmat_geometry_core::EntityIdRange::new(2, 2)],
                ),
            ],
            diagnostics: Vec::new(),
        }
    }

    fn thin_box_geometry() -> GeometryAsset {
        let mut geometry = cube_geometry();
        geometry.geometry_id = "geo_tet_thin_box".to_string();
        for vertex in geometry.surface_meshes[0].vertices.iter_mut().skip(4) {
            vertex[2] = 0.2;
        }
        geometry
    }

    #[test]
    fn structured_tet_mesher_generates_valid_analysis_mesh() {
        let geometry = cube_geometry();
        let mesh = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
            .expect("cube should produce an analysis mesh");

        validate_analysis_mesh(&mesh, Default::default()).expect("analysis mesh should validate");
        assert_eq!(mesh.schema_version, ANALYSIS_MESH_SCHEMA_VERSION);
        assert_eq!(mesh.nodes.len(), 27);
        assert_eq!(mesh.volume_elements.len(), 48);
        assert_eq!(mesh.boundary_faces.len(), 48);
        assert!(mesh.boundary_faces.iter().any(|face| {
            face.region_ids
                .iter()
                .any(|region| region == "region_fixed")
        }));
        assert!(
            mesh.boundary_faces
                .iter()
                .any(|face| face.region_ids.iter().any(|region| region == "region_load"))
        );
        assert!(
            mesh.quality
                .elements
                .iter()
                .all(|quality| quality.volume_m3 > 0.0)
        );
        assert!(mesh.volume_elements.iter().all(|element| {
            tet_volume(
                [
                    element.node_ids[0],
                    element.node_ids[1],
                    element.node_ids[2],
                    element.node_ids[3],
                ],
                &mesh.nodes,
            ) > 0.0
        }));
    }

    #[test]
    fn target_size_controls_structured_tet_density() {
        let geometry = cube_geometry();
        let coarse = generate_analysis_mesh(
            &geometry,
            VolumeMeshingOptions {
                kind: MeshKindRequest::Solid,
                target_size: MeshTargetSize::LengthM(1.0),
                ..VolumeMeshingOptions::default()
            },
        )
        .expect("coarse mesh should generate");
        let fine = generate_analysis_mesh(
            &geometry,
            VolumeMeshingOptions {
                kind: MeshKindRequest::Solid,
                target_size: MeshTargetSize::LengthM(0.25),
                max_elements: 10_000,
                ..VolumeMeshingOptions::default()
            },
        )
        .expect("fine mesh should generate");

        assert!(fine.volume_elements.len() > coarse.volume_elements.len());
        assert!(fine.nodes.len() > coarse.nodes.len());
    }

    #[test]
    fn sizing_field_controls_structured_tet_density() {
        let geometry = cube_geometry();
        let base_options = VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        };
        let mut base_options = base_options;
        base_options.refinement.focus.curvature = false;
        base_options.refinement.focus.small_features = false;
        let coarse = generate_analysis_mesh(&geometry, base_options.clone())
            .expect("coarse mesh should generate");
        let sizing = MeshSizingField {
            samples: vec![SizingSample {
                position_m: [0.5, 0.5, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.stress_gradient".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let refined = generate_analysis_mesh_with_sizing(&geometry, base_options, &sizing)
            .expect("sizing-driven mesh should generate");

        assert!(refined.volume_elements.len() > coarse.volume_elements.len());
        assert_eq!(refined.sizing.samples.len(), 1);
        assert_eq!(
            refined.sizing.samples[0].reason.as_deref(),
            Some("structural.stress_gradient")
        );
    }

    #[test]
    fn sizing_field_refinement_respects_element_budget() {
        let geometry = cube_geometry();
        let sizing = MeshSizingField {
            samples: vec![SizingSample {
                position_m: [0.5, 0.5, 0.5],
                target_size_m: 0.01,
                reason: Some("structural.stress_gradient".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let mesh = generate_analysis_mesh_with_sizing(
            &geometry,
            VolumeMeshingOptions {
                target_size: MeshTargetSize::LengthM(1.0),
                max_elements: 48,
                ..VolumeMeshingOptions::default()
            },
            &sizing,
        )
        .expect("budgeted sizing-driven mesh should generate");

        assert!(mesh.volume_elements.len() <= 48);
        assert_eq!(mesh.volume_elements.len(), 48);
    }

    #[test]
    fn curvature_focus_adds_geometry_sizing_samples() {
        let geometry = cube_geometry();
        let mut coarse_options = VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        };
        coarse_options.refinement.focus.curvature = false;
        coarse_options.refinement.focus.small_features = false;
        let coarse = generate_analysis_mesh(&geometry, coarse_options.clone())
            .expect("coarse mesh should generate");

        coarse_options.refinement.focus.curvature = true;
        let focused = generate_analysis_mesh(&geometry, coarse_options)
            .expect("curvature-focused mesh should generate");

        assert!(focused.volume_elements.len() > coarse.volume_elements.len());
        assert!(focused
            .sizing
            .samples
            .iter()
            .any(|sample| sample.reason.as_deref() == Some("geometry.curvature")));
    }

    #[test]
    fn small_feature_focus_adds_geometry_sizing_samples() {
        let geometry = thin_box_geometry();
        let mut coarse_options = VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        };
        coarse_options.refinement.focus.curvature = false;
        coarse_options.refinement.focus.small_features = false;
        let coarse = generate_analysis_mesh(&geometry, coarse_options.clone())
            .expect("coarse thin-box mesh should generate");

        coarse_options.refinement.focus.small_features = true;
        let focused = generate_analysis_mesh(&geometry, coarse_options)
            .expect("small-feature-focused mesh should generate");

        assert!(focused.volume_elements.len() > coarse.volume_elements.len());
        assert!(focused
            .sizing
            .samples
            .iter()
            .any(|sample| sample.reason.as_deref() == Some("geometry.small_features")));
    }

    #[test]
    fn invalid_open_shell_returns_meshing_error() {
        let mut geometry = cube_geometry();
        geometry.surface_meshes[0].triangles.pop();
        geometry.meshes[0].element_count -= 1;

        let err = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
            .expect_err("open shell should fail");

        assert!(matches!(err, MeshingError::BoundaryInput(_)));
        assert!(err.to_string().contains("incidence"));
    }

    #[test]
    fn unsupported_element_kind_is_rejected() {
        let geometry = cube_geometry();
        let err = generate_analysis_mesh(
            &geometry,
            VolumeMeshingOptions {
                element: VolumeElementKind::Hex8,
                ..VolumeMeshingOptions::default()
            },
        )
        .expect_err("hex backend is not implemented");

        assert_eq!(
            err,
            MeshingError::UnsupportedElementKind(VolumeElementKind::Hex8)
        );
    }
}
