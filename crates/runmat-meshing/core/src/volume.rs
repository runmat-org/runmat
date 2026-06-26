use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    artifact::{
        AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode, AnalysisVolumeElement,
        ANALYSIS_MESH_SCHEMA_VERSION,
    },
    boundary::{BoundaryMeshInput, BoundaryMeshInputError},
    options::{MeshProfile, MeshTargetSize, VolumeMeshingOptions},
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance, SourceEntityKind},
    quality::{AnalysisMeshQualityReport, ElementQuality},
    sizing::MeshSizingField,
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
        if !matches!(options.element, VolumeElementKind::Tet4) {
            return Err(MeshingError::UnsupportedElementKind(options.element));
        }
        if options.max_elements == 0 {
            return Err(MeshingError::InvalidElementBudget);
        }
        if input.region_ids.is_empty() {
            return Err(MeshingError::EmptyBoundaryRegions);
        }

        let divisions = grid_divisions(input, options)?;
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

        Ok(AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: format!("analysis_{}", input.mesh_id),
            nodes,
            volume_elements,
            boundary_faces,
            boundary_edges: Vec::new(),
            quality,
            sizing: MeshSizingField {
                global_target_size_m: target_size_m(input, options, divisions),
                min_size_m: None,
                max_size_m: None,
                samples: Vec::new(),
            },
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

fn grid_divisions(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
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
    Ok(requested.clamp(1, max_by_budget))
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
    let ad = sub(a, d);
    let bd = sub(b, d);
    let cd = sub(c, d);
    dot(ad, cross(bd, cd)) / 6.0
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{validate_analysis_mesh, MeshKindRequest};
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
        assert!(mesh.boundary_faces.iter().any(|face| face
            .region_ids
            .iter()
            .any(|region| region == "region_fixed")));
        assert!(mesh
            .boundary_faces
            .iter()
            .any(|face| face.region_ids.iter().any(|region| region == "region_load")));
        assert!(mesh
            .quality
            .elements
            .iter()
            .all(|quality| quality.volume_m3 > 0.0));
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
