use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    cad_eval::{project_to_face, CadEvaluationModel},
    source_topology::{SourceTopologyFace, SourceTopologyModel},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceDiscretizationOptions {
    pub preserve_source_faces: bool,
    pub centroid_subdivision: bool,
}

impl Default for SurfaceDiscretizationOptions {
    fn default() -> Self {
        Self {
            preserve_source_faces: true,
            centroid_subdivision: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceNode {
    pub node_id: u32,
    pub source_vertex_id: u32,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceElement {
    pub element_id: u32,
    pub source_face_id: u32,
    #[serde(default)]
    pub cad_face_id: Option<String>,
    pub source_edge_ids: [u32; 3],
    pub node_ids: [u32; 3],
    #[serde(default)]
    pub parametric_node_uv: [[f64; 2]; 3],
    #[serde(default)]
    pub max_projection_error_m: f64,
    pub region_ids: Vec<String>,
    pub area_m2: f64,
    pub unit_normal: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceDiscretization {
    pub nodes: Vec<SurfaceNode>,
    pub elements: Vec<SurfaceElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceDiscretizationError {
    MissingFaceVertex { face_id: u32, node_id: u32 },
    MissingCadFaceFrame { source_face_id: u32 },
}

impl std::fmt::Display for SurfaceDiscretizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFaceVertex { face_id, node_id } => write!(
                formatter,
                "source face {face_id} references missing topology vertex {node_id}"
            ),
            Self::MissingCadFaceFrame { source_face_id } => write!(
                formatter,
                "source face {source_face_id} does not have a CAD evaluation frame"
            ),
        }
    }
}

impl std::error::Error for SurfaceDiscretizationError {}

pub fn discretize_topology_surfaces(
    topology: &SourceTopologyModel,
    _options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    let nodes = topology
        .vertices
        .iter()
        .map(|vertex| SurfaceNode {
            node_id: vertex.vertex_id,
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();

    let mut elements = Vec::<SurfaceElement>::with_capacity(topology.faces.len());
    for face in &topology.faces {
        validate_face_vertices(topology, face)?;
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: None,
            source_edge_ids: face.edge_ids,
            node_ids: face.node_ids,
            parametric_node_uv: [[0.0, 0.0]; 3],
            max_projection_error_m: 0.0,
            region_ids: face.region_ids.clone(),
            area_m2: face.area_m2,
            unit_normal: face.unit_normal,
        });
    }

    Ok(SurfaceDiscretization { nodes, elements })
}

pub fn discretize_cad_surfaces(
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    let mut nodes = topology
        .vertices
        .iter()
        .map(|vertex| SurfaceNode {
            node_id: vertex.vertex_id,
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();
    let frames_by_source_face = cad_evaluation
        .face_frames
        .iter()
        .map(|frame| (frame.source_face_id, frame))
        .collect::<BTreeMap<_, _>>();

    let element_capacity = if options.centroid_subdivision {
        topology.faces.len() * 3
    } else {
        topology.faces.len()
    };
    let mut elements = Vec::<SurfaceElement>::with_capacity(element_capacity);
    for face in &topology.faces {
        validate_face_vertices(topology, face)?;
        let frame = frames_by_source_face.get(&face.face_id).ok_or(
            SurfaceDiscretizationError::MissingCadFaceFrame {
                source_face_id: face.face_id,
            },
        )?;
        let mut parametric_node_uv = [[0.0_f64, 0.0_f64]; 3];
        let mut max_projection_error_m = 0.0_f64;
        let mut corner_points = [[0.0_f64, 0.0_f64, 0.0_f64]; 3];
        for (index, node_id) in face.node_ids.into_iter().enumerate() {
            let point = topology
                .vertices
                .get(node_id as usize)
                .filter(|vertex| vertex.vertex_id == node_id)
                .map(|vertex| vertex.coordinates_m)
                .ok_or(SurfaceDiscretizationError::MissingFaceVertex {
                    face_id: face.face_id,
                    node_id,
                })?;
            corner_points[index] = point;
            let projection = project_to_face(frame, point);
            parametric_node_uv[index] = projection.uv;
            max_projection_error_m = max_projection_error_m.max(projection.distance_m);
        }

        if options.centroid_subdivision {
            let centroid = triangle_centroid(corner_points);
            let centroid_projection = project_to_face(frame, centroid);
            max_projection_error_m = max_projection_error_m.max(centroid_projection.distance_m);
            append_centroid_subdivision(
                face,
                frame,
                parametric_node_uv,
                centroid,
                centroid_projection.uv,
                max_projection_error_m,
                &mut nodes,
                &mut elements,
            );
        } else {
            elements.push(SurfaceElement {
                element_id: elements.len() as u32,
                source_face_id: face.face_id,
                cad_face_id: Some(frame.face_id.clone()),
                source_edge_ids: face.edge_ids,
                node_ids: face.node_ids,
                parametric_node_uv,
                max_projection_error_m,
                region_ids: face.region_ids.clone(),
                area_m2: face.area_m2,
                unit_normal: frame.unit_normal,
            });
        }
    }

    Ok(SurfaceDiscretization { nodes, elements })
}

fn append_centroid_subdivision(
    face: &SourceTopologyFace,
    frame: &crate::CadFaceEvaluationFrame,
    corner_uv: [[f64; 2]; 3],
    centroid_m: [f64; 3],
    centroid_uv: [f64; 2],
    corner_projection_error_m: f64,
    nodes: &mut Vec<SurfaceNode>,
    elements: &mut Vec<SurfaceElement>,
) {
    let centroid_node_id = nodes.len() as u32;
    nodes.push(SurfaceNode {
        node_id: centroid_node_id,
        source_vertex_id: u32::MAX,
        coordinates_m: centroid_m,
    });
    let child_specs = [
        (
            [face.node_ids[0], face.node_ids[1], centroid_node_id],
            [face.edge_ids[0], face.edge_ids[1], face.edge_ids[2]],
            [corner_uv[0], corner_uv[1], centroid_uv],
        ),
        (
            [face.node_ids[1], face.node_ids[2], centroid_node_id],
            [face.edge_ids[1], face.edge_ids[2], face.edge_ids[0]],
            [corner_uv[1], corner_uv[2], centroid_uv],
        ),
        (
            [face.node_ids[2], face.node_ids[0], centroid_node_id],
            [face.edge_ids[2], face.edge_ids[0], face.edge_ids[1]],
            [corner_uv[2], corner_uv[0], centroid_uv],
        ),
    ];
    for (node_ids, source_edge_ids, parametric_node_uv) in child_specs {
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: Some(frame.face_id.clone()),
            source_edge_ids,
            node_ids,
            parametric_node_uv,
            max_projection_error_m: corner_projection_error_m,
            region_ids: face.region_ids.clone(),
            area_m2: face.area_m2 / 3.0,
            unit_normal: frame.unit_normal,
        });
    }
}

fn triangle_centroid(points: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (points[0][0] + points[1][0] + points[2][0]) / 3.0,
        (points[0][1] + points[1][1] + points[2][1]) / 3.0,
        (points[0][2] + points[1][2] + points[2][2]) / 3.0,
    ]
}

fn validate_face_vertices(
    topology: &SourceTopologyModel,
    face: &SourceTopologyFace,
) -> Result<(), SurfaceDiscretizationError> {
    for node_id in face.node_ids {
        if topology
            .vertices
            .get(node_id as usize)
            .is_none_or(|vertex| vertex.vertex_id != node_id)
        {
            return Err(SurfaceDiscretizationError::MissingFaceVertex {
                face_id: face.face_id,
                node_id,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        SourceTopologyEdge, SourceTopologyFace, SourceTopologyModel, SourceTopologyVertex,
    };

    #[test]
    fn discretizes_source_faces_as_surface_elements() {
        let surface = discretize_topology_surfaces(
            &single_triangle_topology(),
            SurfaceDiscretizationOptions::default(),
        )
        .expect("surface should discretize");

        assert_eq!(surface.nodes.len(), 3);
        assert_eq!(surface.elements.len(), 1);
        assert_eq!(surface.elements[0].source_face_id, 7);
        assert_eq!(surface.elements[0].cad_face_id, None);
        assert_eq!(surface.elements[0].source_edge_ids, [0, 1, 2]);
        assert_eq!(surface.elements[0].parametric_node_uv, [[0.0, 0.0]; 3]);
        assert_eq!(surface.elements[0].max_projection_error_m, 0.0);
        assert_eq!(surface.elements[0].region_ids, vec!["face_a".to_string()]);
        assert!((surface.elements[0].area_m2 - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn discretizes_surfaces_with_cad_face_ownership() {
        let topology = single_triangle_topology();
        let cad_topology =
            crate::build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
        let cad_evaluation =
            crate::build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

        let surface = discretize_cad_surfaces(
            &topology,
            &cad_evaluation,
            SurfaceDiscretizationOptions::default(),
        )
        .expect("cad-owned surface should discretize");

        assert_eq!(surface.elements.len(), 1);
        assert_eq!(
            surface.elements[0].cad_face_id,
            Some("cad_face_7".to_string())
        );
        assert_eq!(surface.elements[0].parametric_node_uv.len(), 3);
        assert_eq!(surface.elements[0].max_projection_error_m, 0.0);
    }

    #[test]
    fn centroid_subdivision_preserves_cad_face_ownership_and_boundary_edges() {
        let topology = single_triangle_topology();
        let cad_topology =
            crate::build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
        let cad_evaluation =
            crate::build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

        let surface = discretize_cad_surfaces(
            &topology,
            &cad_evaluation,
            SurfaceDiscretizationOptions {
                centroid_subdivision: true,
                ..SurfaceDiscretizationOptions::default()
            },
        )
        .expect("cad-owned surface should subdivide");

        assert_eq!(surface.nodes.len(), 4);
        assert_eq!(surface.elements.len(), 3);
        assert!(surface
            .elements
            .iter()
            .all(|element| element.cad_face_id == Some("cad_face_7".to_string())));
        assert!(surface.elements.iter().any(|element| {
            element.node_ids[0..2] == [0, 1] && element.source_edge_ids[0] == 0
        }));
        assert_eq!(surface.nodes[3].coordinates_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    }

    #[test]
    fn rejects_missing_face_vertices() {
        let mut topology = single_triangle_topology();
        topology.vertices.pop();

        let err = discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
            .expect_err("missing face vertex should fail");

        assert_eq!(
            err,
            SurfaceDiscretizationError::MissingFaceVertex {
                face_id: 7,
                node_id: 2,
            }
        );
    }

    fn single_triangle_topology() -> SourceTopologyModel {
        SourceTopologyModel {
            mesh_id: "surface".to_string(),
            source_geometry_id: "geo".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
            vertices: vec![
                SourceTopologyVertex {
                    vertex_id: 0,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                SourceTopologyVertex {
                    vertex_id: 1,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
                SourceTopologyVertex {
                    vertex_id: 2,
                    coordinates_m: [0.0, 1.0, 0.0],
                },
            ],
            edges: vec![
                SourceTopologyEdge {
                    edge_id: 0,
                    node_ids: [0, 1],
                    adjacent_face_ids: vec![7],
                    region_ids: vec!["face_a".to_string()],
                    length_m: 1.0,
                },
                SourceTopologyEdge {
                    edge_id: 1,
                    node_ids: [1, 2],
                    adjacent_face_ids: vec![7],
                    region_ids: vec!["face_a".to_string()],
                    length_m: 2.0_f64.sqrt(),
                },
                SourceTopologyEdge {
                    edge_id: 2,
                    node_ids: [0, 2],
                    adjacent_face_ids: vec![7],
                    region_ids: vec!["face_a".to_string()],
                    length_m: 1.0,
                },
            ],
            faces: vec![SourceTopologyFace {
                face_id: 7,
                source_triangle_id: 11,
                node_ids: [0, 1, 2],
                edge_ids: [0, 1, 2],
                region_ids: vec!["face_a".to_string()],
                area_m2: 0.5,
                unit_normal: [0.0, 0.0, 1.0],
            }],
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [1.0, 1.0, 0.0],
            region_ids: vec!["face_a".to_string()],
        }
    }

    fn geometry_for_topology() -> runmat_geometry_core::GeometryAsset {
        runmat_geometry_core::GeometryAsset {
            geometry_id: "geo".to_string(),
            source: runmat_geometry_core::GeometrySource {
                path: "/fixtures/surface.step".to_string(),
                sha256: "surface".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: runmat_geometry_core::SourceGeometry {
                kind: runmat_geometry_core::SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
                cad_evaluators: Vec::new(),
            },
            tessellation_profile: runmat_geometry_core::TessellationProfile::default(),
            units: runmat_geometry_core::UnitSystem::Meter,
            revision: 1,
            meshes: Vec::new(),
            surface_meshes: Vec::new(),
            regions: Vec::new(),
            region_entity_mappings: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}
