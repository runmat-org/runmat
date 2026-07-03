use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    math::{cross, dot, norm, sub, triangle_area, MeshingTolerance, Triangle3},
    SurfaceDiscretization, SurfaceElement,
};
use runmat_meshing_cad::SourceTopologyModel;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceRecoveryOptions {
    pub require_closed: bool,
    pub max_area_relative_error: f64,
    pub min_normal_alignment: f64,
}

impl Default for SurfaceRecoveryOptions {
    fn default() -> Self {
        Self {
            require_closed: true,
            max_area_relative_error: 1.0e-8,
            min_normal_alignment: 1.0 - 1.0e-8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceRecoveryReport {
    pub surface_element_count: usize,
    pub recovered_edge_count: usize,
    pub open_edge_count: usize,
    pub nonmanifold_edge_count: usize,
    pub max_area_relative_error: f64,
    pub min_normal_alignment: f64,
    pub source_face_coverage_ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceRecoveryError {
    EmptySurface,
    InvalidOptions,
    MissingSurfaceNode {
        element_id: u32,
        node_id: u32,
    },
    NonFiniteSurfaceNode {
        node_id: u32,
    },
    DegenerateElement {
        element_id: u32,
    },
    AreaMismatch {
        element_id: u32,
        relative_error: f64,
        max_relative_error: f64,
    },
    SourceFaceAreaMismatch {
        source_face_id: u32,
        relative_error: f64,
        max_relative_error: f64,
    },
    NormalMismatch {
        element_id: u32,
        alignment: f64,
        min_alignment: f64,
    },
    MissingSourceFace {
        source_face_id: u32,
    },
    UncoveredSourceFace {
        source_face_id: u32,
    },
    OpenEdge {
        edge: [u32; 2],
        count: usize,
    },
    NonManifoldEdge {
        edge: [u32; 2],
        count: usize,
    },
}

impl std::fmt::Display for SurfaceRecoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySurface => write!(formatter, "surface recovery input has no elements"),
            Self::InvalidOptions => write!(
                formatter,
                "surface recovery options must use finite area and normal thresholds"
            ),
            Self::MissingSurfaceNode {
                element_id,
                node_id,
            } => write!(
                formatter,
                "surface element {element_id} references missing node {node_id}"
            ),
            Self::NonFiniteSurfaceNode { node_id } => {
                write!(formatter, "surface node {node_id} has non-finite coordinates")
            }
            Self::DegenerateElement { element_id } => {
                write!(formatter, "surface element {element_id} is degenerate")
            }
            Self::AreaMismatch {
                element_id,
                relative_error,
                max_relative_error,
            } => write!(
                formatter,
                "surface element {element_id} area relative error {relative_error:.6e} exceeds {max_relative_error:.6e}"
            ),
            Self::SourceFaceAreaMismatch {
                source_face_id,
                relative_error,
                max_relative_error,
            } => write!(
                formatter,
                "source face {source_face_id} recovered area relative error {relative_error:.6e} exceeds {max_relative_error:.6e}"
            ),
            Self::NormalMismatch {
                element_id,
                alignment,
                min_alignment,
            } => write!(
                formatter,
                "surface element {element_id} normal alignment {alignment:.6e} is below {min_alignment:.6e}"
            ),
            Self::MissingSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is not present in topology")
            }
            Self::UncoveredSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is not covered by surface mesh")
            }
            Self::OpenEdge { edge, count } => write!(
                formatter,
                "surface edge {}-{} has incidence {count}, expected 2",
                edge[0], edge[1]
            ),
            Self::NonManifoldEdge { edge, count } => write!(
                formatter,
                "surface edge {}-{} has non-manifold incidence {count}, expected 2",
                edge[0], edge[1]
            ),
        }
    }
}

impl std::error::Error for SurfaceRecoveryError {}

pub fn validate_surface_recovery(
    topology: &SourceTopologyModel,
    surface: &SurfaceDiscretization,
    options: SurfaceRecoveryOptions,
) -> Result<SurfaceRecoveryReport, SurfaceRecoveryError> {
    validate_options(options)?;
    if surface.elements.is_empty() {
        return Err(SurfaceRecoveryError::EmptySurface);
    }

    let tolerance = MeshingTolerance::from_bounds(topology.bounds_min_m, topology.bounds_max_m);
    let source_faces = topology
        .faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    let mut covered_source_faces = BTreeSet::<u32>::new();
    let mut edge_incidence = BTreeMap::<[u32; 2], usize>::new();
    let mut recovered_area_by_source_face = BTreeMap::<u32, f64>::new();
    let mut max_area_relative_error = 0.0_f64;
    let mut min_normal_alignment = f64::INFINITY;

    for element in &surface.elements {
        let source_face = source_faces.get(&element.source_face_id).ok_or(
            SurfaceRecoveryError::MissingSourceFace {
                source_face_id: element.source_face_id,
            },
        )?;
        let points = surface_element_points(surface, element)?;
        let area_m2 = triangle_area(points);
        if !area_m2.is_finite() || area_m2 <= tolerance.length_epsilon(1.0).powi(2) {
            return Err(SurfaceRecoveryError::DegenerateElement {
                element_id: element.element_id,
            });
        }

        let expected_area = element.area_m2;
        let relative_error = if expected_area > 0.0 && expected_area.is_finite() {
            (area_m2 - expected_area).abs() / expected_area
        } else {
            0.0
        };
        max_area_relative_error = max_area_relative_error.max(relative_error);
        if relative_error > options.max_area_relative_error {
            return Err(SurfaceRecoveryError::AreaMismatch {
                element_id: element.element_id,
                relative_error,
                max_relative_error: options.max_area_relative_error,
            });
        }

        *recovered_area_by_source_face
            .entry(element.source_face_id)
            .or_default() += area_m2;
        let actual_normal =
            triangle_unit_normal(points).ok_or(SurfaceRecoveryError::DegenerateElement {
                element_id: element.element_id,
            })?;
        let normal_alignment = dot(actual_normal, source_face.unit_normal).abs();
        let normal_alignment = if normal_alignment.is_finite() {
            normal_alignment
        } else {
            -1.0
        };
        min_normal_alignment = min_normal_alignment.min(normal_alignment);
        if normal_alignment < options.min_normal_alignment {
            return Err(SurfaceRecoveryError::NormalMismatch {
                element_id: element.element_id,
                alignment: normal_alignment,
                min_alignment: options.min_normal_alignment,
            });
        }

        if norm(source_face.unit_normal) <= tolerance.length_epsilon(1.0) {
            return Err(SurfaceRecoveryError::NormalMismatch {
                element_id: element.element_id,
                alignment: 0.0,
                min_alignment: options.min_normal_alignment,
            });
        }

        covered_source_faces.insert(element.source_face_id);
        for edge in triangle_edges(element.node_ids) {
            *edge_incidence.entry(edge).or_default() += 1;
        }
    }

    for source_face in source_faces.keys() {
        if !covered_source_faces.contains(source_face) {
            return Err(SurfaceRecoveryError::UncoveredSourceFace {
                source_face_id: *source_face,
            });
        }
    }

    for source_face in topology.faces.iter() {
        let recovered_area = recovered_area_by_source_face
            .get(&source_face.face_id)
            .copied()
            .unwrap_or(0.0);
        let expected_area = source_face.area_m2;
        let relative_error = if expected_area > 0.0 && expected_area.is_finite() {
            (recovered_area - expected_area).abs() / expected_area
        } else {
            0.0
        };
        max_area_relative_error = max_area_relative_error.max(relative_error);
        if relative_error > options.max_area_relative_error {
            return Err(SurfaceRecoveryError::SourceFaceAreaMismatch {
                source_face_id: source_face.face_id,
                relative_error,
                max_relative_error: options.max_area_relative_error,
            });
        }
    }

    let mut open_edge_count = 0_usize;
    let mut nonmanifold_edge_count = 0_usize;
    let mut recovered_edge_count = 0_usize;
    for (edge, count) in &edge_incidence {
        match *count {
            2 => recovered_edge_count += 1,
            0 | 1 => {
                open_edge_count += 1;
                if options.require_closed {
                    return Err(SurfaceRecoveryError::OpenEdge {
                        edge: *edge,
                        count: *count,
                    });
                }
            }
            _ => {
                nonmanifold_edge_count += 1;
                if options.require_closed {
                    return Err(SurfaceRecoveryError::NonManifoldEdge {
                        edge: *edge,
                        count: *count,
                    });
                }
            }
        }
    }

    Ok(SurfaceRecoveryReport {
        surface_element_count: surface.elements.len(),
        recovered_edge_count,
        open_edge_count,
        nonmanifold_edge_count,
        max_area_relative_error,
        min_normal_alignment: if min_normal_alignment.is_finite() {
            min_normal_alignment
        } else {
            1.0
        },
        source_face_coverage_ratio: covered_source_faces.len() as f64 / source_faces.len() as f64,
    })
}

fn validate_options(options: SurfaceRecoveryOptions) -> Result<(), SurfaceRecoveryError> {
    if !options.max_area_relative_error.is_finite()
        || options.max_area_relative_error < 0.0
        || !options.min_normal_alignment.is_finite()
        || !(0.0..=1.0).contains(&options.min_normal_alignment)
    {
        return Err(SurfaceRecoveryError::InvalidOptions);
    }
    Ok(())
}

fn triangle_unit_normal(points: Triangle3) -> Option<[f64; 3]> {
    let normal = cross(sub(points[1], points[0]), sub(points[2], points[0]));
    let length = norm(normal);
    (length > 0.0).then_some([normal[0] / length, normal[1] / length, normal[2] / length])
}

fn surface_element_points(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
) -> Result<Triangle3, SurfaceRecoveryError> {
    Ok([
        surface_node(surface, element, element.node_ids[0])?,
        surface_node(surface, element, element.node_ids[1])?,
        surface_node(surface, element, element.node_ids[2])?,
    ])
}

fn surface_node(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
    node_id: u32,
) -> Result<[f64; 3], SurfaceRecoveryError> {
    let node = surface
        .nodes
        .get(node_id as usize)
        .filter(|node| node.node_id == node_id)
        .ok_or(SurfaceRecoveryError::MissingSurfaceNode {
            element_id: element.element_id,
            node_id,
        })?;
    if node.coordinates_m.iter().any(|value| !value.is_finite()) {
        return Err(SurfaceRecoveryError::NonFiniteSurfaceNode { node_id });
    }
    Ok(node.coordinates_m)
}

fn triangle_edges(triangle: [u32; 3]) -> [[u32; 2]; 3] {
    [
        sorted_edge(triangle[0], triangle[1]),
        sorted_edge(triangle[1], triangle[2]),
        sorted_edge(triangle[2], triangle[0]),
    ]
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{discretize_topology_surfaces, SurfaceDiscretizationOptions};
    use runmat_meshing_cad::{SourceTopologyFace, SourceTopologyModel, SourceTopologyVertex};

    #[test]
    fn validates_closed_surface_recovery() {
        let topology = cube_topology();
        let surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");

        let report =
            validate_surface_recovery(&topology, &surface, SurfaceRecoveryOptions::default())
                .expect("surface recovery should validate");

        assert_eq!(report.surface_element_count, 12);
        assert_eq!(report.open_edge_count, 0);
        assert_eq!(report.nonmanifold_edge_count, 0);
        assert_eq!(report.source_face_coverage_ratio, 1.0);
        assert!(report.min_normal_alignment >= 1.0 - 1.0e-8);
    }

    #[test]
    fn rejects_surface_with_open_edge() {
        let topology = cube_topology();
        let mut surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");
        surface.elements.pop();

        let err = validate_surface_recovery(
            &topology,
            &surface,
            SurfaceRecoveryOptions {
                require_closed: true,
                ..SurfaceRecoveryOptions::default()
            },
        )
        .expect_err("open surface should fail recovery");

        assert!(matches!(
            err,
            SurfaceRecoveryError::UncoveredSourceFace { .. }
                | SurfaceRecoveryError::OpenEdge { .. }
        ));
    }

    #[test]
    fn rejects_surface_area_mismatch() {
        let topology = cube_topology();
        let mut surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");
        surface.nodes[1].coordinates_m = [2.0, 0.0, 0.0];

        let err = validate_surface_recovery(
            &topology,
            &surface,
            SurfaceRecoveryOptions {
                max_area_relative_error: 1.0e-12,
                ..SurfaceRecoveryOptions::default()
            },
        )
        .expect_err("area mismatch should fail recovery");

        assert!(matches!(err, SurfaceRecoveryError::AreaMismatch { .. }));
    }

    fn cube_topology() -> SourceTopologyModel {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(vertex_id, coordinates_m)| SourceTopologyVertex {
            vertex_id: vertex_id as u32,
            coordinates_m,
        })
        .collect::<Vec<_>>();
        let faces = [
            ([0, 2, 1], [0.0, 0.0, -1.0]),
            ([0, 3, 2], [0.0, 0.0, -1.0]),
            ([4, 5, 6], [0.0, 0.0, 1.0]),
            ([4, 6, 7], [0.0, 0.0, 1.0]),
            ([0, 1, 5], [0.0, -1.0, 0.0]),
            ([0, 5, 4], [0.0, -1.0, 0.0]),
            ([1, 2, 6], [1.0, 0.0, 0.0]),
            ([1, 6, 5], [1.0, 0.0, 0.0]),
            ([2, 3, 7], [0.0, 1.0, 0.0]),
            ([2, 7, 6], [0.0, 1.0, 0.0]),
            ([3, 0, 4], [-1.0, 0.0, 0.0]),
            ([3, 4, 7], [-1.0, 0.0, 0.0]),
        ]
        .into_iter()
        .enumerate()
        .map(|(face_id, (node_ids, unit_normal))| SourceTopologyFace {
            face_id: face_id as u32,
            source_triangle_id: face_id as u32,
            node_ids,
            edge_ids: [
                face_id as u32 * 3,
                face_id as u32 * 3 + 1,
                face_id as u32 * 3 + 2,
            ],
            region_ids: Vec::new(),
            area_m2: 0.5,
            unit_normal,
        })
        .collect::<Vec<_>>();

        SourceTopologyModel {
            mesh_id: "surface_recovery_cube".to_string(),
            source_geometry_id: "geo_surface_recovery_cube".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
            vertices,
            edges: Vec::new(),
            faces,
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [1.0, 1.0, 1.0],
            region_ids: Vec::new(),
        }
    }
}
