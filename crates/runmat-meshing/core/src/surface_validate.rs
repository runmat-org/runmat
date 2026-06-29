use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    predicate::{dot, norm, point_triangle_distance, triangle_area, Triangle3},
    source_topology::{SourceTopologyEdge, SourceTopologyModel},
    surface::{SurfaceDiscretization, SurfaceElement, INTERNAL_SOURCE_EDGE_ID},
    tolerance::MeshingTolerance,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceValidationOptions {
    pub max_projection_error_m: f64,
    pub min_orientation_alignment: f64,
    pub require_source_edge_conformity: bool,
}

impl Default for SurfaceValidationOptions {
    fn default() -> Self {
        Self {
            max_projection_error_m: 1.0e-8,
            min_orientation_alignment: 1.0 - 1.0e-8,
            require_source_edge_conformity: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceValidationReport {
    pub source_face_count: usize,
    pub surface_element_count: usize,
    pub source_edge_loop_count: usize,
    pub closed_source_edge_loop_count: usize,
    pub conforming_source_edge_count: usize,
    pub missing_source_edge_count: usize,
    pub max_projection_error_m: f64,
    pub min_orientation_alignment: f64,
    pub face_coverage_ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceValidationError {
    InvalidOptions,
    EmptySurface,
    MissingSurfaceNode {
        element_id: u32,
        node_id: u32,
    },
    MissingSourceFace {
        source_face_id: u32,
    },
    MissingSourceEdge {
        source_edge_id: u32,
    },
    EdgeConformityFailed {
        source_edge_id: u32,
        source_edge_node_ids: [u32; 2],
        recovered_segment_count: usize,
    },
    OpenSourceLoop {
        source_edge_id: u32,
        endpoint_id: u32,
        incidence_count: usize,
    },
    DegenerateElement {
        element_id: u32,
    },
    ProjectionError {
        element_id: u32,
        error_m: f64,
        max_error_m: f64,
    },
    OrientationMismatch {
        element_id: u32,
        source_face_id: u32,
        alignment: f64,
        min_alignment: f64,
    },
    UncoveredSourceFace {
        source_face_id: u32,
    },
}

impl std::fmt::Display for SurfaceValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions => write!(
                formatter,
                "surface validation options must use finite projection and orientation thresholds"
            ),
            Self::EmptySurface => write!(formatter, "surface validation input has no elements"),
            Self::MissingSurfaceNode {
                element_id,
                node_id,
            } => write!(
                formatter,
                "surface element {element_id} references missing node {node_id}"
            ),
            Self::MissingSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is missing")
            }
            Self::MissingSourceEdge { source_edge_id } => {
                write!(formatter, "source edge {source_edge_id} is missing")
            }
            Self::EdgeConformityFailed {
                source_edge_id,
                source_edge_node_ids,
                recovered_segment_count,
            } => write!(
                formatter,
                "source edge {source_edge_id} with endpoints {:?} is not represented by a matching surface element edge; recovered surface segment count is {recovered_segment_count}",
                source_edge_node_ids
            ),
            Self::OpenSourceLoop {
                source_edge_id,
                endpoint_id,
                incidence_count,
            } => write!(
                formatter,
                "source edge {source_edge_id} endpoint {endpoint_id} has loop incidence {incidence_count}, expected 2"
            ),
            Self::DegenerateElement { element_id } => {
                write!(formatter, "surface element {element_id} is degenerate")
            }
            Self::ProjectionError {
                element_id,
                error_m,
                max_error_m,
            } => write!(
                formatter,
                "surface element {element_id} projection error {error_m:.6e} m exceeds {max_error_m:.6e} m"
            ),
            Self::OrientationMismatch {
                element_id,
                source_face_id,
                alignment,
                min_alignment,
            } => write!(
                formatter,
                "surface element {element_id} on source face {source_face_id} orientation alignment {alignment:.6e} is below {min_alignment:.6e}"
            ),
            Self::UncoveredSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is not covered")
            }
        }
    }
}

impl std::error::Error for SurfaceValidationError {}

pub fn validate_surface_discretization(
    topology: &SourceTopologyModel,
    surface: &SurfaceDiscretization,
    options: SurfaceValidationOptions,
) -> Result<SurfaceValidationReport, SurfaceValidationError> {
    validate_options(options)?;
    if surface.elements.is_empty() {
        return Err(SurfaceValidationError::EmptySurface);
    }

    let tolerance = MeshingTolerance::from_bounds(topology.bounds_min_m, topology.bounds_max_m);
    let source_faces = topology
        .faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    let source_edges = topology
        .edges
        .iter()
        .map(|edge| (edge.edge_id, edge))
        .collect::<BTreeMap<_, _>>();
    let surface_edges = surface_edge_source_ids(surface);

    let mut covered_source_faces = BTreeSet::<u32>::new();
    let mut conforming_source_edges = BTreeSet::<u32>::new();
    let mut max_projection_error_m = 0.0_f64;
    let mut min_orientation_alignment = f64::INFINITY;

    for element in &surface.elements {
        let source_face = source_faces.get(&element.source_face_id).ok_or(
            SurfaceValidationError::MissingSourceFace {
                source_face_id: element.source_face_id,
            },
        )?;
        let points = surface_element_points(surface, element)?;
        if triangle_area(points) <= tolerance.length_epsilon(1.0).powi(2) {
            return Err(SurfaceValidationError::DegenerateElement {
                element_id: element.element_id,
            });
        }
        let source_points = topology_face_points(topology, source_face.node_ids)?;
        let projection_error_m = points
            .into_iter()
            .map(|point| point_triangle_distance(point, source_points))
            .fold(0.0_f64, f64::max);
        max_projection_error_m = max_projection_error_m.max(projection_error_m);
        if projection_error_m > options.max_projection_error_m.max(tolerance.absolute_m) {
            return Err(SurfaceValidationError::ProjectionError {
                element_id: element.element_id,
                error_m: projection_error_m,
                max_error_m: options.max_projection_error_m,
            });
        }

        let surface_normal =
            unit_normal(points).ok_or(SurfaceValidationError::DegenerateElement {
                element_id: element.element_id,
            })?;
        let alignment = dot(surface_normal, source_face.unit_normal);
        min_orientation_alignment = min_orientation_alignment.min(alignment);
        if alignment < options.min_orientation_alignment {
            return Err(SurfaceValidationError::OrientationMismatch {
                element_id: element.element_id,
                source_face_id: element.source_face_id,
                alignment,
                min_alignment: options.min_orientation_alignment,
            });
        }

        covered_source_faces.insert(element.source_face_id);
        for source_edge_id in element.source_edge_ids {
            if source_edge_id == INTERNAL_SOURCE_EDGE_ID {
                continue;
            }
            source_edges
                .get(&source_edge_id)
                .ok_or(SurfaceValidationError::MissingSourceEdge { source_edge_id })?;
        }
    }

    for (source_edge_id, source_edge) in &source_edges {
        if source_edge_is_recovered_by_chain(
            surface_edges
                .get(source_edge_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            source_edge,
        ) {
            conforming_source_edges.insert(*source_edge_id);
        }
    }

    if options.require_source_edge_conformity {
        for edge in &topology.edges {
            if !conforming_source_edges.contains(&edge.edge_id) {
                return Err(SurfaceValidationError::EdgeConformityFailed {
                    source_edge_id: edge.edge_id,
                    source_edge_node_ids: edge.node_ids,
                    recovered_segment_count: surface_edges
                        .get(&edge.edge_id)
                        .map(Vec::len)
                        .unwrap_or_default(),
                });
            }
        }
    }

    for source_face in source_faces.keys() {
        if !covered_source_faces.contains(source_face) {
            return Err(SurfaceValidationError::UncoveredSourceFace {
                source_face_id: *source_face,
            });
        }
    }

    let (source_edge_loop_count, closed_source_edge_loop_count) =
        count_closed_source_edge_loops(&topology.edges)?;

    Ok(SurfaceValidationReport {
        source_face_count: topology.faces.len(),
        surface_element_count: surface.elements.len(),
        source_edge_loop_count,
        closed_source_edge_loop_count,
        conforming_source_edge_count: conforming_source_edges.len(),
        missing_source_edge_count: topology
            .edges
            .len()
            .saturating_sub(conforming_source_edges.len()),
        max_projection_error_m,
        min_orientation_alignment: if min_orientation_alignment.is_finite() {
            min_orientation_alignment
        } else {
            1.0
        },
        face_coverage_ratio: covered_source_faces.len() as f64 / topology.faces.len() as f64,
    })
}

fn validate_options(options: SurfaceValidationOptions) -> Result<(), SurfaceValidationError> {
    if !options.max_projection_error_m.is_finite()
        || options.max_projection_error_m < 0.0
        || !options.min_orientation_alignment.is_finite()
        || !(0.0..=1.0).contains(&options.min_orientation_alignment)
    {
        return Err(SurfaceValidationError::InvalidOptions);
    }
    Ok(())
}

fn count_closed_source_edge_loops(
    edges: &[SourceTopologyEdge],
) -> Result<(usize, usize), SurfaceValidationError> {
    let mut endpoint_incidence = BTreeMap::<u32, usize>::new();
    let mut endpoint_edges = BTreeMap::<u32, Vec<u32>>::new();
    let mut edges_by_id = BTreeMap::<u32, &SourceTopologyEdge>::new();
    for edge in edges {
        edges_by_id.insert(edge.edge_id, edge);
        *endpoint_incidence.entry(edge.node_ids[0]).or_default() += 1;
        *endpoint_incidence.entry(edge.node_ids[1]).or_default() += 1;
        endpoint_edges
            .entry(edge.node_ids[0])
            .or_default()
            .push(edge.edge_id);
        endpoint_edges
            .entry(edge.node_ids[1])
            .or_default()
            .push(edge.edge_id);
    }

    let mut visited_edges = BTreeSet::<u32>::new();
    let mut component_count = 0_usize;
    let mut closed_count = 0_usize;
    for edge in edges {
        if !visited_edges.insert(edge.edge_id) {
            continue;
        }

        component_count += 1;
        let mut closed = true;
        let mut component_edge_ids = Vec::<u32>::new();
        let mut stack = vec![edge.edge_id];
        while let Some(edge_id) = stack.pop() {
            component_edge_ids.push(edge_id);
            let component_edge =
                edges_by_id
                    .get(&edge_id)
                    .ok_or(SurfaceValidationError::MissingSourceEdge {
                        source_edge_id: edge_id,
                    })?;
            for endpoint_id in component_edge.node_ids {
                let incidence_count = endpoint_incidence.get(&endpoint_id).copied().unwrap_or(0);
                if incidence_count < 2 {
                    closed = false;
                }
                if let Some(adjacent_edges) = endpoint_edges.get(&endpoint_id) {
                    for adjacent_edge_id in adjacent_edges {
                        if visited_edges.insert(*adjacent_edge_id) {
                            stack.push(*adjacent_edge_id);
                        }
                    }
                }
            }
        }

        if closed {
            closed_count += 1;
        } else {
            for endpoint_id in edge.node_ids {
                let incidence_count = endpoint_incidence.get(&endpoint_id).copied().unwrap_or(0);
                if incidence_count < 2 {
                    return Err(SurfaceValidationError::OpenSourceLoop {
                        source_edge_id: edge.edge_id,
                        endpoint_id,
                        incidence_count,
                    });
                }
            }
            for edge_id in component_edge_ids {
                let component_edge =
                    edges_by_id
                        .get(&edge_id)
                        .ok_or(SurfaceValidationError::MissingSourceEdge {
                            source_edge_id: edge_id,
                        })?;
                for endpoint_id in component_edge.node_ids {
                    let incidence_count =
                        endpoint_incidence.get(&endpoint_id).copied().unwrap_or(0);
                    if incidence_count < 2 {
                        return Err(SurfaceValidationError::OpenSourceLoop {
                            source_edge_id: edge_id,
                            endpoint_id,
                            incidence_count,
                        });
                    }
                }
            }
        }
    }

    Ok((component_count, closed_count))
}

#[cfg(test)]
fn source_edge(edge_id: u32, node_ids: [u32; 2]) -> SourceTopologyEdge {
    SourceTopologyEdge {
        edge_id,
        node_ids,
        adjacent_face_ids: Vec::new(),
        region_ids: Vec::new(),
        length_m: 1.0,
    }
}

fn surface_edge_source_ids(surface: &SurfaceDiscretization) -> BTreeMap<u32, Vec<[u32; 2]>> {
    let mut edges = BTreeMap::<u32, Vec<[u32; 2]>>::new();
    for element in &surface.elements {
        for (source_edge_id, node_ids) in element.source_edge_ids.into_iter().zip([
            sorted_edge(element.node_ids[0], element.node_ids[1]),
            sorted_edge(element.node_ids[1], element.node_ids[2]),
            sorted_edge(element.node_ids[2], element.node_ids[0]),
        ]) {
            if source_edge_id != INTERNAL_SOURCE_EDGE_ID {
                edges.entry(source_edge_id).or_default().push(node_ids);
            }
        }
    }
    edges
}

fn source_edge_is_recovered_by_chain(
    segments: &[[u32; 2]],
    source_edge: &SourceTopologyEdge,
) -> bool {
    let source_edge_nodes = sorted_edge(source_edge.node_ids[0], source_edge.node_ids[1]);
    if segments.contains(&source_edge_nodes) {
        return true;
    }
    let mut adjacency = BTreeMap::<u32, Vec<u32>>::new();
    for segment in segments {
        adjacency.entry(segment[0]).or_default().push(segment[1]);
        adjacency.entry(segment[1]).or_default().push(segment[0]);
    }
    let mut stack = vec![source_edge.node_ids[0]];
    let mut visited = BTreeSet::<u32>::new();
    while let Some(node_id) = stack.pop() {
        if !visited.insert(node_id) {
            continue;
        }
        if node_id == source_edge.node_ids[1] {
            return true;
        }
        if let Some(next) = adjacency.get(&node_id) {
            stack.extend(next.iter().copied());
        }
    }
    false
}

fn topology_face_points(
    topology: &SourceTopologyModel,
    node_ids: [u32; 3],
) -> Result<Triangle3, SurfaceValidationError> {
    Ok([
        topology_node(topology, node_ids[0])?,
        topology_node(topology, node_ids[1])?,
        topology_node(topology, node_ids[2])?,
    ])
}

fn topology_node(
    topology: &SourceTopologyModel,
    node_id: u32,
) -> Result<[f64; 3], SurfaceValidationError> {
    topology
        .vertices
        .get(node_id as usize)
        .filter(|vertex| vertex.vertex_id == node_id)
        .map(|vertex| vertex.coordinates_m)
        .ok_or(SurfaceValidationError::MissingSurfaceNode {
            element_id: u32::MAX,
            node_id,
        })
}

fn surface_element_points(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
) -> Result<Triangle3, SurfaceValidationError> {
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
) -> Result<[f64; 3], SurfaceValidationError> {
    surface
        .nodes
        .get(node_id as usize)
        .filter(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
        .ok_or(SurfaceValidationError::MissingSurfaceNode {
            element_id: element.element_id,
            node_id,
        })
}

fn unit_normal(points: Triangle3) -> Option<[f64; 3]> {
    let normal = crate::predicate::cross(
        crate::predicate::sub(points[1], points[0]),
        crate::predicate::sub(points[2], points[0]),
    );
    let length = norm(normal);
    (length > 0.0).then_some([normal[0] / length, normal[1] / length, normal[2] / length])
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
    use crate::{
        discretize_topology_surfaces, predicate::distance, SourceTopologyEdge, SourceTopologyFace,
        SourceTopologyModel, SourceTopologyVertex, SurfaceDiscretizationOptions,
    };

    #[test]
    fn validates_surface_projection_and_source_edge_loops() {
        let topology = cube_topology();
        let surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");

        let report = validate_surface_discretization(
            &topology,
            &surface,
            SurfaceValidationOptions::default(),
        )
        .expect("surface validation should pass");

        assert_eq!(report.source_face_count, 12);
        assert_eq!(report.surface_element_count, 12);
        assert_eq!(report.source_edge_loop_count, 1);
        assert_eq!(report.closed_source_edge_loop_count, 1);
        assert_eq!(report.conforming_source_edge_count, 18);
        assert_eq!(report.missing_source_edge_count, 0);
        assert_eq!(report.max_projection_error_m, 0.0);
        assert_eq!(report.face_coverage_ratio, 1.0);
    }

    #[test]
    fn rejects_surface_projection_drift() {
        let topology = cube_topology();
        let mut surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");
        surface.nodes[0].coordinates_m = [0.0, 0.0, 0.1];

        let err = validate_surface_discretization(
            &topology,
            &surface,
            SurfaceValidationOptions {
                max_projection_error_m: 1.0e-6,
                ..SurfaceValidationOptions::default()
            },
        )
        .expect_err("projection drift should fail");

        assert!(matches!(
            err,
            SurfaceValidationError::ProjectionError { .. }
        ));
    }

    #[test]
    fn orientation_mismatch_reports_source_face() {
        let topology = cube_topology();
        let mut surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");
        let source_face_id = surface.elements[0].source_face_id;
        surface.elements[0].node_ids.swap(1, 2);

        let err = validate_surface_discretization(
            &topology,
            &surface,
            SurfaceValidationOptions::default(),
        )
        .expect_err("flipped surface orientation should fail");

        assert_eq!(
            err,
            SurfaceValidationError::OrientationMismatch {
                element_id: 0,
                source_face_id,
                alignment: -1.0,
                min_alignment: SurfaceValidationOptions::default().min_orientation_alignment,
            }
        );
        assert!(err
            .to_string()
            .contains(&format!("source face {source_face_id}")));
    }

    #[test]
    fn edge_conformity_failure_reports_recovered_segment_count() {
        let topology = cube_topology();
        let mut surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");
        let edge = topology.edges[0].clone();
        for element in &mut surface.elements {
            for source_edge_id in &mut element.source_edge_ids {
                if *source_edge_id == edge.edge_id {
                    *source_edge_id = INTERNAL_SOURCE_EDGE_ID;
                }
            }
        }

        let err = validate_surface_discretization(
            &topology,
            &surface,
            SurfaceValidationOptions::default(),
        )
        .expect_err("missing source edge recovery should fail");

        assert_eq!(
            err,
            SurfaceValidationError::EdgeConformityFailed {
                source_edge_id: edge.edge_id,
                source_edge_node_ids: edge.node_ids,
                recovered_segment_count: 0,
            }
        );
        let message = err.to_string();
        assert!(message.contains(&format!("source edge {}", edge.edge_id)));
        assert!(message.contains("recovered surface segment count is 0"));
    }

    #[test]
    fn rejects_open_source_loop() {
        let mut topology = cube_topology();
        topology.edges[0].node_ids = [100, 101];
        let surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");

        let err = validate_surface_discretization(
            &topology,
            &surface,
            SurfaceValidationOptions {
                require_source_edge_conformity: false,
                ..SurfaceValidationOptions::default()
            },
        )
        .expect_err("open source loop should fail");

        assert!(matches!(err, SurfaceValidationError::OpenSourceLoop { .. }));
    }

    #[test]
    fn counts_disconnected_closed_source_edge_loop_components() {
        let edges = vec![
            source_edge(0, [0, 1]),
            source_edge(1, [1, 2]),
            source_edge(2, [2, 0]),
            source_edge(3, [3, 4]),
            source_edge(4, [4, 5]),
            source_edge(5, [5, 3]),
        ];

        let (loop_count, closed_loop_count) =
            count_closed_source_edge_loops(&edges).expect("closed loops should count");

        assert_eq!(loop_count, 2);
        assert_eq!(closed_loop_count, 2);
    }

    #[test]
    fn rejects_disconnected_open_source_edge_component() {
        let edges = vec![
            source_edge(0, [0, 1]),
            source_edge(1, [1, 2]),
            source_edge(2, [2, 0]),
            source_edge(3, [3, 4]),
        ];

        let err =
            count_closed_source_edge_loops(&edges).expect_err("open component should fail closed");

        assert!(matches!(
            err,
            SurfaceValidationError::OpenSourceLoop {
                source_edge_id: 3,
                ..
            }
        ));
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
        let face_nodes = [
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
        ];
        let face_normals = [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ];
        let mut edge_ids = BTreeMap::<[u32; 2], u32>::new();
        let mut edge_faces = BTreeMap::<[u32; 2], Vec<u32>>::new();
        let mut faces = Vec::<SourceTopologyFace>::new();
        for (face_id, node_ids) in face_nodes.into_iter().enumerate() {
            let mut face_edge_ids = [0_u32; 3];
            for (index, edge) in [
                sorted_edge(node_ids[0], node_ids[1]),
                sorted_edge(node_ids[1], node_ids[2]),
                sorted_edge(node_ids[2], node_ids[0]),
            ]
            .into_iter()
            .enumerate()
            {
                let next_edge_id = edge_ids.len() as u32;
                let edge_id = *edge_ids.entry(edge).or_insert(next_edge_id);
                face_edge_ids[index] = edge_id;
                edge_faces.entry(edge).or_default().push(face_id as u32);
            }
            faces.push(SourceTopologyFace {
                face_id: face_id as u32,
                source_triangle_id: face_id as u32,
                node_ids,
                edge_ids: face_edge_ids,
                region_ids: Vec::new(),
                area_m2: 0.5,
                unit_normal: face_normals[face_id],
            });
        }
        let mut edges = edge_ids
            .into_iter()
            .map(|(node_ids, edge_id)| SourceTopologyEdge {
                edge_id,
                node_ids,
                adjacent_face_ids: edge_faces.remove(&node_ids).unwrap_or_default(),
                region_ids: Vec::new(),
                length_m: distance(
                    vertices[node_ids[0] as usize].coordinates_m,
                    vertices[node_ids[1] as usize].coordinates_m,
                ),
            })
            .collect::<Vec<_>>();
        edges.sort_by_key(|edge| edge.edge_id);

        SourceTopologyModel {
            mesh_id: "surface_validate_cube".to_string(),
            source_geometry_id: "geo_surface_validate_cube".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
            vertices,
            edges,
            faces,
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [1.0, 1.0, 1.0],
            region_ids: Vec::new(),
        }
    }
}
