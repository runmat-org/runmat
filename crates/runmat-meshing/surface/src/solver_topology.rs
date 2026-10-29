//! Canonical solver topology for shell (sheet triangle) and beam (edge segment) meshes.
//!
//! A surface mesh publishes every exact-surface triangle as a linear boundary face with no
//! adjacent volume, plus every triangle edge as a linear boundary edge; property realization
//! turns faces with shell sections into MITC3 shells and edges with beam sections into beams.
//! A curve mesh publishes only the exact-edge segments, so beam frames carry no unused nodes.

use std::collections::BTreeMap;

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId};
use runmat_meshing_core::{
    solver_boundary_edge_identity, solver_boundary_face_identity,
    sort_solver_node_exact_parameters, BoundaryEdgeOrder, BoundaryFaceRole, BoundaryTriangleOrder,
    FieldTopologyLocation, FieldTopologyMap, MeshDimension, MeshingCancellationSignal,
    MeshingContractError, MeshingRequest, SolverBoundaryEdge, SolverBoundaryFace, SolverMeshNode,
    SolverMeshTopology, SolverNodeExactParameter, StableDigest, ValidatedSolverMeshTopology,
};

use crate::ExactSurfaceMesh;

const MAX_PROVENANCE: usize = 32;

/// Builds and independently validates the solver topology of a surface or curve request.
pub fn build_surface_solver_topology(
    surface: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    request: &MeshingRequest,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<ValidatedSolverMeshTopology, MeshingContractError> {
    request.validate()?;
    let dimension = request.dimension;
    if dimension.is_solid() {
        return Err(invalid(
            "solid requests are projected from their tetrahedral volume",
        ));
    }
    let vertices = topology
        .vertices
        .iter()
        .map(|vertex| (&vertex.id, vertex))
        .collect::<BTreeMap<_, _>>();
    let edges = topology
        .edges
        .iter()
        .map(|edge| (&edge.id, edge))
        .collect::<BTreeMap<_, _>>();

    // Curve meshes keep only the nodes of exact-edge segments.
    let mut segment_nodes = BTreeMap::<StableDigest, ()>::new();
    for segment in &surface.boundary_segments {
        for node in segment.node_ids {
            segment_nodes.insert(node, ());
        }
    }
    let mut nodes = Vec::new();
    let mut node_index = BTreeMap::<StableDigest, u64>::new();
    for (work, node) in surface.nodes.iter().enumerate() {
        checkpoint(work, cancellation)?;
        if dimension == MeshDimension::Curve && !segment_nodes.contains_key(&node.node_id) {
            continue;
        }
        let mut provenance = Vec::<PersistentEntityId>::new();
        let mut parameters = Vec::<SolverNodeExactParameter>::new();
        for node_use in &node.uses {
            if dimension == MeshDimension::Surface {
                provenance.push(node_use.source_face_id.clone());
                parameters.push(SolverNodeExactParameter::Surface {
                    source_face_id: node_use.source_face_id.clone(),
                    chart_id: node_use.chart_id,
                    evaluator_uv: node_use.evaluator_uv,
                });
            }
            for edge_parameter in &node_use.exact_edge_parameters {
                provenance.push(edge_parameter.source_edge_id.clone());
                parameters.push(SolverNodeExactParameter::Curve {
                    source_edge_id: edge_parameter.source_edge_id.clone(),
                    parameter: edge_parameter.parameter,
                });
                // A node at an exact-edge endpoint is that topological vertex.
                let edge = edges
                    .get(&edge_parameter.source_edge_id)
                    .ok_or_else(|| invalid("surface node references an absent exact edge"))?;
                for vertex_id in [&edge.start_vertex_id, &edge.end_vertex_id]
                    .into_iter()
                    .flatten()
                {
                    let vertex = vertices
                        .get(vertex_id)
                        .ok_or_else(|| invalid("exact edge references an absent vertex"))?;
                    let tolerance = vertex
                        .tolerance_m
                        .max(edge.tolerance_m)
                        .max(request.tolerance.absolute_floor_m);
                    if distance(vertex.point_m.canonical_value(), node.point_m)
                        <= tolerance.canonical_value()
                    {
                        provenance.push(vertex.id.clone());
                    }
                }
            }
        }
        provenance.sort();
        provenance.dedup();
        sort_solver_node_exact_parameters(&mut parameters);
        parameters.dedup();
        if provenance.is_empty() || provenance.len() > MAX_PROVENANCE {
            return Err(invalid(
                "surface node provenance is empty or exceeds its bound",
            ));
        }
        let node_id = nodes.len() as u64 + 1;
        node_index.insert(node.node_id, node_id);
        nodes.push(SolverMeshNode {
            node_id,
            stable_identity: node.node_id,
            coordinates_m: runmat_types::physical::Position::new(node.point_m),
            provenance,
            exact_parameters: parameters,
        });
    }
    let solver_node = |identity: &StableDigest| {
        node_index
            .get(identity)
            .copied()
            .ok_or_else(|| invalid("surface element references an absent node"))
    };
    let identity_of = |id: u64| nodes[id as usize - 1].stable_identity;

    let mut boundary_faces = Vec::new();
    if dimension == MeshDimension::Surface {
        for (work, triangle) in surface.triangles.iter().enumerate() {
            checkpoint(work, cancellation)?;
            let node_ids = [
                solver_node(&triangle.node_ids[0])?,
                solver_node(&triangle.node_ids[1])?,
                solver_node(&triangle.node_ids[2])?,
            ];
            boundary_faces.push(SolverBoundaryFace {
                face_id: boundary_faces.len() as u64 + 1,
                stable_identity: solver_boundary_face_identity(node_ids.map(identity_of)),
                order: BoundaryTriangleOrder::Tri3,
                node_ids: node_ids.to_vec(),
                adjacent_volume_element_ids: Vec::new(),
                role: BoundaryFaceRole::Exterior,
                provenance: vec![triangle.source_face_id.clone()],
            });
        }
    }

    // Exact-edge provenance of each segment, keyed by its sorted solver node pair.
    let mut segment_edges = BTreeMap::<[u64; 2], Vec<PersistentEntityId>>::new();
    for segment in &surface.boundary_segments {
        let mut pair = [
            solver_node(&segment.node_ids[0])?,
            solver_node(&segment.node_ids[1])?,
        ];
        pair.sort_unstable();
        let entities = segment_edges.entry(pair).or_default();
        if !entities.contains(&segment.source_edge_id) {
            entities.push(segment.source_edge_id.clone());
        }
    }
    let mut edge_rows = BTreeMap::<[u64; 2], (Vec<u64>, Vec<PersistentEntityId>)>::new();
    match dimension {
        MeshDimension::Surface => {
            for face in &boundary_faces {
                for [left, right] in [[0, 1], [1, 2], [2, 0]] {
                    let mut pair = [face.node_ids[left], face.node_ids[right]];
                    pair.sort_unstable();
                    let row = edge_rows.entry(pair).or_default();
                    row.0.push(face.face_id);
                    match segment_edges.get(&pair) {
                        Some(entities) => row.1.extend(entities.iter().cloned()),
                        None => row.1.extend(face.provenance.iter().cloned()),
                    }
                }
            }
        }
        MeshDimension::Curve => {
            for (pair, entities) in &segment_edges {
                edge_rows.insert(*pair, (Vec::new(), entities.clone()));
            }
        }
        MeshDimension::Solid => {
            return Err(invalid(
                "solid requests are projected from their tetrahedral volume",
            ))
        }
    }
    let mut boundary_edges = Vec::with_capacity(edge_rows.len());
    for (work, (pair, (mut adjacent, mut provenance))) in edge_rows.into_iter().enumerate() {
        checkpoint(work, cancellation)?;
        adjacent.sort_unstable();
        adjacent.dedup();
        provenance.sort();
        provenance.dedup();
        if provenance.len() > MAX_PROVENANCE {
            return Err(invalid("surface edge provenance exceeds its bound"));
        }
        boundary_edges.push(SolverBoundaryEdge {
            edge_id: boundary_edges.len() as u64 + 1,
            stable_identity: solver_boundary_edge_identity(pair.map(identity_of)),
            order: BoundaryEdgeOrder::Line2,
            node_ids: pair.to_vec(),
            adjacent_boundary_face_ids: adjacent,
            provenance,
        });
    }
    let field_topologies = [
        ("nodes", FieldTopologyLocation::Node, nodes.len()),
        ("elements", FieldTopologyLocation::VolumeElement, 0),
        (
            "boundary_faces",
            FieldTopologyLocation::BoundaryFace,
            boundary_faces.len(),
        ),
        (
            "boundary_edges",
            FieldTopologyLocation::BoundaryEdge,
            boundary_edges.len(),
        ),
    ]
    .into_iter()
    .map(|(topology_id, location, count)| FieldTopologyMap {
        topology_id: topology_id.into(),
        location,
        ordered_entity_ids: (1..=count as u64).collect(),
    })
    .collect();
    ValidatedSolverMeshTopology::new(
        SolverMeshTopology {
            nodes,
            volume_elements: Vec::new(),
            neighbors: Vec::new(),
            boundary_faces,
            boundary_edges,
            regions: Vec::new(),
            conformal_interfaces: Vec::new(),
            contacts: Vec::new(),
            field_topologies,
        },
        request.clone(),
        cancellation,
    )
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn checkpoint(
    work: usize,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    if work.is_multiple_of(1024) && cancellation.is_cancelled() {
        return Err(invalid("cancelled"));
    }
    Ok(())
}

fn invalid(reason: &str) -> MeshingContractError {
    MeshingContractError::invalid("surface solver topology", reason)
}
