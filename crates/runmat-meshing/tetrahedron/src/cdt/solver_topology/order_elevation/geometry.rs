use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    sort_solver_node_exact_parameters, MeshingCancellationSignal, SolverMeshNode,
    SolverMeshTopology, SolverNodeExactParameter, StableDigest,
};
use runmat_meshing_surface::ExactFaceMeshNode;

use super::{sorted_edge, MidpointMap, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};
use crate::cdt::solver_topology::{
    checkpoint, error, DelaunayExactEvaluation, DelaunaySolverTopologyError,
    DelaunaySolverTopologyErrorKind, DelaunaySolverTopologyInput, DelaunaySolverTopologyOptions,
};

mod evaluation;
use evaluation::{
    curve_parameter, evaluate_surface, midpoint_geometry, pcurve_uvs, require_matching_points,
};
pub(super) use evaluation::{edge_geometry, surface_geometry, EdgeGeometryRequest};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct SurfaceUse {
    pub(super) source_face_id: PersistentEntityId,
    pub(super) chart_id: StableDigest,
}

struct VolumeEdgeInventory {
    edges: BTreeSet<[u64; 2]>,
    provenance: BTreeMap<[u64; 2], BTreeSet<PersistentEntityId>>,
}

struct BoundaryParameterCompletion<'a> {
    node_index: &'a BTreeMap<u64, usize>,
    surface_uses: &'a BTreeMap<[u64; 2], Vec<SurfaceUse>>,
    exact_edges: &'a BTreeMap<[u64; 2], PersistentEntityId>,
    evaluation: DelaunayExactEvaluation<'a>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &'a dyn MeshingCancellationSignal,
}

pub(super) fn append_midpoint_nodes(
    input: &DelaunaySolverTopologyInput<'_>,
    topology: &mut SolverMeshTopology,
    evaluation: DelaunayExactEvaluation<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<MidpointMap, DelaunaySolverTopologyError> {
    let mut node_index = BTreeMap::new();
    for (index, node) in topology.nodes.iter().enumerate() {
        boundary_parameter_checkpoint(cancellation)?;
        node_index.insert(node.node_id, index);
    }
    let edge_inventory = volume_edge_inventory(topology, cancellation)?;
    let final_node_count = topology
        .nodes
        .len()
        .checked_add(edge_inventory.edges.len())
        .ok_or_else(|| resource("Tet10 node inventory overflows the platform index space"))?;
    if final_node_count as u64 > input.request.resources.maximum_nodes {
        return Err(resource(format!(
            "Tet10 elevation requires {final_node_count} nodes but the hard limit is {}",
            input.request.resources.maximum_nodes
        )));
    }
    let surface_uses = boundary_surface_uses(input, topology, cancellation)?;
    let exact_edges = boundary_exact_edges(topology, cancellation)?;
    complete_boundary_surface_parameters(
        input,
        topology,
        BoundaryParameterCompletion {
            node_index: &node_index,
            surface_uses: &surface_uses,
            exact_edges: &exact_edges,
            evaluation,
            options,
            cancellation,
        },
    )?;
    let mut midpoint_by_edge = BTreeMap::new();
    for (work, edge) in edge_inventory.edges.into_iter().enumerate() {
        checkpoint(work as u64, options, cancellation)?;
        let left = topology
            .nodes
            .get(
                *node_index
                    .get(&edge[0])
                    .ok_or_else(|| invalid("edge node is absent"))?,
            )
            .ok_or_else(|| invalid("edge node index is invalid"))?;
        let right = topology
            .nodes
            .get(
                *node_index
                    .get(&edge[1])
                    .ok_or_else(|| invalid("edge node is absent"))?,
            )
            .ok_or_else(|| invalid("edge node index is invalid"))?;
        let uses = surface_uses.get(&edge).cloned().unwrap_or_default();
        let exact_edge = exact_edges.get(&edge);
        let (coordinates_m, exact_parameters) =
            midpoint_geometry(input, left, right, exact_edge, &uses, evaluation, options)?;
        let provenance =
            midpoint_provenance(edge_inventory.provenance.get(&edge), &uses, exact_edge)?;
        let node_id = topology.nodes.len() as u64 + 1;
        topology.nodes.push(SolverMeshNode {
            node_id,
            stable_identity: runmat_meshing_core::solver_midside_node_identity(
                edge.map(|node_id| topology.nodes[node_index[&node_id]].stable_identity),
            ),
            coordinates_m,
            provenance,
            exact_parameters,
        });
        midpoint_by_edge.insert(edge, node_id);
    }
    Ok(midpoint_by_edge)
}

fn volume_edge_inventory(
    topology: &SolverMeshTopology,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<VolumeEdgeInventory, DelaunaySolverTopologyError> {
    let mut edges = BTreeSet::new();
    let mut provenance = BTreeMap::<[u64; 2], BTreeSet<PersistentEntityId>>::new();
    for element in &topology.volume_elements {
        boundary_parameter_checkpoint(cancellation)?;
        let owners = element
            .provenance
            .iter()
            .cloned()
            .chain(std::iter::once(element.region_id.clone()))
            .collect::<Vec<_>>();
        for local_edge in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
            let edge = sorted_edge([
                element.node_ids[local_edge[0]],
                element.node_ids[local_edge[1]],
            ]);
            edges.insert(edge);
            provenance
                .entry(edge)
                .or_default()
                .extend(owners.iter().cloned());
        }
    }
    Ok(VolumeEdgeInventory { edges, provenance })
}

fn complete_boundary_surface_parameters(
    input: &DelaunaySolverTopologyInput<'_>,
    topology: &mut SolverMeshTopology,
    completion: BoundaryParameterCompletion<'_>,
) -> Result<(), DelaunaySolverTopologyError> {
    let BoundaryParameterCompletion {
        node_index,
        surface_uses,
        exact_edges,
        evaluation,
        options,
        cancellation,
    } = completion;
    let mut source_nodes = BTreeMap::new();
    for source in &input.exact_surface.nodes {
        boundary_parameter_checkpoint(cancellation)?;
        source_nodes.insert(source.node_id, source);
    }
    for (edge, edge_id) in exact_edges {
        let Some(uses) = surface_uses.get(edge) else {
            continue;
        };
        for node_id in edge {
            boundary_parameter_checkpoint(cancellation)?;
            let node = topology
                .nodes
                .get_mut(
                    *node_index
                        .get(node_id)
                        .ok_or_else(|| invalid("edge node is absent"))?,
                )
                .ok_or_else(|| invalid("edge node index is invalid"))?;
            let parameter = curve_parameter(node, edge_id)?;
            for surface_use in uses {
                for evaluator_uv in pcurve_uvs(input, edge_id, parameter, surface_use, evaluation)?
                {
                    let point =
                        evaluate_surface(input, surface_use, evaluator_uv, evaluation, options)?;
                    require_matching_points(input, node.coordinates_m, point)?;
                    insert_surface_parameter(node, surface_use, evaluator_uv)?;
                }
            }
        }
    }
    for (edge, uses) in surface_uses {
        for node_id in edge {
            boundary_parameter_checkpoint(cancellation)?;
            let node = topology
                .nodes
                .get_mut(
                    *node_index
                        .get(node_id)
                        .ok_or_else(|| invalid("surface edge node is absent"))?,
                )
                .ok_or_else(|| invalid("surface edge node index is invalid"))?;
            for surface_use in uses {
                for evaluator_uv in derive_surface_parameters(
                    input,
                    &source_nodes,
                    node,
                    surface_use,
                    cancellation,
                )? {
                    insert_surface_parameter(node, surface_use, evaluator_uv)?;
                }
            }
        }
    }
    Ok(())
}

fn derive_surface_parameters(
    input: &DelaunaySolverTopologyInput<'_>,
    source_nodes: &BTreeMap<StableDigest, &ExactFaceMeshNode>,
    node: &SolverMeshNode,
    surface_use: &SurfaceUse,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<[f64; 2]>, DelaunaySolverTopologyError> {
    let mut parameters =
        node.exact_parameters
            .iter()
            .filter_map(|parameter| match parameter {
                SolverNodeExactParameter::Surface {
                    source_face_id,
                    chart_id,
                    evaluator_uv,
                } if source_face_id == &surface_use.source_face_id
                    && chart_id == &surface_use.chart_id =>
                {
                    Some(*evaluator_uv)
                }
                SolverNodeExactParameter::Curve { .. }
                | SolverNodeExactParameter::Surface { .. } => None,
            })
            .collect::<Vec<_>>();
    for triangle in input.exact_surface.triangles.iter().filter(|triangle| {
        triangle.source_face_id == surface_use.source_face_id
            && triangle.chart_id == surface_use.chart_id
    }) {
        boundary_parameter_checkpoint(cancellation)?;
        let sources = triangle
            .node_ids
            .map(|identity| source_nodes.get(&identity).copied());
        let [Some(first), Some(second), Some(third)] = sources else {
            return Err(invalid(
                "source surface triangle references an absent exact surface node",
            ));
        };
        let points = [first.point_m, second.point_m, third.point_m];
        let tolerance_m = input
            .request
            .tolerance
            .source_tolerance_m
            .max(input.request.tolerance.absolute_floor_m)
            .max(input.request.tolerance.maximum_healing_displacement_m);
        let Some(weights) = triangle_coordinates(points, node.coordinates_m, tolerance_m) else {
            continue;
        };
        let uses = [first, second, third].map(|source| {
            source
                .uses
                .iter()
                .filter(|candidate| {
                    candidate.source_face_id == surface_use.source_face_id
                        && candidate.chart_id == surface_use.chart_id
                })
                .collect::<Vec<_>>()
        });
        if uses.iter().any(Vec::is_empty) {
            continue;
        }
        for first_use in &uses[0] {
            for second_use in &uses[1] {
                for third_use in &uses[2] {
                    parameters.push(std::array::from_fn(|axis| {
                        weights[0] * first_use.evaluator_uv[axis]
                            + weights[1] * second_use.evaluator_uv[axis]
                            + weights[2] * third_use.evaluator_uv[axis]
                    }));
                }
            }
        }
    }
    parameters.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
    });
    parameters.dedup_by(|left, right| left.map(f64::to_bits) == right.map(f64::to_bits));
    if parameters.is_empty() {
        return Err(invalid(format!(
            "surface edge node {} cannot be located in its authoritative source triangle",
            node.node_id
        )));
    }
    Ok(parameters)
}

fn boundary_parameter_checkpoint(
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySolverTopologyError> {
    if cancellation.is_cancelled() {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::Cancelled,
            "cancelled",
        ));
    }
    Ok(())
}

fn triangle_coordinates(
    points: [[f64; 3]; 3],
    point: [f64; 3],
    tolerance_m: f64,
) -> Option<[f64; 3]> {
    let subtract =
        |left: [f64; 3], right: [f64; 3]| std::array::from_fn(|axis| left[axis] - right[axis]);
    let dot = |left: [f64; 3], right: [f64; 3]| {
        left.into_iter().zip(right).map(|(a, b)| a * b).sum::<f64>()
    };
    let first_edge = subtract(points[1], points[0]);
    let second_edge = subtract(points[2], points[0]);
    let offset = subtract(point, points[0]);
    let first_squared = dot(first_edge, first_edge);
    let cross = dot(first_edge, second_edge);
    let second_squared = dot(second_edge, second_edge);
    let first_offset = dot(offset, first_edge);
    let second_offset = dot(offset, second_edge);
    let denominator = first_squared * second_squared - cross * cross;
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let second = (second_squared * first_offset - cross * second_offset) / denominator;
    let third = (first_squared * second_offset - cross * first_offset) / denominator;
    let first = 1.0 - second - third;
    let weights = [first, second, third];
    let scale = first_squared.max(second_squared).sqrt();
    if !scale.is_finite() || scale == 0.0 || !tolerance_m.is_finite() || tolerance_m < 0.0 {
        return None;
    }
    let coordinate_tolerance = (tolerance_m / scale).max(64.0 * f64::EPSILON);
    if weights
        .iter()
        .any(|weight| *weight < -coordinate_tolerance || *weight > 1.0 + coordinate_tolerance)
    {
        return None;
    }
    let reconstructed = std::array::from_fn(|axis| {
        first * points[0][axis] + second * points[1][axis] + third * points[2][axis]
    });
    let residual = subtract(point, reconstructed);
    (dot(residual, residual).sqrt() <= tolerance_m.max(64.0 * f64::EPSILON * scale))
        .then_some(weights)
}

fn insert_surface_parameter(
    node: &mut SolverMeshNode,
    surface_use: &SurfaceUse,
    evaluator_uv: [f64; 2],
) -> Result<(), DelaunaySolverTopologyError> {
    if node.exact_parameters.iter().any(|parameter| {
        matches!(
            parameter,
            SolverNodeExactParameter::Surface {
                source_face_id,
                chart_id,
                evaluator_uv: existing,
            } if source_face_id == &surface_use.source_face_id
                && chart_id == &surface_use.chart_id
                && existing.map(f64::to_bits) == evaluator_uv.map(f64::to_bits)
        )
    }) {
        return Ok(());
    }
    // A boundary vertex can have several authoritative parameter images on
    // one surface chart: periodic seams differ by a period, while singular
    // poles can admit many values along the collapsed parameter direction.
    // Preserve every distinct coedge image. Midpoint construction later
    // chooses the nearest compatible pair instead of inventing one value.
    node.exact_parameters
        .push(SolverNodeExactParameter::Surface {
            source_face_id: surface_use.source_face_id.clone(),
            chart_id: surface_use.chart_id,
            evaluator_uv,
        });
    sort_solver_node_exact_parameters(&mut node.exact_parameters);
    Ok(())
}

pub(super) fn boundary_surface_uses(
    input: &DelaunaySolverTopologyInput<'_>,
    topology: &SolverMeshTopology,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<BTreeMap<[u64; 2], Vec<SurfaceUse>>, DelaunaySolverTopologyError> {
    let mut uses = BTreeMap::<[u64; 2], BTreeSet<SurfaceUse>>::new();
    for face in &topology.boundary_faces {
        boundary_parameter_checkpoint(cancellation)?;
        let facet = input
            .volume_mesh
            .provenance
            .facets
            .get(face.face_id as usize - 1)
            .ok_or_else(|| invalid("boundary face has no recovered facet lineage"))?;
        let source_face_id = facet
            .entity_ids
            .iter()
            .find(|entity| entity.kind == PersistentEntityKind::Face)
            .ok_or_else(|| invalid("recovered facet has no exact face"))?
            .clone();
        let use_record = SurfaceUse {
            source_face_id,
            chart_id: facet.chart_id,
        };
        for pair in [[0, 1], [1, 2], [2, 0]] {
            uses.entry(sorted_edge([
                face.node_ids[pair[0]],
                face.node_ids[pair[1]],
            ]))
            .or_default()
            .insert(use_record.clone());
        }
    }
    Ok(uses
        .into_iter()
        .map(|(edge, uses)| (edge, uses.into_iter().collect()))
        .collect())
}

pub(super) fn boundary_exact_edges(
    topology: &SolverMeshTopology,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<BTreeMap<[u64; 2], PersistentEntityId>, DelaunaySolverTopologyError> {
    let mut result = BTreeMap::new();
    for edge in &topology.boundary_edges {
        boundary_parameter_checkpoint(cancellation)?;
        let exact = edge
            .provenance
            .iter()
            .filter(|entity| entity.kind == PersistentEntityKind::Edge)
            .collect::<Vec<_>>();
        if exact.len() > 1 {
            return Err(invalid(
                "one boundary mesh edge has multiple exact edge owners",
            ));
        }
        if let Some(exact) = exact.first() {
            result.insert(
                sorted_edge([edge.node_ids[0], edge.node_ids[1]]),
                (*exact).clone(),
            );
        }
    }
    Ok(result)
}

fn midpoint_provenance(
    volume_provenance: Option<&BTreeSet<PersistentEntityId>>,
    surface_uses: &[SurfaceUse],
    exact_edge: Option<&PersistentEntityId>,
) -> Result<Vec<PersistentEntityId>, DelaunaySolverTopologyError> {
    let mut provenance = volume_provenance.cloned().unwrap_or_default();
    provenance.extend(
        surface_uses
            .iter()
            .map(|use_record| use_record.source_face_id.clone()),
    );
    if let Some(edge) = exact_edge {
        provenance.insert(edge.clone());
    }
    if provenance.is_empty()
        || provenance.len() > super::super::construction::MAX_PROVENANCE_PER_ENTITY
    {
        return Err(invalid(
            "elevated node provenance is empty or exceeds its hard bound",
        ));
    }
    Ok(provenance.into_iter().collect())
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidGeometry, reason)
}

fn resource(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::ResourceLimit, reason)
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::MeshingCancellationSignal;

    use super::{boundary_parameter_checkpoint, triangle_coordinates};

    struct Cancelled;

    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    #[test]
    fn boundary_parameter_recovery_observes_cancellation() {
        let failure = boundary_parameter_checkpoint(&Cancelled)
            .expect_err("boundary parameter recovery must stop when cancellation is requested");
        assert_eq!(
            failure.kind,
            crate::cdt::solver_topology::DelaunaySolverTopologyErrorKind::Cancelled
        );
    }

    #[test]
    fn triangle_coordinates_use_the_governed_geometry_tolerance() {
        let triangle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let weights = triangle_coordinates(triangle, [0.25, 0.25, 1.0e-10], 1.0e-8)
            .expect("point within the source tolerance must retain its surface lineage");
        assert!((weights[0] - 0.5).abs() < 1.0e-12);
        assert!((weights[1] - 0.25).abs() < 1.0e-12);
        assert!((weights[2] - 0.25).abs() < 1.0e-12);
        assert!(triangle_coordinates(triangle, [0.25, 0.25, 1.0e-4], 1.0e-8).is_none());
    }

    #[test]
    fn triangle_coordinates_scale_the_inside_test_for_small_geometry() {
        let triangle = [[0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0], [0.0, 1.0e-4, 0.0]];
        assert!(triangle_coordinates(triangle, [-1.0e-10, 5.0e-5, 0.0], 1.0e-9).is_some());
        assert!(triangle_coordinates(triangle, [-1.0e-6, 5.0e-5, 0.0], 1.0e-9).is_none());
    }
}
