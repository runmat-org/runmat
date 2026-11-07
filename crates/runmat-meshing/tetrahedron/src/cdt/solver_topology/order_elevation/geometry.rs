use super::PreparedEvaluation;

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    sort_solver_node_exact_parameters, MeshingCancellationSignal, SolverMeshNode,
    SolverMeshTopology, SolverNodeExactParameter, StableDigest,
};

use super::{sorted_edge, MidpointMap, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};
use crate::cdt::solver_topology::{
    checkpoint, error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyOptions, ProjectionInput,
};

mod edge_inventory;
mod sources;
pub(super) use sources::ParameterSources;
pub(super) mod boundary_sources;
pub(super) use boundary_sources::{BoundaryExactEdges, BoundarySurfaceUses};
mod completion;
mod evaluation;
use evaluation::{
    curve_parameter, evaluate_surface_point, midpoint_geometry, pcurve_uvs, require_matching_curve_points,
};
pub(super) use evaluation::{
    edge_geometry, surface_geometry, surface_parameters, EdgeGeometryRequest,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct SurfaceUse<'a> {
    pub(super) source_face_id: &'a PersistentEntityId,
    pub(super) chart_id: StableDigest,
}

struct BoundaryParameterCompletion<'source, 'view> {
    surface_uses: &'view BoundarySurfaceUses<'view>,
    exact_edges: &'view BoundaryExactEdges<'view>,
    evaluation: PreparedEvaluation<'source, 'view>,
    cancellation: &'view dyn MeshingCancellationSignal,
}

pub(super) fn append_midpoint_nodes<'source, 'control>(
    input: &ProjectionInput<'source>,
    topology: &mut SolverMeshTopology,
    evaluation: PreparedEvaluation<'source, 'control>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<MidpointMap<'control>, DelaunaySolverTopologyError> {
    let owner_count = input
        .exact_topology
        .regions
        .len()
        .checked_mul(2)
        .ok_or_else(|| resource("owner source inventory overflow"))?;
    let owner_bytes = owner_count
        .checked_mul(std::mem::size_of::<&PersistentEntityId>())
        .ok_or_else(|| resource("owner source storage overflow"))?;
    let owner_storage = runmat_geometry_core::GeometryAllocationReservation::new(
        evaluation.control,
        owner_bytes as u64,
    )
    .map_err(error::geometry)?;
    let mut owner_sources = input
        .exact_topology
        .regions
        .iter()
        .flat_map(|region| [&region.id, &region.solid_id])
        .collect::<Vec<_>>();
    let owner_scratch_storage = runmat_geometry_core::GeometryAllocationReservation::new(
        evaluation.control,
        owner_bytes as u64,
    )
    .map_err(error::geometry)?;
    let mut scratch = owner_sources.clone();
    runmat_cpu_execution::sort_copy_with_checkpoint(
        &mut owner_sources,
        &mut scratch,
        |a: &&PersistentEntityId, b: &&PersistentEntityId| a.cmp(b),
        || evaluation.control.checkpoint().map_err(error::geometry),
    )?;
    drop(scratch);
    drop(owner_scratch_storage);
    owner_sources.dedup();
    let edge_inventory =
        edge_inventory::EdgeOwners::prepare(topology, &owner_sources, evaluation.control)?;
    drop(owner_sources);
    drop(owner_storage);
    let final_node_count = topology
        .nodes
        .len()
        .checked_add(edge_inventory.groups.len())
        .ok_or_else(|| resource("Tet10 node inventory overflows the platform index space"))?;
    if final_node_count as u64 > input.request.limits.maximum_nodes {
        return Err(resource(format!(
            "Tet10 elevation requires {final_node_count} nodes but the hard limit is {}",
            input.request.limits.maximum_nodes
        )));
    }
    let storage = super::super::storage::Storage(Some(evaluation.control));
    storage.reserve(&mut topology.nodes, final_node_count)?;
    let surface_uses =
        boundary_sources::surface_uses(input, topology, evaluation.control, cancellation)?;
    let exact_edges =
        boundary_sources::exact_edges(input, topology, evaluation.control, cancellation)?;
    completion::complete_boundary_surface_parameters(
        input,
        topology,
        BoundaryParameterCompletion {
            surface_uses: &surface_uses,
            exact_edges: &exact_edges,
            evaluation,
            cancellation,
        },
    )?;
    let mut midpoint_by_edge = MidpointMap::new(edge_inventory.groups.len(), evaluation.control)?;
    for (work, (edge, owners)) in edge_inventory.groups.iter().enumerate() {
        let edge = *edge;
        checkpoint(work as u64, options, cancellation)?;
        let left = topology
            .nodes
            .get(node_slot(&topology.nodes, edge[0])?)
            .ok_or_else(|| invalid("edge node index is invalid"))?;
        let right = topology
            .nodes
            .get(node_slot(&topology.nodes, edge[1])?)
            .ok_or_else(|| invalid("edge node index is invalid"))?;
        let uses = surface_uses.get(&edge).unwrap_or_default();
        let exact_edge = exact_edges.get(&edge);
        let (coordinates_m, exact_parameters) =
            midpoint_geometry(input, left, right, exact_edge, uses, evaluation, options)?;
        let provenance = midpoint_provenance(
            edge_inventory.owners(owners.clone()),
            uses,
            exact_edge,
            storage,
        )?;
        let node_id = topology.nodes.len() as u64 + 1;
        topology.nodes.push(SolverMeshNode {
            node_id,
            stable_identity: runmat_meshing_core::solver_midside_node_identity([
                left.stable_identity,
                right.stable_identity,
            ]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates_m),
            provenance,
            exact_parameters,
        });
        midpoint_by_edge.push(edge, node_id)?;
    }
    Ok(midpoint_by_edge)
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
    storage: super::super::storage::Storage<'_>,
) -> Result<(), DelaunaySolverTopologyError> {
    if node.exact_parameters.iter().any(|parameter| {
        matches!(
            parameter,
            SolverNodeExactParameter::Surface {
                source_face_id,
                chart_id,
                evaluator_uv: existing,
            } if source_face_id == surface_use.source_face_id
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
    let parameter = SolverNodeExactParameter::Surface {
        source_face_id: storage.entity(surface_use.source_face_id)?,
        chart_id: surface_use.chart_id,
        evaluator_uv,
    };
    storage.push(&mut node.exact_parameters, parameter)?;
    sort_solver_node_exact_parameters(&mut node.exact_parameters);
    Ok(())
}

fn midpoint_provenance<'a>(
    volume_provenance: impl Iterator<Item = &'a PersistentEntityId>,
    surface_uses: &[SurfaceUse<'a>],
    exact_edge: Option<&'a PersistentEntityId>,
    storage: super::super::storage::Storage<'_>,
) -> Result<Vec<PersistentEntityId>, DelaunaySolverTopologyError> {
    let mut values = [None; super::super::construction::MAX_PROVENANCE_PER_ENTITY];
    let mut count = 0;
    for entity in volume_provenance
        .chain(surface_uses.iter().map(|usage| usage.source_face_id))
        .chain(exact_edge)
    {
        if values[..count].contains(&Some(entity)) {
            continue;
        }
        if count == values.len() {
            return Err(invalid("elevated node provenance exceeds its hard bound"));
        }
        values[count] = Some(entity);
        count += 1;
    }
    if count == 0 {
        return Err(invalid("elevated node provenance is empty"));
    }
    values[..count].sort_unstable();
    storage.entities(
        values[..count]
            .iter()
            .map(|value| value.expect("populated provenance")),
    )
}

/// This constructor assigns contiguous one-based IDs. Check that invariant
/// rather than retaining a second mesh-sized ID-to-position tree.
pub(super) fn node_slot(
    nodes: &[SolverMeshNode],
    id: u64,
) -> Result<usize, DelaunaySolverTopologyError> {
    let slot = id
        .checked_sub(1)
        .and_then(|id| usize::try_from(id).ok())
        .ok_or_else(|| invalid("solver node ID has no canonical position"))?;
    if nodes.get(slot).is_none_or(|node| node.node_id != id) {
        return Err(invalid(
            "solver node ID differs from its canonical position",
        ));
    }
    Ok(slot)
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
