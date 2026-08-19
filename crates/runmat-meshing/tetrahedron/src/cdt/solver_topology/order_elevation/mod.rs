use std::collections::BTreeMap;

use runmat_meshing_core::{
    MeshingCancellationSignal, SolverMeshTopology, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

use super::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyInput, DelaunaySolverTopologyOptions,
};

mod connectivity;
mod geometry;
mod jacobian;
mod optimization;

pub(super) fn elevate(
    input: &DelaunaySolverTopologyInput<'_>,
    mut topology: SolverMeshTopology,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SolverMeshTopology, DelaunaySolverTopologyError> {
    let evaluation = input.exact_evaluation.ok_or_else(|| {
        error::failure(
            DelaunaySolverTopologyErrorKind::InvalidOptions,
            "Tet10 elevation requires exact curve, pcurve, surface, and trim evaluation",
        )
    })?;
    let midpoint_by_edge =
        geometry::append_midpoint_nodes(input, &mut topology, evaluation, options, cancellation)?;
    connectivity::elevate(&mut topology, &midpoint_by_edge)?;
    let mut jacobian_work = 0_u64;
    jacobian::validate(
        &topology,
        input.request.resources.maximum_search_work,
        input.request.resources.maximum_recursion_depth,
        options.cancellation_check_interval,
        cancellation,
        &mut jacobian_work,
    )?;
    optimization::optimize(
        input,
        &mut topology,
        &midpoint_by_edge,
        evaluation,
        options,
        cancellation,
        &mut jacobian_work,
    )?;
    // Re-certify the complete joined topology independently after all accepted rounds.
    jacobian::validate(
        &topology,
        input.request.resources.maximum_search_work,
        input.request.resources.maximum_recursion_depth,
        options.cancellation_check_interval,
        cancellation,
        &mut jacobian_work,
    )?;
    Ok(topology)
}

pub(super) fn sorted_edge(mut nodes: [u64; 2]) -> [u64; 2] {
    nodes.sort_unstable();
    nodes
}

pub(super) type MidpointMap = BTreeMap<[u64; 2], u64>;
