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
    let mut jacobian_work = 0_u64;
    let mut candidate_work = 0_u64;
    let evaluation = input.exact_evaluation.ok_or_else(|| {
        error::failure(
            DelaunaySolverTopologyErrorKind::InvalidOptions,
            "Tet10 elevation requires exact curve, pcurve, surface, and trim evaluation",
        )
    })?;
    let midpoint_by_edge =
        geometry::append_midpoint_nodes(input, &mut topology, evaluation, options, cancellation)?;
    connectivity::elevate(&mut topology, &midpoint_by_edge, cancellation)?;
    let mut invalid_elements = jacobian::invalid_element_indices(
        &topology,
        input.request.resources.maximum_search_work,
        input.request.resources.maximum_recursion_depth,
        options.cancellation_check_interval,
        cancellation,
        &mut jacobian_work,
    )?;
    let initial_invalid_element_count = invalid_elements.len();
    let mut moved_node_count = 0_u64;
    for sweep in 0..options.maximum_curved_optimization_sweeps {
        let improve_valid_nodes = sweep == 0 && invalid_elements.is_empty();
        let moved = optimization::optimize(
            input,
            &mut topology,
            &midpoint_by_edge,
            evaluation,
            optimization::OptimizationWork {
                options,
                cancellation,
                jacobian_work: &mut jacobian_work,
                candidate_work: &mut candidate_work,
            },
            improve_valid_nodes,
            &invalid_elements,
        )?;
        moved_node_count = moved_node_count.checked_add(moved).ok_or_else(|| {
            error::failure(
                DelaunaySolverTopologyErrorKind::ResourceLimit,
                "curved-node movement counter overflowed",
            )
        })?;
        invalid_elements = jacobian::invalid_element_indices(
            &topology,
            input.request.resources.maximum_search_work,
            input.request.resources.maximum_recursion_depth,
            options.cancellation_check_interval,
            cancellation,
            &mut jacobian_work,
        )?;
        if invalid_elements.is_empty() {
            return Ok(topology);
        }
        if moved == 0 {
            let worst_element =
                jacobian::worst_invalid_element(&topology, &invalid_elements, cancellation)?
                    .ok_or_else(|| {
                        error::failure(
                            DelaunaySolverTopologyErrorKind::InvalidMesh,
                            "Tet10 invalid-element inventory has no worst element",
                        )
                    })?;
            let patch_moved = optimization::repair_corner_star(
                input,
                &mut topology,
                worst_element,
                evaluation,
                optimization::OptimizationWork {
                    options,
                    cancellation,
                    jacobian_work: &mut jacobian_work,
                    candidate_work: &mut candidate_work,
                },
            )?;
            moved_node_count = moved_node_count.checked_add(patch_moved).ok_or_else(|| {
                error::failure(
                    DelaunaySolverTopologyErrorKind::ResourceLimit,
                    "curved-node movement counter overflowed",
                )
            })?;
            if patch_moved > 0 {
                invalid_elements = jacobian::invalid_element_indices(
                    &topology,
                    input.request.resources.maximum_search_work,
                    input.request.resources.maximum_recursion_depth,
                    options.cancellation_check_interval,
                    cancellation,
                    &mut jacobian_work,
                )?;
                if invalid_elements.is_empty() {
                    return Ok(topology);
                }
                continue;
            }
            let relocated = optimization::relocate_interior_corner(
                &mut topology,
                worst_element,
                input.request.resources.maximum_search_work,
                input.request.resources.maximum_recursion_depth,
                optimization::OptimizationWork {
                    options,
                    cancellation,
                    jacobian_work: &mut jacobian_work,
                    candidate_work: &mut candidate_work,
                },
            )?;
            moved_node_count = moved_node_count.checked_add(relocated).ok_or_else(|| {
                error::failure(
                    DelaunaySolverTopologyErrorKind::ResourceLimit,
                    "curved-node movement counter overflowed",
                )
            })?;
            if relocated > 0 {
                invalid_elements = jacobian::invalid_element_indices(
                    &topology,
                    input.request.resources.maximum_search_work,
                    input.request.resources.maximum_recursion_depth,
                    options.cancellation_check_interval,
                    cancellation,
                    &mut jacobian_work,
                )?;
                if invalid_elements.is_empty() {
                    return Ok(topology);
                }
                continue;
            }
            return jacobian::validate(
                &topology,
                input.request.resources.maximum_search_work,
                input.request.resources.maximum_recursion_depth,
                options.cancellation_check_interval,
                cancellation,
                &mut jacobian_work,
            )
            .map(|()| topology);
        }
    }
    Err(error::failure(
        DelaunaySolverTopologyErrorKind::ResourceLimit,
        format!(
            "curved-node optimization did not certify the Tet10 mesh within {} hard sweeps ({} initially invalid elements, {} remaining, {} accepted node movements)",
            options.maximum_curved_optimization_sweeps,
            initial_invalid_element_count,
            invalid_elements.len(),
            moved_node_count,
        ),
    ))
}

pub(super) fn sorted_edge(mut nodes: [u64; 2]) -> [u64; 2] {
    nodes.sort_unstable();
    nodes
}

pub(super) type MidpointMap = BTreeMap<[u64; 2], u64>;
