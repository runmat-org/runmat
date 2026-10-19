use runmat_meshing_core::{
    MeshingCancellationSignal, SolverMeshTopology, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

use super::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyOptions, ProjectionInput,
};

mod affine;
mod connectivity;
mod geometry;
mod jacobian;
mod optimization;

#[derive(Clone, Copy)]
struct PreparedEvaluation<'source, 'view> {
    evaluator: &'view dyn crate::cdt::solver_topology::DelaunayExactEvaluator,
    control: &'view dyn runmat_geometry_core::GeometryEvaluationControl,
    sources: &'view geometry::ParameterSources<'source>,
}

pub(super) fn elevate(
    input: &ProjectionInput<'_>,
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
    let sources = geometry::ParameterSources::prepare(input, evaluation.control)
        .map_err(|failure| phase_error("exact parameter preparation", failure))?;
    let evaluation = PreparedEvaluation {
        evaluator: evaluation.evaluator,
        control: evaluation.control,
        sources: &sources,
    };
    let midpoint_by_edge =
        geometry::append_midpoint_nodes(input, &mut topology, evaluation, options, cancellation)
            .map_err(|failure| phase_error("exact midpoint construction", failure))?;
    connectivity::elevate(
        &mut topology,
        &midpoint_by_edge,
        cancellation,
        evaluation.control,
    )?;
    let mut invalid_elements = jacobian::invalid_element_indices(
        &topology,
        input.request.resources.maximum_search_work,
        input.request.resources.maximum_recursion_depth,
        options.cancellation_check_interval,
        cancellation,
        jacobian::ValidationWork {
            counter: &mut jacobian_work,
            storage: super::storage::Storage(Some(evaluation.control)),
        },
    )?;
    if invalid_elements.is_empty()
        && affine::retains_affine_edge_nodes(&topology, &midpoint_by_edge, evaluation.control)?
    {
        tracing::info!(
            event_code = "meshing.tet10.affine_geometry",
            node_count = topology.nodes.len(),
            element_count = topology.volume_elements.len(),
            jacobian_work,
            "retained certified affine edge geometry without optional shape search"
        );
        return Ok(topology);
    }
    let optimization_topology = optimization::OptimizationTopology::prepare(
        input,
        &topology,
        evaluation.control,
        cancellation,
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
            optimization::OptimizationTargets {
                improve_valid_nodes,
                invalid_elements: &invalid_elements,
                indexes: &optimization_topology,
            },
        )
        .map_err(|failure| phase_error("curved-node optimization", failure))?;
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
            jacobian::ValidationWork {
                counter: &mut jacobian_work,
                storage: super::storage::Storage(Some(evaluation.control)),
            },
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
                &optimization_topology,
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
                    jacobian::ValidationWork {
                        counter: &mut jacobian_work,
                        storage: super::storage::Storage(Some(evaluation.control)),
                    },
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
                &optimization_topology,
                evaluation.control,
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
                    jacobian::ValidationWork {
                        counter: &mut jacobian_work,
                        storage: super::storage::Storage(Some(evaluation.control)),
                    },
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
                jacobian::ValidationWork {
                    counter: &mut jacobian_work,
                    storage: super::storage::Storage(Some(evaluation.control)),
                },
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

mod midpoints;
use midpoints::MidpointMap;

fn phase_error(
    phase: &str,
    mut failure: DelaunaySolverTopologyError,
) -> DelaunaySolverTopologyError {
    failure.reason = format!("{phase}: {}", failure.reason);
    failure
}
