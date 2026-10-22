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
) -> Result<(SolverMeshTopology, runmat_meshing_core::SolverElevationReport), DelaunaySolverTopologyError> {
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
    // The exact midside placements before any optimization move: each is the edge's image on its
    // curve or surface, so its distance from the chord is the straight edge's geometric error.
    let exact_midsides = topology
        .nodes
        .iter()
        .filter(|node| !node.exact_parameters.is_empty())
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut invalid_elements = jacobian::invalid_element_indices(
        &topology,
        jacobian::ELEMENT_SEARCH_WORK,
        input.request.limits.maximum_recursion_depth,
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
        return Ok((topology, runmat_meshing_core::SolverElevationReport::default()));
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
            jacobian::ELEMENT_SEARCH_WORK,
            input.request.limits.maximum_recursion_depth,
            options.cancellation_check_interval,
            cancellation,
            jacobian::ValidationWork {
                counter: &mut jacobian_work,
                storage: super::storage::Storage(Some(evaluation.control)),
            },
        )?;
        if invalid_elements.is_empty() {
            return Ok((topology, runmat_meshing_core::SolverElevationReport::default()));
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
                    jacobian::ELEMENT_SEARCH_WORK,
                    input.request.limits.maximum_recursion_depth,
                    options.cancellation_check_interval,
                    cancellation,
                    jacobian::ValidationWork {
                        counter: &mut jacobian_work,
                        storage: super::storage::Storage(Some(evaluation.control)),
                    },
                )?;
                if invalid_elements.is_empty() {
                    return Ok((topology, runmat_meshing_core::SolverElevationReport::default()));
                }
                continue;
            }
            let relocated = optimization::relocate_interior_corner(
                &mut topology,
                worst_element,
                jacobian::ELEMENT_SEARCH_WORK,
                input.request.limits.maximum_recursion_depth,
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
                    jacobian::ELEMENT_SEARCH_WORK,
                    input.request.limits.maximum_recursion_depth,
                    options.cancellation_check_interval,
                    cancellation,
                    jacobian::ValidationWork {
                        counter: &mut jacobian_work,
                        storage: super::storage::Storage(Some(evaluation.control)),
                    },
                )?;
                if invalid_elements.is_empty() {
                    return Ok((topology, runmat_meshing_core::SolverElevationReport::default()));
                }
                continue;
            }
            break;
        }
    }
    // Curving cannot certify the remaining elements. Their edges go back to straight chords,
    // which the certified linear mesh already places within the requested chordal deviation;
    // a straight-sided Tet10 has the linear element's constant positive Jacobian. Straightening
    // a shared edge can uncurve a neighbor, so repeat until every element certifies (all
    // straight is the bound).
    let remaining_after_curving = invalid_elements.len();
    let mut straightened = 0_u64;
    let mut straightened_elements = std::collections::BTreeSet::new();
    let maximum_deviation_m = input.request.quality.surface.maximum_chordal_deviation_m;
    while !invalid_elements.is_empty() {
        straightened_elements.extend(invalid_elements.iter().copied());
        let changed = straighten_element_edges(
            &mut topology,
            &invalid_elements,
            &exact_midsides,
            maximum_deviation_m,
            cancellation,
        )?;
        if changed == 0 {
            return Err(error::failure(
                DelaunaySolverTopologyErrorKind::InvalidMesh,
                format!(
                    "{} Tet10 elements remain uncertified with straight edges ({} initially invalid elements, {} accepted node movements)",
                    invalid_elements.len(),
                    initial_invalid_element_count,
                    moved_node_count,
                ),
            ));
        }
        straightened += changed;
        invalid_elements = jacobian::invalid_element_indices(
            &topology,
            jacobian::ELEMENT_SEARCH_WORK,
            input.request.limits.maximum_recursion_depth,
            options.cancellation_check_interval,
            cancellation,
            jacobian::ValidationWork {
                counter: &mut jacobian_work,
                storage: super::storage::Storage(Some(evaluation.control)),
            },
        )?;
    }
    tracing::info!(
        event_code = "meshing.tet10.straightened_edges",
        initially_invalid = initial_invalid_element_count,
        remaining_after_curving,
        straightened_edges = straightened,
        moved_node_count,
        "Tet10 edges that could not be curved validly were kept straight"
    );
    Ok((
        topology,
        runmat_meshing_core::SolverElevationReport {
            straightened_elements: straightened_elements.len() as u64,
            straightened_edges: straightened,
        },
    ))
}

/// Moves each curved midside node of the given elements to its edge's chord midpoint. Returns
/// how many nodes moved.
/// A boundary midside keeps its exact parameters as the node's foot on its source entity; the
/// straight node lies off that entity by the chord's sagitta (its distance from the edge's exact
/// midside image), which must stay within the requested chordal deviation (checked here for
/// every straightened boundary node).
fn straighten_element_edges(
    topology: &mut SolverMeshTopology,
    elements: &[usize],
    exact_midsides: &std::collections::BTreeMap<u64, [f64; 3]>,
    maximum_deviation_m: f64,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<u64, DelaunaySolverTopologyError> {
    let index_of = |topology: &SolverMeshTopology, id: u64| {
        topology
            .nodes
            .binary_search_by_key(&id, |node| node.node_id)
            .map_err(|_| {
                error::failure(
                    DelaunaySolverTopologyErrorKind::InvalidMesh,
                    "Tet10 straightening references an absent node",
                )
            })
    };
    let mut moved = 0_u64;
    for &element_index in elements {
        if cancellation.is_cancelled() {
            return Err(error::failure(
                DelaunaySolverTopologyErrorKind::Cancelled,
                "cancelled",
            ));
        }
        let node_ids = topology.volume_elements[element_index].node_ids.clone();
        for (local_edge, corners) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.iter().enumerate() {
            let [left, right] = [
                index_of(topology, node_ids[corners[0]])?,
                index_of(topology, node_ids[corners[1]])?,
            ];
            let middle = index_of(topology, node_ids[4 + local_edge])?;
            let chord = std::array::from_fn(|axis| {
                0.5 * (topology.nodes[left].coordinates_m[axis]
                    + topology.nodes[right].coordinates_m[axis])
            });
            if topology.nodes[middle].coordinates_m != chord {
                let curved = exact_midsides
                    .get(&topology.nodes[middle].node_id)
                    .copied()
                    .unwrap_or(topology.nodes[middle].coordinates_m);
                let sagitta = (0..3)
                    .map(|axis| (curved[axis] - chord[axis]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if !topology.nodes[middle].exact_parameters.is_empty()
                    && !(sagitta <= maximum_deviation_m * (1.0 + 1.0e-9))
                {
                    return Err(error::failure(
                        DelaunaySolverTopologyErrorKind::InvalidMesh,
                        format!(
                            "a straight Tet10 boundary edge would leave its exact geometry by {sagitta:e} m, beyond the chordal deviation {maximum_deviation_m:e} m"
                        ),
                    ));
                }
                topology.nodes[middle].coordinates_m = chord;
                moved += 1;
            }
        }
    }
    Ok(moved)
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
