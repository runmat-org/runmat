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
    let mut invalid_elements = with_folded_faces(&topology, jacobian::invalid_element_indices(
        &topology,
        jacobian::ELEMENT_SEARCH_WORK,
        input.request.limits.maximum_recursion_depth,
        options.cancellation_check_interval,
        cancellation,
        jacobian::ValidationWork {
            counter: &mut jacobian_work,
            storage: super::storage::Storage(Some(evaluation.control)),
        },
    )?)?;
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
        invalid_elements = with_folded_faces(&topology, jacobian::invalid_element_indices(
            &topology,
            jacobian::ELEMENT_SEARCH_WORK,
            input.request.limits.maximum_recursion_depth,
            options.cancellation_check_interval,
            cancellation,
            jacobian::ValidationWork {
                counter: &mut jacobian_work,
                storage: super::storage::Storage(Some(evaluation.control)),
            },
        )?)?;
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
                invalid_elements = with_folded_faces(&topology, jacobian::invalid_element_indices(
                    &topology,
                    jacobian::ELEMENT_SEARCH_WORK,
                    input.request.limits.maximum_recursion_depth,
                    options.cancellation_check_interval,
                    cancellation,
                    jacobian::ValidationWork {
                        counter: &mut jacobian_work,
                        storage: super::storage::Storage(Some(evaluation.control)),
                    },
                )?)?;
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
                invalid_elements = with_folded_faces(&topology, jacobian::invalid_element_indices(
                    &topology,
                    jacobian::ELEMENT_SEARCH_WORK,
                    input.request.limits.maximum_recursion_depth,
                    options.cancellation_check_interval,
                    cancellation,
                    jacobian::ValidationWork {
                        counter: &mut jacobian_work,
                        storage: super::storage::Storage(Some(evaluation.control)),
                    },
                )?)?;
                if invalid_elements.is_empty() {
                    return Ok((topology, runmat_meshing_core::SolverElevationReport::default()));
                }
                continue;
            }
            break;
        }
    }
    // Curving cannot certify the remaining elements. Relax their edges toward straight chords:
    // a straight-sided Tet10 has the linear element's constant positive Jacobian. Interior
    // midsides carry no geometric error, so they straighten first. A boundary midside then moves
    // toward its chord only as far as the requested chordal deviation allows: it stays within
    // that distance of the edge's exact midside image, and goes fully straight only when the
    // chord's sagitta is itself within the deviation. Relaxing a shared edge can uncurve a
    // neighbor, so repeat until every element certifies or nothing can move further.
    let remaining_after_curving = invalid_elements.len();
    let mut straightened = 0_u64;
    let mut straightened_elements = std::collections::BTreeSet::new();
    let maximum_deviation_m = input.request.quality.surface.maximum_chordal_deviation_m;
    while !invalid_elements.is_empty() {
        straightened_elements.extend(invalid_elements.iter().copied());
        let mut changed = relax_element_edges(
            &mut topology,
            &invalid_elements,
            &exact_midsides,
            maximum_deviation_m,
            MidsideClass::Interior,
            cancellation,
        )?;
        if changed == 0 {
            changed = relax_element_edges(
                &mut topology,
                &invalid_elements,
                &exact_midsides,
                maximum_deviation_m,
                MidsideClass::Boundary,
                cancellation,
            )?;
        }
        if changed == 0 {
            return Err(error::failure(
                DelaunaySolverTopologyErrorKind::InvalidMesh,
                format!(
                    "{} Tet10 elements remain uncertified with interior edges straight and boundary midsides at the chordal deviation {maximum_deviation_m:e} m from their exact geometry ({} of {} elements initially invalid, {} still invalid after {} curved-node movements); first: {}",
                    invalid_elements.len(),
                    initial_invalid_element_count,
                    topology.volume_elements.len(),
                    remaining_after_curving,
                    moved_node_count,
                    invalid_elements
                        .iter()
                        .take(1)
                        .map(|element| describe_element(&topology, *element, &exact_midsides))
                        .collect::<Vec<_>>()
                        .join("; "),
                ),
            ));
        }
        straightened += changed;
        invalid_elements = with_folded_faces(&topology, jacobian::invalid_element_indices(
            &topology,
            jacobian::ELEMENT_SEARCH_WORK,
            input.request.limits.maximum_recursion_depth,
            options.cancellation_check_interval,
            cancellation,
            jacobian::ValidationWork {
                counter: &mut jacobian_work,
                storage: super::storage::Storage(Some(evaluation.control)),
            },
        )?)?;
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum MidsideClass {
    /// Midsides without exact parameters: interior edges, no geometric error when straight.
    Interior,
    /// Midsides on a curve or surface, bounded by the requested chordal deviation.
    Boundary,
}

/// Moves the given elements' midside nodes of one class toward their edges' chord midpoints.
/// Returns how many nodes moved.
/// An interior midside goes to its chord. A boundary midside keeps its exact parameters as the
/// node's foot on its source entity and moves from the edge's exact midside image toward the
/// chord by at most the requested chordal deviation, so it never leaves its exact geometry by
/// more than that deviation.
fn relax_element_edges(
    topology: &mut SolverMeshTopology,
    elements: &[usize],
    exact_midsides: &std::collections::BTreeMap<u64, [f64; 3]>,
    maximum_deviation_m: f64,
    class: MidsideClass,
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
            let on_geometry = !topology.nodes[middle].exact_parameters.is_empty();
            if on_geometry != (class == MidsideClass::Boundary) {
                continue;
            }
            let chord: [f64; 3] = std::array::from_fn(|axis| {
                0.5 * (topology.nodes[left].coordinates_m[axis]
                    + topology.nodes[right].coordinates_m[axis])
            });
            let target = if on_geometry {
                let curved = exact_midsides
                    .get(&topology.nodes[middle].node_id)
                    .copied()
                    .unwrap_or(topology.nodes[middle].coordinates_m);
                let sagitta = (0..3)
                    .map(|axis| (curved[axis] - chord[axis]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if sagitta <= maximum_deviation_m {
                    chord
                } else {
                    let fraction = maximum_deviation_m / sagitta;
                    std::array::from_fn(|axis| {
                        curved[axis] + fraction * (chord[axis] - curved[axis])
                    })
                }
            } else {
                chord
            };
            if topology.nodes[middle].coordinates_m != target {
                topology.nodes[middle].coordinates_m = target;
                moved += 1;
            }
        }
    }
    Ok(moved)
}

/// Corner roles, smallest altitude and largest boundary-edge sagitta of one element, for a
/// failure that names why its curved geometry cannot be certified.
fn describe_element(
    topology: &SolverMeshTopology,
    element: usize,
    exact_midsides: &std::collections::BTreeMap<u64, [f64; 3]>,
) -> String {
    let node = |id: u64| topology.nodes.iter().find(|node| node.node_id == id);
    let ids = &topology.volume_elements[element].node_ids;
    let corners = ids[..4]
        .iter()
        .filter_map(|id| node(*id))
        .map(|node| (node.coordinates_m, !node.exact_parameters.is_empty()))
        .collect::<Vec<_>>();
    if corners.len() != 4 {
        return format!("element {element} has absent corners");
    }
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    let norm = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let p = [corners[0].0, corners[1].0, corners[2].0, corners[3].0];
    let edges = [sub(p[1], p[0]), sub(p[2], p[0]), sub(p[3], p[0])];
    let volume6 = {
        let c = cross(edges[0], edges[1]);
        (c[0] * edges[2][0] + c[1] * edges[2][1] + c[2] * edges[2][2]).abs()
    };
    let largest_face = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        .iter()
        .map(|[a, b, c]| norm(cross(sub(p[*b], p[*a]), sub(p[*c], p[*a]))))
        .fold(0.0_f64, f64::max);
    let altitude = if largest_face > 0.0 { volume6 / largest_face } else { 0.0 };
    let mut sagitta = 0.0_f64;
    let mut worst = String::new();
    for (local_edge, edge) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.iter().enumerate() {
        if let Some(curved) = exact_midsides.get(&ids[4 + local_edge]) {
            let chord: [f64; 3] = std::array::from_fn(|axis| 0.5 * (p[edge[0]][axis] + p[edge[1]][axis]));
            let edge_sagitta = norm(sub(*curved, chord));
            if edge_sagitta > sagitta {
                sagitta = edge_sagitta;
                let parameters = |id: u64| {
                    node(id)
                        .map(|node| {
                            node.exact_parameters
                                .iter()
                                .map(|parameter| match parameter {
                                    runmat_meshing_core::SolverNodeExactParameter::Curve {
                                        source_edge_id,
                                        parameter,
                                    } => format!(
                                        "curve {}@{parameter:.6}",
                                        source_edge_id.source_topology_id.chars().rev().take(6).collect::<String>()
                                    ),
                                    runmat_meshing_core::SolverNodeExactParameter::Surface {
                                        source_face_id,
                                        evaluator_uv,
                                        ..
                                    } => format!(
                                        "face {}@({:.6},{:.6})",
                                        source_face_id.source_topology_id.chars().rev().take(6).collect::<String>(),
                                        evaluator_uv[0],
                                        evaluator_uv[1]
                                    ),
                                })
                                .collect::<Vec<_>>()
                                .join(" ")
                        })
                        .unwrap_or_default()
                };
                worst = format!(
                    " (edge chord {:e} m; endpoint parameters [{}] and [{}]; midside parameters [{}])",
                    norm(sub(p[edge[1]], p[edge[0]])),
                    parameters(ids[edge[0]]),
                    parameters(ids[edge[1]]),
                    parameters(ids[4 + local_edge]),
                );
            }
        }
    }
    format!(
        "element {element}: {} of 4 corners on exact geometry, smallest altitude {altitude:e} m, largest boundary-edge sagitta {sagitta:e} m{worst}",
        corners.iter().filter(|corner| corner.1).count(),
    )
}

/// Elements with a folded curved boundary face, merged into the uncertified inventory. A
/// boundary face's curved Jacobian, projected on its corner normal, must stay positive over
/// the whole face (the exact minimum of that quadratic, the same criterion consumers apply
/// before projecting onto the surface); a folded face is never published.
fn with_folded_faces<'storage>(
    topology: &SolverMeshTopology,
    mut invalid: super::storage::ReservedBuffer<'storage, usize>,
) -> Result<super::storage::ReservedBuffer<'storage, usize>, DelaunaySolverTopologyError> {
    let folded = folded_boundary_face_elements(topology)?;
    if folded.is_empty() {
        return Ok(invalid);
    }
    let present = invalid.iter().copied().collect::<std::collections::BTreeSet<_>>();
    for index in folded.into_iter().collect::<std::collections::BTreeSet<_>>() {
        if !present.contains(&index) {
            invalid.push(index)?;
        }
    }
    invalid.sort_unstable();
    Ok(invalid)
}

fn folded_boundary_face_elements(
    topology: &SolverMeshTopology,
) -> Result<Vec<usize>, DelaunaySolverTopologyError> {
    let position = |id: u64| {
        topology
            .nodes
            .binary_search_by_key(&id, |node| node.node_id)
            .map(|index| topology.nodes[index].coordinates_m)
            .map_err(|_| {
                error::failure(
                    DelaunaySolverTopologyErrorKind::InvalidMesh,
                    "Tet10 boundary face references an absent node",
                )
            })
    };
    let mut folded = Vec::new();
    for face in &topology.boundary_faces {
        if face.node_ids.len() != 6 {
            continue;
        }
        let mut coordinates = [[0.0; 3]; 6];
        for (slot, id) in face.node_ids.iter().enumerate() {
            coordinates[slot] = position(*id)?;
        }
        if signed_face_area_minimum(&coordinates).is_none_or(|minimum| minimum <= 0.0) {
            for element_id in &face.adjacent_volume_element_ids {
                if let Some(index) = topology
                    .volume_elements
                    .iter()
                    .position(|element| element.element_id == *element_id)
                {
                    folded.push(index);
                }
            }
        }
    }
    Ok(folded)
}

/// Exact minimum over the face of the six-node triangle's tangent cross product projected on
/// its corner normal (corners 0..3, midsides 01, 12, 20).
fn signed_face_area_minimum(coordinates: &[[f64; 3]; 6]) -> Option<f64> {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let normal = cross(sub(coordinates[1], coordinates[0]), sub(coordinates[2], coordinates[0]));
    let signed_area = |u: f64, v: f64| {
        let l1 = 1.0 - u - v;
        let derivatives: [[f64; 2]; 6] = [
            [1.0 - 4.0 * l1, 1.0 - 4.0 * l1],
            [4.0 * u - 1.0, 0.0],
            [0.0, 4.0 * v - 1.0],
            [4.0 * (l1 - u), -4.0 * u],
            [4.0 * v, 4.0 * u],
            [-4.0 * v, 4.0 * (l1 - v)],
        ];
        let tangent = |axis: usize| -> [f64; 3] {
            std::array::from_fn(|component| {
                (0..6)
                    .map(|node| derivatives[node][axis] * coordinates[node][component])
                    .sum::<f64>()
            })
        };
        dot(cross(tangent(0), tangent(1)), normal)
    };
    let [q00, q10, q01, qh0, q0h, qhh] = [
        signed_area(0.0, 0.0),
        signed_area(1.0, 0.0),
        signed_area(0.0, 1.0),
        signed_area(0.5, 0.0),
        signed_area(0.0, 0.5),
        signed_area(0.5, 0.5),
    ];
    if [q00, q10, q01, qh0, q0h, qhh].iter().any(|value| !value.is_finite()) {
        return None;
    }
    let a = 2.0 * (q10 + q00 - 2.0 * qh0);
    let b = 2.0 * (q01 + q00 - 2.0 * q0h);
    let d = q10 - q00 - a;
    let e = q01 - q00 - b;
    let c = 4.0 * (qhh - 0.25 * (a + b) - 0.5 * (d + e) - q00);
    let edge_minimum = |start: f64, end: f64, midpoint: f64| {
        let curvature = 2.0 * (start + end - 2.0 * midpoint);
        if curvature <= 0.0 {
            return start.min(end);
        }
        let slope = end - start - curvature;
        let position = (-slope / (2.0 * curvature)).clamp(0.0, 1.0);
        (curvature * position + slope) * position + start
    };
    let mut minimum = edge_minimum(q00, q10, qh0)
        .min(edge_minimum(q00, q01, q0h))
        .min(edge_minimum(q10, q01, qhh));
    let determinant = 4.0 * a * b - c * c;
    if a > 0.0 && determinant > 0.0 {
        let u = (c * e - 2.0 * b * d) / determinant;
        let v = (c * d - 2.0 * a * e) / determinant;
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            minimum = minimum.min(a * u * u + b * v * v + c * u * v + d * u + e * v + q00);
        }
    }
    minimum.is_finite().then_some(minimum)
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

#[cfg(test)]
mod relaxation_tests {
    use super::*;
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    // witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: A six-node boundary face whose curved Jacobian folds against its corner
    //   normal is uncertified; the bracket Tet10 study exercises the whole elevation.
    #[test]
    fn folded_curved_boundary_face_is_uncertified() {
        let mut face = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        assert!(signed_face_area_minimum(&face).unwrap() > 0.0);
        // A gently curved face stays positive; a midside pushed past the far corner folds it.
        face[4][2] = 0.1;
        assert!(signed_face_area_minimum(&face).unwrap() > 0.0);
        face[3][0] = 1.5;
        assert!(signed_face_area_minimum(&face).unwrap() <= 0.0);
    }

    // witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: An uncertifiable Tet10 relaxes interior midsides to chords and boundary
    //   midsides only to the chordal deviation from their exact image; real parts are
    //   exercised by the hv-electrode and coax studies.
    #[test]
    fn relaxation_keeps_boundary_midsides_within_the_chordal_deviation() {
        let mut topology = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10).topology;
        let midside = |topology: &SolverMeshTopology, local_edge: usize| {
            let id = topology.volume_elements[0].node_ids[4 + local_edge];
            topology.nodes.iter().position(|node| node.node_id == id).unwrap()
        };
        let chord = |topology: &SolverMeshTopology, local_edge: usize| -> [f64; 3] {
            let corners = TETRAHEDRON_MIDSIDE_EDGE_CORNERS[local_edge];
            let point = |corner: usize| {
                let id = topology.volume_elements[0].node_ids[corners[corner]];
                topology.nodes.iter().find(|node| node.node_id == id).unwrap().coordinates_m
            };
            std::array::from_fn(|axis| 0.5 * (point(0)[axis] + point(1)[axis]))
        };
        let (boundary, interior) = (midside(&topology, 0), midside(&topology, 1));
        let boundary_chord = chord(&topology, 0);
        let interior_chord = chord(&topology, 1);
        // A boundary midside 0.3 m off its chord, and an interior midside moved off its chord.
        let curved = [boundary_chord[0], boundary_chord[1] - 0.3, boundary_chord[2]];
        topology.nodes[boundary].coordinates_m = curved;
        topology.nodes[interior].exact_parameters.clear();
        topology.nodes[interior].coordinates_m[2] += 0.2;
        let exact = std::collections::BTreeMap::from([(topology.nodes[boundary].node_id, curved)]);
        let relax = |topology: &mut SolverMeshTopology, class, deviation| {
            relax_element_edges(topology, &[0], &exact, deviation, class, &runmat_meshing_core::NeverCancelled)
                .unwrap()
        };
        assert_eq!(relax(&mut topology, MidsideClass::Interior, 0.1), 1);
        assert_eq!(topology.nodes[interior].coordinates_m, interior_chord);
        assert_eq!(topology.nodes[boundary].coordinates_m, curved);
        // The sagitta (0.3 m) exceeds the deviation (0.1 m): the node moves 0.1 m toward the chord.
        assert!(relax(&mut topology, MidsideClass::Boundary, 0.1) >= 1);
        let offset = (0..3)
            .map(|axis| (topology.nodes[boundary].coordinates_m[axis] - curved[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((offset - 0.1).abs() < 1.0e-12, "{offset}");
        assert_eq!(topology.nodes[interior].coordinates_m, interior_chord);
        // Within the deviation the edge goes fully straight.
        relax(&mut topology, MidsideClass::Boundary, 0.5);
        assert_eq!(topology.nodes[boundary].coordinates_m, boundary_chord);
    }
}
