//! Complete each original image job once, preserving first-occurrence order.
use super::*;
use crate::cdt::solver_topology::storage::Storage;
use runmat_geometry_core::{ExactSurfaceEvaluator, GeometryAllocationScope};
pub(super) fn complete_boundary_surface_parameters(
    input: &ProjectionInput<'_>,
    topology: &mut SolverMeshTopology,
    completion: BoundaryParameterCompletion<'_, '_>,
) -> Result<(), DelaunaySolverTopologyError> {
    let BoundaryParameterCompletion {
        surface_uses,
        exact_edges,
        evaluation,
        cancellation,
    } = completion;
    let temporaries = GeometryAllocationScope::new(evaluation.control);
    let scratch = Storage(Some(&temporaries));
    let retained = Storage(Some(evaluation.control));
    let curve_count = exact_edges.iter().try_fold(0usize, |count, (edge, _)| {
        count
            .checked_add(
                surface_uses
                    .get(edge)
                    .map_or(0, |uses| uses.len())
                    .checked_mul(2)
                    .ok_or_else(|| invalid("boundary image job count overflow"))?,
            )
            .ok_or_else(|| invalid("boundary image job count overflow"))
    })?;
    let mut curve_jobs = scratch.vector(curve_count)?;
    for (edge, id) in exact_edges.iter() {
        if let Some(uses) = surface_uses.get(edge) {
            for node in edge {
                for usage in uses {
                    curve_jobs.push(((*node, id, *usage), curve_jobs.len()));
                }
            }
        }
    }
    unique_first(&mut curve_jobs, scratch, cancellation)?;
    for ((node_id, id, usage), _) in curve_jobs {
        boundary_parameter_checkpoint(cancellation)?;
        let slot = node_slot(&topology.nodes, node_id)?;
        let node = &mut topology.nodes[slot];
        let parameter = curve_parameter(node, id)?;
        let edge = evaluation.sources.edge(id)?;
        // The curve point, not the node, is what the pcurve images must reproduce: a node the
        // volume stage inserted on a boundary segment lies on the chord, off the curve.
        let curve = evaluate_curve_point(id, parameter, evaluation)?;
        require_node_near_curve(
            input,
            edge,
            parameter,
            node.coordinates_m.canonical_value(),
            curve,
        )?;
        // Pcurve images lie on the trim boundary by construction; only agreement is checked.
        for uv in pcurve_uvs(id, parameter, &usage, evaluation)? {
            let point = evaluate_surface_point(&usage, uv, evaluation)?;
            require_matching_curve_points(input, edge, curve, point)?;
            insert_surface_parameter(node, &usage, uv, retained)?;
        }
    }
    let surface_count = surface_uses.iter().try_fold(0usize, |count, (_, uses)| {
        count
            .checked_add(
                uses.len()
                    .checked_mul(2)
                    .ok_or_else(|| invalid("boundary image job count overflow"))?,
            )
            .ok_or_else(|| invalid("boundary image job count overflow"))
    })?;
    let mut surface_jobs = scratch.vector(surface_count)?;
    for (edge, uses) in surface_uses.iter() {
        for node in edge {
            for usage in uses {
                surface_jobs.push(((*node, *usage), surface_jobs.len()));
            }
        }
    }
    unique_first(&mut surface_jobs, scratch, cancellation)?;
    let mut triangle_candidates = Vec::new();
    for ((node_id, usage), _) in surface_jobs {
        boundary_parameter_checkpoint(cancellation)?;
        let slot = node_slot(&topology.nodes, node_id)?;
        let node = &mut topology.nodes[slot];
        let parameter_scope = GeometryAllocationScope::new(evaluation.control);
        let parameters = derive_surface_parameters(
            input,
            node,
            &usage,
            evaluation,
            cancellation,
            SurfaceParameterScratch {
                candidates: &mut triangle_candidates,
                parameters: Storage(Some(&parameter_scope)),
                query_control: &temporaries,
            },
        )?;
        for uv in parameters {
            insert_surface_parameter(node, &usage, uv, retained)?;
        }
    }
    drop(triangle_candidates);
    drop(temporaries);
    Ok(())
}
fn unique_first<T: Copy + Ord>(
    rows: &mut Vec<(T, usize)>,
    storage: Storage<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySolverTopologyError> {
    let mut scratch = storage.collect(rows.iter().copied())?;
    runmat_cpu_execution::sort_copy_with_checkpoint(
        rows,
        &mut scratch,
        |a: &(T, usize), b: &(T, usize)| a.cmp(b),
        || boundary_parameter_checkpoint(cancellation),
    )?;
    rows.dedup_by(|later, earlier| later.0 == earlier.0);
    scratch.truncate(rows.len());
    runmat_cpu_execution::sort_copy_with_checkpoint(
        rows,
        &mut scratch,
        |a: &(T, usize), b: &(T, usize)| a.1.cmp(&b.1),
        || boundary_parameter_checkpoint(cancellation),
    )?;
    storage.release_backing(scratch)?;
    Ok(())
}

struct SurfaceParameterScratch<'a> {
    candidates: &'a mut Vec<usize>,
    parameters: Storage<'a>,
    query_control: &'a dyn runmat_geometry_core::GeometryEvaluationControl,
}
fn derive_surface_parameters(
    input: &ProjectionInput<'_>,
    node: &SolverMeshNode,
    surface_use: &SurfaceUse,
    evaluation: PreparedEvaluation<'_, '_>,
    cancellation: &dyn MeshingCancellationSignal,
    scratch: SurfaceParameterScratch<'_>,
) -> Result<Vec<[f64; 2]>, DelaunaySolverTopologyError> {
    let SurfaceParameterScratch {
        candidates: triangle_candidates,
        parameters: parameter_storage,
        query_control,
    } = scratch;
    let source_nodes = &evaluation.sources.nodes;
    let face = evaluation.sources.face(surface_use.source_face_id)?;
    let periods =
        ExactSurfaceEvaluator::periodicity(evaluation.evaluator, &face.surface_evaluator_id)
            .map_err(error::geometry)?;
    let bounds =
        ExactSurfaceEvaluator::parameter_bounds(evaluation.evaluator, &face.surface_evaluator_id)
            .map_err(error::geometry)?;
    if source_nodes
        .get(&node.stable_identity)
        .is_some_and(|source| source.point_m == node.coordinates_m.canonical_value())
        && node.exact_parameters.iter().any(|parameter| {
            matches!(parameter,
            SolverNodeExactParameter::Surface { source_face_id, chart_id, .. }
            if source_face_id == surface_use.source_face_id && chart_id == &surface_use.chart_id)
        })
    {
        // Construction copied every original chart image. No new parameter is
        // needed for an unchanged source node; retain its existing records in place.
        return Ok(Vec::new());
    }
    let mut parameters =
        parameter_storage.collect(node.exact_parameters.iter().filter_map(|parameter| {
            match parameter {
                SolverNodeExactParameter::Surface {
                    source_face_id,
                    chart_id,
                    evaluator_uv,
                } if source_face_id == surface_use.source_face_id
                    && chart_id == &surface_use.chart_id =>
                {
                    Some(*evaluator_uv)
                }
                SolverNodeExactParameter::Curve { .. }
                | SolverNodeExactParameter::Surface { .. } => None,
            }
        }))?;
    evaluation.sources.triangle_candidates(
        surface_use,
        node.coordinates_m.canonical_value(),
        query_control,
        triangle_candidates,
    )?;
    for &triangle_index in triangle_candidates.iter() {
        let triangle = &input.exact_surface.triangles[triangle_index];
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
            .canonical_value()
            .max(input.request.tolerance.absolute_floor_m.canonical_value())
            .max(
                input
                    .request
                    .tolerance
                    .maximum_healing_displacement_m
                    .canonical_value(),
            );
        let Some(weights) =
            triangle_coordinates(points, node.coordinates_m.canonical_value(), tolerance_m)
        else {
            continue;
        };
        let matching = |candidate: &&runmat_meshing_surface::ExactFaceMeshNodeUse| {
            &candidate.source_face_id == surface_use.source_face_id
                && candidate.chart_id == surface_use.chart_id
        };
        for first_use in first.uses.iter().filter(matching) {
            for second_use in second.uses.iter().filter(matching) {
                for third_use in third.uses.iter().filter(matching) {
                    // A source triangle on a periodic seam carries images on both sides of
                    // it; interpolate one consistent set (each vertex on the first vertex's
                    // side), never a mixture that lands across the surface.
                    let first_uv = first_use.evaluator_uv;
                    let second_uv =
                        evaluation::nearest_translate(first_uv, second_use.evaluator_uv, periods);
                    let third_uv =
                        evaluation::nearest_translate(first_uv, third_use.evaluator_uv, periods);
                    parameter_storage.push(
                        &mut parameters,
                        evaluation::wrap_into_period(
                            std::array::from_fn(|axis| {
                                weights[0] * first_uv[axis]
                                    + weights[1] * second_uv[axis]
                                    + weights[2] * third_uv[axis]
                            }),
                            periods,
                            bounds,
                        ),
                    )?;
                }
            }
        }
    }
    parameters.sort_unstable_by(|left, right| {
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
