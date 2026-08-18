use runmat_meshing_core::{predicate::orient2d, PredicateSign};

use crate::{ExactFaceDelaunayError, ExactFaceDelaunayTriangle, ExactFacePslg};

use super::recover::RecoveryControl;
use crate::exact_cdt::topology::{canonical_triangle, oriented_triangle};

/// Triangulates one simple cavity side. The input is an open chain whose implicit
/// closing edge joins the last vertex to the first protected-segment endpoint.
pub(super) fn triangulate_polygon(
    chain: &[u32],
    pslg: &ExactFacePslg,
    control: &mut RecoveryControl<'_>,
) -> Result<Vec<ExactFaceDelaunayTriangle>, ExactFaceDelaunayError> {
    if chain.len() < 2 {
        return Err(control.unsatisfied("recovery cavity side has fewer than two vertices"));
    }
    if chain.len() == 2 {
        return Ok(Vec::new());
    }

    let mut polygon = chain.to_vec();
    make_counterclockwise(&mut polygon, pslg, control)?;
    let mut triangles = Vec::with_capacity(polygon.len().saturating_sub(2));

    while polygon.len() > 3 {
        control.checkpoint()?;
        let mut candidates = (0..polygon.len()).collect::<Vec<_>>();
        candidates.sort_by_key(|index| polygon[*index]);
        let mut selected = None;
        for index in candidates {
            let previous = polygon[(index + polygon.len() - 1) % polygon.len()];
            let current = polygon[index];
            let next = polygon[(index + 1) % polygon.len()];
            control.consume_predicates(1)?;
            if orientation([previous, current, next], pslg, control)? != PredicateSign::Positive {
                continue;
            }

            let mut contains_vertex = false;
            for vertex in polygon.iter().copied() {
                if [previous, current, next].contains(&vertex) {
                    continue;
                }
                control.consume_predicates(3)?;
                let signs = [
                    orientation([previous, current, vertex], pslg, control)?,
                    orientation([current, next, vertex], pslg, control)?,
                    orientation([next, previous, vertex], pslg, control)?,
                ];
                if signs.iter().all(|sign| *sign != PredicateSign::Negative) {
                    contains_vertex = true;
                    break;
                }
            }
            if !contains_vertex {
                selected = Some((index, [previous, current, next]));
                break;
            }
        }

        let Some((index, triangle)) = selected else {
            return Err(control.unsatisfied("recovery cavity side has no strict deterministic ear"));
        };
        triangles.push(ExactFaceDelaunayTriangle {
            vertex_indices: canonical_triangle(triangle),
        });
        polygon.remove(index);
    }

    control.consume_predicates(1)?;
    let triangle = oriented_triangle([polygon[0], polygon[1], polygon[2]], pslg)
        .map_err(|error| control.predicate_error(error))?
        .ok_or_else(|| control.unsatisfied("recovery cavity closes with a degenerate triangle"))?;
    triangles.push(ExactFaceDelaunayTriangle {
        vertex_indices: canonical_triangle(triangle),
    });
    Ok(triangles)
}

fn make_counterclockwise(
    polygon: &mut [u32],
    pslg: &ExactFacePslg,
    control: &mut RecoveryControl<'_>,
) -> Result<(), ExactFaceDelaunayError> {
    let extreme = (0..polygon.len())
        .min_by(|left, right| {
            let left_uv = pslg.vertices[polygon[*left] as usize].uv;
            let right_uv = pslg.vertices[polygon[*right] as usize].uv;
            left_uv[0]
                .total_cmp(&right_uv[0])
                .then_with(|| left_uv[1].total_cmp(&right_uv[1]))
                .then_with(|| polygon[*left].cmp(&polygon[*right]))
        })
        .expect("polygon has at least three vertices");

    let mut winding = PredicateSign::Zero;
    for distance in 1..polygon.len() {
        let previous = polygon[(extreme + polygon.len() - distance) % polygon.len()];
        for forward_distance in 1..polygon.len() {
            let next = polygon[(extreme + forward_distance) % polygon.len()];
            if previous == next {
                continue;
            }
            control.consume_predicates(1)?;
            winding = orientation([previous, polygon[extreme], next], pslg, control)?;
            if winding != PredicateSign::Zero {
                break;
            }
        }
        if winding != PredicateSign::Zero {
            break;
        }
    }

    match winding {
        PredicateSign::Positive => Ok(()),
        PredicateSign::Negative => {
            polygon.reverse();
            Ok(())
        }
        PredicateSign::Zero => Err(control.unsatisfied("recovery cavity side has zero area")),
    }
}

fn orientation(
    vertices: [u32; 3],
    pslg: &ExactFacePslg,
    control: &RecoveryControl<'_>,
) -> Result<PredicateSign, ExactFaceDelaunayError> {
    orient2d(vertices.map(|index| pslg.vertices[index as usize].uv))
        .map_err(|error| control.predicate_error(error))
}
