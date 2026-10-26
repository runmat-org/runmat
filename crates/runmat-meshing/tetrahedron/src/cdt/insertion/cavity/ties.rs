//! One symbolic in-sphere tie policy for cavity selection and legalization.
//! Solver coordinates remain unchanged; exact physical admission is independent.
use super::*;
use runmat_meshing_core::quality::predicate::SpatialPredicateError;

pub(in crate::cdt::insertion) fn circumsphere_relation(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<PredicateSign, DelaunayInsertionError> {
    work.predicate()?;
    let points = vertices.map(|vertex| topology.nodes()[vertex as usize]);
    let relation = topology
        .insphere_nodes([points[0], points[1], points[2], points[3], node])
        .map_err(predicate_error)?;
    if relation != PredicateSign::Zero {
        return Ok(relation);
    }
    symbolic_circumsphere_relation(topology, vertices, node)
}

fn symbolic_circumsphere_relation(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
) -> Result<PredicateSign, DelaunayInsertionError> {
    let corners = vertices.map(|vertex| topology.nodes()[vertex as usize]);
    symbolic_relation(
        [corners[0], corners[1], corners[2], corners[3], node],
        |points| topology.orient_nodes(points),
    )
}

pub(in crate::cdt::insertion) fn materialized_relation(
    corners: [DelaunayVolumeNode; 4],
    node: DelaunayVolumeNode,
) -> Result<PredicateSign, DelaunayInsertionError> {
    let points = [corners[0], corners[1], corners[2], corners[3], node];
    let relation = crate::cdt::topology::insphere_materialized_volume_nodes(points)
        .map_err(predicate_error)?;
    if relation != PredicateSign::Zero {
        return Ok(relation);
    }
    symbolic_relation(
        points,
        crate::cdt::topology::orient_materialized_volume_nodes,
    )
}

fn symbolic_relation(
    points: [DelaunayVolumeNode; 5],
    orient: impl Fn([DelaunayVolumeNode; 4]) -> Result<PredicateSign, SpatialPredicateError>,
) -> Result<PredicateSign, DelaunayInsertionError> {
    let mut points = points.to_vec();
    let mut inversions = 0_u8;
    for left in 0..points.len() {
        for right in (left + 1)..points.len() {
            inversions ^= u8::from(
                compare_symbolic_nodes(&points[left], &points[right])
                    == std::cmp::Ordering::Greater,
            );
        }
    }
    points.sort_unstable_by(compare_symbolic_nodes);
    let parity_is_odd = inversions != 0;
    let leading = orient([points[1], points[2], points[3], points[4]]).map_err(predicate_error)?;
    if leading != PredicateSign::Zero {
        return Ok(if parity_is_odd {
            reverse_sign(leading)
        } else {
            leading
        });
    }
    let trailing = orient([points[0], points[2], points[3], points[4]]).map_err(predicate_error)?;
    if trailing == PredicateSign::Zero {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "symbolic circumsphere perturbation encountered five affinely degenerate nodes",
        ));
    }
    Ok(if parity_is_odd {
        trailing
    } else {
        reverse_sign(trailing)
    })
}

fn compare_symbolic_nodes(
    left: &DelaunayVolumeNode,
    right: &DelaunayVolumeNode,
) -> std::cmp::Ordering {
    left.coordinates_m
        .canonical_value()
        .iter()
        .zip(right.coordinates_m.canonical_value().iter())
        .find_map(|(left, right)| (left != right).then(|| left.total_cmp(right)))
        .unwrap_or_else(|| left.identity.cmp(&right.identity))
}

fn reverse_sign(sign: PredicateSign) -> PredicateSign {
    match sign {
        PredicateSign::Negative => PredicateSign::Positive,
        PredicateSign::Zero => PredicateSign::Zero,
        PredicateSign::Positive => PredicateSign::Negative,
    }
}

#[cfg(test)]
mod tests;
