use runmat_meshing_core::{
    quality::predicate::{diametral_sphere_position, PredicateSign},
    MeshingCancellationSignal,
};

use super::{
    error, DelaunayConstraints, DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind,
    DelaunaySegmentRecoveryOptions,
};

/// Classifies constraint vertices that are the acute endpoint of any pair of incident segments.
///
/// For incident endpoints `p-q-r`, the angle at `q` is acute exactly when `q` lies outside the
/// sphere with diameter `p-r`. Reusing the exact diametral predicate keeps this classification
/// deterministic for nearly orthogonal and exactly symmetric input.
pub(super) fn classify_acute_vertices(
    constraints: &DelaunayConstraints,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<bool>, DelaunaySegmentRecoveryError> {
    let mut neighbors = vec![Vec::<u32>::new(); constraints.nodes.len()];
    for segment in &constraints.segments {
        let [left, right] = segment.vertex_indices;
        neighbors[left as usize].push(right);
        neighbors[right as usize].push(left);
    }
    let mut acute = vec![false; constraints.nodes.len()];
    let mut checks = 0_u64;
    for (center, adjacent) in neighbors.iter_mut().enumerate() {
        adjacent.sort_unstable();
        adjacent.dedup();
        'pairs: for first in 0..adjacent.len() {
            for second in (first + 1)..adjacent.len() {
                checks = checks.checked_add(1).ok_or_else(|| {
                    error(
                        DelaunaySegmentRecoveryErrorKind::ResourceLimit,
                        None,
                        "acute-vertex classification count overflow",
                    )
                })?;
                if checks.is_multiple_of(options.constraints.cancellation_check_interval)
                    && cancellation.is_cancelled()
                {
                    return Err(error(
                        DelaunaySegmentRecoveryErrorKind::Cancelled,
                        None,
                        "cancelled",
                    ));
                }
                let endpoints = [
                    constraints.nodes[adjacent[first] as usize]
                        .coordinates_m
                        .canonical_value(),
                    constraints.nodes[adjacent[second] as usize]
                        .coordinates_m
                        .canonical_value(),
                ];
                let point = constraints.nodes[center].coordinates_m.canonical_value();
                let relation = diametral_sphere_position(endpoints, point).map_err(|failure| {
                    error(
                        DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
                        None,
                        format!("acute-vertex predicate failed: {failure:?}"),
                    )
                })?;
                if relation == PredicateSign::Positive {
                    acute[center] = true;
                    break 'pairs;
                }
            }
        }
    }
    Ok(acute)
}
