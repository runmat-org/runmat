use super::{DelaunaySegmentRecoveryError, RecoveryWork};
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use runmat_meshing_core::{Aabb3, StableDigest};

pub(super) fn find_representable_node(
    topology: &impl TetrahedralTopology,
    coordinates: [f64; 3],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<StableDigest>, DelaunaySegmentRecoveryError> {
    // A global scale only broadens the candidate box. The original pairwise
    // tolerance decides reuse, including at a rounding/scale boundary.
    let scale = coordinates
        .into_iter()
        .map(f64::abs)
        .fold(topology.coordinate_scale_m(), f64::max);
    let tolerance = representable_location_tolerance(scale);
    let bounds = Aabb3 {
        min_m: coordinates.map(|value| (value - tolerance).next_down()),
        max_m: coordinates.map(|value| (value + tolerance).next_up()),
    };
    let finite = bounds
        .min_m
        .into_iter()
        .chain(bounds.max_m)
        .all(f64::is_finite);
    let candidates = topology.node_candidates_in_bounds(finite.then_some(bounds), &mut || {
        work.search_location_step(constraint_index)
    })?;
    for index in candidates {
        work.search_location_step(constraint_index)?;
        let node = topology.nodes()[index as usize];
        if coordinates_share_representable_location(
            node.coordinates_m.canonical_value(),
            coordinates,
        ) {
            return Ok(Some(node.identity));
        }
    }
    Ok(None)
}

pub(in crate::cdt) fn coordinates_share_representable_location(
    left: [f64; 3],
    right: [f64; 3],
) -> bool {
    let scale = left
        .into_iter()
        .chain(right)
        .map(f64::abs)
        .fold(1.0_f64, f64::max);
    let tolerance = representable_location_tolerance(scale);
    left.into_iter()
        .zip(right)
        .all(|(left, right)| (left - right).abs() <= tolerance)
}

fn representable_location_tolerance(scale_m: f64) -> f64 {
    128.0 * f64::EPSILON * scale_m.max(1.0)
}

#[cfg(test)]
mod tests;
