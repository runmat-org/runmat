//! Source-independent strict containment. The shared 3D broad phase is only
//! a conservative candidate filter; original adaptive predicates own the verdict.
use super::*;
use runmat_meshing_core::quality::predicate::{orient3d, PredicateSign};
use runmat_meshing_core::{Aabb3, PointSpatialIndex, SpatialPoint};

pub(super) fn contains_embedded_node(
    topology: &DelaunayVolumeTopology,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<bool, DelaunayVolumeSliverError> {
    let mut work = 0u64;
    let mut checkpoint = || {
        super::super::checkpoint(work, options, cancellation)?;
        work = work.saturating_add(1);
        Ok::<_, DelaunayVolumeSliverError>(())
    };
    let entries = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            checkpoint()?;
            SpatialPoint::new(node.coordinates_m.canonical_value(), index).ok_or_else(|| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "embedded-node index requires finite canonical coordinates",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let index = PointSpatialIndex::from_entries_with_checkpoint(entries, &mut checkpoint)?;
    for tetrahedron in &topology.tetrahedra {
        checkpoint()?;
        let points = tetrahedron.vertex_indices.map(|vertex| {
            topology.nodes[vertex as usize]
                .coordinates_m
                .canonical_value()
        });
        let bounds = bounds(points);
        // The open region also prunes entire coplanar occupied branches.
        // Stream payloads into one inventory; preserve the original x/index
        // order for predicate failures and the first embedded-node verdict.
        let mut candidates = candidate_nodes(&index, bounds, &mut checkpoint)?;
        candidates.sort_unstable_by(|left, right| {
            topology.nodes[*left].coordinates_m.canonical_value()[0]
                .total_cmp(&topology.nodes[*right].coordinates_m.canonical_value()[0])
                .then_with(|| left.cmp(right))
        });
        for node_index in candidates {
            checkpoint()?;
            if tetrahedron.vertex_indices.contains(&(node_index as u32)) {
                continue;
            }
            let node = &topology.nodes[node_index];
            let mut interior = true;
            for replace in 0..4 {
                let mut simplex = points;
                simplex[replace] = node.coordinates_m.canonical_value();
                let sign = orient3d(simplex).map_err(|failure| {
                    error(
                        DelaunayVolumeSliverErrorKind::InvalidTopology,
                        format!("optimized topology containment predicate failed: {failure:?}"),
                    )
                })?;
                if sign != PredicateSign::Positive {
                    interior = false;
                    break;
                }
            }
            if interior {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
fn bounds(points: [[f64; 3]; 4]) -> Aabb3 {
    Aabb3 {
        min_m: std::array::from_fn(|axis| {
            points
                .iter()
                .map(|point| point[axis])
                .min_by(f64::total_cmp)
                .expect("four corners")
        }),
        max_m: std::array::from_fn(|axis| {
            points
                .iter()
                .map(|point| point[axis])
                .max_by(f64::total_cmp)
                .expect("four corners")
        }),
    }
}
fn candidate_nodes(
    index: &PointSpatialIndex<usize>,
    bounds: Aabb3,
    checkpoint: &mut impl FnMut() -> Result<(), DelaunayVolumeSliverError>,
) -> Result<Vec<usize>, DelaunayVolumeSliverError> {
    // Positive tetrahedra have nonzero extent in every coordinate. A strict
    // interior point is a convex combination with four positive weights, hence
    // is strictly between every coordinate's extrema. Degenerate occupied point
    // boxes still intersect this open region when their coordinate is interior.
    let intersects = |occupied: Aabb3| {
        (0..3).all(|axis| {
            occupied.max_m[axis] > bounds.min_m[axis] && occupied.min_m[axis] < bounds.max_m[axis]
        })
    };
    let mut candidates = Vec::new();
    index.visit_region_with_checkpoint(
        bounds,
        &intersects,
        &mut |entry| candidates.push(entry.payload),
        checkpoint,
    )?;
    Ok(candidates)
}

#[cfg(test)]
mod tests;
