use std::collections::BTreeSet;

use runmat_meshing_core::quality::predicate::Point3Construction;

use super::DelaunayVolumeNode;

const MAXIMUM_ULP_OFFSET: i8 = 2;

/// Enumerates nearby binary64 representations in a stable, distance-first
/// order. Exact constructions keep their provenance; only the coordinates
/// consumed by downstream solvers move, by at most two ulps per axis.
pub(super) fn representable_nodes(node: DelaunayVolumeNode) -> Vec<DelaunayVolumeNode> {
    if node.construction == Point3Construction::Explicit {
        return vec![node];
    }

    let mut candidates = Vec::new();
    let mut positions = BTreeSet::new();
    for radius in 0..=MAXIMUM_ULP_OFFSET {
        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    if x.abs().max(y.abs()).max(z.abs()) != radius {
                        continue;
                    }
                    let coordinates_m = [
                        offset_ulps(node.coordinates_m[0], x),
                        offset_ulps(node.coordinates_m[1], y),
                        offset_ulps(node.coordinates_m[2], z),
                    ];
                    let key = coordinates_m.map(|coordinate| coordinate.to_bits());
                    if positions.insert(key) {
                        candidates.push(DelaunayVolumeNode {
                            coordinates_m,
                            ..node
                        });
                    }
                }
            }
        }
    }
    candidates
}

fn offset_ulps(mut value: f64, offset: i8) -> f64 {
    for _ in 0..offset.unsigned_abs() {
        value = if offset < 0 {
            value.next_down()
        } else {
            value.next_up()
        };
    }
    value
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{quality::predicate::Point3Construction, StableDigest};

    use super::*;

    #[test]
    fn constructed_nodes_enumerate_nominal_then_bounded_neighbors() {
        let node = DelaunayVolumeNode {
            identity: StableDigest::from_bytes([7; 32]),
            coordinates_m: [1.0, 2.0, 3.0],
            construction: Point3Construction::Segment {
                endpoints: [[0.0; 3], [2.0, 4.0, 6.0]],
                parameter_numerator: 1,
                parameter_exponent: 1,
            },
        };
        let candidates = representable_nodes(node);
        assert_eq!(candidates[0], node);
        assert_eq!(candidates.len(), 125);
        assert!(candidates.iter().all(|candidate| {
            candidate.identity == node.identity && candidate.construction == node.construction
        }));
    }

    #[test]
    fn explicit_nodes_are_not_perturbed() {
        let node = DelaunayVolumeNode::explicit(StableDigest::from_bytes([8; 32]), [1.0; 3]);
        assert_eq!(representable_nodes(node), vec![node]);
    }
}
