use std::collections::BTreeSet;

use runmat_meshing_core::quality::predicate::Point3Construction;

use super::DelaunayVolumeNode;

const MAXIMUM_ULP_OFFSET: i8 = 2;

/// Enumerates nearby binary64 representations in a stable, distance-first
/// order. Exact constructions keep their provenance; only the coordinates
/// consumed by downstream solvers move, by at most two ulps per axis.
pub(super) fn representable_nodes(
    node: DelaunayVolumeNode,
) -> impl Iterator<Item = DelaunayVolumeNode> {
    // Store only the bounded offset recipe once. Most insertions accept the
    // nominal representation and never need the other 124 constructed nodes.
    static OFFSETS: std::sync::OnceLock<Vec<[i8; 3]>> = std::sync::OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        let mut offsets = Vec::new();
        for radius in 0..=MAXIMUM_ULP_OFFSET {
            for x in -radius..=radius {
                for y in -radius..=radius {
                    for z in -radius..=radius {
                        if x.abs().max(y.abs()).max(z.abs()) == radius {
                            offsets.push([x, y, z]);
                        }
                    }
                }
            }
        }
        offsets
    });
    let count = if node.construction == Point3Construction::Explicit {
        1
    } else {
        offsets.len()
    };
    let mut positions = BTreeSet::new();
    offsets[..count].iter().filter_map(move |offset| {
        let coordinates_m = std::array::from_fn(|axis| {
            offset_ulps(node.coordinates_m.canonical_value()[axis], offset[axis])
        });
        positions
            .insert(coordinates_m.map(f64::to_bits))
            .then_some(DelaunayVolumeNode {
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
                ..node
            })
    })
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
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([1.0, 2.0, 3.0]),
            construction: Point3Construction::Segment {
                endpoints: [[0.0; 3], [2.0, 4.0, 6.0]],
                parameter_numerator: 1,
                parameter_exponent: 1,
            },
        };
        let candidates = representable_nodes(node).collect::<Vec<_>>();
        assert_eq!(candidates[0], node);
        assert_eq!(candidates.len(), 125);
        assert!(candidates.iter().all(|candidate| {
            candidate.identity == node.identity && candidate.construction == node.construction
        }));
    }

    #[test]
    fn explicit_nodes_are_not_perturbed() {
        let node = DelaunayVolumeNode::explicit(StableDigest::from_bytes([8; 32]), [1.0; 3]);
        assert_eq!(representable_nodes(node).collect::<Vec<_>>(), vec![node]);
    }
}

#[cfg(test)]
mod sequence_tests {
    use super::*;
    use runmat_meshing_core::StableDigest;
    #[test]
    fn lazy_materializations_preserve_distance_first_sequence_and_prefix() {
        for coordinates_m in [
            [1.0, 2.0, 3.0],
            [-0.0, 0.0, f64::MIN_POSITIVE],
            [f64::MAX, -f64::MAX, -1.0],
        ] {
            let node = DelaunayVolumeNode {
                identity: StableDigest::from_bytes([9; 32]),
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
                construction: Point3Construction::TetrahedronCentroid {
                    vertices: [[0.0; 3]; 4],
                },
            };
            let mut expected = Vec::new();
            let mut seen = BTreeSet::new();
            for radius in 0..=MAXIMUM_ULP_OFFSET {
                for x in -radius..=radius {
                    for y in -radius..=radius {
                        for z in -radius..=radius {
                            if x.abs().max(y.abs()).max(z.abs()) != radius {
                                continue;
                            }
                            let point = [
                                offset_ulps(coordinates_m[0], x),
                                offset_ulps(coordinates_m[1], y),
                                offset_ulps(coordinates_m[2], z),
                            ];
                            if seen.insert(point.map(f64::to_bits)) {
                                expected.push(point.map(f64::to_bits));
                            }
                        }
                    }
                }
            }
            assert_eq!(
                representable_nodes(node)
                    .map(|point| point.coordinates_m.canonical_value().map(f64::to_bits))
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(representable_nodes(node).take(1).count(), 1);
        }
    }
}
