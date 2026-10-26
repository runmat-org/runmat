//! Recover complete planar, concave extrusion constraints under geometry and order changes.
//! These are segment-recovery witnesses; volume quality and physics require native studies.
use super::{original_constraints, recover_constraints};

#[derive(Clone, Copy, Debug)]
enum Condition {
    Rotated,
    TranslatedAndScaled,
    HalfThickness,
    DoubleThickness,
    ReversedSegments,
    ReversedNodes,
    RelabeledNodes,
}

fn exercise(condition: Condition) {
    eprintln!("complete fork constraint family: {condition:?}");
    let mut constraints = original_constraints();
    match condition {
        Condition::Rotated => {
            for node in &mut constraints.nodes {
                let [x, y, z] = node.coordinates_m.canonical_value();
                node.coordinates_m = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([z, x, y]);
            }
        }
        Condition::TranslatedAndScaled => {
            for node in &mut constraints.nodes {
                node.coordinates_m = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(std::array::from_fn(|axis| {
                    node.coordinates_m.canonical_value()[axis] * 1024.0 + [0.125, -0.25, 0.5][axis]
                }));
            }
        }
        Condition::HalfThickness | Condition::DoubleThickness => {
            let factor = if matches!(condition, Condition::HalfThickness) {
                0.5
            } else {
                2.0
            };
            for node in &mut constraints.nodes {
                {
                    let quantity_change = factor;
                    let quantity_index = 1;
                    let quantity_slot = &mut node.coordinates_m;
                    let mut quantity_coordinates = quantity_slot.canonical_value();
                    quantity_coordinates[quantity_index] *= quantity_change;
                    *quantity_slot = runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(quantity_coordinates);
                };
            }
        }
        Condition::ReversedSegments => constraints.segments.reverse(),
        Condition::ReversedNodes => {
            constraints.nodes.reverse();
            let count = constraints.nodes.len() as u32;
            for segment in &mut constraints.segments {
                segment.vertex_indices = segment.vertex_indices.map(|index| count - 1 - index);
            }
        }
        Condition::RelabeledNodes => {
            for node in &mut constraints.nodes {
                let mut bytes = *node.identity.bytes();
                // A bijection changes symbolic ordering without merging nodes.
                bytes[0] ^= 0xa5;
                node.identity = super::StableDigest::from_bytes(bytes);
            }
        }
    }
    // Recovery consumes canonical constraints. Reordering import inventory
    // must restore that contract before it reaches the numerical kernel.
    let mut order = (0..constraints.nodes.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| constraints.nodes[*index].identity);
    let mut remap = vec![0; order.len()];
    for (canonical, original) in order.iter().copied().enumerate() {
        remap[original] = canonical as u32;
    }
    constraints.nodes = order
        .into_iter()
        .map(|index| constraints.nodes[index].clone())
        .collect();
    for segment in &mut constraints.segments {
        segment.vertex_indices = segment.vertex_indices.map(|index| remap[index as usize]);
        segment.vertex_indices.sort_unstable();
    }
    constraints
        .segments
        .sort_by_key(|segment| segment.vertex_indices);
    assert_eq!(
        (constraints.nodes.len(), constraints.segments.len()),
        (424, 1266)
    );
    recover_constraints(constraints);
}

#[test]
fn planar_concave_extrusion_rotation() {
    exercise(Condition::Rotated);
}

#[test]
fn planar_concave_extrusion_translation_and_scale() {
    exercise(Condition::TranslatedAndScaled);
}

#[test]
fn planar_concave_extrusion_half_thickness() {
    exercise(Condition::HalfThickness);
}

#[test]
fn planar_concave_extrusion_double_thickness() {
    exercise(Condition::DoubleThickness);
}

#[test]
fn planar_concave_extrusion_reversed_segments() {
    exercise(Condition::ReversedSegments);
}

#[test]
fn planar_concave_extrusion_reversed_nodes() {
    exercise(Condition::ReversedNodes);
}

#[test]
fn planar_concave_extrusion_relabeled_nodes() {
    exercise(Condition::RelabeledNodes);
}
