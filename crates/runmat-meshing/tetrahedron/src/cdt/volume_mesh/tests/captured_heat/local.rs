//! Authenticated, explicitly partial recovery discriminators. Passing these
//! neighborhoods cannot establish complete native surface or study acceptance.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[test]
#[ignore = "requires the authenticated native heat-sink boundary diagnostic fixture"]
fn authenticated_heat_neighborhoods_recover_without_unbounded_refinement() {
    let original = authenticated_constraints();
    let original_frame = frame::CanonicalVolumeFrame::from_constraints(&original).unwrap();
    for seed in [3490, 23313, 14580] {
        let mut selected = BTreeSet::from(original.segments[seed].vertex_indices);
        // Three complete adjacency layers retain coupled incident segments.
        for _ in 0..3 {
            let previous = selected.clone();
            for segment in &original.segments {
                if segment
                    .vertex_indices
                    .iter()
                    .any(|node| previous.contains(node))
                {
                    selected.extend(segment.vertex_indices);
                }
            }
        }
        // Original axis-extreme points provide physical 3D hull context when a
        // selected patch is planar. No artificial points or altered coordinates.
        for axis in 0..3 {
            for maximum in [false, true] {
                let index = (0..original.nodes.len())
                    .max_by(|&left, &right| {
                        let order = original.nodes[left].coordinates_m.canonical_value()[axis]
                            .total_cmp(
                                &original.nodes[right].coordinates_m.canonical_value()[axis],
                            );
                        if maximum {
                            order
                        } else {
                            order.reverse()
                        }
                    })
                    .unwrap();
                selected.insert(index as u32);
            }
        }
        assert!(selected.len() <= 1024, "bounded diagnostic neighborhood");
        let indices = selected
            .iter()
            .enumerate()
            .map(|(index, &original)| (original, index as u32))
            .collect::<BTreeMap<_, _>>();
        let constraints = DelaunayConstraints {
            nodes: selected
                .iter()
                .map(|&index| original.nodes[index as usize].clone())
                .collect(),
            segments: original
                .segments
                .iter()
                .filter(|segment| {
                    segment
                        .vertex_indices
                        .iter()
                        .all(|node| selected.contains(node))
                })
                .map(|segment| DelaunayConstraintSegment {
                    vertex_indices: segment.vertex_indices.map(|node| indices[&node]),
                    source_edge_id: segment.source_edge_id.clone(),
                    source_edge_parameters: segment.source_edge_parameters,
                })
                .collect(),
            facets: Vec::new(),
        };
        let initial = crate::cdt::point_set::build_delaunay_volume_point_set_with_order(
            constraints.volume_nodes(),
            Default::default(),
            &NeverCancelled,
            |left, right| original_frame.compare_nodes(left, right),
        )
        .unwrap();
        // A tighter diagnostic work cap detects local runaway promptly. The
        // complete original replay retains its original one-billion-step cap.
        let options = DelaunaySegmentRecoveryOptions {
            maximum_search_steps: 10_000_000,
            ..Default::default()
        };
        let recovered = tracing::subscriber::with_default(
            crate::cdt::test_fixtures::diagnostics::RecoveryDiagnostics::segment_recovery(),
            || {
                recover_delaunay_segments_from_point_set(
                    initial,
                    &constraints,
                    options,
                    &NeverCancelled,
                )
            },
        )
        .unwrap_or_else(|failure| {
            panic!(
                "seed={seed}, nodes={}, segments={}: {failure:?}",
                constraints.nodes.len(),
                constraints.segments.len()
            )
        });
        crate::cdt::validate_delaunay_segment_recovery(
            &recovered,
            &constraints,
            options,
            &NeverCancelled,
        )
        .unwrap();
        eprintln!(
            "authenticated neighborhood seed={seed} nodes={} segments={} Steiner={} passes={}",
            constraints.nodes.len(),
            constraints.segments.len(),
            recovered.steiner_node_identities.len(),
            recovered.recovery_passes
        );
    }
}
