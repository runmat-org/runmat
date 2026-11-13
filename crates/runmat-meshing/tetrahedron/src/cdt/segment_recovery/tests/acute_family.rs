//! Complete simultaneous recovery on independently specified acute ray pairs.
//! This small geometric family is separate from captured application evidence.
use super::*;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Six independently specified acute-ray geometries recover all simultaneous chains with
//   original endpoints and strictly ordered full parameter spans, across three scales and two
//   identity orders. A small feature family does not establish arbitrary assembly complexity,
//   complete native thermal or modal acceptance, resource saturation or a distributed meshing
//   envelope.
#[test]
fn acute_incident_rays_recover_simultaneously_across_scale_and_identity_order() {
    for scale in [1.0e-3, 1., 1.0e3] {
        for reverse in [false, true] {
            let angle = 0.73_f64.to_radians();
            let coordinates = [
                [0., 0., 0.],
                [1., 0., 0.],
                [0.37 * angle.cos(), 0.37 * angle.sin(), 0.],
                [0.37 * angle.cos(), -0.37 * angle.sin(), 0.],
                [-1., -1., 0.],
                [-1., 1., 0.],
                [2., 1., 0.],
                [2., -1., 0.],
                [0., 0., 1.],
            ];
            let mut constraints = DelaunayConstraints {
                nodes: coordinates
                    .into_iter()
                    .enumerate()
                    .map(|(index, point)| DelaunayConstraintNode {
                        identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
                        coordinates_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(
                            point.map(|value| value * scale)
                        ),
                        source_vertex_id: None,
                    })
                    .collect(),
                segments: [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
                    .into_iter()
                    .map(|vertex_indices| DelaunayConstraintSegment {
                        vertex_indices,
                        source_edge_id: None,
                        source_edge_parameters: None,
                    })
                    .collect(),
                facets: Vec::new(),
            };
            if reverse {
                constraints.nodes.reverse();
                for (index, node) in constraints.nodes.iter_mut().enumerate() {
                    node.identity = StableDigest::from_bytes([(index + 1) as u8; 32]);
                }
                for segment in &mut constraints.segments {
                    segment.vertex_indices = segment.vertex_indices.map(|index| 8 - index);
                    segment.vertex_indices.sort_unstable();
                }
                constraints
                    .segments
                    .sort_by_key(|segment| segment.vertex_indices);
            }
            let initial = build_delaunay_volume_point_set(
                constraints.volume_nodes(),
                DelaunayPointSetOptions::default(),
                &NeverCancelled,
            )
            .unwrap();
            let options = DelaunaySegmentRecoveryOptions::default();
            let mut work = RecoveryWork::new(options, &NeverCancelled);
            assert!(
                !edge_exists(
                    &initial,
                    constraints.nodes[if reverse { 8 } else { 0 }].identity,
                    constraints.nodes[if reverse { 7 } else { 1 }].identity,
                    0,
                    &mut work
                )
                .unwrap(),
                "family must exercise missing-edge recovery"
            );
            let recovered = recover_delaunay_segments_from_point_set(
                initial,
                &constraints,
                options,
                &NeverCancelled,
            )
            .unwrap();
            validate_delaunay_segment_recovery(&recovered, &constraints, options, &NeverCancelled)
                .unwrap();
            assert!(!recovered.steiner_node_identities.is_empty());
            // Independent analytic closure: all original rays span the complete
            // parameter interval, in strict order, with unchanged endpoints.
            for (segment, chain) in constraints.segments.iter().zip(&recovered.segments) {
                assert_eq!(
                    chain.nodes.first().unwrap().identity,
                    constraints.nodes[segment.vertex_indices[0] as usize].identity
                );
                assert_eq!(
                    chain.nodes.last().unwrap().identity,
                    constraints.nodes[segment.vertex_indices[1] as usize].identity
                );
                let parameters = chain
                    .nodes
                    .iter()
                    .map(|node| {
                        node.parameter_numerator as f64
                            / 2.0_f64.powi(i32::from(node.parameter_exponent))
                    })
                    .collect::<Vec<_>>();
                assert_eq!((parameters[0], *parameters.last().unwrap()), (0., 1.));
                assert!(parameters.windows(2).all(|pair| pair[0] < pair[1]));
            }
        }
    }
}
