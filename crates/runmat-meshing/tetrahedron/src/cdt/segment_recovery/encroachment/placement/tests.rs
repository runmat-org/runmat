use super::*;
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Rejected radial placement uses inherited acute-corner spacing in both endpoint
//   orientations at physical scales 0.001/1/1000, followed by separate exact child-sphere
//   admission. The analytic discriminator distinguishes this rule from whole-fragment bisection and
//   orthogonal projection. Does not prove global corner protection, arbitrary feature-size bounds
//   or complete native recovery termination.
#[test]
fn rejected_corner_radius_uses_inherited_spacing_in_both_orientations() {
    for scale in [0.001, 1.0, 1000.0] {
        for reverse in [false, true] {
            let map = |x: f64| if reverse { 1.0 - x } else { x };
            let coordinates = |x, y| [x * scale, y * scale, 0.0];
            let node = |id, point| {
                crate::cdt::DelaunayVolumeNode::explicit(StableDigest::from_bytes([id; 32]), point)
            };
            let endpoints = [node(1, coordinates(0., 0.)), node(2, coordinates(1., 0.))];
            let context = SegmentContext {
                constraint_index: 3,
                first_identity: endpoints[0].identity,
                last_identity: endpoints[1].identity,
                first_coordinates: endpoints[0].coordinates_m.canonical_value(),
                last_coordinates: endpoints[1].coordinates_m.canonical_value(),
            };
            let witness = node(3, coordinates(map(0.8), 0.21));
            let hints = preferred(
                endpoints.map(|n| n.coordinates_m.canonical_value()),
                witness.coordinates_m.canonical_value(),
                context,
                [0., 1.],
                if reverse {
                    [false, true]
                } else {
                    [true, false]
                },
            );
            // Independent analytic spacing: the rejected radius is sqrt(.6841).
            // Its witness gap is sqrt((radius-.8)^2 + .21^2). Rule 3 advances
            // from the acute endpoint by radius-gap, not the whole midpoint.
            let radius = 0.6841_f64.sqrt();
            let gap = ((radius - 0.8).powi(2) + 0.21_f64.powi(2)).sqrt();
            assert!((hints.parameters[0] - map(radius - gap)).abs() < 1.0e-14);
            assert!((0.61..0.62).contains(&map(hints.parameters[0])));
            let interval = [
                DyadicNode::endpoint(endpoints[0].identity, false),
                DyadicNode::endpoint(endpoints[1].identity, true),
            ];
            let mut work = RecoveryWork::new(Default::default(), &NeverCancelled);
            let split = super::super::clearance::choose(
                endpoints,
                witness,
                context,
                interval,
                hints.parameters,
                hints.radial_origin,
                &mut work,
            )
            .unwrap();
            assert!((0.61..0.62).contains(&map(split.parameter().unwrap())));
        }
    }
}

#[test]
fn corner_spacing_uses_fragment_endpoint_without_resetting_the_original_reference() {
    let context = SegmentContext {
        constraint_index: 3,
        first_identity: StableDigest::from_bytes([1; 32]),
        last_identity: StableDigest::from_bytes([2; 32]),
        first_coordinates: [0., 0., 0.],
        last_coordinates: [1., 0., 0.],
    };
    let witness = [0.8, 0.21, 0.];
    let hints = preferred(
        [[0.5, 0., 0.], [1., 0., 0.]],
        witness,
        context,
        [0.5, 1.],
        [true, false],
    );
    // The near-to-rejected radius interval is too short for the subtract-gap
    // branch. Bisect that interval, retaining the original corner reference.
    assert!((hints.parameters[0] - (0.5 + 0.5 * (0.6841_f64.sqrt() - 0.5))).abs() < 1.0e-14);
    assert_ne!(hints.parameters[0], 0.75);
}

#[test]
fn accepted_radial_hint_retains_the_original_acute_reference_on_both_sides() {
    for right_acute in [false, true] {
        let context = SegmentContext {
            constraint_index: 0,
            first_identity: StableDigest::from_bytes([1; 32]),
            last_identity: StableDigest::from_bytes([2; 32]),
            first_coordinates: [0., 0., 0.],
            last_coordinates: [1., 0., 0.],
        };
        let witness = if right_acute {
            [0.63, 0.003, 0.]
        } else {
            [0.37, 0.003, 0.]
        };
        let hints = preferred(
            [context.first_coordinates, context.last_coordinates],
            witness,
            context,
            [0., 1.],
            [!right_acute, right_acute],
        );
        assert_eq!(
            hints.radial_origin,
            Some(if right_acute {
                context.last_coordinates
            } else {
                context.first_coordinates
            })
        );
    }
}
