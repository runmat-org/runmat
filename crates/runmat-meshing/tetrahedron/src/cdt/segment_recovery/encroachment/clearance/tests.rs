use super::*;
use runmat_meshing_core::NeverCancelled;

fn problem(
    scale: f64,
    angle: f64,
) -> (
    [DelaunayVolumeNode; 2],
    DelaunayVolumeNode,
    SegmentContext,
    [DyadicNode; 2],
) {
    let node = |id, point| DelaunayVolumeNode::explicit(StableDigest::from_bytes([id; 32]), point);
    let endpoints = [node(1, [0., 0., 0.]), node(2, [scale, 0., 0.])];
    let witness = node(
        3,
        [0.37 * angle.cos() * scale, 0.37 * angle.sin() * scale, 0.],
    );
    let context = SegmentContext {
        constraint_index: 7,
        first_identity: endpoints[0].identity,
        last_identity: endpoints[1].identity,
        first_coordinates: endpoints[0].coordinates_m.canonical_value(),
        last_coordinates: endpoints[1].coordinates_m.canonical_value(),
    };
    let interval = [
        DyadicNode::endpoint(endpoints[0].identity, false),
        DyadicNode::endpoint(endpoints[1].identity, true),
    ];
    (endpoints, witness, context, interval)
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Off-centre dyadic rounding clears the original witness from both materialized child
//   diametral spheres at 0.2/0.73/2 degrees and physical scales 0.001/1/1000. Independent analytic
//   dot products verify the exact predicate verdicts; work remains charged to the original search
//   ceiling. Excludes all-witness clearance, midpoint policy, complete recovery termination and
//   native resource qualification.
#[test]
fn acute_offcentre_rounding_requires_exact_child_sphere_clearance_across_scales() {
    for scale in [1.0e-3, 1.0, 1.0e3] {
        for degrees in [0.2_f64, 0.73, 2.0] {
            let (endpoints, witness, context, interval) = problem(scale, degrees.to_radians());
            let mut work = RecoveryWork::new(Default::default(), &NeverCancelled);
            let coarse = interval[0]
                .near_parameter(interval[1], 0.37, context)
                .unwrap();
            assert!(!admits(endpoints, witness, context, coarse, &mut work).unwrap());
            let split = choose(
                endpoints,
                witness,
                context,
                interval,
                [
                    0.37,
                    0.37,
                    witness.coordinates_m.canonical_value()[0] / scale,
                ],
                None,
                &mut work,
            )
            .unwrap();
            let point = interpolate(
                context.first_coordinates,
                context.last_coordinates,
                split.parameter().unwrap(),
            );
            // Independent analytic dot products: negative means encroachment.
            for child in [
                [endpoints[0].coordinates_m.canonical_value(), point],
                [point, endpoints[1].coordinates_m.canonical_value()],
            ] {
                let dot = (0..3)
                    .map(|axis| {
                        (witness.coordinates_m.canonical_value()[axis] - child[0][axis])
                            * (witness.coordinates_m.canonical_value()[axis] - child[1][axis])
                    })
                    .sum::<f64>();
                assert!(dot >= 0., "angle={degrees}, scale={scale}, dot={dot}");
            }
            assert!(admits(endpoints, witness, context, split, &mut work).unwrap());
            assert!(work.search_inventory()[2] > 0);
        }
    }
}

#[test]
fn clearance_rejects_an_invalid_preference_then_admits_a_geometric_alternative() {
    let (endpoints, mut witness, context, interval) = problem(1., 0.1);
    witness.coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([0.8, 0.21, 0.]);
    let mut work = RecoveryWork::new(Default::default(), &NeverCancelled);
    let split = choose(
        endpoints,
        witness,
        context,
        interval,
        [0.5, (0.8_f64.powi(2) + 0.21_f64.powi(2)).sqrt(), 0.8],
        None,
        &mut work,
    )
    .unwrap();
    assert!(admits(endpoints, witness, context, split, &mut work).unwrap());
    assert_ne!(split.parameter().unwrap(), 0.5);
}

#[test]
fn clearance_work_uses_the_original_search_ceiling_and_retries_after_rejection() {
    let (endpoints, witness, context, interval) = problem(1., 0.73_f64.to_radians());
    let options = super::super::super::DelaunaySegmentRecoveryOptions {
        maximum_search_steps: 1,
        ..Default::default()
    };
    let mut limited = RecoveryWork::new(options, &NeverCancelled);
    assert_eq!(
        choose(
            endpoints,
            witness,
            context,
            interval,
            [0.37; 3],
            None,
            &mut limited
        )
        .err()
        .expect("unchanged search budget must reject")
        .kind,
        DelaunaySegmentRecoveryErrorKind::ResourceLimit
    );
    let mut retry = RecoveryWork::new(Default::default(), &NeverCancelled);
    assert!(choose(endpoints, witness, context, interval, [0.37; 3], None, &mut retry).is_ok());
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An independent acute-ray discriminator exhibits coarse child-clearance-only radial
//   erosion and requires the guarded split to retain the reference radius, clear both child spheres
//   and approach the reference radius to floating-point resolution, in mirrored geometry at scales
//   0.001, 1 and 1000. The intentionally smaller rejected-radius spacing rule, full simultaneous
//   constraint recovery, cancellation and original native performance require separate evidence.
#[test]
fn acute_protection_radius_does_not_erode_through_dyadic_rounding() {
    for scale in [1.0e-3, 1., 1.0e3] {
        for mirrored in [false, true] {
            let (mut endpoints, mut witness, mut context, interval) =
                problem(scale, 0.73_f64.to_radians());
            if mirrored {
                for node in &mut endpoints {
                    {
                        let quantity_change = scale - node.coordinates_m.canonical_value()[0];
                        let quantity_index = 0;
                        let quantity_slot = &mut node.coordinates_m;
                        let mut quantity_coordinates = quantity_slot.canonical_value();
                        quantity_coordinates[quantity_index] = quantity_change;
                        *quantity_slot = runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(quantity_coordinates);
                    };
                }
                {
                    let quantity_change = scale - witness.coordinates_m.canonical_value()[0];
                    let quantity_index = 0;
                    let quantity_slot = &mut witness.coordinates_m;
                    let mut quantity_coordinates = quantity_slot.canonical_value();
                    quantity_coordinates[quantity_index] = quantity_change;
                    *quantity_slot = runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(quantity_coordinates);
                };
                context.first_coordinates = endpoints[0].coordinates_m.canonical_value();
                context.last_coordinates = endpoints[1].coordinates_m.canonical_value();
            }
            let origin = endpoints[0].coordinates_m.canonical_value();
            let mut work = RecoveryWork::new(Default::default(), &NeverCancelled);
            let unguarded = interval[0]
                .near_parameter_with_clearance(interval[1], 0.37, &mut |candidate| {
                    admits(endpoints, witness, context, candidate, &mut work)
                })
                .unwrap()
                .unwrap();
            // This independently exhibits the defect: child clearance alone can
            // admit a radius smaller than the incident ray's reference radius.
            let radius_squared = |point: [f64; 3]| {
                (0..3)
                    .map(|axis| (point[axis] - origin[axis]).powi(2))
                    .sum::<f64>()
            };
            let reference_squared = radius_squared(witness.coordinates_m.canonical_value());
            let unguarded_point = interpolate(
                context.first_coordinates,
                context.last_coordinates,
                unguarded.parameter().unwrap(),
            );
            assert!(radius_squared(unguarded_point) < reference_squared);
            let guarded = choose(
                endpoints,
                witness,
                context,
                interval,
                [0.37; 3],
                Some(origin),
                &mut work,
            )
            .unwrap();
            let guarded_point = interpolate(
                context.first_coordinates,
                context.last_coordinates,
                guarded.parameter().unwrap(),
            );
            assert!(radius_squared(guarded_point) >= reference_squared);
            // The valid coarse point is only a bracket. Preserve the physical
            // radius to floating-point resolution rather than rounding it up
            // by a fraction of the current fragment at every coupled split.
            assert!(
                (radius_squared(guarded_point).sqrt() - reference_squared.sqrt()).abs()
                    <= 8.0 * f64::EPSILON * scale
            );
            assert!(admits(endpoints, witness, context, guarded, &mut work).unwrap());
        }
    }
}
