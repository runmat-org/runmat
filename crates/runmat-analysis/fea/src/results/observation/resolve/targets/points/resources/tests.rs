use super::*;
use runmat_analysis_core::{Position, RecoveryPolicy};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn original_point_targets_fit_identity_and_inventory_admission() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(order)).unwrap();
        let topology = ObservationTopology::from_projection(&projection).unwrap();
        let mut spec = crate::results::observation::tests::spec(
            "counted-path",
            ObservationTarget::Path {
                positions: [[0.1, 0.1, 0.1], [0.2, 0.3, 0.1], [0.05, 0.15, 0.4]]
                    .into_iter()
                    .map(Position::new)
                    .collect(),
            },
        );
        spec.output_location = FieldLocation::Point;
        spec.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
        let bytes = interpolated_points_resident_bytes(&projection, 3).unwrap();
        let run = |cap| {
            super::super::resolve_interpolated_points(
                &spec,
                RecoveredFieldEntityDomain::Nodes,
                &topology,
                ObservationLimits {
                    maximum_peak_resident_bytes: cap,
                    ..Default::default()
                },
            )
        };
        let (target, retained, peak) = crate::topology::measure_test_allocations(|| run(bytes));
        let target = target.unwrap();
        assert_eq!(target.selected_entity_count(), 3);
        assert!(
            peak as u64 <= bytes,
            "point construction {peak} exceeds {bytes}"
        );
        assert!(retained < peak);
        let (short, retained, short_peak) =
            crate::topology::measure_test_allocations(|| run(bytes - 1));
        assert!(matches!(
            short,
            Err(ObservationError::Limit(
                "point target construction residency"
            ))
        ));
        assert_eq!(retained, 0);
        // Profiling may encode fixed-length digest identifiers; no point inventory is allocated.
        assert!((short_peak as u64) < projection.volume_point_location_resident_bytes().unwrap());
    }
}
