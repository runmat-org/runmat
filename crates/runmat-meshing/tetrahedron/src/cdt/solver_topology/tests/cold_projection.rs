use super::*;
use crate::cdt::encode_delaunay_volume_mesh;

#[test]
fn cold_projection_preserves_both_orders_and_rejects_changed_context_and_bytes() {
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let request = request(order);
        let options = volume_options();
        let volume = crate::cdt::construct_delaunay_volume_mesh(
            &topology,
            &surface,
            &request.metric,
            options,
            &NeverCancelled,
        )
        .unwrap();
        let bytes = encode_delaunay_volume_mesh(
            &volume,
            &topology,
            &surface,
            &request.metric,
            options,
            &NeverCancelled,
        )
        .unwrap();
        let evaluation = Some(DelaunayExactEvaluation {
            evaluator: &test_evaluator::EVALUATOR,
            control: &test_evaluator::CONTROL,
        });
        let expected = build_delaunay_solver_topology(
            DelaunaySolverTopologyInput {
                exact_topology: &topology,
                exact_surface: &surface,
                volume_mesh: &volume,
                volume_metric: &request.metric,
                volume_options: options,
                request: &request,
                exact_evaluation: evaluation,
            },
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let input = || DelaunaySolverProjectionInput {
            exact_topology: &topology,
            exact_surface: &surface,
            volume_metric: &request.metric,
            volume_options: options,
            request: &request,
            exact_evaluation: evaluation,
        };
        let actual = decode_delaunay_solver_topology(
            bytes.clone(),
            input(),
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        assert_eq!(actual.into_topology(), expected);
        let mut corrupted = bytes.clone();
        corrupted.push(0);
        assert!(matches!(
            decode_delaunay_solver_topology(
                corrupted,
                input(),
                DelaunaySolverTopologyOptions::default(),
                &NeverCancelled
            ),
            Err(DelaunaySolverProjectionError::Decode(_))
        ));
        let mut changed = input();
        changed.volume_options.quality.maximum_metric_edge_length = 0.01;
        assert!(matches!(
            decode_delaunay_solver_topology(
                bytes,
                changed,
                DelaunaySolverTopologyOptions::default(),
                &NeverCancelled
            ),
            Err(DelaunaySolverProjectionError::Decode(_))
        ));
    }
}

#[test]
fn cold_projection_admits_both_orders_and_transfers_only_original_retained_storage() {
    use crate::cdt::test_fixtures::allocation::AllocationBudget;
    use runmat_geometry_core::{GeometryAllocationReservation, GeometryAllocationScope};
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let request = request(order);
        let options = volume_options();
        let volume = crate::cdt::construct_delaunay_volume_mesh(
            &topology,
            &surface,
            &request.metric,
            options,
            &NeverCancelled,
        )
        .unwrap();
        let bytes = encode_delaunay_volume_mesh(
            &volume,
            &topology,
            &surface,
            &request.metric,
            options,
            &NeverCancelled,
        )
        .unwrap();
        let budget = AllocationBudget::new(512 << 20);
        let scope = GeometryAllocationScope::new(&budget);
        let wire = GeometryAllocationReservation::new(&scope, bytes.capacity() as u64).unwrap();
        let admitted = decode_delaunay_solver_topology(
            bytes,
            DelaunaySolverProjectionInput {
                exact_topology: &topology,
                exact_surface: &surface,
                volume_options: options,
                volume_metric: &request.metric,
                request: &request,
                exact_evaluation: Some(DelaunayExactEvaluation {
                    evaluator: &test_evaluator::EVALUATOR,
                    control: &scope,
                }),
            },
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        drop(wire);
        let projection = admitted
            .into_projection(runmat_meshing_core::GeometryRevisionRef {
                source_unit_evidence:
                    runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(
                        runmat_geometry_core::LengthUnit::Meter,
                        "canonical solver fixture",
                    ),
                source_digest: runmat_meshing_core::StableDigest::from_bytes([17; 32]),
                geometry_revision: 1,
                persistent_mapping_version: 1,
            })
            .unwrap();
        let retained = projection
            .retained_storage_bytes(u64::MAX, &mut || Ok(()))
            .unwrap();
        let ownership = scope
            .transfer_reservation(retained)
            .expect("all original output storage was admitted before construction");
        drop(scope);
        assert_eq!(budget.live_bytes(), retained);
        projection.validate().unwrap();
        drop(projection);
        drop(ownership);
        assert_eq!(budget.live_bytes(), 0);
        assert!(budget.peak_bytes() < 512 << 20);
    }
}

#[path = "cold_projection/scale.rs"]
mod scale;
