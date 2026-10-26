//! Cold projection scale is separate from CAD meshing and PDE qualification.
use super::*;
use crate::cdt::test_fixtures::allocation::AllocationBudget;
use runmat_geometry_core::{GeometryAllocationReservation, GeometryAllocationScope};
#[path = "scale/fixture.rs"]
mod fixture;

#[test]
fn analytic_refinement_projects_both_orders_with_original_volume_and_lattice() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        projection_case(1, order);
    }
}
#[test]
#[ignore = "explicit complete cold-projection scale gate under unchanged 512 MiB component ceiling"]
fn representative_cold_projection_fits_default_component_admission_for_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        projection_case(6, order);
    }
}
fn projection_case(levels: usize, order: ElementOrder) {
    let (cad, surface, volume) = fixture::refined(levels);
    let mut request = request(order);
    request.limits.maximum_nodes = 1_000_000;
    request.limits.maximum_elements = 1_000_000;
    let options = volume_options();
    let bytes = encode_delaunay_volume_mesh(
        &volume,
        &cad,
        &surface,
        &request.metric,
        options,
        &NeverCancelled,
    )
    .unwrap();
    drop(volume);
    // Original borrowed CAD and accepted surface are separate input owners. This
    // gate measures cold wire + typed volume + constructor scratch/retained output.
    let budget = AllocationBudget::new(512 << 20);
    let scope = GeometryAllocationScope::new(&budget);
    let wire = GeometryAllocationReservation::new(&scope, bytes.capacity() as u64).unwrap();
    let topology = decode_delaunay_solver_topology(
        bytes,
        DelaunaySolverProjectionInput {
            exact_topology: &cad,
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
    let projection = topology
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
    let ownership = scope.transfer_reservation(retained).unwrap();
    drop(scope);
    assert_eq!(budget.live_bytes(), retained);
    projection.validate().unwrap();
    assert_eq!(
        projection.topology.volume_elements.len(),
        8usize.pow(levels as u32)
    );
    let divisions = (1usize << levels) * if order == ElementOrder::Tet10 { 2 } else { 1 };
    assert_eq!(
        projection.topology.nodes.len(),
        (divisions + 1) * (divisions + 2) * (divisions + 3) / 6
    );
    let volume: f64 = projection
        .topology
        .volume_elements
        .iter()
        .map(|element| {
            let points = std::array::from_fn(|index| {
                projection.topology.nodes[(element.node_ids[index] - 1) as usize]
                    .coordinates_m
                    .canonical_value()
            });
            runmat_meshing_core::quality::predicate::tetrahedron_signed_volume(points).abs()
        })
        .sum();
    assert!(
        (volume - 1.0 / 6.0).abs() < 1.0e-11,
        "analytic volume changed: {volume}"
    );
    eprintln!("cold-projection level={levels} order={order:?} nodes={} cells={} retained={retained} requested_peak={} component_ceiling={} borrowed_CAD_and_surface_excluded=true",
        projection.topology.nodes.len(),projection.topology.volume_elements.len(),budget.peak_bytes(),512u64<<20);
    drop(projection);
    drop(ownership);
    assert_eq!(budget.live_bytes(), 0);
    assert!(budget.peak_bytes() <= 512 << 20);
}
