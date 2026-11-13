use super::*;
use crate::cdt::test_fixtures::allocation::AllocationBudget;
use runmat_geometry_core::GeometryAllocationScope;
use runmat_meshing_core::NeverCancelled;

#[test]
fn cold_source_transfer_releases_dropped_evidence_and_retains_actual_nested_capacities() {
    let (geometry, surface) = crate::cdt::constraints::tests::tetrahedron();
    let metric = crate::cdt::test_fixtures::volume::metric_request();
    let options = crate::cdt::test_fixtures::volume::volume_options();
    let mesh = crate::cdt::construct_delaunay_volume_mesh(
        &geometry,
        &surface,
        &metric,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let bytes = crate::cdt::encode_delaunay_volume_mesh(
        &mesh,
        &geometry,
        &surface,
        &metric,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let budget = AllocationBudget::new(u64::MAX);
    let scope = GeometryAllocationScope::new(&budget);
    let decoded = crate::cdt::volume_mesh::decode_delaunay_volume_mesh_admitted(
        &bytes,
        &geometry,
        &surface,
        &metric,
        options,
        &NeverCancelled,
        Some(&scope),
    )
    .unwrap();
    let mut source = super::super::compact_source(decoded);
    let retained = retained_bytes(&source.topology, &source.provenance, &budget).unwrap();
    let before = budget.live_bytes();
    assert!(before > retained);
    source.storage = Some(scope.transfer_reservation(retained).unwrap());
    drop(scope);
    assert_eq!(budget.live_bytes(), retained);
    assert_eq!(source.topology, mesh.topology);
    assert_eq!(source.provenance, mesh.provenance);
    drop(source);
    assert_eq!(budget.live_bytes(), 0);
    assert_eq!(budget.peak_bytes(), before);
}
