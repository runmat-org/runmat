use super::*;
use crate::cdt::test_fixtures::allocation::AllocationBudget;
use runmat_geometry_core::GeometryAllocationScope;
use runmat_meshing_core::NeverCancelled;

// witness: FEA-MESH-EXECUTION-001/bounded_publication_and_cold_materialization @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Cold volume decode admits before materializing and releases on failure;
//   million-element scale is not asserted here.
#[test]
fn cold_current_and_legacy_decode_admit_before_materialization_and_release_on_failure() {
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    let metric = crate::cdt::volume_mesh::tests::metric_request();
    let options = crate::cdt::volume_mesh::tests::volume_options();
    let mesh = crate::cdt::construct_delaunay_volume_mesh(
        &topology,
        &surface,
        &metric,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let current = encode_delaunay_volume_mesh(
        &mesh,
        &topology,
        &surface,
        &metric,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let legacy = encode_delaunay_volume_mesh_with_schema_version(&mesh, 3).unwrap();
    for bytes in [&current, &legacy] {
        let baseline = AllocationBudget::new(u64::MAX);
        {
            let scope = GeometryAllocationScope::new(&baseline);
            let decoded = decode(
                bytes,
                &topology,
                &surface,
                &metric,
                options,
                &NeverCancelled,
                Some(&scope),
            )
            .unwrap();
            assert_eq!(decoded, mesh);
            assert!(baseline.live_bytes() > 0);
            drop(decoded);
        }
        assert_eq!(baseline.live_bytes(), 0);
        let exact = baseline.peak_bytes();
        for limit in [0, exact - 1, exact] {
            let budget = AllocationBudget::new(limit);
            {
                let scope = GeometryAllocationScope::new(&budget);
                let decoded = decode(
                    bytes,
                    &topology,
                    &surface,
                    &metric,
                    options,
                    &NeverCancelled,
                    Some(&scope),
                );
                if limit == exact {
                    assert_eq!(decoded.unwrap(), mesh);
                } else {
                    assert_eq!(
                        decoded.unwrap_err().kind,
                        DelaunayVolumeMeshCodecErrorKind::ResourceLimit
                    );
                }
            }
            assert_eq!(budget.live_bytes(), 0);
        }
        let budget = AllocationBudget::new(exact);
        let mut corrupt = bytes.clone();
        corrupt.push(0);
        {
            let scope = GeometryAllocationScope::new(&budget);
            assert!(decode(
                &corrupt,
                &topology,
                &surface,
                &metric,
                options,
                &NeverCancelled,
                Some(&scope)
            )
            .is_err());
        }
        assert_eq!(budget.live_bytes(), 0);
    }
}
