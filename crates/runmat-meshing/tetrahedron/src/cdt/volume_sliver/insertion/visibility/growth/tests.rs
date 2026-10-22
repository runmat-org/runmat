use super::*;
use runmat_meshing_core::NeverCancelled;
use runmat_meshing_size::metric::MetricTensor3;
mod reference;
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Incremental visibility and quality growth retain the independent scan reference's
//   exact cavity under isotropic and off-diagonal metrics. Each fixed-point face metric is resolved
//   once; the scan reference performs repeated queries. Canonical decision order, protected
//   boundary and shared smoothing checks remain separate. This is an algorithmic work witness, not
//   whole-model throughput or study acceptance.
#[test]
fn incremental_visibility_matches_scan_reference_and_resolves_each_face_metric_once() {
    let (topology, provenance) = crate::cdt::test_fixtures::coupled_quality::fixture(false, 0);
    let protected = provenance
        .facets
        .iter()
        .map(|face| face.node_identities)
        .collect();
    let source_index = 0;
    let neighbor_index = topology.tetrahedra[0]
        .neighbors
        .into_iter()
        .flatten()
        .next()
        .unwrap();
    for metric in [
        MetricTensor3::isotropic_length_m(1.).unwrap(),
        MetricTensor3 {
            xx: 4.,
            yy: 9.,
            zz: 16.,
            xy: 0.5,
            xz: 0.25,
            yz: 0.75,
        },
    ] {
        let calls = std::cell::RefCell::new(BTreeMap::new());
        let metrics = |face| {
            *calls.borrow_mut().entry(face).or_insert(0_usize) += 1;
            Ok(metric)
        };
        let node =
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([242; 32]), [0.2, 0.13, 0.11]);
        let options = DelaunayVolumeSliverOptions::default();
        let request = |metrics| VisibilityCavityRequest {
            node,
            source_index,
            neighbor_index,
            relocated_vertices: &[],
            protected_faces: &protected,
            quality_growth: Some((metrics, 1.)),
            options,
            cancellation: &NeverCancelled,
        };
        let actual = grow_visibility_cavity(&topology, request(&metrics))
            .unwrap()
            .unwrap();
        assert!(!calls.borrow().is_empty());
        assert!(calls.borrow().values().all(|count| *count == 1));
        let actual_queries = calls.borrow().values().sum::<usize>();
        calls.borrow_mut().clear();
        let expected = reference::grow(&topology, request(&metrics))
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.0.len(), topology.tetrahedra.len());
        assert!(calls.borrow().values().sum::<usize>() > actual_queries);
    }
}
