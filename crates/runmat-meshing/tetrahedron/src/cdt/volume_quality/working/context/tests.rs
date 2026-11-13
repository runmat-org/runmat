use super::*;
use crate::cdt::volume_quality::tests::{metric_contexts, metric_request, region, topology};
#[test]
fn raw_context_cache_is_bounded_and_preserves_uncached_resolution() {
    let topology = topology();
    let provenance = metric_contexts(&topology);
    let request = metric_request();
    let prepared = PreparedLocalQuality::prepare(
        &request,
        &provenance,
        DelaunayVolumeQualityOptions::default(),
    )
    .unwrap();
    let field = ResolvedMetricField::new(&request).unwrap();
    for index in 0..1024 {
        let entities = vec![region(&format!("independent-{index}"))];
        assert_eq!(
            prepared.resolve_raw(&entities).unwrap(),
            field.resolve(&entities.iter().cloned().collect()).unwrap()
        );
        assert!(prepared.raw_cache.borrow().len() <= 128);
    }
    let entities = (0..65)
        .map(|index| region(&format!("large-{index}")))
        .collect::<Vec<_>>();
    let count = prepared.raw_cache.borrow().len();
    assert_eq!(
        prepared.resolve_raw(&entities).unwrap(),
        field.resolve(&entities.iter().cloned().collect()).unwrap()
    );
    assert_eq!(prepared.raw_cache.borrow().len(), count);
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Prospective Steiner children resolve their actual surviving boundary entities through
//   the same metric authority as final evaluation. Losing a constrained face restores the declared
//   global metric rather than retaining a parent constraint. This scope check does not establish
//   mesh convergence.
#[test]
fn steiner_children_do_not_inherit_a_disappearing_boundary_scope() {
    let (topology, provenance) = crate::cdt::test_fixtures::coupled_quality::fixture(false, 0);
    let request = crate::cdt::test_fixtures::coupled_quality::metric(&provenance, true);
    let entity = match &request.contributions[0].scope {
        runmat_meshing_size::metric::MetricContributionScope::Entity { entity_id } => entity_id,
        _ => panic!("fixture must use one scoped face"),
    };
    let facet = provenance
        .facets
        .iter()
        .find(|facet| facet.entity_ids.contains(entity))
        .unwrap();
    let prepared =
        PreparedLocalQuality::prepare(&request, &provenance, Default::default()).unwrap();
    let region = topology.tetrahedra[0].region_id.as_ref().unwrap();
    assert_eq!(
        prepared
            .insertion_metric(facet.node_identities, region)
            .unwrap(),
        request.contributions[0].metric
    );
    let mut child = facet.node_identities;
    child[2] = StableDigest::from_bytes([240; 32]);
    assert_eq!(
        prepared.insertion_metric(child, region).unwrap(),
        request.global_metric,
        "a new child without the constrained face must resolve its own scope"
    );
}
