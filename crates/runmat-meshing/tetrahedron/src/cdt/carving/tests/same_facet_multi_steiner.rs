use super::*;
use crate::cdt::{
    build_delaunay_volume_provenance, evaluate_delaunay_volume_quality,
    validate_delaunay_facet_recovery, validate_delaunay_volume_provenance_sources,
    validate_delaunay_volume_quality, DelaunayVolumeProvenanceOptions,
    DelaunayVolumeQualityOptions,
};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};

#[test]
fn same_facet_multi_insertion_reaches_validated_carved_volume() {
    let seed_constraints = crate::cdt::test_fixtures::same_facet_multi_steiner_constraints();
    let initial = build_delaunay_volume_point_set(
        seed_constraints.volume_nodes(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let constraints =
        crate::cdt::test_fixtures::closed_same_facet_multi_steiner_constraints(&initial);
    let segments = recover_delaunay_segments(
        initial,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let facets = recover_delaunay_facets(
        segments,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let target_index = constraints
        .facets
        .iter()
        .position(|facet| facet.facet_id == StableDigest::from_bytes([90; 32]))
        .unwrap() as u32;
    assert_eq!(facets.steiner_insertions.len(), 2);
    assert!(facets
        .steiner_insertions
        .iter()
        .all(|insertion| insertion.constraint_index == target_index));
    validate_delaunay_facet_recovery(
        &facets,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let carving_options = DelaunayCarvingOptions::default();
    let carving =
        carve_delaunay_volume(&facets, &constraints, carving_options, &NeverCancelled).unwrap();
    assert_eq!(
        carving
            .topology
            .incidence
            .regions
            .iter()
            .map(|region| region.region_id.source_topology_id.as_str())
            .collect::<Vec<_>>(),
        vec!["lower", "upper"]
    );
    assert!(carving
        .topology
        .incidence
        .unassigned_tetrahedron_indices
        .is_empty());
    validate_delaunay_carving(
        &facets,
        &constraints,
        &carving,
        carving_options,
        &NeverCancelled,
    )
    .unwrap();

    let provenance_options = DelaunayVolumeProvenanceOptions::default();
    let provenance = build_delaunay_volume_provenance(
        &facets,
        &constraints,
        &carving,
        carving_options,
        provenance_options,
        &NeverCancelled,
    )
    .unwrap();
    assert!(provenance.facets.iter().any(|facet| {
        facet
            .entity_ids
            .iter()
            .any(|entity| entity.source_topology_id == "interface")
    }));
    validate_delaunay_volume_provenance_sources(
        &facets,
        &constraints,
        &carving,
        &provenance,
        carving_options,
        provenance_options,
        &NeverCancelled,
    )
    .unwrap();

    let metric = MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(20.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
    };
    let quality_options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 2.0,
        maximum_radius_edge_ratio: 100.0,
        minimum_metric_scaled_jacobian: 1.0e-12,
        ..DelaunayVolumeQualityOptions::default()
    };
    let quality = evaluate_delaunay_volume_quality(
        &carving.topology,
        &metric,
        &provenance,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(quality.tetrahedra.len(), carving.topology.tetrahedra.len());
    validate_delaunay_volume_quality(
        &carving.topology,
        &metric,
        &provenance,
        &quality,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
}
