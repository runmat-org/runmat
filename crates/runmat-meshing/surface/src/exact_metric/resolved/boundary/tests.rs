use super::*;
use runmat_meshing_core::{
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind,
};
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Prepared surface support retains each edge/region tensor separately from the combined
//   face envelope, without modifying the original request. Interior support proposals pass shared
//   exact insertion and spacing; whole workflow and original CAD convergence require separate
//   evidence. This witness does not prove universal boundary feasibility or whole-task memory
//   demand.
#[test]
fn boundary_support_preserves_edge_local_tensors_instead_of_using_the_face_envelope() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let incidence = TopologyMetricIncidence::new(&topology);
    let face = &topology.faces[0];
    let entities = incidence.incident_face_entities(face);
    let edges = entities
        .iter()
        .filter(|entity| entity.kind == PersistentEntityKind::Edge)
        .take(2)
        .cloned()
        .collect::<Vec<_>>();
    let first = MetricTensor3 {
        xx: 4.005,
        yy: 4.005,
        zz: 0.01,
        xy: 3.995,
        xz: 0.,
        yz: 0.,
    };
    let second = MetricTensor3 {
        xx: 0.01,
        yy: 4.005,
        zz: 4.005,
        xy: 0.,
        xz: 0.,
        yz: 3.995,
    };
    let request = MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(10.).unwrap(),
        maximum_grading_ratio: 1.3,
        contributions: edges
            .iter()
            .zip([first, second])
            .map(|(edge, metric)| MetricContribution {
                source: MetricSourceKind::Proximity,
                scope: MetricContributionScope::Entity {
                    entity_id: edge.clone(),
                },
                metric,
            })
            .collect(),
        cells_across_gap: None,
    };
    let field = ResolvedFaceMetricField::new(&topology, &request).unwrap();
    assert_eq!(
        field
            .volume_edge_metrics(&face.id, &edges[0])
            .collect::<Vec<_>>(),
        vec![first]
    );
    assert_eq!(
        field
            .volume_edge_metrics(&face.id, &edges[1])
            .collect::<Vec<_>>(),
        vec![second]
    );
    let face_metric = field.by_face.get(&face.id).unwrap().resolved.metric;
    assert_ne!(face_metric, first);
    assert_ne!(face_metric, second);
    assert_eq!(request.contributions[0].metric, first);
    assert_eq!(request.contributions[1].metric, second);
}
