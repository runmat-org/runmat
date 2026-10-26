use super::*;
use runmat_geometry_core::*;
use runmat_meshing_core::{
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricSourceKind,
    MetricTensor3,
};
use std::collections::BTreeSet;
struct Control;
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Every boundary edge satisfies the shared incident face metric trace under three
//   occurrence transforms, with stable repeated queries and source provenance. This component
//   fixture does not establish complete native surface recovery or original study acceptance.
#[test]
fn curve_boundary_trace_satisfies_every_adjacent_fine_face_before_surface_execution() {
    for transform in [
        GeometryTransform([
            1., 0., 0., 3., 0., 1., 0., -4., 0., 0., 1., 2., 0., 0., 0., 1.,
        ]),
        GeometryTransform([
            0., 0., 1., 3., 1., 0., 0., -4., 0., 1., 0., 2., 0., 0., 0., 1.,
        ]),
        GeometryTransform([
            1., 0., 0., 3., 0., 0.8, -0.6, -4., 0., 0.6, 0.8, 2., 0., 0., 0., 1.,
        ]),
    ] {
        let (document, mut topology, registry) = runmat_geometry_fixtures::exact_tetrahedron();
        topology.instances[0].transform = transform;
        let source = topology.edges[0].id.clone();
        let faces = topology
            .coedges
            .iter()
            .filter(|c| c.edge_id == source)
            .map(|c| &c.face_id)
            .collect::<BTreeSet<_>>();
        let required = MetricTensor3::isotropic_length_m(0.05).unwrap();
        let request = MetricFieldRequest {
            entity_supports: Vec::new(),
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(1.).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: vec![MetricContribution {
                source: MetricSourceKind::Feature,
                scope: MetricContributionScope::Entity {
                    entity_id: source.clone(),
                },
                metric: required,
            }],
            cells_across_gap: None,
        };
        let field = ResolvedCurveMetricField::new(&topology, &request).unwrap();
        let GeometryModel::ExactBRep { model } = &document.model else {
            panic!("exact fixture")
        };
        let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
        let mut checked = 0;
        for edge in &topology.edges {
            if !topology
                .coedges
                .iter()
                .any(|c| c.edge_id == edge.id && faces.contains(&c.face_id))
            {
                continue;
            }
            let range =
                ExactCurveEvaluator::parameter_range(&evaluator, &edge.curve_evaluator_id).unwrap();
            let parameter = range.parameter_at_fraction(0.5).unwrap();
            let raw = ExactCurveEvaluator::derivatives(
                &evaluator,
                &edge.curve_evaluator_id,
                parameter,
                &Control,
            )
            .unwrap();
            let placement = topology.world_transform_for(&edge.id).unwrap();
            let tangent = placement.transform_vector(raw.first_m);
            let norm = tangent[0].hypot(tangent[1]).hypot(tangent[2]);
            let unit = tangent.map(|x| x / norm);
            let query = CurveMetricQuery {
                edge_id: &edge.id,
                parameter,
                point_m: placement.transform_point(raw.point_m),
                unit_tangent: unit,
            };
            let evaluated = field.evaluate(query).unwrap();
            let [x, y, z] = unit;
            let m = evaluated.metric;
            let density = m.xx * x * x
                + m.yy * y * y
                + m.zz * z * z
                + 2. * (m.xy * x * y + m.xz * x * z + m.yz * y * z);
            assert!(
                density >= required.xx * (1. - 1e-12),
                "curve {:?} is coarser than its adjacent constant fine face",
                edge.id
            );
            assert_eq!(
                field.evaluate(query).unwrap(),
                evaluated,
                "querying cannot accumulate new density"
            );
            assert!(evaluated
                .active_sources
                .contains(&MetricSourceKind::Feature));
            checked += 1;
        }
        assert!(checked > 2);
    }
}
