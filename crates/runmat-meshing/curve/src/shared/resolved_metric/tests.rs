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
            let tangent = placement
                .transform_vector(raw.first_m.with_canonical_storage(|values, _axes| *values));
            let norm = tangent[0].hypot(tangent[1]).hypot(tangent[2]);
            let unit = tangent.map(|x| x / norm);
            let query = CurveMetricQuery {
                edge_id: &edge.id,
                parameter,
                point_m: placement.transform_point(*raw.point_m.canonical_values()),
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

fn edge_point(
    evaluator: &PortableExactEvaluator<'_>,
    topology: &ExactBRepTopology,
    edge: &ExactEdge,
    fraction: f64,
) -> (f64, [f64; 3], [f64; 3]) {
    let range = ExactCurveEvaluator::parameter_range(evaluator, &edge.curve_evaluator_id).unwrap();
    let parameter = range.parameter_at_fraction(fraction).unwrap();
    let raw =
        ExactCurveEvaluator::derivatives(evaluator, &edge.curve_evaluator_id, parameter, &Control)
            .unwrap();
    let placement = topology.world_transform_for(&edge.id).unwrap();
    let tangent =
        placement.transform_vector(raw.first_m.with_canonical_storage(|values, _axes| *values));
    let norm = tangent[0].hypot(tangent[1]).hypot(tangent[2]);
    (
        parameter,
        placement.transform_point(*raw.point_m.canonical_values()),
        tangent.map(|x| x / norm),
    )
}

/// A fine edge-local source sizes the edges that meet its edge only near the feature: at the
/// shared vertex they take its size, and away from it the size grows with distance at the
/// grading ratio. Grading by adjacency alone would cap each neighbor's whole length.
#[test]
fn an_edge_local_source_refines_its_neighbors_only_near_its_support() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_tetrahedron();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("exact fixture")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let source = &topology.edges[0];
    let fine = 0.01;
    let request = MetricFieldRequest {
        entity_supports: vec![runmat_meshing_size::metric::MetricEntitySupport {
            entity_id: source.id.clone(),
            points_m: (0..9)
                .map(|sample| edge_point(&evaluator, &topology, source, f64::from(sample) / 8.).1)
                .collect(),
        }],
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(1.).unwrap(),
        maximum_grading_ratio: 1.3,
        contributions: vec![MetricContribution {
            source: MetricSourceKind::Proximity,
            scope: MetricContributionScope::Entity {
                entity_id: source.id.clone(),
            },
            metric: MetricTensor3::isotropic_length_m(fine).unwrap(),
        }],
        cells_across_gap: None,
    };
    let field = ResolvedCurveMetricField::new(&topology, &request).unwrap();
    let size = |edge: &ExactEdge, fraction: f64| {
        let (parameter, point_m, unit_tangent) = edge_point(&evaluator, &topology, edge, fraction);
        let metric = field
            .evaluate(CurveMetricQuery {
                edge_id: &edge.id,
                parameter,
                point_m,
                unit_tangent,
            })
            .unwrap()
            .metric;
        let [x, y, z] = unit_tangent;
        let density = metric.xx * x * x
            + metric.yy * y * y
            + metric.zz * z * z
            + 2. * (metric.xy * x * y + metric.xz * x * z + metric.yz * y * z);
        (density.sqrt().recip(), point_m)
    };
    for fraction in [0.0, 0.5, 1.0] {
        assert!(size(source, fraction).0 <= fine * (1. + 1e-9));
    }
    let shares_vertex = |edge: &ExactEdge| {
        edge.id != source.id
            && [&edge.start_vertex_id, &edge.end_vertex_id]
                .into_iter()
                .flatten()
                .any(|vertex| {
                    source.start_vertex_id.as_ref() == Some(vertex)
                        || source.end_vertex_id.as_ref() == Some(vertex)
                })
    };
    let mut checked = 0;
    for neighbor in topology.edges.iter().filter(|edge| shares_vertex(edge)) {
        let shared_at_start = [&source.start_vertex_id, &source.end_vertex_id]
            .into_iter()
            .flatten()
            .any(|vertex| neighbor.start_vertex_id.as_ref() == Some(vertex));
        let (near, far) = if shared_at_start { (0.0, 1.0) } else { (1.0, 0.0) };
        assert!(size(neighbor, near).0 <= fine * (1. + 1e-6));
        let (far_size, far_point) = size(neighbor, far);
        let distance = (0..=64)
            .map(|sample| {
                let point = edge_point(&evaluator, &topology, source, f64::from(sample) / 64.).1;
                (0..3)
                    .map(|axis| (point[axis] - far_point[axis]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(distance > 10. * fine, "fixture far vertex is near the source");
        assert!(
            far_size >= fine + 0.3 * (distance - fine) * 0.9,
            "{:?} is refined at {far_size} far from the source",
            neighbor.id
        );
        checked += 1;
    }
    assert!(checked >= 2);
}
