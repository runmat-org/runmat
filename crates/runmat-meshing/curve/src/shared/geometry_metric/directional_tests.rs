//! Real derivative/occurrence paths, independent of matrix-entry orientation.
use super::*;
use runmat_geometry_core::*;
use runmat_meshing_core::{CurveQualityTargets, MetricTensor3, SurfaceQualityTargets};

fn density(metric: MetricTensor3, v: [f64; 3]) -> f64 {
    let [x, y, z] = v;
    metric.xx * x * x
        + metric.yy * y * y
        + metric.zz * z * z
        + 2.0 * (metric.xy * x * y + metric.xz * x * z + metric.yz * y * z)
}
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
//   boundary: Portable exact cylindrical derivatives pass through occurrence transforms, principal
//   physical directions and the production source derivation. Three translated/rotated placements
//   preserve the flat extrusion direction, refine curved directions, and retain deterministic cold
//   request identity. No complete CAD surface, volume, PDE or native study acceptance is
//   established.
#[test]
fn cylindrical_curvature_refines_the_cross_section_without_refining_the_extrusion_axis() {
    for (transform, axis, radial) in [
        (
            GeometryTransform([
                1., 0., 0., 3., 0., 1., 0., -4., 0., 0., 1., 2., 0., 0., 0., 1.,
            ]),
            [0., 0., 1.],
            [1., 0., 0.],
        ),
        (
            GeometryTransform([
                0., 0., 1., 3., 1., 0., 0., -4., 0., 1., 0., 2., 0., 0., 0., 1.,
            ]),
            [1., 0., 0.],
            [0., 1., 0.],
        ),
        (
            GeometryTransform([
                1., 0., 0., 3., 0., 0.8, -0.6, -4., 0., 0.6, 0.8, 2., 0., 0., 0., 1.,
            ]),
            [0., -0.6, 0.8],
            [1., 0., 0.],
        ),
    ] {
        let (document, mut topology, mut registry) = runmat_geometry_fixtures::exact_circle();
        topology.instances[0].transform = transform;
        topology.faces[0].periodic_u = true;
        registry.pcurves[0].implementation = ExactPcurveImplementation::Portable {
            definition: ExactPcurveDefinition::Line {
                origin_uv: [0., 0.],
                direction_uv_per_parameter: [1., 0.],
                domain: ParameterRange {
                    start: 0.,
                    end: std::f64::consts::TAU,
                },
            },
        };
        registry.surfaces[0].implementation = ExactSurfaceImplementation::Portable {
            definition: ExactSurfaceDefinition::Cylinder {
                origin_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.; 3]),
                x_axis: [1., 0., 0.],
                y_axis: [0., 1., 0.],
                axis_m_per_v: [0., 0., 1.],
                radius_m: runmat_types::physical::Length::new(1.),
                domains: [
                    ParameterRange {
                        start: 0.,
                        end: std::f64::consts::TAU,
                    },
                    ParameterRange {
                        start: -1.,
                        end: 1.,
                    },
                ],
            },
        };
        let GeometryModel::ExactBRep { model } = &document.model else {
            panic!("exact fixture")
        };
        let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
        let request = MetricFieldRequest {
            entity_supports: Vec::new(),
            located_sources: Vec::new(),
            combination: runmat_meshing_core::MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(1.).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: vec![],
            cells_across_gap: None,
        };
        let curve_quality = CurveQualityTargets {
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.001),
            maximum_tangent_change_degrees: 5.,
            minimum_metric_edge_length: 0.01,
            maximum_metric_edge_length: 1.5,
        };
        let surface_quality = SurfaceQualityTargets {
            minimum_metric_edge_length: 0.1,
            minimum_metric_angle_degrees: 20.,
            maximum_physical_aspect_ratio: 10.,
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.001),
            maximum_normal_deviation_degrees: 5.,
        };
        let derived = derive_curve_geometry_metric(
            &topology,
            &evaluator,
            &Control,
            &request,
            quality_targets(curve_quality, surface_quality),
        )
        .unwrap();
        derived.validate().unwrap();
        for contribution in &derived.contributions {
            assert!(
                (density(contribution.metric, axis) - 1.).abs() < 1e-10,
                "unrelated extrusion axis inherited {:?} curvature",
                contribution.source
            );
        }
        let face = derived
            .contributions
            .iter()
            .find(|c| c.source == MetricSourceKind::Face)
            .unwrap();
        let length = curvature_target_size(1., 0.001, 5.).unwrap();
        assert!(density(face.metric, radial) >= length.powi(-2) * (1. - 1e-12));
        let cold: MetricFieldRequest =
            serde_json::from_slice(&serde_json::to_vec(&derived).unwrap()).unwrap();
        assert_eq!(cold, derived);
        assert_eq!(
            derive_curve_geometry_metric(
                &topology,
                &evaluator,
                &Control,
                &request,
                quality_targets(curve_quality, surface_quality),
            )
            .unwrap(),
            derived
        );
    }
}

/// Curve and surface targets with the default volume target (radius-edge ratio 10).
fn quality_targets(
    curve: runmat_meshing_core::CurveQualityTargets,
    surface: runmat_meshing_core::SurfaceQualityTargets,
) -> runmat_meshing_core::MeshingQualityTargets {
    runmat_meshing_core::MeshingQualityTargets {
        curve,
        surface,
        volume: runmat_meshing_core::VolumeQualityTargets {
            maximum_radius_edge_ratio: 10.0,
            minimum_scaled_jacobian: 0.001,
            maximum_metric_edge_length: 1.0,
        },
    }
}
