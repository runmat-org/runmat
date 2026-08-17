use std::sync::atomic::{AtomicU64, Ordering};

use super::super::super::exact_topology_tests::{model, topology};
use super::super::tests::registry;
use super::super::*;
use super::vector::{cross, normalize};
use super::*;

struct Control {
    iterations: AtomicU64,
    search_work: AtomicU64,
    allocation_bytes: AtomicU64,
}

impl Control {
    fn generous() -> Self {
        Self {
            iterations: AtomicU64::new(10_000_000),
            search_work: AtomicU64::new(10_000_000),
            allocation_bytes: AtomicU64::new(10_000_000),
        }
    }

    fn limited(iterations: u64, search_work: u64, allocation_bytes: u64) -> Self {
        Self {
            iterations: AtomicU64::new(iterations),
            search_work: AtomicU64::new(search_work),
            allocation_bytes: AtomicU64::new(allocation_bytes),
        }
    }

    fn consume(remaining: &AtomicU64, count: u64) -> Result<(), GeometryEvaluationError> {
        remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_sub(count)
            })
            .map(|_| ())
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::BudgetExceeded,
                    "test surface evaluation budget exceeded",
                )
            })
    }
}

impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(&self.iterations, count)
    }

    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(&self.search_work, count)
    }

    fn consume_allocation_bytes(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(&self.allocation_bytes, count)
    }
}

fn range(start: f64, end: f64) -> ParameterRange {
    ParameterRange { start, end }
}

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn plane_answers_complete_queries_and_clamps_projection_to_bounds() {
    let registry = registry();
    let evaluator = PortableExactEvaluator::new(&registry, &topology(), &model()).unwrap();
    let id = SurfaceEvaluatorId::new("surface:1").unwrap();
    let control = Control::generous();
    assert_eq!(
        evaluator.parameter_bounds(&id).unwrap(),
        [range(-2.0, 2.0), range(-2.0, 2.0)]
    );
    assert_eq!(evaluator.periodicity(&id).unwrap(), [None, None]);
    let derivatives =
        ExactSurfaceEvaluator::derivatives(&evaluator, &id, [1.0, -1.0], &control).unwrap();
    assert_eq!(derivatives.point_m, [1.0, -1.0, 0.0]);
    assert_eq!(derivatives.du_m, [1.0, 0.0, 0.0]);
    assert_eq!(derivatives.dv_m, [0.0, 1.0, 0.0]);
    assert_eq!(
        evaluator.unit_normal(&id, [0.0, 0.0], &control).unwrap(),
        [0.0, 0.0, 1.0]
    );
    let curvature = evaluator
        .principal_curvature(&id, [0.0, 0.0], &control)
        .unwrap();
    assert_eq!(curvature.minimum_1_per_m, 0.0);
    assert_eq!(curvature.maximum_1_per_m, 0.0);
    let projection = evaluator
        .closest_point(&id, [3.0, -3.0, 2.0], 1.0e-10, &control)
        .unwrap();
    assert_eq!(projection.uv, [2.0, -2.0]);
    assert_eq!(projection.point_m, [2.0, -2.0, 0.0]);
    assert_near(projection.distance_m, 6.0f64.sqrt(), 1.0e-12);
}

#[test]
fn analytic_surfaces_produce_exact_partials_normals_and_curvature() {
    let mut topology = topology();
    topology.faces[0].periodic_u = true;
    let mut registry = registry();
    let domains = [
        range(-std::f64::consts::PI, std::f64::consts::PI),
        range(-1.0, 1.0),
    ];
    registry.surfaces[0].implementation = ExactSurfaceImplementation::Portable {
        definition: ExactSurfaceDefinition::Cylinder {
            origin_m: [0.0; 3],
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0],
            axis_m_per_v: [0.0, 0.0, 1.0],
            radius_m: 2.0,
            domains,
        },
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, &model()).unwrap();
    let id = SurfaceEvaluatorId::new("surface:1").unwrap();
    let control = Control::generous();
    assert_eq!(
        evaluator.periodicity(&id).unwrap(),
        [Some(std::f64::consts::TAU), None]
    );
    let cylinder =
        ExactSurfaceEvaluator::derivatives(&evaluator, &id, [0.0, 0.5], &control).unwrap();
    assert_eq!(cylinder.point_m, [2.0, 0.0, 0.5]);
    assert_eq!(cylinder.du_m, [0.0, 2.0, 0.0]);
    assert_eq!(cylinder.dvv_m, [0.0; 3]);
    assert_eq!(
        evaluator.unit_normal(&id, [0.0, 0.5], &control).unwrap(),
        [1.0, 0.0, 0.0]
    );
    let curvature = evaluator
        .principal_curvature(&id, [0.0, 0.5], &control)
        .unwrap();
    assert_near(curvature.minimum_1_per_m, -0.5, 1.0e-14);
    assert_eq!(curvature.maximum_1_per_m, 0.0);

    registry.surfaces[0].implementation = ExactSurfaceImplementation::Portable {
        definition: ExactSurfaceDefinition::Cone {
            apex_m: [0.0; 3],
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            semi_angle_rad: std::f64::consts::FRAC_PI_4,
            domains: [
                range(-std::f64::consts::PI, std::f64::consts::PI),
                range(0.5, 3.0),
            ],
        },
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, &model()).unwrap();
    let cone = ExactSurfaceEvaluator::derivatives(&evaluator, &id, [0.0, 2.0], &control).unwrap();
    assert_near(cone.point_m[0], 2.0f64.sqrt(), 1.0e-14);
    assert_near(cone.point_m[2], 2.0f64.sqrt(), 1.0e-14);
    assert_near(cone.du_m[1], 2.0f64.sqrt(), 1.0e-14);
    assert_near(cone.dv_m[0], std::f64::consts::FRAC_1_SQRT_2, 1.0e-14);
    assert_near(cone.dv_m[2], std::f64::consts::FRAC_1_SQRT_2, 1.0e-14);
    assert_near(cone.duv_m[1], std::f64::consts::FRAC_1_SQRT_2, 1.0e-14);

    registry.surfaces[0].implementation = ExactSurfaceImplementation::Portable {
        definition: ExactSurfaceDefinition::Sphere {
            center_m: [0.0; 3],
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0],
            z_axis: [0.0, 0.0, 1.0],
            radius_m: 2.0,
            domains: [
                range(-std::f64::consts::PI, std::f64::consts::PI),
                range(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
            ],
        },
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, &model()).unwrap();
    assert_eq!(
        ExactSurfaceEvaluator::point(&evaluator, &id, [0.0, 0.0], &control).unwrap(),
        [2.0, 0.0, 0.0]
    );
    let curvature = evaluator
        .principal_curvature(&id, [0.0, 0.0], &control)
        .unwrap();
    assert_near(curvature.minimum_1_per_m, -0.5, 1.0e-14);
    assert_near(curvature.maximum_1_per_m, -0.5, 1.0e-14);
    let projection = evaluator
        .closest_point(&id, [3.0, 0.0, 0.0], 1.0e-10, &control)
        .unwrap();
    assert_near(projection.distance_m, 1.0, 1.0e-12);

    topology.faces[0].periodic_v = true;
    registry.surfaces[0].implementation = ExactSurfaceImplementation::Portable {
        definition: ExactSurfaceDefinition::Torus {
            center_m: [0.0; 3],
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0],
            z_axis: [0.0, 0.0, 1.0],
            major_radius_m: 3.0,
            minor_radius_m: 1.0,
            domains: [
                range(-std::f64::consts::PI, std::f64::consts::PI),
                range(-std::f64::consts::PI, std::f64::consts::PI),
            ],
        },
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, &model()).unwrap();
    assert_eq!(
        ExactSurfaceEvaluator::point(&evaluator, &id, [0.0, 0.0], &control).unwrap(),
        [4.0, 0.0, 0.0]
    );
    let curvature = evaluator
        .principal_curvature(&id, [0.0, 0.0], &control)
        .unwrap();
    assert_near(curvature.minimum_1_per_m, -1.0, 1.0e-14);
    assert_near(curvature.maximum_1_per_m, -0.25, 1.0e-14);
}

#[test]
fn tensor_product_rational_surface_derivatives_are_deterministic() {
    let mut registry = registry();
    registry.surfaces[0].implementation = ExactSurfaceImplementation::Portable {
        definition: ExactSurfaceDefinition::Nurbs {
            definition: NurbsSurface3 {
                u_degree: 1,
                v_degree: 1,
                u_knots: vec![0.0, 0.0, 1.0, 1.0],
                v_knots: vec![0.0, 0.0, 1.0, 1.0],
                u_control_count: 2,
                v_control_count: 2,
                control_points_m: vec![
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                weights: vec![1.0, 1.0, 1.0, 2.0],
                domains: [range(0.0, 1.0), range(0.0, 1.0)],
                periodic_u: false,
                periodic_v: false,
            },
        },
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology(), &model()).unwrap();
    let id = SurfaceEvaluatorId::new("surface:1").unwrap();
    let value =
        ExactSurfaceEvaluator::derivatives(&evaluator, &id, [0.5, 0.5], &Control::generous())
            .unwrap();
    assert_eq!(value.point_m, [0.6, 0.6, 0.4]);
    assert_near(value.du_m[0], 0.96, 1.0e-14);
    assert_near(value.du_m[1], 0.16, 1.0e-14);
    assert_near(value.du_m[2], 0.64, 1.0e-14);
    assert_near(value.dv_m[0], 0.16, 1.0e-14);
    assert_near(value.dv_m[1], 0.96, 1.0e-14);
    assert_near(value.dv_m[2], 0.64, 1.0e-14);
    let repeated =
        ExactSurfaceEvaluator::derivatives(&evaluator, &id, [0.5, 0.5], &Control::generous())
            .unwrap();
    assert_eq!(value, repeated);
    let normal = normalize(&cross(&value.du_m, &value.dv_m)).unwrap();
    let query = std::array::from_fn(|axis| value.point_m[axis] + 0.1 * normal[axis]);
    let projection = evaluator
        .closest_point(&id, query, 1.0e-10, &Control::generous())
        .unwrap();
    assert_near(projection.uv[0], 0.5, 1.0e-10);
    assert_near(projection.uv[1], 0.5, 1.0e-10);
    assert_near(projection.distance_m, 0.1, 1.0e-10);
    let error = ExactSurfaceEvaluator::derivatives(
        &evaluator,
        &id,
        [0.5, 0.5],
        &Control::limited(u64::MAX, u64::MAX, 0),
    )
    .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::BudgetExceeded);
}

#[test]
fn surface_failures_preserve_domain_kernel_and_budget_categories() {
    let mut kernel_registry = registry();
    kernel_registry.surfaces[0].implementation = ExactSurfaceImplementation::Kernel {
        reference: KernelEvaluatorRef {
            entity_token: "face:1".into(),
            representation_digest: [7; 32],
        },
    };
    let evaluator = PortableExactEvaluator::new(&kernel_registry, &topology(), &model()).unwrap();
    let id = SurfaceEvaluatorId::new("surface:1").unwrap();
    let error = ExactSurfaceEvaluator::point(&evaluator, &id, [0.0, 0.0], &Control::generous())
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::KernelUnavailable);

    let registry = registry();
    let evaluator = PortableExactEvaluator::new(&registry, &topology(), &model()).unwrap();
    let error = ExactSurfaceEvaluator::point(&evaluator, &id, [3.0, 0.0], &Control::generous())
        .unwrap_err();
    assert_eq!(
        error.kind,
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    );
    let error = evaluator
        .closest_point(
            &id,
            [0.0, 0.0, 1.0],
            1.0e-10,
            &Control::limited(u64::MAX, 0, u64::MAX),
        )
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::BudgetExceeded);
}
