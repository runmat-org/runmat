use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::super::super::exact_topology_tests::{model, topology};
use super::super::tests::registry;
use super::super::*;
use super::*;

struct BudgetControl {
    cancelled: AtomicBool,
    iterations: AtomicU64,
    search_work: AtomicU64,
}

impl BudgetControl {
    fn new(iterations: u64, search_work: u64) -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            iterations: AtomicU64::new(iterations),
            search_work: AtomicU64::new(search_work),
        }
    }

    fn cancelled() -> Self {
        let control = Self::new(u64::MAX, u64::MAX);
        control.cancelled.store(true, Ordering::Relaxed);
        control
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
                    "test evaluation budget exceeded",
                )
            })
    }
}

impl GeometryEvaluationControl for BudgetControl {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::Cancelled,
                "test evaluation cancelled",
            ));
        }
        Ok(())
    }

    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(&self.iterations, count)
    }

    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        Self::consume(&self.search_work, count)
    }
}

fn generous_control() -> BudgetControl {
    BudgetControl::new(10_000_000, 10_000_000)
}

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn analytic_circle_answers_complete_curve_and_pcurve_queries() {
    let registry = registry();
    let evaluator = PortableExactEvaluatorV2::new(&registry, &topology(), &model()).unwrap();
    let control = generous_control();
    let curve_id = CurveEvaluatorIdV2::new("curve:1").unwrap();
    let at_quarter = ExactCurveEvaluatorV2::derivatives(
        &evaluator,
        &curve_id,
        std::f64::consts::FRAC_PI_2,
        &control,
    )
    .unwrap();
    assert_near(at_quarter.point_m[0], 0.0, 1.0e-14);
    assert_near(at_quarter.point_m[1], 1.0, 1.0e-14);
    assert_eq!(
        evaluator.unit_tangent(&curve_id, 0.0, &control).unwrap(),
        [0.0, 1.0, 0.0]
    );
    assert_near(
        evaluator
            .curvature_1_per_m(&curve_id, 0.3, &control)
            .unwrap(),
        1.0,
        1.0e-14,
    );
    assert_near(
        evaluator
            .arc_length_m(
                &curve_id,
                ParameterRangeV2 {
                    start: 0.0,
                    end: std::f64::consts::TAU,
                },
                1.0e-12,
                &control,
            )
            .unwrap(),
        std::f64::consts::TAU,
        1.0e-14,
    );
    let projection = evaluator
        .inverse_project(&curve_id, [2.0, 0.0, 0.0], 1.0e-10, &control)
        .unwrap();
    assert_near(projection.parameter, 0.0, 1.0e-10);
    assert_near(projection.distance_m, 1.0, 1.0e-12);

    let pcurve_id = PcurveEvaluatorIdV2::new("pcurve:1").unwrap();
    let pcurve = ExactPcurveEvaluatorV2::derivatives(
        &evaluator,
        &pcurve_id,
        std::f64::consts::FRAC_PI_2,
        &control,
    )
    .unwrap();
    assert_near(pcurve.point_uv[0], 0.0, 1.0e-14);
    assert_near(pcurve.point_uv[1], 1.0, 1.0e-14);
}

#[test]
fn rational_bspline_derivatives_length_and_projection_are_deterministic() {
    let mut topology = topology();
    topology.edges[0].is_periodic = false;
    topology.edges[0].is_closed = false;
    let mut end_vertex = topology.vertices[0].clone();
    end_vertex.id.source_topology_id = "vertex-end".into();
    end_vertex.point_m = [1.0, 0.0, 0.0];
    topology.edges[0].end_vertex_id = Some(end_vertex.id.clone());
    topology.vertices.push(end_vertex);
    let mut registry = registry();
    registry.curves[0].implementation = ExactCurveImplementationV2::Portable {
        definition: ExactCurveDefinitionV2::Nurbs {
            definition: NurbsCurve3V2 {
                degree: 2,
                knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                control_points_m: vec![[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]],
                weights: vec![1.0, 1.0, 1.0],
                domain: ParameterRangeV2 {
                    start: 0.0,
                    end: 1.0,
                },
                periodic: false,
            },
        },
    };
    registry.pcurves[0].implementation = ExactPcurveImplementationV2::Portable {
        definition: ExactPcurveDefinitionV2::Nurbs {
            definition: NurbsCurve2V2 {
                degree: 2,
                knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                control_points_uv: vec![[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
                weights: vec![1.0, 0.5, 1.0],
                domain: ParameterRangeV2 {
                    start: 0.0,
                    end: 1.0,
                },
                periodic: false,
            },
        },
    };
    let mut summary = model();
    summary.vertex_count = 2;
    let evaluator = PortableExactEvaluatorV2::new(&registry, &topology, &summary).unwrap();
    let control = generous_control();
    let id = CurveEvaluatorIdV2::new("curve:1").unwrap();
    let value = ExactCurveEvaluatorV2::derivatives(&evaluator, &id, 0.5, &control).unwrap();
    assert_eq!(value.point_m, [0.5, 0.5, 0.0]);
    assert_eq!(value.first_m, [1.0, 0.0, 0.0]);
    assert_eq!(value.second_m, [0.0, -4.0, 0.0]);
    assert_near(
        evaluator.curvature_1_per_m(&id, 0.5, &control).unwrap(),
        4.0,
        1.0e-12,
    );
    let length = evaluator
        .arc_length_m(
            &id,
            ParameterRangeV2 {
                start: 0.0,
                end: 1.0,
            },
            1.0e-10,
            &control,
        )
        .unwrap();
    assert_near(length, 1.478_942_857_5, 1.0e-9);
    let repeated_length = evaluator
        .arc_length_m(
            &id,
            ParameterRangeV2 {
                start: 0.0,
                end: 1.0,
            },
            1.0e-10,
            &generous_control(),
        )
        .unwrap();
    assert_eq!(length.to_bits(), repeated_length.to_bits());
    let projection = evaluator
        .inverse_project(&id, [0.5, 0.6, 0.0], 1.0e-10, &control)
        .unwrap();
    assert_near(projection.parameter, 0.5, 1.0e-8);
    assert_near(projection.distance_m, 0.1, 1.0e-8);
    let pcurve = ExactPcurveEvaluatorV2::derivatives(
        &evaluator,
        &PcurveEvaluatorIdV2::new("pcurve:1").unwrap(),
        0.5,
        &control,
    )
    .unwrap();
    assert_near(pcurve.point_uv[0], 0.5, 1.0e-14);
    assert_near(pcurve.point_uv[1], 1.0 / 3.0, 1.0e-14);
    assert_near(pcurve.first_uv[0], 4.0 / 3.0, 1.0e-14);

    let error = evaluator
        .arc_length_m(
            &id,
            ParameterRangeV2 {
                start: 0.0,
                end: 1.0,
            },
            1.0e-12,
            &BudgetControl::new(5, u64::MAX),
        )
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::BudgetExceeded);
}

#[test]
fn analytic_line_projection_clamps_to_its_exact_domain() {
    let mut topology = topology();
    topology.edges[0].is_periodic = false;
    topology.edges[0].is_closed = false;
    topology.vertices[0].point_m = [1.0, 0.0, 0.0];
    let mut end_vertex = topology.vertices[0].clone();
    end_vertex.id.source_topology_id = "vertex-end".into();
    end_vertex.point_m = [5.0, 0.0, 0.0];
    topology.edges[0].end_vertex_id = Some(end_vertex.id.clone());
    topology.vertices.push(end_vertex);
    let mut registry = registry();
    let domain = ParameterRangeV2 {
        start: 0.0,
        end: 2.0,
    };
    registry.curves[0].implementation = ExactCurveImplementationV2::Portable {
        definition: ExactCurveDefinitionV2::Line {
            origin_m: [1.0, 0.0, 0.0],
            direction_m_per_parameter: [2.0, 0.0, 0.0],
            domain,
        },
    };
    registry.pcurves[0].implementation = ExactPcurveImplementationV2::Portable {
        definition: ExactPcurveDefinitionV2::Line {
            origin_uv: [0.0, 0.0],
            direction_uv_per_parameter: [1.0, 0.0],
            domain,
        },
    };
    let mut summary = model();
    summary.vertex_count = 2;
    let evaluator = PortableExactEvaluatorV2::new(&registry, &topology, &summary).unwrap();
    let id = CurveEvaluatorIdV2::new("curve:1").unwrap();
    let control = generous_control();
    assert_eq!(
        ExactCurveEvaluatorV2::point(&evaluator, &id, 1.0, &control).unwrap(),
        [3.0, 0.0, 0.0]
    );
    assert_eq!(
        evaluator
            .arc_length_m(&id, domain, 1.0e-12, &control)
            .unwrap(),
        4.0
    );
    let projection = evaluator
        .inverse_project(&id, [8.0, 1.0, 0.0], 1.0e-12, &control)
        .unwrap();
    assert_eq!(projection.parameter, 2.0);
    assert_eq!(projection.point_m, [5.0, 0.0, 0.0]);
    assert_near(projection.distance_m, 10.0f64.sqrt(), 1.0e-14);
}

#[test]
fn kernel_ownership_cancellation_and_budgets_fail_explicitly() {
    let mut kernel_registry = registry();
    kernel_registry.curves[0].implementation = ExactCurveImplementationV2::Kernel {
        reference: KernelEvaluatorRefV2 {
            entity_token: "edge:1".into(),
            representation_digest: [7; 32],
        },
    };
    let evaluator = PortableExactEvaluatorV2::new(&kernel_registry, &topology(), &model()).unwrap();
    let id = CurveEvaluatorIdV2::new("curve:1").unwrap();
    let error =
        ExactCurveEvaluatorV2::point(&evaluator, &id, 0.0, &generous_control()).unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::KernelUnavailable);

    let registry = registry();
    let evaluator = PortableExactEvaluatorV2::new(&registry, &topology(), &model()).unwrap();
    let error = ExactCurveEvaluatorV2::point(&evaluator, &id, 0.0, &BudgetControl::cancelled())
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::Cancelled);
    let error = evaluator
        .inverse_project(&id, [0.0, 2.0, 0.0], 1.0e-12, &BudgetControl::new(1, 1))
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::BudgetExceeded);
    let error = evaluator
        .inverse_project(
            &id,
            [0.0, 2.0, 0.0],
            1.0e-12,
            &BudgetControl::new(u64::MAX, 0),
        )
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::BudgetExceeded);
    let error =
        ExactCurveEvaluatorV2::point(&evaluator, &id, -1.0, &generous_control()).unwrap_err();
    assert_eq!(
        error.kind,
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    );
}
