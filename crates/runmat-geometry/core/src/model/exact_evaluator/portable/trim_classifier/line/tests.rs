use super::*;
use crate::model::exact_evaluator::portable::test_support::BudgetControl;

fn line(first: [f64; 2], last: [f64; 2]) -> ExactPcurveDefinition {
    ExactPcurveDefinition::Line {
        origin_uv: first,
        direction_uv_per_parameter: [last[0] - first[0], last[1] - first[1]],
        domain: ParameterRange {
            start: 0.0,
            end: 1.0,
        },
    }
}
fn classify_polygon(
    vertices: &[[f64; 2]],
    query: [f64; 2],
    tolerance: f64,
    control: &BudgetControl,
) -> (bool, bool) {
    let mut winding = 0.0;
    for (index, &first) in vertices.iter().enumerate() {
        let last = vertices[(index + 1) % vertices.len()];
        match classify_line(
            &line(first, last),
            ParameterRange {
                start: 0.0,
                end: 1.0,
            },
            query,
            tolerance,
            1.0e-10 / vertices.len() as f64,
            control,
        )
        .unwrap()
        .expect("ordinary straight segment takes analytic path")
        {
            LineLocation::Boundary => return (false, true),
            LineLocation::Winding(angle) => winding += angle,
        }
    }
    let turns = winding / std::f64::consts::TAU;
    assert!((turns - turns.round()).abs() < 1.0e-10);
    (turns.round().abs() == 1.0, false)
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Straight portable trims use analytic projection and winding, matching the explicit
//   concave-polygon inside/outside/boundary references across the tested scales and orientations
//   with zero allocation budget. Curved trims keep the general method; this is finite
//   portable-kernel evidence, not native CAD or whole-study qualification.
#[test]
fn analytic_lines_classify_concave_boundaries_across_scale_and_orientation_without_allocation() {
    let original = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ];
    for scale in [1.0e-6, 1.0, 1.0e6] {
        for shift in [0.0, 10.0 * scale] {
            for reverse in [false, true] {
                let mut polygon =
                    original.map(|point| point.map(|coordinate| coordinate * scale + shift));
                if reverse {
                    polygon.reverse();
                }
                let control = BudgetControl::allocation_limited(0);
                for (point, expected) in [
                    ([0.25, 0.25], true),
                    ([1.5, 0.25], true),
                    ([0.25, 1.5], true),
                    ([1.5, 1.5], false),
                    ([-0.5, 0.5], false),
                ] {
                    let point = point.map(|coordinate| coordinate * scale + shift);
                    assert_eq!(
                        classify_polygon(&polygon, point, 1.0e-9 * scale, &control),
                        (expected, false)
                    );
                }
                let boundary = [shift + scale, shift];
                assert_eq!(
                    classify_polygon(&polygon, boundary, 1.0e-9 * scale, &control),
                    (false, true)
                );
            }
        }
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Near-tolerance straight-edge decisions fall back to the existing general method.
//   Search denial and cancellation remain typed failures; the fast path does not enlarge the
//   boundary tolerance or suppress resource checks.
#[test]
fn analytic_line_decisions_fall_back_near_the_tolerance_and_preserve_resource_failures() {
    let definition = line([0.0, 0.0], [1.0, 0.0]);
    let range = ParameterRange {
        start: 0.0,
        end: 1.0,
    };
    let tolerance = 1.0e-9;
    assert!(classify_line(
        &definition,
        range,
        [0.5, tolerance],
        tolerance,
        1.0e-10,
        &BudgetControl::generous()
    )
    .unwrap()
    .is_none());
    let failure = classify_line(
        &definition,
        range,
        [0.5, 0.0],
        tolerance,
        1.0e-10,
        &BudgetControl::new(100, 0),
    )
    .err()
    .unwrap();
    assert_eq!(failure.kind, GeometryEvaluationErrorKind::BudgetExceeded);
    let failure = classify_line(
        &definition,
        range,
        [0.5, 0.0],
        tolerance,
        1.0e-10,
        &BudgetControl::cancelled(),
    )
    .err()
    .unwrap();
    assert_eq!(failure.kind, GeometryEvaluationErrorKind::Cancelled);
}
