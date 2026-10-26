//! Exercise the actual borrowed SQP/KKT mechanism without a user callback.
use super::super::{
    domain::BoxGeometry,
    merit::{self, Criterion, Phase, Proposals, Selection},
    oracle::{Controls, ModelView},
    rbf::{self, Data, Fit, Generation, Training, TrainingOwner},
    refinement::{self, Session},
    schedule, IDENTITY,
};
use crate::builtins::math::optim::{
    common::{global_time::TimeLimit, solver_timing::SolverTiming, ScalarCooperation},
    fmincon::internal::Limits,
    intlinprog::feasibility::{Domain, Input},
};
use crate::execution::RootedValue;
use clarabel::algebra::CscMatrix;
use runmat_value::Value;

struct Samples;
impl Proposals for Samples {
    fn len(&self) -> usize {
        2
    }
    fn dimensions(&self) -> usize {
        2
    }
    fn coordinate(&self, row: usize, axis: usize) -> f64 {
        let x = [0.2, 0.8][row];
        if axis == 0 {
            x
        } else {
            1.0 - x
        }
    }
}
struct Plane<'a>(Data<'a>);
impl Training for Plane<'_> {
    fn valid(&self) -> bool {
        Training::valid(&self.0)
    }
    fn owner(&self) -> &TrainingOwner {
        self.0.owner
    }
    fn generation(&self) -> Generation {
        self.0.generation
    }
    fn rows(&self) -> &[usize] {
        self.0.rows
    }
    fn dimensions(&self) -> usize {
        self.0.dimensions
    }
    fn lanes(&self) -> usize {
        self.0.lanes
    }
    fn affine_axes(&self) -> Option<&[usize]> {
        Some(&[0])
    }
    fn coordinate(&self, row: usize, axis: usize) -> f64 {
        Training::coordinate(&self.0, row, axis)
    }
    fn measurement(&self, row: usize, lane: usize) -> f64 {
        Training::measurement(&self.0, row, lane)
    }
}

#[test]
fn borrowed_original_linear_rows_refine_merit_without_invoking_expensive_source() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let clock = SolverTiming::start(Some(&runtime)).unwrap();
        let mut domain = Domain::new(
            Input {
                matrix: CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]),
                rhs: vec![1.0],
                equalities: 1,
                lower: vec![0.0, 0.0],
                upper: vec![1.0, 1.0],
                integers: vec![],
            },
            clock.clone(),
            TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        let shape = BoxGeometry::new(&domain, &mut cooperation).await.unwrap();
        let data = Plane(Data {
            owner: &TrainingOwner::new(),
            points: &[0.0, 1.0, 0.5, 0.5, 1.0, 0.0],
            values: &[0.09, 0.04, 0.49],
            rows: &[0, 1, 2],
            dimensions: 2,
            lanes: 1,
            generation: Generation(1),
        });
        let Fit::Poised(model) = rbf::fit(&data, &mut cooperation).await.unwrap() else {
            panic!("one-dimensional three-site model");
        };
        let Selection::Point(selected) = merit::select(
            &model,
            &data,
            &data,
            &Samples,
            merit::SelectionControls {
                phase: Phase::Objective {
                    objective_lane: 0,
                    inequality_start: 1,
                },
                weight: 0.95,
                constraint_tolerance: 1e-3,
                min_distance: 1e-8,
            },
            &mut cooperation,
        )
        .await
        .unwrap() else {
            panic!("unevaluated merit samples");
        };
        assert!(matches!(selected.criterion, Criterion::Objective(_)));
        // Any accidental selected-source call fails, rather than returning a
        // fake value that could conceal objective replay inside refinement.
        let source =
            RootedValue::in_optional_context(Value::FunctionHandle("error".into()), Some(&runtime))
                .unwrap();
        let result = {
            let mut session = Session::new(
                &domain,
                &source,
                ModelView {
                    model: &model,
                    training: &data,
                    evaluated: &data,
                    domain: shape.view(&domain).unwrap(),
                    criterion: selected.criterion,
                    inequality_start: 1,
                },
                Controls {
                    tolerance: 1e-3,
                    clock: &clock,
                    time_limit: TimeLimit::Unbounded,
                },
                Limits {
                    iterations: 30,
                    evaluations: 256,
                    feasibility: 1e-8,
                    optimality: 1e-5,
                    step: 1e-10,
                },
                &mut cooperation,
            )
            .await
            .unwrap();
            let anchor = [0.8, 0.2];
            refinement::run(
                schedule::Refinement::MultiStart,
                shape.view(&domain).unwrap(),
                &anchor,
                0.8,
                &mut session,
                &mut cooperation,
            )
            .await
            .unwrap()
            .expect("an eligible modeled refinement")
        };
        assert!(result.merit.is_some_and(f64::is_finite));
        assert!(result.violation.is_some_and(|value| value <= 1e-7));
        assert!(
            result.point[0] < 0.5,
            "the actual SQP must improve the sampled merit"
        );
        assert!((result.point[0] + result.point[1] - 1.0).abs() < 1e-7);
        assert!(
            domain
                .classify(&result.point, 1e-7, &mut cooperation)
                .await
                .unwrap()
                .feasible
        );
        assert_eq!(
            domain.repair_nodes(),
            0,
            "SQP borrows the original model, not a second integer solve"
        );
        let mut integer = Domain::new(
            Input {
                matrix: CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]),
                rhs: vec![1.0],
                equalities: 1,
                lower: vec![0.0; 2],
                upper: vec![1.0; 2],
                integers: vec![0],
            },
            clock.clone(),
            TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        let integer_shape = BoxGeometry::new(&integer, &mut cooperation).await.unwrap();
        let result = {
            let mut session = Session::new(
                &integer,
                &source,
                ModelView {
                    model: &model,
                    training: &data,
                    evaluated: &data,
                    domain: integer_shape.view(&integer).unwrap(),
                    criterion: selected.criterion,
                    inequality_start: 1,
                },
                Controls {
                    tolerance: 1e-3,
                    clock: &clock,
                    time_limit: TimeLimit::Unbounded,
                },
                Limits {
                    iterations: 30,
                    evaluations: 256,
                    feasibility: 1e-8,
                    optimality: 1e-5,
                    step: 1e-10,
                },
                &mut cooperation,
            )
            .await
            .unwrap();
            refinement::run(
                schedule::Refinement::TreeReset,
                integer_shape.view(&integer).unwrap(),
                &[1.0, 0.0],
                0.2,
                &mut session,
                &mut cooperation,
            )
            .await
            .unwrap()
            .expect("an eligible integer modeled branch")
        };
        assert_eq!(result.point[0], 0.0);
        assert!((result.point[1] - 1.0).abs() < 1e-7);
        assert!(
            integer
                .classify(&result.point, 1e-7, &mut cooperation)
                .await
                .unwrap()
                .feasible
        );
        assert_eq!(integer.repair_nodes(), 0);
    });
}
