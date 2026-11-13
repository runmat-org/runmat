use super::super::{common, IDENTITY};
use super::*;
use common::{global_time::TimeLimit, solver_timing::SolverTiming};
use num_rational::BigRational;

#[test]
fn original_row_metadata_and_exact_step_preserve_equality_bounds_and_clock() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let timing = SolverTiming::start(Some(&runtime)).unwrap();
        let model = mip::Domain::new(
            mip::Input {
                matrix: CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1., 1.]),
                rhs: vec![1.],
                equalities: 1,
                lower: vec![0., 0.],
                upper: vec![1., 1.],
                integers: vec![],
            },
            timing.clone(),
            TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        let mut domain = Domain::new(model, &mut cooperation).await.unwrap();
        let mut scratch = domain.row_scratch(&cooperation).unwrap();
        domain
            .original_row(0, &mut scratch, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(scratch, [(0, 1.), (1, 1.)]);
        let direction = [
            BigRational::from_integer(1.into()),
            BigRational::from_integer((-1).into()),
        ];
        let step = domain
            .step(
                &[0.25, 0.75],
                &direction,
                step::Controls {
                    maximum: 1.,
                    draw: 0.5,
                    tolerance: 0.,
                    timing: &timing,
                    deadline: TimeLimit::Unbounded,
                },
                &mut cooperation,
            )
            .await
            .unwrap();
        let step::Step::Point(point) = step else {
            panic!("exact bounded equality step");
        };
        assert_eq!(point, [0.625, 0.375]);
        assert!(
            domain
                .classify(&point, 0., &mut cooperation)
                .await
                .unwrap()
                .feasible
        );
        assert_eq!(domain.repair_nodes(), 0);
        domain.retune(TimeLimit::Unbounded, 0.).unwrap();
        let small = domain
            .classify(&[0.25 + 1e-9, 0.75], 0., &mut cooperation)
            .await
            .unwrap();
        assert!(small.feasible && small.violation > 0.);
        let large = domain
            .classify(&[0.25 + 1e-6, 0.75], 0., &mut cooperation)
            .await
            .unwrap();
        assert!(!large.feasible);
        use super::super::evaluation::generation::Violation;
        assert!(Violation::Finite {
            maximum: 0.,
            sum: 0.,
            linear: small.violation
        }
        .admitted(0.));
        assert!(
            !Violation::Finite {
                maximum: 1e-9,
                sum: 1e-9,
                linear: 0.
            }
            .admitted(0.),
            "the documented linear floor cannot widen nonlinear feasibility"
        );
        let stopped = domain
            .step(
                &point,
                &direction,
                step::Controls {
                    maximum: 1.,
                    draw: 0.5,
                    tolerance: 0.,
                    timing: &timing,
                    deadline: TimeLimit::Days(0.),
                },
                &mut cooperation,
            )
            .await
            .unwrap();
        assert!(matches!(stopped, step::Step::TimeLimit));
        let zero = [
            BigRational::from_integer(0.into()),
            BigRational::from_integer(0.into()),
        ];
        let unchanged = domain
            .step(
                &[-0., 1.],
                &zero,
                step::Controls {
                    maximum: 1.,
                    draw: 0.5,
                    tolerance: 0.,
                    timing: &timing,
                    deadline: TimeLimit::Unbounded,
                },
                &mut cooperation,
            )
            .await
            .unwrap();
        assert!(
            matches!(unchanged, step::Step::Blocked),
            "zero direction must not manufacture a -0 to +0 mutation"
        );
    });
}
