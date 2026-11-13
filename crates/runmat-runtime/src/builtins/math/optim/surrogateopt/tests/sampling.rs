//! Actual original-domain random proposals at narrow and opposite-MAX spans.
use super::super::{
    domain::BoxGeometry,
    region::{Neighbor, Region, Scope},
    sampling::{self, Kind, Problem},
    IDENTITY,
};
use crate::builtins::math::optim::{
    common::{global_time::TimeLimit, solver_timing::SolverTiming, ScalarCooperation},
    intlinprog::feasibility::{Domain, Input},
};
use clarabel::algebra::CscMatrix;

#[test]
fn adaptive_integer_sampling_uses_original_membership_and_finite_large_span() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let clock = SolverTiming::start(Some(&runtime)).unwrap();
        for (lower, upper, scale, radius) in
            [(-3.0, 3.0, 1e-5, 1.0), (-f64::MAX, f64::MAX, 0.8, f64::MAX)]
        {
            let mut domain = Domain::new(
                Input {
                    matrix: CscMatrix::new(0, 1, vec![0, 0], vec![], vec![]),
                    rhs: vec![],
                    equalities: 0,
                    lower: vec![lower],
                    upper: vec![upper],
                    integers: vec![0],
                },
                clock.clone(),
                TimeLimit::Unbounded,
                1e-8,
                &mut cooperation,
            )
            .await
            .unwrap();
            let shape = BoxGeometry::new(&domain, &mut cooperation).await.unwrap();
            for _ in 0..8 {
                let point = {
                    let view = shape.view(&domain).unwrap();
                    let problem = Problem::classify(&view, &mut cooperation).await.unwrap();
                    assert_eq!(problem, Problem::Integer);
                    assert_eq!(problem.sampler(0).unwrap(), Kind::Orthogonal);
                    assert_eq!(problem.sampler(1).unwrap(), Kind::Crossover);
                    sampling::random_point(&view, &[0.0], scale, &mut cooperation)
                        .await
                        .unwrap()
                };
                assert!(point[0].is_finite());
                assert_eq!(point[0].fract(), 0.0);
                assert!(point[0].abs() <= radius);
                assert!(
                    domain
                        .classify(&point, 0.0, &mut cooperation)
                        .await
                        .unwrap()
                        .feasible
                );
            }
            assert_eq!(domain.repair_nodes(), 0);
            let mut region = Region::new(
                shape.view(&domain).unwrap(),
                &[0.0],
                Scope::Local(scale),
                &mut cooperation,
            )
            .await
            .unwrap();
            assert_eq!(
                region.neighbor(0, 0.0, Neighbor::Lower).unwrap(),
                Some(-1.0)
            );
            assert_eq!(region.neighbor(0, 0.0, Neighbor::Upper).unwrap(), Some(1.0));
            if upper == f64::MAX {
                let large = 2.0_f64.powi(53);
                assert_eq!(
                    region.neighbor(0, large, Neighbor::Upper).unwrap(),
                    Some(large.next_up())
                );
                assert_eq!(
                    region.neighbor(0, -large, Neighbor::Lower).unwrap(),
                    Some((-large).next_down())
                );
            }
            assert!(region.pin_integer(0, 1.0).unwrap());
            assert_eq!(region.bounds(), (&[1.0][..], &[1.0][..]));
            region.restore(&mut cooperation).await.unwrap();
            assert!(region.bounds().0[0] <= -1.0 && region.bounds().1[0] >= 1.0);
        }
    });
}
