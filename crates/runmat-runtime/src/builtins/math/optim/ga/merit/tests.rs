use super::*;
use crate::builtins::math::optim::ga::IDENTITY;

#[test]
fn actual_shifted_barrier_and_equality_merit_are_not_penalty_aliases() {
    futures::executor::block_on(async {
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, None);
        let mut merit = AugmentedLagrangian::new(&[Some(0.0)], 1, 10.0, 100.0, &mut cooperation)
            .await
            .unwrap();
        // lambda=10, shift=1, ceq=.5: the equality term is 10*.25/2.
        let expected = 3.0 - 10.0 * 2.0_f64.ln() + 1.25;
        assert!((merit.score(3.0, &[-1.0], &[0.5]).unwrap() - expected).abs() < 1e-13);
        assert!(merit.score(3.0, &[1.0], &[0.0]).is_none());
        assert!(matches!(merit.domain(&[1.0]), BarrierDomain::Exterior(0.0)));
        assert!(matches!(merit.domain(&[-1.0]), BarrierDomain::Interior));
        assert_eq!(merit.complementarity(&[0.0]), Some(0.0));
        assert_eq!(
            merit
                .update(&[0.0], &[0.0], true, &mut cooperation)
                .await
                .unwrap(),
            Update::Multipliers
        );
        assert_eq!(
            merit
                .update(&[0.0], &[1.0], false, &mut cooperation)
                .await
                .unwrap(),
            Update::Penalty
        );
        assert_eq!(merit.penalty, 1000.0);
        assert_eq!(penalty_score(Some(-3.0), 0.0, true, Some(-1.0)), Some(-3.0));
        assert_eq!(
            penalty_score(Some(-1e10), 2.0, false, Some(-1.0)),
            Some(1.0)
        );
        assert_eq!(penalty_score(Some(5.0), 2.0, false, None), Some(2.0));
        let unknown = AugmentedLagrangian::new(&[None], 0, 10.0, 100.0, &mut cooperation)
            .await
            .unwrap();
        assert!(matches!(
            unknown.domain(&[100.0]),
            BarrierDomain::Exterior(_)
        ));
    });
}
