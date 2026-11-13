use super::*;
use crate::builtins::math::optim::ga::IDENTITY;

#[test]
fn rank_expectation_and_real_selection_preserve_member_identity() {
    futures::executor::block_on(async {
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, None);
        let ranked = order(
            &[
                Score::Defined(2.0),
                Score::Undefined,
                Score::Defined(-4.0),
                Score::Defined(2.0),
            ],
            &mut cooperation,
        )
        .await
        .unwrap();
        assert_eq!(ranked.as_slice(), [2, 0, 3, 1]);
        let restored = order(
            &[
                Score::RestoreDomain(0.0),
                Score::Defined(1e100),
                Score::RestoreDomain(1.0),
                Score::Undefined,
            ],
            &mut cooperation,
        )
        .await
        .unwrap();
        assert_eq!(restored.as_slice(), [1, 0, 2, 3]);
        let expectation = rank_expectation(&ranked, 6, &mut cooperation)
            .await
            .unwrap();
        assert!((expectation.iter().sum::<f64>() - 6.0).abs() < 1e-12);
        assert!(expectation[2] > expectation[0] && expectation[0] > expectation[3]);
        let parents = stochastic_uniform(&[0.0, 2.0, 0.0, 2.0], 4, 0.25, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(parents, [1, 1, 3, 3]);
        let edge = stochastic_uniform(
            &[1.0, 1.0, 0.0],
            2,
            f64::from_bits(1.0_f64.to_bits() - 1),
            &mut cooperation,
        )
        .await
        .unwrap();
        assert_eq!(edge, [0, 1]);
        let selected = tournament(&expectation, &[0.9, 0.0, 0.6, 0.4], 2, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(selected, [0, 2]);
        assert_eq!(
            tournament(&[0., 10., 1.], &[0., 0.4], 2, &mut cooperation)
                .await
                .unwrap(),
            [1],
            "custom/proportional expectation ordering must reach the actual tournament"
        );
    });
}
