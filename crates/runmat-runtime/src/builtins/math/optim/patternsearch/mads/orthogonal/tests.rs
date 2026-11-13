use super::*;
use num_rational::BigRational;
use num_traits::One;

#[tokio::test]
async fn exact_halton_adjustment_and_householder_match_original_construction() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
        Some(&runtime),
    );
    let mut sequence = Sequence::new(2, &mut cooperation).await.unwrap();
    let ray = sequence.direction(3, &mut cooperation).await.unwrap();
    assert_eq!(
        ray,
        [
            BigRational::new((-1).into(), 4.into()),
            BigRational::new((-5).into(), 9.into())
        ]
    );
    let q = adjusted::direction(&ray, 3, &mut cooperation)
        .await
        .unwrap();
    assert_eq!(q, [BigInt::from(-1), BigInt::from(-2)]);
    let mut model = Householder {
        q,
        norm: BigInt::from(5),
        negative: Some(vec![BigInt::one(), BigInt::from(7)]),
    };
    assert!(certificate::admissible(&model, &mut cooperation)
        .await
        .unwrap());
    let first = model.direction(0, &mut cooperation).await.unwrap().unwrap();
    let second = model.direction(1, &mut cooperation).await.unwrap().unwrap();
    assert_eq!(first, [3.0, -4.0]);
    assert_eq!(second, [-4.0, -3.0]);
    assert_eq!(
        first.iter().zip(&second).map(|(a, b)| a * b).sum::<f64>(),
        0.0
    );
    assert_eq!(
        model.direction(2, &mut cooperation).await.unwrap().unwrap(),
        [1.0, 7.0]
    );
    model.negative.as_mut().unwrap().fill(BigInt::zero());
    // Even an exactly represented zero completion must not pass positivity.
    assert!(!certificate::admissible(&model, &mut cooperation)
        .await
        .unwrap());

    let coarse = sequence.direction(2, &mut cooperation).await.unwrap();
    assert_eq!(coarse[0], BigRational::new(3.into(), 4.into()));
    assert_eq!(sequence.direction(3, &mut cooperation).await.unwrap(), ray);
    let next_coarse = sequence.direction(2, &mut cooperation).await.unwrap();
    assert_eq!(next_coarse[0], BigRational::new((-7).into(), 8.into()));
}

#[tokio::test]
async fn emitted_wide_frames_are_certified_without_matrix_storage_or_random_draws() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
        Some(&runtime),
    );
    let _same_turn =
        crate::builtins::common::random::try_acquire_turn(Some(&runtime), "original_test_turn")
            .unwrap();
    for mesh in [1.0, 0.0625, 2f64.powi(-240), f64::from_bits(1)] {
        for basis in [Basis::Maximal, Basis::Minimal] {
            let mut sequence = Sequence::new(2, &mut cooperation).await.unwrap();
            let level = Level::for_mesh(mesh).unwrap();
            let model = Householder::new(&mut sequence, level, basis, &mut cooperation)
                .await
                .unwrap()
                .unwrap();
            assert!(model.norm > BigInt::zero() && model.norm <= level.integer());
            assert_eq!(
                model.count().unwrap(),
                if basis == Basis::Maximal { 4 } else { 3 }
            );
            for index in 0..model.count().unwrap() {
                let ray = model
                    .direction(index, &mut cooperation)
                    .await
                    .unwrap()
                    .unwrap();
                assert!(ray.iter().all(|value| value.is_finite()));
                assert!(ray.iter().any(|&value| value != 0.0));
            }
        }
    }
}
