use super::super::super::{
    linear::Linear,
    timing::{Deadline, TimeLimit},
};
use super::super::Bounds;
use super::*;
use crate::builtins::math::optim::common::{
    cubic_rbf, feasible_cone::IndependentAxes, ScalarCooperation,
};
use clarabel::algebra::CscMatrix;
use runmat_value::{ExactArrayStorage, Tensor, Value};

#[tokio::test]
async fn real_cubic_model_proposes_unmeasured_improvement_and_respects_original_equality_metric() {
    let mut cooperation =
        ScalarCooperation::for_builtin(runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity, None);
    let mut history = History::new(1, 0).unwrap();
    for (index, x) in [0.0f64, 1., 2.].into_iter().enumerate() {
        history.attempted(index + 1).unwrap();
        history
            .record(
                index + 1,
                Coordinates::point(&Value::Num(x), 1).unwrap(),
                (x - 1.2).powi(2),
                ExactArrayStorage::empty(),
                ExactArrayStorage::empty(),
                &mut cooperation,
            )
            .await
            .unwrap();
    }
    let bounds = Bounds::new(vec![f64::NEG_INFINITY], vec![f64::INFINITY]).unwrap();
    let geometry = Linear::new(
        CscMatrix::new(0, 1, vec![0, 0], vec![], vec![]),
        vec![],
        0,
        &mut cooperation,
    )
    .await
    .unwrap();
    let window = bounds
        .window(&history, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (&window.lower[..], &window.upper[..]),
        (&[0.][..], &[2.][..])
    );
    let scores = history.records().iter().map(|p| p.f).collect::<Vec<_>>();
    let training = history
        .training(&scores, None, &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    let policy = cubic_rbf::Policy {
        identity: runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
        invalid: &runmat_builtins::PATTERNSEARCH_ERROR_INVALID_ARGUMENT,
    };
    let cubic_rbf::Fit::Poised(model) = cubic_rbf::fit(policy, &training, &mut cooperation)
        .await
        .unwrap()
    else {
        panic!("poised real samples")
    };
    let deadline = Deadline::new(TimeLimit::Unbounded, None, false).unwrap();
    let candidate = model::propose(
        &training,
        &model,
        &window,
        &geometry,
        1e-6,
        &deadline,
        &mut cooperation,
    )
    .await
    .unwrap()
    .unwrap();
    assert!((candidate[0] - 1.2).powi(2) < scores[1]);
    assert!(!history
        .contains(&candidate, &mut cooperation)
        .await
        .unwrap());

    // Coupled equalities reduce only P's independent affine axes. Full physical
    // coordinates stay present in Training and in the radial metric.
    let geometry = Linear::new(
        CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1., -1.]),
        vec![0.],
        1,
        &mut cooperation,
    )
    .await
    .unwrap();
    let bounds = Bounds::new(vec![f64::NEG_INFINITY; 2], vec![f64::INFINITY; 2]).unwrap();
    let IndependentAxes::Axes(axes) = bounds.axes(&geometry, &mut cooperation).await.unwrap()
    else {
        panic!("one free equality axis")
    };
    assert_eq!(axes.len(), 1);
    let mut history = History::new(2, 0).unwrap();
    for (index, x) in [0.0f64, 1., 2.].into_iter().enumerate() {
        history.attempted(index + 1).unwrap();
        history
            .record(
                index + 1,
                Coordinates::point(
                    &Value::Tensor(Tensor::new(vec![x, x], vec![2, 1]).unwrap()),
                    2,
                )
                .unwrap(),
                (x - 1.2).powi(2),
                ExactArrayStorage::empty(),
                ExactArrayStorage::empty(),
                &mut cooperation,
            )
            .await
            .unwrap();
    }
    let scores = history.records().iter().map(|p| p.f).collect::<Vec<_>>();
    let training = history
        .training(&scores, Some(&axes), &mut cooperation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cubic_rbf::Training::dimensions(&training), 2);
    assert_eq!(cubic_rbf::Training::coordinate(&training, 2, 0), 2.);
    assert_eq!(cubic_rbf::Training::coordinate(&training, 2, 1), 2.);
    assert!(matches!(
        cubic_rbf::fit(policy, &training, &mut cooperation)
            .await
            .unwrap(),
        cubic_rbf::Fit::Poised(_)
    ));
}
