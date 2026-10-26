use super::*;
use futures::executor::block_on;

#[test]
fn inverse_updates_and_bounded_history_are_real_distinct_models() {
    let mut cooperation = ScalarCooperation::new(runmat_builtins::OptimSolver::Fminunc, None);
    for kind in [
        Update::Bfgs,
        Update::Dfp,
        Update::LimitedMemory {
            history: std::num::NonZeroUsize::new(1).unwrap(),
        },
        Update::Steepest,
    ] {
        let mut model = Model::new(kind, 2, 10).unwrap();
        block_on(model.update(vec![1., 0.], vec![2., 0.], &mut cooperation)).unwrap();
        let direction = block_on(model.direction(&[2., 3.], &mut cooperation)).unwrap();
        let expected = match kind {
            Update::Bfgs | Update::Dfp => [-1., -3.],
            Update::LimitedMemory { .. } => [-1., -1.5],
            Update::Steepest => [-2., -3.],
        };
        for (actual, expected) in direction.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }
        if let Update::LimitedMemory { .. } = kind {
            block_on(model.update(vec![0., 1.], vec![0., 4.], &mut cooperation)).unwrap();
            assert_eq!(
                block_on(model.direction(&[2., 3.], &mut cooperation)).unwrap(),
                vec![-0.5, -0.75]
            );
            let Model::Limited { pairs, .. } = &model else {
                panic!("limited model")
            };
            assert_eq!(pairs.len(), 1);
        }
    }
    let n = MAX_DENSE_BFGS_VARIABLES + 1;
    for (kind, expected) in [
        (Update::Bfgs, [-7. / 9., -31. / 9.]),
        (Update::Dfp, [-13. / 15., -49. / 15.]),
    ] {
        let mut model = Model::new(kind, 2, 1).unwrap();
        block_on(model.update(vec![1., 1.], vec![2., 1.], &mut cooperation)).unwrap();
        let direction = block_on(model.direction(&[2., 3.], &mut cooperation)).unwrap();
        for (actual, expected) in direction.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }
    }
    assert!(Model::new(Update::Bfgs, n, 1).is_err());
    for kind in [Update::Bfgs, Update::Dfp] {
        let mut model = Model::new(kind, 1, 1).unwrap();
        block_on(model.update(vec![1e200], vec![1e-100], &mut cooperation)).unwrap();
        let Model::Dense { values, .. } = &model else {
            panic!("dense")
        };
        assert!(values[0].is_finite());
        assert!((values[0] / 1e300 - 1.).abs() < 1e-14);
        // A genuinely nonrepresentable candidate must not partially mutate H.
        let mut model = Model::new(kind, 2, 1).unwrap();
        block_on(model.update(vec![1e300, 1.], vec![1e-150, 1.], &mut cooperation)).unwrap();
        let Model::Dense { values, .. } = &model else {
            panic!("dense")
        };
        assert_eq!(values, &vec![1., 0., 0., 1.]);
    }
    for kind in [
        Update::Steepest,
        Update::LimitedMemory {
            history: std::num::NonZeroUsize::new(10).unwrap(),
        },
    ] {
        let model = Model::new(kind, n, 1).unwrap();
        assert!(!matches!(model, Model::Dense { .. }));
        assert_eq!(
            block_on(model.direction(&vec![1.; n], &mut cooperation)).unwrap(),
            vec![-1.; n]
        );
    }
}
