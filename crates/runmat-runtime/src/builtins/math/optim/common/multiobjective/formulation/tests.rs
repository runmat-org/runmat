use super::*;
use clarabel::algebra::CscMatrix;
use futures::executor::block_on;

#[test]
fn signed_goals_keep_raw_weights_zero_hard_rows_and_original_jacobian() {
    block_on(async {
        let mut cooperation = ScalarCooperation::for_builtin(
            runmat_builtins::BuiltinCatalogIdentity {
                name: "fgoalattain",
            },
            None,
        );
        let specification = Specification::Goal {
            goals: vec![1.0, 3.0, 2.0],
            weights: vec![2.0, -1.0, 0.0],
            equal: 0,
        };
        let model = specification
            .bind(Kind::GoalAttain, 3, &mut cooperation)
            .await
            .unwrap();
        let Specification::Goal { weights, .. } = &specification else {
            unreachable!()
        };
        assert_eq!(
            weights,
            &[2.0, -1.0, 0.0],
            "raw caller weights are not normalized in place"
        );
        assert_eq!(model.rows[1].direction, Direction::AtLeast);
        assert_eq!(model.rows[1].weight, 1.0);
        assert_eq!(
            model
                .worst(&[2.0, 2.0, 2.0], &mut cooperation)
                .await
                .unwrap(),
            Some(1.0)
        );
        assert_eq!(
            model
                .residuals(&[2.0, 2.0, 2.0], 1.0, &mut cooperation)
                .await
                .unwrap()
                .unwrap(),
            vec![-1.0, 0.0, 0.0]
        );
        let original = CscMatrix::new(3, 1, vec![0, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let lifted = model
            .jacobian(original, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(lifted.nzval, vec![1.0, -2.0, 3.0, -2.0, -1.0]);
        let projected = model
            .original_jacobian(&lifted, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(projected.colptr, vec![0, 1, 2, 3]);
        assert_eq!(projected.nzval, vec![1.0, 2.0, 3.0]);
        let equality = Specification::Goal {
            goals: vec![1.0],
            weights: vec![-2.0],
            equal: 1,
        }
        .bind(Kind::GoalAttain, 1, &mut cooperation)
        .await
        .unwrap();
        assert_eq!(
            equality.worst(&[3.0], &mut cooperation).await.unwrap(),
            Some(1.0)
        );
        assert_eq!(
            equality
                .residuals(&[3.0], 1.0, &mut cooperation)
                .await
                .unwrap()
                .unwrap(),
            vec![-4.0, 0.0]
        );
        let hard = Specification::Goal {
            goals: vec![1.0],
            weights: vec![0.0],
            equal: 1,
        }
        .bind(Kind::GoalAttain, 1, &mut cooperation)
        .await
        .unwrap();
        assert_eq!(hard.worst(&[1.0], &mut cooperation).await.unwrap(), None);
        assert_eq!(
            hard.residuals(&[2.0], 99.0, &mut cooperation)
                .await
                .unwrap()
                .unwrap(),
            vec![1.0, -1.0]
        );
        assert!(model
            .residuals(&[f64::MAX, -f64::MAX, 2.0], f64::MAX, &mut cooperation)
            .await
            .unwrap()
            .is_none());
        assert!(Specification::Goal {
            goals: vec![0.0],
            weights: vec![f64::NEG_INFINITY],
            equal: 0
        }
        .bind(Kind::GoalAttain, 1, &mut cooperation)
        .await
        .is_err());
    });
}
