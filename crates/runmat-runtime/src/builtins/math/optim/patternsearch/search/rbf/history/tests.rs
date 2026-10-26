use super::*;

#[tokio::test]
async fn actual_oriented_backing_raw_constraints_duplicate_sites_and_reset_epochs() {
    for axis in [PointAxis::Rows, PointAxis::Columns] {
        let mut cooperation = ScalarCooperation::for_builtin(
            runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
            None,
        );
        let data = if axis == PointAxis::Rows {
            vec![1., 3., 2., 4.]
        } else {
            vec![1., 2., 3., 4.]
        };
        let mut backing = Tensor::new(data, vec![2, 2]).unwrap();
        let mut history = History::new(2, 0).unwrap();
        history.attempted(2).unwrap();
        for point in 0..2 {
            let c = ExactArrayStorage::shared_unbounded(vec![point as f64 - 0.5]);
            let address = c.as_ptr();
            history
                .record(
                    point + 1,
                    Coordinates::batch(&backing, axis, point, 2, 2).unwrap(),
                    point as f64,
                    c.clone(),
                    ExactArrayStorage::empty(),
                    &mut cooperation,
                )
                .await
                .unwrap();
            assert_eq!(history.records[point].inequality.as_ptr(), address);
        }
        assert!(backing.try_as_f64_slice_mut_unique().is_none());
        assert!(history.contains(&[1., 2.], &mut cooperation).await.unwrap());
        assert!(history.contains(&[3., 4.], &mut cooperation).await.unwrap());
        assert!(!history.contains(&[1., 3.], &mut cooperation).await.unwrap());
        let previous = history.generation;
        history.attempted(3).unwrap();
        history
            .record(
                3,
                Coordinates::point(
                    &Value::Tensor(Tensor::new(vec![1., 2.], vec![1, 2]).unwrap()),
                    2,
                )
                .unwrap(),
                -2.,
                ExactArrayStorage::empty(),
                ExactArrayStorage::empty(),
                &mut cooperation,
            )
            .await
            .unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history.records[0].f, -2.);
        assert!(history.generation > previous);
        // Undefined trials still advance the true reset epoch. A late result
        // from an earlier wave epoch cannot repopulate the new history.
        history.attempted(20).unwrap();
        assert_eq!(history.len(), 0);
        history
            .record(
                19,
                Coordinates::batch(&backing, axis, 0, 2, 2).unwrap(),
                1.,
                ExactArrayStorage::empty(),
                ExactArrayStorage::empty(),
                &mut cooperation,
            )
            .await
            .unwrap();
        assert_eq!(history.len(), 0);
        assert!(backing.try_as_f64_slice_mut_unique().is_some());
    }
}
