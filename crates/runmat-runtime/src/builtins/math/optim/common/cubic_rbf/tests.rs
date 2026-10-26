use super::*;
const POLICY: Policy = Policy {
    identity: runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
    invalid: &runmat_builtins::PATTERNSEARCH_ERROR_INVALID_ARGUMENT,
};

#[test]
fn cubic_tail_interpolates_two_lanes_without_copying_or_replaying_trials() {
    futures::executor::block_on(async {
        let mut cooperation = ScalarCooperation::for_builtin(POLICY.identity, None);
        // The rejected NaN trial remains in history but is not a training row.
        let points = [0.0, 0.5, 1.0, f64::NAN];
        let values = [0.0, 1.0, 0.25, 2.0, 1.0, 3.0, f64::NAN, f64::NAN];
        let rows = [0, 1, 2];
        let data = Data {
            owner: &TrainingOwner::new(),
            points: &points,
            values: &values,
            rows: &rows,
            dimensions: 1,
            lanes: 2,
            generation: Generation(1),
        };
        let Fit::Poised(model) = fit(POLICY, &data, &mut cooperation).await.unwrap() else {
            panic!("three distinct sites must produce a poised cubic model");
        };
        let mut predicted = [0.0; 2];
        for row in 0..3 {
            assert_eq!(
                model
                    .predict_into(
                        &data,
                        &points[row..row + 1],
                        &mut predicted,
                        &mut cooperation
                    )
                    .await
                    .unwrap(),
                Prediction::Finite
            );
            for lane in 0..2 {
                assert!((predicted[lane] - values[row * 2 + lane]).abs() < 1e-12);
            }
        }
        model
            .predict_into(&data, &[0.25], &mut predicted, &mut cooperation)
            .await
            .unwrap();
        // Cubic interpolation with a linear tail, not a polynomial/GP alias.
        assert!((predicted[0] - 0.078125).abs() < 1e-12);
        assert!((predicted[1] - 1.5).abs() < 1e-12);
        let mut second_lane = [0.0];
        assert_eq!(
            model
                .predict_lanes_into(&data, &[0.25], 1..2, &mut second_lane, &mut cooperation)
                .await
                .unwrap(),
            Prediction::Finite
        );
        assert!((second_lane[0] - 1.5).abs() < 1e-12);
        predicted = [123.0, 456.0];
        let changed = Data {
            generation: Generation(2),
            ..data
        };
        assert!(model
            .predict_into(&changed, &[0.25], &mut predicted, &mut cooperation)
            .await
            .is_err());
        assert_eq!(predicted, [123.0, 456.0]);
        let foreign = Data {
            owner: &TrainingOwner::new(),
            ..data
        };
        let error = model
            .predict_into(&foreign, &[0.25], &mut predicted, &mut cooperation)
            .await
            .expect_err("different canonical history");
        assert_eq!(error.identifier.as_deref(), POLICY.invalid.identifier);
        assert_eq!(predicted, [123.0, 456.0]);
        equality_tail_uses_full_radial_geometry(&mut cooperation).await;
    });
}

#[test]
fn duplicate_and_overflowing_sites_never_publish_zero_models() {
    futures::executor::block_on(async {
        let mut cooperation = ScalarCooperation::for_builtin(POLICY.identity, None);
        let duplicate = Data {
            owner: &TrainingOwner::new(),
            points: &[0.0, 0.5, 0.5],
            values: &[1.0, 2.0, 2.0],
            rows: &[0, 1, 2],
            dimensions: 1,
            lanes: 1,
            generation: Generation(1),
        };
        assert!(matches!(
            fit(POLICY, &duplicate, &mut cooperation).await.unwrap(),
            Fit::Unpoised
        ));
        let overflow = Data {
            owner: &TrainingOwner::new(),
            points: &[0.0, f64::MAX],
            values: &[0.0, 1.0],
            rows: &[0, 1],
            dimensions: 1,
            lanes: 1,
            generation: Generation(2),
        };
        assert!(matches!(
            fit(POLICY, &overflow, &mut cooperation).await.unwrap(),
            Fit::Undefined
        ));
    });
}

struct Tail<'a> {
    data: Data<'a>,
    axes: &'a [usize],
}
impl Training for Tail<'_> {
    fn valid(&self) -> bool {
        self.data.valid()
    }
    fn owner(&self) -> &TrainingOwner {
        self.data.owner
    }
    fn generation(&self) -> Generation {
        self.data.generation
    }
    fn rows(&self) -> &[usize] {
        self.data.rows
    }
    fn dimensions(&self) -> usize {
        self.data.dimensions
    }
    fn lanes(&self) -> usize {
        self.data.lanes
    }
    fn affine_axes(&self) -> Option<&[usize]> {
        Some(self.axes)
    }
    fn coordinate(&self, row: usize, axis: usize) -> f64 {
        self.data.coordinate(row, axis)
    }
    fn measurement(&self, row: usize, lane: usize) -> f64 {
        self.data.measurement(row, lane)
    }
}
async fn equality_tail_uses_full_radial_geometry(cooperation: &mut ScalarCooperation) {
    let owner = TrainingOwner::new();
    let rows = [0, 1, 2, 3, 4];
    let values = [0.0, 0.25, 0.25, 0.5, 0.125];
    // Original equality z=x+y. Its affine tail has two independent columns,
    // but z still contributes to every physical normalized radial distance.
    let full = Tail {
        data: Data {
            owner: &owner,
            points: &[
                0., 0., 0., 0.5, 0., 0.5, 0., 0.5, 0.5, 0.5, 0.5, 1., 0.25, 0.25, 0.5,
            ],
            values: &values,
            rows: &rows,
            dimensions: 3,
            lanes: 1,
            generation: Generation(7),
        },
        axes: &[0, 1],
    };
    let Fit::Poised(model) = fit(POLICY, &full, cooperation).await.unwrap() else {
        panic!("independent equality tail must retain the physical radial metric");
    };
    let mut output = [0.0];
    for row in 0..rows.len() {
        let point = &full.data.points[row * 3..row * 3 + 3];
        assert_eq!(
            model
                .predict_into(&full, point, &mut output, cooperation)
                .await
                .unwrap(),
            Prediction::Finite
        );
        assert!((output[0] - values[row]).abs() < 1e-11);
    }
    model
        .predict_into(&full, &[0.1, 0.2, 0.3], &mut output, cooperation)
        .await
        .unwrap();
    let physical = output[0];
    let projected = Data {
        points: &[0., 0., 0.5, 0., 0., 0.5, 0.5, 0.5, 0.25, 0.25],
        dimensions: 2,
        ..full.data
    };
    let Fit::Poised(projected_model) = fit(POLICY, &projected, cooperation).await.unwrap() else {
        panic!("comparison projected model");
    };
    projected_model
        .predict_into(&projected, &[0.1, 0.2], &mut output, cooperation)
        .await
        .unwrap();
    assert!(
        (physical - output[0]).abs() > 1e-8,
        "radial norm must not silently drop a dependent physical coordinate"
    );
    let changed = Tail {
        data: full.data,
        axes: &[0, 2],
    };
    output = [123.0];
    assert!(model
        .predict_into(&changed, &[0.1, 0.2, 0.3], &mut output, cooperation)
        .await
        .is_err());
    assert_eq!(output, [123.0]);
    let invalid = Tail {
        data: full.data,
        axes: &[0, 0],
    };
    assert_eq!(
        fit(POLICY, &invalid, cooperation)
            .await
            .err()
            .unwrap()
            .identifier
            .as_deref(),
        POLICY.invalid.identifier
    );
}
