use super::{rbf::*, IDENTITY};
use crate::builtins::math::optim::common::ScalarCooperation;
mod evaluation;
mod refinement;
mod sampling;

#[test]
fn sampler_qr_streams_orthogonal_directions_and_coordinate_rings_halve() {
    use super::frame::{Frame, Kind, Rings};
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let frame = Frame::new(Kind::RandomOrthogonal, 3, None, &mut cooperation)
            .await
            .unwrap()
            .expect("Gaussian frame must be numerically full-rank");
        let mut columns = [[0.0; 3]; 3];
        for (column, output) in columns.iter_mut().enumerate() {
            frame
                .direction_into(column * 2, output, &mut cooperation)
                .await
                .unwrap();
        }
        for a in 0..3 {
            for b in 0..3 {
                let product: f64 = columns[a].iter().zip(&columns[b]).map(|(a, b)| a * b).sum();
                assert!((product - if a == b { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        let mut opposite = [0.0; 3];
        frame
            .direction_into(1, &mut opposite, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(opposite, columns[0].map(|v| -v));
        frame
            .direction_into(6, &mut opposite, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(opposite, [1.0; 3]);
        let coordinate = Frame::new(Kind::Coordinate, 2, None, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        let mut rings = crate::builtins::math::optim::common::retained_control::Retained::new(
            Rings::new(coordinate, 0.2).unwrap(),
            &cooperation,
            super::CONTROL_STORAGE,
        )
        .unwrap();
        let mut output = [0.0; 2];
        for _ in 0..6 {
            assert_eq!(
                rings
                    .next_into(&mut output, &mut cooperation)
                    .await
                    .unwrap(),
                Some(0.2)
            );
        }
        assert_eq!(
            rings
                .next_into(&mut output, &mut cooperation)
                .await
                .unwrap(),
            Some(0.1)
        );
        assert_eq!(output, [1.0, 0.0]);
        let mut rings = rings.into_inner();
        assert_eq!(
            rings
                .next_into(&mut output, &mut cooperation)
                .await
                .unwrap(),
            Some(0.1)
        );
        assert_eq!(output, [-1.0, 0.0]);
    });
}

#[test]
fn callback_generation_preserves_sparse_rows_and_rejected_nan_without_dense_mirrors() {
    use super::observation::{decode, Form, Objective, Quality};
    use crate::execution::RootedValueSequence;
    use runmat_value::{SparseTensor, Tensor, UnboundedStructValueBuilder, Value, ValueSequence};
    futures::executor::block_on(async {
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, None);
        let mut fields = UnboundedStructValueBuilder::new();
        fields.insert(
            "Fval",
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap()),
        );
        fields.insert(
            "Ineq",
            Value::SparseTensor(
                SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![f64::NAN, -2.0]).unwrap(),
            ),
        );
        fields.insert("ignored", Value::String("not a numerical output".into()));
        let returned = RootedValueSequence::new(
            ValueSequence::single(Value::Struct(fields.finish())).unwrap(),
        )
        .unwrap();
        let batch = decode(
            returned,
            Form::Vectorized(std::num::NonZeroUsize::new(2).unwrap()),
            1e-3,
            &mut cooperation,
        )
        .await
        .unwrap();
        assert_eq!(batch[0].quality, Quality::RejectedNaN);
        assert_eq!(batch[1].quality, Quality::Finite);
        assert!(matches!(batch[1].objective(), Objective::Measured(3.0)));
        assert_eq!(batch[1].inequality(0), 0.0);
        assert_eq!(batch[1].inequality(1), -2.0);

        // A large sparse vector is validated by its actual stored values,
        // not by materializing or visiting a million implicit zeros.
        let mut fields = UnboundedStructValueBuilder::new();
        fields.insert(
            "Ineq",
            Value::SparseTensor(
                SparseTensor::new(1_000_000, 1, vec![0, 1], vec![999_999], vec![-1.0]).unwrap(),
            ),
        );
        let returned = RootedValueSequence::new(
            ValueSequence::single(Value::Struct(fields.finish())).unwrap(),
        )
        .unwrap();
        let point = decode(returned, Form::Point, 1e-3, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(point[0].quality, Quality::Finite);
        assert!(matches!(point[0].objective(), Objective::Absent));
        assert_eq!(point[0].inequality(999_999), -1.0);
        assert_eq!(point[0].measurement().unwrap().nonlinear.maximum, Some(0.0));
        assert_eq!(point[0].measurement().unwrap().nonlinear.violated, 0);
        assert_eq!(batch[1].measurement().unwrap().nonlinear.maximum, Some(0.0));
        assert!(batch[0].measurement().is_none());
    });
}

#[test]
fn adaptive_merit_excludes_measured_sites_and_scale_counts_only_measured_steps() {
    use super::merit::{self, Improvement, Phase, Proposals, Search, Selection};
    struct Grid<'a>(&'a [f64]);
    impl Proposals for Grid<'_> {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn dimensions(&self) -> usize {
            1
        }
        fn coordinate(&self, row: usize, _axis: usize) -> f64 {
            self.0[row]
        }
    }
    futures::executor::block_on(async {
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, None);
        let data = Data {
            owner: &TrainingOwner::new(),
            points: &[0.0, 1.0],
            values: &[0.0, 1.0],
            rows: &[0, 1],
            dimensions: 1,
            lanes: 1,
            generation: Generation(1),
        };
        let Fit::Poised(model) = fit(&data, &mut cooperation).await.unwrap() else {
            panic!("linear two-site interpolation must be poised");
        };
        let phase = Phase::Objective {
            objective_lane: 0,
            inequality_start: 1,
        };
        let choice = merit::select(
            &model,
            &data,
            &data,
            &Grid(&[0.0, 0.25, 0.75, 1.0]),
            merit::SelectionControls {
                phase,
                weight: 0.5,
                constraint_tolerance: 1e-3,
                min_distance: 1e-6,
            },
            &mut cooperation,
        )
        .await
        .unwrap();
        assert!(matches!(
            choice,
            Selection::Point(merit::Selected {
                index: merit::CandidateIndex(1),
                ..
            })
        ));
        assert!(matches!(
            merit::select(
                &model,
                &data,
                &data,
                &Grid(&[0.0, 1.0]),
                merit::SelectionControls {
                    phase,
                    weight: 0.5,
                    constraint_tolerance: 1e-3,
                    min_distance: 1e-6,
                },
                &mut cooperation
            )
            .await
            .unwrap(),
            Selection::Reset(merit::NoCandidate::TooClose)
        ));
        assert!(
            matches!(
                merit::select(
                    &model,
                    &data,
                    &data,
                    &Grid(&[0.0, 1.0]),
                    merit::SelectionControls {
                        phase,
                        weight: 0.5,
                        constraint_tolerance: 1e-3,
                        min_distance: 0.0,
                    },
                    &mut cooperation
                )
                .await
                .unwrap(),
                Selection::Point(_)
            ),
            "zero minimum distance does not reject an exact site implicitly"
        );
    });
    let mut search = Search::new(2);
    for expected in [0.3, 0.5, 0.8, 0.95, 0.3] {
        assert_eq!(search.weight(), expected);
        search.selected();
    }
    assert_eq!(
        search.scale(),
        0.2,
        "selection is not an evaluated improvement"
    );
    for _ in 0..3 {
        search.measured(Improvement::Success);
    }
    assert_eq!(search.scale(), 0.4);
    for _ in 0..5 {
        search.measured(Improvement::Failure);
    }
    assert_eq!(search.scale(), 0.2);
}

#[test]
fn original_equalities_reduce_coordinates_without_copying_the_linear_model() {
    use crate::builtins::math::optim::common::{
        global_time::TimeLimit, solver_timing::SolverTiming,
    };
    use crate::builtins::math::optim::intlinprog::feasibility::{Domain, Input, Repair};
    use clarabel::algebra::CscMatrix;
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let timing = SolverTiming::start(Some(&runtime)).unwrap();
        let mut domain = Domain::new(
            Input {
                matrix: CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]),
                rhs: vec![1.0],
                equalities: 1,
                lower: vec![0.0, 0.0],
                upper: vec![1.0, 1.0],
                integers: vec![],
            },
            timing.clone(),
            TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        assert!(matches!(
            domain
                .repair(&[0.25, 0.75], &mut cooperation)
                .await
                .unwrap(),
            Repair::Unchanged
        ));
        let geometry = super::domain::BoxGeometry::new(&domain, &mut cooperation)
            .await
            .unwrap();
        let view = geometry.view(&domain).unwrap();
        let super::affine::Coordinates::Axes(axes) =
            super::affine::coordinates(&view, &mut cooperation)
                .await
                .unwrap()
        else {
            panic!("one independent coordinate on the line x1+x2=1");
        };
        assert_eq!(axes.len(), 1);
        assert!(axes[0] < 2);
        let Some(span) = super::affine::span_in(&view, &view, &mut cooperation)
            .await
            .unwrap()
        else {
            panic!("actual equality nullspace requires a thin original QR span");
        };
        let frame = super::frame::Frame::new(
            super::frame::Kind::RandomOrthogonal,
            2,
            Some(&span),
            &mut cooperation,
        )
        .await
        .unwrap()
        .unwrap();
        let mut direction = [0.0; 2];
        for index in 0..frame.count() {
            frame
                .direction_into(index, &mut direction, &mut cooperation)
                .await
                .unwrap();
            assert!((direction[0] + direction[1]).abs() < 1e-12);
            assert!((direction[0].hypot(direction[1]) - 1.0).abs() < 1e-12);
        }
        assert_eq!(domain.repair_nodes(), 0);
        let sampler = super::gss::Geometry::new(&view, &mut cooperation)
            .await
            .unwrap();
        let cone = sampler
            .tangent(&view, &[0.25, 0.75], 0.2, &mut cooperation)
            .await
            .unwrap();
        let ray = cone
            .oriented_direction(
                0,
                crate::builtins::math::optim::common::feasible_cone::Basis::Maximal,
                false,
                &mut cooperation,
            )
            .await
            .unwrap()
            .unwrap();
        let super::gss::Proposal::Point { point, .. } = sampler
            .candidate(&view, &[0.25, 0.75], &ray, 0.2, 1e-8, &mut cooperation)
            .await
            .unwrap()
        else {
            panic!("a nonzero feasible ray exists on the original equality");
        };
        assert_ne!(point, [0.25, 0.75]);
        assert!(
            domain
                .classify(&point, 1e-8, &mut cooperation)
                .await
                .unwrap()
                .feasible
        );

        let mut domain = Domain::new(
            Input {
                matrix: CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]),
                rhs: vec![0.25, 0.75],
                equalities: 2,
                lower: vec![0.0, 0.0],
                upper: vec![1.0, 1.0],
                integers: vec![],
            },
            timing.clone(),
            TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        assert!(matches!(
            domain
                .repair(&[0.25, 0.75], &mut cooperation)
                .await
                .unwrap(),
            Repair::Unchanged
        ));
        let geometry = super::domain::BoxGeometry::new(&domain, &mut cooperation)
            .await
            .unwrap();
        let view = geometry.view(&domain).unwrap();
        assert!(matches!(
            super::affine::coordinates(&view, &mut cooperation)
                .await
                .unwrap(),
            super::affine::Coordinates::Unique
        ));
        let mut integer = Domain::new(
            Input {
                matrix: CscMatrix::new(0, 2, vec![0, 0, 0], vec![], vec![]),
                rhs: vec![],
                equalities: 0,
                lower: vec![0.1, 2.0],
                upper: vec![1.5, 2.0],
                integers: vec![0],
            },
            timing,
            TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        let shape = super::domain::BoxGeometry::new(&integer, &mut cooperation)
            .await
            .unwrap();
        let view = shape.view(&integer).unwrap();
        assert_eq!(view.dimensions(), 0);
        let mut point = [0.0; 2];
        view.denormalize_into(&[], &mut point, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(point, [1.0, 2.0]);
        assert!(
            integer
                .classify(&point, 1e-8, &mut cooperation)
                .await
                .unwrap()
                .feasible
        );
        assert_eq!(domain.repair_nodes(), 0);
    });
}
